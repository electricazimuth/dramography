#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-End Book Processing Pipeline (Multi-File Support + Rolling Context)
Stages:
1. Summaries & Extraction (Iterative with Rolling Context)
2. Grammar/Entity Consolidation (Hybrid: Programmatic Merge + LLM Refinement)
3. Detail Extraction (Iterative)
4. God View & Swim Lane Generation (Iterative, Dual Prompt, JSON Output)
5. Combine God View & Swim Lane JSON Outputs

Usage:
    python run_pipeline_steps.py config.yaml [--step STEP]
"""

import os
import sys
import time
import signal
import subprocess
import argparse
import json
import yaml
import requests
import re
import logging
import difflib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Tuple, Set

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class JSONParser:
    """Helper to extract and validate JSON from LLM text."""
    
    @staticmethod
    def clean_and_validate(text: str) -> Optional[str]:
        extracted_text = text.strip()

        # Strategy 1: Look for markdown code blocks
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()

        # Strategy 2: If finding simple brackets helps
        if not (extracted_text.startswith('{') or extracted_text.startswith('[')):
            start_idx = extracted_text.find('{')
            end_idx = extracted_text.rfind('}')
            if start_idx != -1 and end_idx != -1:
                extracted_text = extracted_text[start_idx : end_idx + 1]

        try:
            data = json.loads(extracted_text)
            if not isinstance(data, (dict, list)) or not data:
                return None
            return json.dumps(data, indent=2)
        except Exception:
            return None

class PipelineConfig:
    """Parses and holds configuration data."""
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r') as f:
            self.data = yaml.safe_load(f)

        # Server
        self.host = self.data['server'].get('host', '127.0.0.1')
        self.port = self.data['server'].get('port', 5000)
        self.api_url = f"http://{self.host}:{self.port}/v1/chat/completions"
        self.health_url = f"http://{self.host}:{self.port}/health"
        self.startup_wait = self.data['server'].get('startup_wait', 300)
        self.cooldown_wait = self.data['server'].get('cooldown_wait', 5)
        self.api_timeout = self.data['server'].get('api_timeout', 600)
        self.max_retries = self.data['server'].get('max_retries', 3)

        # Backend
        self.backend_type = self.data['backend'].get('type', 'llamacpp')
        self.server_bin = self.data['backend'].get('llama_cpp_server_path')
        
        raw_args = self.data['backend'].get('llama_cpp_args', [])
        self.base_args = self._clean_base_args(raw_args)

        # Paths
        self.model_base_dir = Path(self.data['paths']['model_dir'])
        self.results_dir = Path(self.data['paths']['results_dir'])
        
        # Files Processing Logic
        processing_cfg = self.data['processing']
        self.txt_files: List[Path] = []
        
        if 'txt_files' in processing_cfg and isinstance(processing_cfg['txt_files'], list):
            self.txt_files = [Path(f) for f in processing_cfg['txt_files']]
        elif 'txt_file' in processing_cfg:
            self.txt_files = [Path(processing_cfg['txt_file'])]
        
        if not self.txt_files:
            raise ValueError("No text files defined in config under processing.txt_files")

        # Processing Settings
        self.split_string = processing_cfg.get('chapter_split_string', "=== section break ===")
        self.force_regenerate = processing_cfg.get('force_regenerate', False)
        self.prefix = processing_cfg.get('prefix', '')
        self.context_token_limit = processing_cfg.get('context_token_limit', 2000)
        
        self.steps = self.data.get('steps', {})

    def _clean_base_args(self, args: List[str]) -> List[str]:
        cleaned = []
        skip_next = False
        for arg in args:
            if skip_next:
                skip_next = False
                continue
            if arg in ['--ctx-size', '-c']:
                skip_next = True
                continue
            cleaned.append(arg)
        return cleaned

    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        return self.steps.get(step_name, {})

class LlamaServer:
    """Manages the lifecycle of the Llama.cpp server process."""
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.process = None
        self.current_model_path = None
        self.current_ctx_size = 0

    def start(self, model_filename: str, ctx_size: int = 4096):
        full_model_path = self.config.model_base_dir / model_filename
        
        if not full_model_path.exists():
            raise FileNotFoundError(f"Model file not found: {full_model_path}")

        if (self.process and 
            self.current_model_path == full_model_path and 
            self.current_ctx_size == ctx_size):
            if self.is_healthy():
                return
            else:
                self.stop()

        self.stop()
        logger.info(f"Starting server with model: {model_filename} | Context Size: {ctx_size}")

        cmd = [
            str(self.config.server_bin),
            "--model", str(full_model_path),
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--ctx-size", str(ctx_size)
        ] + self.config.base_args

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace' 
        )
        
        self.current_model_path = full_model_path
        self.current_ctx_size = ctx_size
        self._wait_for_startup()

    def stop(self):
        if self.process:
            logger.info("Stopping LLM server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None
            self.current_model_path = None
            self.current_ctx_size = 0
            time.sleep(self.config.cooldown_wait)

    def is_healthy(self) -> bool:
        try:
            resp = requests.get(self.config.health_url, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                return data.get('status') in ['ok', 'loading']
        except Exception:
            return False
        return False

    def _wait_for_startup(self):
        start = time.time()
        while time.time() - start < self.config.startup_wait:
            if self.process.poll() is not None:
                _, stderr = self.process.communicate()
                raise RuntimeError(f"Server crashed on startup.\nSTDERR: {stderr}")
            if self.is_healthy():
                return
            time.sleep(2)
        raise TimeoutError("Server failed to start within the allocated time.")

class LLMClient:
    """Handles prompt generation requests with Retries and Validation."""
    def __init__(self, config: PipelineConfig):
        self.config = config

    def generate(self, prompt: str, step_config: Dict[str, Any], debug_context: str = "unknown", 
                 save_dir: Path = None, is_json: bool = True) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": step_config.get('max_tokens', 2048),
            "temperature": step_config.get('temperature', 0.8),
            "top_k": step_config.get('top_k', 64),
            "top_p": step_config.get('top_p', 0.95),
            "seed": step_config.get('seed', -1)
        }

        attempts = 0
        max_attempts = self.config.max_retries + 1

        while attempts < max_attempts:
            attempts += 1
            logger.info(f"Generation Attempt {attempts}/{max_attempts} for {debug_context}")

            try:
                response = requests.post(
                    self.config.api_url, 
                    json=payload, 
                    timeout=self.config.api_timeout
                )
                response.raise_for_status()
                
                result = response.json()
                choice = result['choices'][0]
                raw_content = choice['message']['content']
                finish_reason = choice.get('finish_reason', 'unknown')

                if is_json:
                    valid_json_str = JSONParser.clean_and_validate(raw_content)
                    if valid_json_str:
                        return valid_json_str
                    logger.warning(f"Attempt produced invalid JSON.")
                else:
                    if raw_content and len(raw_content.strip()) > 0:
                        return raw_content
                    logger.warning("Received empty text response.")

                if finish_reason == 'length':
                    logger.error("DIAGNOSTIC: Generation stopped because max_tokens was reached.")

            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error: {e}")
            
            if attempts < max_attempts:
                time.sleep(3)
        
        raise RuntimeError(f"Failed to generate valid output after {max_attempts} attempts.")

class EntityProcessor:
    """Handles programmatic merging and cleaning of entities before LLM processing."""
    def __init__(self, fuzz_threshold=0.9):
        self.threshold = fuzz_threshold

    def is_match(self, name1: str, name2: str) -> bool:
        if not name1 or not name2: return False
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        if n1 == n2 or n1 in n2 or n2 in n1:
            return True
        matcher = difflib.SequenceMatcher(None, n1, n2)
        return matcher.ratio() > self.threshold

    def merge_lists(self, existing_list: List[str], new_list: List[str]) -> List[str]:
        result = list(existing_list)
        lower_map = {x.lower(): x for x in existing_list}
        for item in new_list:
            if item.lower() not in lower_map:
                result.append(item)
                lower_map[item.lower()] = item
        return result

    def process_entities(self, all_chapters_data: List[dict]) -> dict:
        combined = {"characters": [], "objects": [], "locations": []}
        for chapter in all_chapters_data:
            chk_entities = chapter.get("entities", {})
            if not isinstance(chk_entities, dict): continue
            for category in ["characters", "objects", "locations"]:
                if category not in chk_entities: continue
                for new_item in chk_entities[category]:
                    self._merge_single_item(combined[category], new_item, category)
        return combined

    def _merge_single_item(self, target_list: List[dict], new_item: dict, category: str):
        new_name = new_item.get("primary_name", "")
        if not new_name: return

        discard_keys = ["relationships", "chapter_summary"]
        if category == "characters": discard_keys.append("status_action")
        elif category == "locations": discard_keys.extend(["description", "events"])
        
        clean_item = {k: v for k, v in new_item.items() if k not in discard_keys}
        new_aliases = clean_item.get("aliases", [])
        
        match_found = None
        for existing in target_list:
            if self.is_match(new_name, existing.get("primary_name", "")):
                match_found = existing
                break

        if match_found:
            match_found["aliases"] = self.merge_lists(match_found.get("aliases", []), new_aliases)
        else:
            target_list.append(clean_item)

# --- I/O Helpers ---

def load_book_chapters(config: PipelineConfig, file_path: Path) -> List[str]:
    if not file_path.exists():
        raise FileNotFoundError(f"Book file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    chapters = content.split(config.split_string)
    chapters = [c.strip() for c in chapters if c.strip()]
    return chapters

def save_result(directory: Path, filename: str, content: str):
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / filename, 'w', encoding='utf-8') as f:
        f.write(content)

def load_result(directory: Path, filename: str) -> str:
    path = directory / filename
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def create_rolling_context(previous_outputs: List[str], max_tokens: int) -> str:
    max_chars = max_tokens * 4
    current_chars = 0
    context_blocks = []
    
    for i, json_str in enumerate(reversed(previous_outputs)):
        original_index = len(previous_outputs) - 1 - i
        chapter_num = original_index + 1
        
        try:
            data = json.loads(json_str)
            summary = data.get('chapter_summary', '')
            entities = data.get('entities', {})
            
            def get_names(key):
                items = entities.get(key, [])
                return [x.get('primary_name') for x in items if x.get('primary_name')]

            all_names = get_names('characters') + get_names('objects') + get_names('locations')
            all_names = list(dict.fromkeys(all_names))
            entity_str = ", ".join(all_names)

            block = f"## Chapter {chapter_num} Summary\n{summary}\n"
            if entity_str: block += f"Entities involved: {entity_str}\n"
            
            block_len = len(block)
            if current_chars + block_len > max_chars:
                break
            
            context_blocks.append(block)
            current_chars += block_len
        except Exception:
            continue

    return "\n".join(reversed(context_blocks))

# --- Step Specific Loaders for Later Steps ---

def get_master_grammar(config: PipelineConfig, current_results_dir: Path) -> str:
    path = current_results_dir / "step2_grammar" / f"{config.prefix}master_grammar.json"
    if not path.exists():
        raise FileNotFoundError(f"Master grammar file missing: {path}. Run Step 2 first.")
    with open(path, 'r', encoding='utf-8') as f:
        return json.dumps(json.load(f), indent=2)

def get_step1_summary(config: PipelineConfig, current_results_dir: Path, chapter_num: int) -> str:
    path = current_results_dir / "step1_summaries" / f"{config.prefix}chapter_{chapter_num:03d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Step 1 Summary missing for Ch {chapter_num}: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f).get("chapter_summary", "Summary not found.")

def get_step3_detail(config: PipelineConfig, current_results_dir: Path, chapter_num: int) -> str:
    path = current_results_dir / "step3_detail" / f"{config.prefix}chapter_{chapter_num:03d}_details.json"
    if not path.exists():
        raise FileNotFoundError(f"Step 3 Detail missing for Ch {chapter_num}: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.dumps(json.load(f), indent=2)

# --- Pipeline Steps ---

def run_step1_summaries(config: PipelineConfig, server: LlamaServer, client: LLMClient, 
                        chapters: List[str], current_results_dir: Path):
    step_name = 'step1_summaries'
    step_cfg = config.get_step_config(step_name)
    output_dir = current_results_dir / "step1_summaries"
    ctx_size = step_cfg.get('ctx_size', 8192)

    logger.info(f"=== Starting Step 1: Summaries ({len(chapters)} chapters) [Ctx: {ctx_size}] ===")
    server.start(step_cfg['llm_file'], ctx_size=ctx_size)
    
    generated_outputs = []
    for i, chapter_text in enumerate(chapters):
        chapter_num = i + 1
        filename = f"{config.prefix}chapter_{chapter_num:03d}.json"
        
        if not config.force_regenerate and (output_dir / filename).exists():
            content = load_result(output_dir, filename)
            if JSONParser.clean_and_validate(content):
                logger.info(f"Skipping Chapter {chapter_num} (Exists & Valid)")
                generated_outputs.append(content)
                continue

        logger.info(f"Processing Chapter {chapter_num}...")
        previous_context = create_rolling_context(generated_outputs, config.context_token_limit) if generated_outputs else "No previous chapters."
        
        prompt = step_cfg['prompt'].replace("{{previous}}", previous_context).replace("{{chapter_number}}", str(chapter_num)).replace("{{chapter}}", chapter_text)

        try:
            output = client.generate(prompt, step_cfg, debug_context=f"step1_ch{chapter_num}", save_dir=output_dir)
            save_result(output_dir, filename, output)
            generated_outputs.append(output)
        except Exception as e:
            logger.error(f"Critical Failure at Chapter {chapter_num}: {e}")
            sys.exit(1)
    return generated_outputs

def run_step2_grammar(config: PipelineConfig, server: LlamaServer, client: LLMClient, 
                      current_results_dir: Path, num_chapters: int):
    step_name = 'step2_grammar'
    step_cfg = config.get_step_config(step_name)
    output_dir = current_results_dir / "step2_grammar"
    filename = f"{config.prefix}master_grammar.json"
    ctx_size = step_cfg.get('ctx_size', 32768)

    logger.info(f"=== Starting Step 2: Grammar Consolidation [Ctx: {ctx_size}] ===")

    if not config.force_regenerate and (output_dir / filename).exists():
        content = load_result(output_dir, filename)
        if JSONParser.clean_and_validate(content):
            logger.info("Skipping Step 2 (Exists & Valid)")
            return content

    # Load Step 1 outputs from disk
    step1_objects = []
    simplified_summaries = []
    for i in range(1, num_chapters + 1):
        path = current_results_dir / "step1_summaries" / f"{config.prefix}chapter_{i:03d}.json"
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
            step1_objects.append(d)
            simplified_summaries.append(d.get('chapter_summary', ''))
            
    combined_summaries = "\n\n".join(simplified_summaries)

    processor = EntityProcessor(fuzz_threshold=0.9)
    merged_entities = processor.process_entities(step1_objects)
    merged_entities_str = json.dumps(merged_entities, indent=2)
    save_result(output_dir, f"{config.prefix}programmatic_entity_merge.json", merged_entities_str)

    server.start(step_cfg['llm_file'], ctx_size=ctx_size)

    prompt = step_cfg['prompt'].replace("{{chapters}}", combined_summaries)
    if "{{entities}}" in prompt:
        prompt = prompt.replace("{{entities}}", merged_entities_str)
    else:
        prompt += f"\n\nHere are the extracted and pre-merged entities:\n{merged_entities_str}\n"

    try:
        output = client.generate(prompt, step_cfg, debug_context="step2_grammar", save_dir=output_dir)
        save_result(output_dir, filename, output)
        return output
    except Exception as e:
        logger.error(f"Critical Failure at Grammar Generation: {e}")
        sys.exit(1)

def run_step3_detail(config: PipelineConfig, server: LlamaServer, client: LLMClient, 
                     chapters: List[str], current_results_dir: Path):
    step_name = 'step3_detail'
    step_cfg = config.get_step_config(step_name)
    output_dir = current_results_dir / "step3_detail"
    ctx_size = step_cfg.get('ctx_size', 8192)

    logger.info(f"=== Starting Step 3: Detailed Extraction ({len(chapters)} chapters) [Ctx: {ctx_size}] ===")

    grammar_output = get_master_grammar(config, current_results_dir)
    server.start(step_cfg['llm_file'], ctx_size=ctx_size)

    # Load previously generated Step 1 files for context
    step1_outputs = []
    for i in range(1, len(chapters) + 1):
        path = current_results_dir / "step1_summaries" / f"{config.prefix}chapter_{i:03d}.json"
        with open(path, 'r', encoding='utf-8') as f:
            step1_outputs.append(f.read())

    for i, chapter_text in enumerate(chapters):
        chapter_num = i + 1
        filename = f"{config.prefix}chapter_{chapter_num:03d}_details.json"

        if not config.force_regenerate and (output_dir / filename).exists():
             if JSONParser.clean_and_validate(load_result(output_dir, filename)):
                logger.info(f"Skipping Chapter {chapter_num} (Exists & Valid)")
                continue

        logger.info(f"Processing Chapter {chapter_num} details...")
        previous_context = create_rolling_context(step1_outputs[:i], config.context_token_limit) if i > 0 else "No previous chapters."

        prompt = step_cfg['prompt'].replace("{{grammar}}", grammar_output).replace("{{previous}}", previous_context).replace("{{chapter_number}}", str(chapter_num)).replace("{{chapter}}", chapter_text)

        try:
            output = client.generate(prompt, step_cfg, debug_context=f"step3_ch{chapter_num}", save_dir=output_dir)
            save_result(output_dir, filename, output)
        except Exception as e:
            logger.error(f"Critical Failure at Chapter {chapter_num} details: {e}")
            sys.exit(1)

def run_step4_godview(config: PipelineConfig, server: LlamaServer, client: LLMClient, 
                      num_chapters: int, current_results_dir: Path):
    """
    Step 4 now runs two separate prompts per chapter:
    1. God View Prompt -> saves as _godview.json
    2. Swim Lane Prompt -> saves as _swimlane.json
    """
    step_name = 'step4_godview'
    step_cfg = config.get_step_config(step_name)
    output_dir = current_results_dir / "step4_godview"
    ctx_size = step_cfg.get('ctx_size', 4096)
    
    # Retrieve prompts from config
    prompt_godview_template = step_cfg.get('godview_prompt')
    prompt_swimlane_template = step_cfg.get('swimlane_prompt')
    
    if not prompt_godview_template and not prompt_swimlane_template:
        logger.error("Step 4 Config Error: Neither 'godview_prompt' nor 'swimlane_prompt' found in step4_godview config.")
        sys.exit(1)

    logger.info(f"=== Starting Step 4: God View & Swim Lane ({num_chapters} chapters) [Ctx: {ctx_size}] ===")

    grammar_json_str = get_master_grammar(config, current_results_dir)
    server.start(step_cfg['llm_file'], ctx_size=ctx_size)

    for i in range(num_chapters):
        chapter_num = i + 1
        
        # Define output filenames
        file_god = f"{config.prefix}chapter_{chapter_num:03d}_godview.json"
        file_swim = f"{config.prefix}chapter_{chapter_num:03d}_swimlane.json"
        
        logger.info(f"Processing Step 4 for Chapter {chapter_num}...")

        try:
            summary_str = get_step1_summary(config, current_results_dir, chapter_num)
            detail_json_str = get_step3_detail(config, current_results_dir, chapter_num)

            # --- Sub-step A: God View ---
            if prompt_godview_template:
                if not config.force_regenerate and (output_dir / file_god).exists() and JSONParser.clean_and_validate(load_result(output_dir, file_god)):
                    logger.info(f"  > God View Skipped (Exists & Valid)")
                else:
                    logger.info(f"  > Generating God View...")
                    prompt = prompt_godview_template\
                        .replace("{{grammar}}", grammar_json_str)\
                        .replace("{{chapter_number}}", str(chapter_num))\
                        .replace("{{summary}}", summary_str)\
                        .replace("{{detail}}", detail_json_str)
                    
                    output = client.generate(prompt, step_cfg, debug_context=f"step4_god_ch{chapter_num}", save_dir=output_dir, is_json=True)
                    save_result(output_dir, file_god, output)

            # --- Sub-step B: Swim Lane ---
            if prompt_swimlane_template:
                if not config.force_regenerate and (output_dir / file_swim).exists() and JSONParser.clean_and_validate(load_result(output_dir, file_swim)):
                    logger.info(f"  > Swim Lane Skipped (Exists & Valid)")
                else:
                    logger.info(f"  > Generating Swim Lane...")
                    prompt = prompt_swimlane_template\
                        .replace("{{grammar}}", grammar_json_str)\
                        .replace("{{chapter_number}}", str(chapter_num))\
                        .replace("{{summary}}", summary_str)\
                        .replace("{{detail}}", detail_json_str)
                    
                    output = client.generate(prompt, step_cfg, debug_context=f"step4_swim_ch{chapter_num}", save_dir=output_dir, is_json=True)
                    save_result(output_dir, file_swim, output)

        except Exception as e:
            logger.error(f"Failure at Chapter {chapter_num}: {e}")
            sys.exit(1)

# --- Step 5 Combining Logic ---

def run_step5_combine(config: PipelineConfig, num_chapters: int, current_results_dir: Path):
    logger.info(f"=== Starting Step 5: Combining JSON Outputs ===")
    
    input_dir = current_results_dir / "step4_godview"
    output_dir = current_results_dir / "step5_combine"
    
    # Containers for God View Combination
    combined_god_scenes = []
    
    # Containers for Swim Lane Combination
    combined_swim_timeline = []
    all_tracked_chars = set()
    all_tracked_objs = set()
    
    source_book_id = "unknown"
    chapters_processed = []

    timestamp_now = datetime.now().isoformat()

    for i in range(num_chapters):
        chapter_num = i + 1
        file_god = f"{config.prefix}chapter_{chapter_num:03d}_godview.json"
        file_swim = f"{config.prefix}chapter_{chapter_num:03d}_swimlane.json"
        
        path_god = input_dir / file_god
        path_swim = input_dir / file_swim
        
        chapters_processed.append(chapter_num)
        
        # --- Process God View JSON ---
        if path_god.exists():
            try:
                with open(path_god, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Update metadata reference
                    if i == 0: source_book_id = data.get("metadata", {}).get("source_book", "unknown")
                    
                    # Extend scenes
                    scenes = data.get("scenes", [])
                    if isinstance(scenes, list):
                        combined_god_scenes.extend(scenes)
                    else:
                        logger.warning(f"Ch {chapter_num} God View 'scenes' is not a list.")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON for God View Ch {chapter_num}")
        else:
            logger.warning(f"Missing God View output for Chapter {chapter_num}")

        # --- Process Swim Lane JSON ---
        if path_swim.exists():
            try:
                with open(path_swim, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Merge tracked items
                    meta = data.get("metadata", {})
                    all_tracked_chars.update(meta.get("tracked_characters", []))
                    all_tracked_objs.update(meta.get("tracked_objects", []))
                    
                    # Extend timeline
                    timeline = data.get("timeline", [])
                    if isinstance(timeline, list):
                        combined_swim_timeline.extend(timeline)
                    else:
                        logger.warning(f"Ch {chapter_num} Swim Lane 'timeline' is not a list.")
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON for Swim Lane Ch {chapter_num}")
        else:
            logger.warning(f"Missing Swim Lane output for Chapter {chapter_num}")

    # --- Construct Final Objects ---

    # 1. Final God View JSON
    final_god_view = {
        "metadata": {
            "chart_type": "god_view",
            "chapters": chapters_processed,
            "source_book": source_book_id,
            "generated_timestamp": timestamp_now
        },
        "scenes": combined_god_scenes
    }

    # 2. Final Swim Lane JSON
    final_swim_lane = {
        "metadata": {
            "chart_type": "swim_lane",
            "chapters": chapters_processed,
            "source_book": source_book_id,
            "generated_timestamp": timestamp_now,
            "tracked_characters": sorted(list(all_tracked_chars)),
            "tracked_objects": sorted(list(all_tracked_objs))
        },
        "timeline": combined_swim_timeline
    }

    # --- Save ---
    
    if combined_god_scenes:
        save_result(output_dir, "god_view_combined.json", json.dumps(final_god_view, indent=2))
        logger.info(f"Saved combined God View JSON ({len(combined_god_scenes)} scenes).")
    
    if combined_swim_timeline:
        save_result(output_dir, "swim_lane_combined.json", json.dumps(final_swim_lane, indent=2))
        logger.info(f"Saved combined Swim Lane JSON ({len(combined_swim_timeline)} timeline events).")

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Multi-Step Book Pipeline")
    parser.add_argument("config_file", help="Path to YAML configuration")
    parser.add_argument("--step", type=str, default="all", choices=["1", "2", "3", "4", "5", "all"],
                        help="Specify a single step to run, or 'all' (default).")
    args = parser.parse_args()

    try:
        config = PipelineConfig(args.config_file)
    except Exception as e:
        logger.error(f"Initialization Error: {e}")
        sys.exit(1)

    server = LlamaServer(config)
    client = LLMClient(config)

    def signal_handler(sig, frame):
        logger.info("Interrupt received, stopping server...")
        server.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        run_all = args.step == "all"
        target_step = args.step

        for txt_file in config.txt_files:
            book_stem = txt_file.stem
            current_results_dir = config.results_dir / book_stem
            chapters = load_book_chapters(config, txt_file)
            num_chapters = len(chapters)
            
            logger.info(f"--------------------------------------------------")
            logger.info(f"PROCESSING FILE: {txt_file} ({num_chapters} chapters)")
            logger.info(f"--------------------------------------------------")

            if run_all or target_step == "1":
                run_step1_summaries(config, server, client, chapters, current_results_dir)

            if run_all or target_step <= "2":
                run_step2_grammar(config, server, client, current_results_dir, num_chapters)

            if run_all or target_step <= "3":
                run_step3_detail(config, server, client, chapters, current_results_dir)
            
            if run_all or target_step <= "4":
                run_step4_godview(config, server, client, num_chapters, current_results_dir)

            if run_all or target_step <= "5":
                # Step 5 requires no LLM, just file parsing.
                run_step5_combine(config, num_chapters, current_results_dir)

            logger.info(f"Finished processing {txt_file.name}")

        logger.info("All Pipeline Tasks Complete!")

    except Exception as e:
        logger.error(f"Pipeline Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server.stop()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2 Experimentation Pipeline
Focus: Comparing standard workflow vs Programmatic Entity Merging for Grammar creation.
Supports iterating through multiple LLMs defined in config.

Usage:
    python run_step2_experiment.py config.yaml
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
from typing import Optional, List, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Core Infrastructure Classes (Reused) ---

class PipelineConfig:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.data = yaml.safe_load(f)

        # Server
        self.host = self.data['server'].get('host', '127.0.0.1')
        self.port = self.data['server'].get('port', 5000)
        self.api_url = f"http://{self.host}:{self.port}/v1/chat/completions"
        self.models_url = f"http://{self.host}:{self.port}/v1/models"
        self.health_url = f"http://{self.host}:{self.port}/health"
        self.startup_wait = self.data['server'].get('startup_wait', 300)
        self.cooldown_wait = self.data['server'].get('cooldown_wait', 5)
        self.api_timeout = self.data['server'].get('api_timeout', 600)
        self.max_retries = self.data['server'].get('max_retries', 3)

        # Backend
        self.server_bin = self.data['backend'].get('llama_cpp_server_path')
        raw_args = self.data['backend'].get('llama_cpp_args', [])
        self.base_args = self._clean_base_args(raw_args)

        # Paths
        self.model_base_dir = Path(self.data['paths']['model_dir'])
        self.results_dir = Path(self.data['paths']['results_dir'])
        self.file_prefix = self.data['processing'].get('prefix', '')

        # Files
        processing_cfg = self.data['processing']
        if 'txt_files' in processing_cfg and isinstance(processing_cfg['txt_files'], list):
            self.txt_files = [Path(f) for f in processing_cfg['txt_files']]
        elif 'txt_file' in processing_cfg:
            self.txt_files = [Path(processing_cfg['txt_file'])]
        else:
            raise ValueError("No text files defined in config")

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
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.process = None
        self.current_model_path = None
        self.current_ctx_size = 0

    def start(self, model_filename: str, ctx_size: int):
        full_model_path = self.config.model_base_dir / model_filename
        if not full_model_path.exists():
            raise FileNotFoundError(f"Model file not found: {full_model_path}")

        # Check if already running the correct config
        if (self.process and self.current_model_path == full_model_path and 
            self.current_ctx_size == ctx_size):
            if self.is_healthy(): return
            else: self.stop()

        self.stop()
        logger.info(f"Starting server: {model_filename} | Ctx: {ctx_size}")

        cmd = [
            str(self.config.server_bin),
            "--model", str(full_model_path),
            "--host", self.config.host,
            "--port", str(self.config.port),
            "--ctx-size", str(ctx_size)
        ] + self.config.base_args

        self.process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', errors='replace'
        )
        self.current_model_path = full_model_path
        self.current_ctx_size = ctx_size
        self._wait_for_startup()

    def stop(self):
        if self.process:
            logger.info("Stopping server...")
            self.process.terminate()
            try: self.process.wait(timeout=10)
            except: self.process.kill()
            self.process = None
            time.sleep(self.config.cooldown_wait)

    def is_healthy(self) -> bool:
        try:
            resp = requests.get(self.config.health_url, timeout=2)
            return resp.status_code == 200 and resp.json().get('status') in ['ok', 'loading']
        except: return False

    def _wait_for_startup(self):
        start = time.time()
        while time.time() - start < self.config.startup_wait:
            if self.process.poll() is not None:
                _, stderr = self.process.communicate()
                raise RuntimeError(f"Server crashed: {stderr}")
            if self.is_healthy(): return
            time.sleep(2)
        raise TimeoutError("Server start timeout")

class LLMClient:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def get_model_name(self) -> str:
        try:
            resp = requests.get(self.config.models_url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if 'data' in data and len(data['data']) > 0:
                    model_id = data['data'][0]['id']
                    return Path(model_id).stem
        except Exception as e:
            logger.warning(f"Could not fetch model name: {e}")
        return "unknown_model"

    def generate(self, prompt: str, step_config: Dict[str, Any]) -> str:
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": step_config.get('max_tokens', 4096),
            "temperature": step_config.get('temperature', 0.8),
            "top_k": step_config.get('top_k', 64),
            "top_p": step_config.get('top_p', 0.95),
            "seed": step_config.get('seed', -1)
        }
        
        try:
            response = requests.post(self.config.api_url, json=payload, timeout=self.config.api_timeout)
            response.raise_for_status()
            content = response.json()['choices'][0]['message']['content']
            
            # Simple validation to ensure we got something
            if not content.strip():
                raise ValueError("Empty response received")
            return content

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

# --- New Logic: Entity Processing & Merging ---
class EntityProcessor:
    def __init__(self, fuzz_threshold=0.9):
        self.threshold = fuzz_threshold

    def is_match(self, name1: str, name2: str) -> bool:
        """Checks if two names are a fuzzy match."""
        if not name1 or not name2: return False
        n1 = name1.lower().strip()
        n2 = name2.lower().strip()
        
        # Exact substring match
        if n1 == n2 or n1 in n2 or n2 in n1:
            return True
            
        # Fuzzy match
        matcher = difflib.SequenceMatcher(None, n1, n2)
        return matcher.ratio() > self.threshold

    def merge_lists(self, existing_list: List[str], new_list: List[str]) -> List[str]:
        """Merges two lists of strings, deduping case-insensitively."""
        result = list(existing_list)
        lower_map = {x.lower(): x for x in existing_list}
        
        for item in new_list:
            if item.lower() not in lower_map:
                result.append(item)
                lower_map[item.lower()] = item
        return result

    def process_entities(self, all_chapters_data: List[dict]) -> dict:
        """
        Combines entities from all chapters into a single structure.
        Dedupes based on primary_name and aliases.
        """
        combined = {
            "characters": [],
            "objects": [],
            "locations": []
        }

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

        # --- Cleaning Logic ---
        # Define fields to explicitly discard based on category
        discard_keys = ["relationships", "chapter_summary"] # Global ignores
        
        if category == "characters":
            discard_keys.append("status_action")
        elif category == "locations":
            discard_keys.extend(["description", "events"])
        
        # Create a clean copy of the item without unwanted fields
        clean_item = {
            k: v for k, v in new_item.items() 
            if k not in discard_keys
        }

        new_aliases = clean_item.get("aliases", [])
        
        # --- Deduplication Logic ---
        match_found = None
        
        for existing in target_list:
            existing_name = existing.get("primary_name", "")
            existing_aliases = existing.get("aliases", [])
            
            # Check 1: Primary name match
            if self.is_match(new_name, existing_name):
                match_found = existing
                break
            
            # Check 2: Name matches an alias
            #for alias in existing_aliases:
            #    if self.is_match(new_name, alias):
            #        match_found = existing
            #        break
            
            # Check 3: Alias matches existing name
            #for new_alias in new_aliases:
            #    if self.is_match(new_alias, existing_name):
            #        match_found = existing
            #        break
            
            if match_found: break

        if match_found:
            # If matched, ONLY merge aliases.
            match_found["aliases"] = self.merge_lists(match_found.get("aliases", []), new_aliases)
        else:
            # If not matched, add the cleaned object
            target_list.append(clean_item)

# --- Helper Functions ---

def load_step1_outputs(results_dir: Path, prefix: str) -> List[dict]:
    """Loads all JSON files from the step1_summaries folder."""
    step1_dir = results_dir / "step1_summaries"
    if not step1_dir.exists():
        logger.error(f"Step 1 directory not found: {step1_dir}")
        return []

    data = []
    # Sort files to ensure chronological order
    files = sorted(step1_dir.glob(f"{prefix}chapter_*.json"))
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fh:
                content = json.load(fh)
                data.append(content)
        except Exception as e:
            logger.warning(f"Error loading {f.name}: {e}")
            
    logger.info(f"Loaded {len(data)} Step 1 JSON files from {step1_dir}")
    return data

def save_output(output_dir: Path, basename: str, content: str, time_taken: float, model_name: str, note: str = "", prompt:str = ""):
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{basename}.txt"
    path = output_dir / filename
    
    header = f"Model: {model_name}\n"
    header += f"Time Taken: {time_taken:.2f}s\n"
    header += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    header += f"Prompt Len: {len(prompt)} chars\n"
    if note:
        header += f"Note: {note}\n"
    header += "=" * 50 + "\n\n"
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + content)
    
    logger.info(f"Saved output to: {path}")

# --- Main Logic ---

def run_experiment(config: PipelineConfig, server: LlamaServer, client: LLMClient):
    step_cfg = config.get_step_config('step2_grammar')
    processor = EntityProcessor(fuzz_threshold=0.9)
    
    # Create Timestamped Output Folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = config.results_dir / f"experiment_step2_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved to: {experiment_dir}")

    # Determine Models to Run
    model_list = step_cfg.get('llm_files', [])
    # Fallback to single file if defined
    if not model_list and 'llm_file' in step_cfg:
        model_list = [step_cfg['llm_file']]
    
    if not model_list:
        logger.error("No LLM files found in configuration (step2_grammar).")
        return

    ctx_size = step_cfg.get('ctx_size', 32768)

    # --- LOOP THROUGH MODELS ---
    for model_file in model_list:
        logger.info(f"==========================================")
        logger.info(f"STARTING MODEL EXPERIMENT: {model_file}")
        logger.info(f"==========================================")

        try:
            # Start Server with specific model
            server.start(model_file, ctx_size=ctx_size)
            
            # Get a clean short name for filenames (remove .gguf extension)
            model_stem = Path(model_file).stem
            
            # Iterate through books
            for txt_file in config.txt_files:
                book_stem = txt_file.stem
                book_results_dir = config.results_dir / book_stem
                
                logger.info(f"--- Processing Book: {book_stem} with {model_stem} ---")
                
                # 1. Load Data
                step1_data = load_step1_outputs(book_results_dir, config.file_prefix)
                if not step1_data:
                    continue

                # 2. Prepare Summaries (Used in both tests)
                summaries_list = [d.get('chapter_summary', '') for d in step1_data]
                combined_summaries = "\n\n".join(summaries_list)

                # --- TEST A: BASELINE ---
                logger.info(f"Running Test A (Baseline) [{model_stem}]...")
                prompt_a = step_cfg['prompt'].replace("{{chapters}}", combined_summaries)
                
                start_time = time.time()
                try:
                    output_a = client.generate(prompt_a, step_cfg)
                    duration_a = time.time() - start_time
                    
                    # Construct filename with model name
                    file_a_name = f"{config.file_prefix}{book_stem}_{model_stem}_TEST_A"
                    save_output(experiment_dir, file_a_name, output_a, duration_a, model_file, "Baseline (Summaries Only)", prompt_a)
                except Exception as e:
                    logger.error(f"Test A Failed for {model_stem}: {e}")

                # --- TEST B: ENHANCED (Pre-merged Entities) ---
                if 'compare_prompt' not in step_cfg:
                    logger.warning("Skipping Test B: 'compare_prompt' not found in config")
                    continue

                logger.info(f"Running Test B (Entity Injection) [{model_stem}]...")
                
                # Merge Entities Programmatically (Determinisitc, but regenerating per loop is safer for clean scope)
                merged_entities = processor.process_entities(step1_data)
                merged_entities_json = json.dumps(merged_entities, indent=2)
                
                # Save the programmatic merge (This will be the same for all models, but we save it to reference)
                merge_filename = f"{config.file_prefix}{book_stem}_programmatic_merge.json"
                with open(experiment_dir / merge_filename, 'w', encoding='utf-8') as f:
                    f.write(merged_entities_json)

                # Prepare Prompt
                prompt_b = step_cfg['compare_prompt']
                prompt_b = prompt_b.replace("{{chapters}}", combined_summaries)
                prompt_b = prompt_b.replace("{{entities}}", merged_entities_json)

                start_time = time.time()
                try:
                    output_b = client.generate(prompt_b, step_cfg)
                    duration_b = time.time() - start_time
                    
                    # Construct filename with model name
                    file_b_name = f"{config.file_prefix}{book_stem}_{model_stem}_TEST_B"
                    save_output(experiment_dir, file_b_name, output_b, duration_b, model_file, "Enhanced (Programmatic Entity Merge)", prompt_b)
                except Exception as e:
                    logger.error(f"Test B Failed for {model_stem}: {e}")

        except Exception as e:
            logger.error(f"Critical error processing model {model_file}: {e}")
        finally:
            # Stop server after finishing this model (before starting the next)
            logger.info(f"Stopping server for {model_file}")
            server.stop()

def main():
    parser = argparse.ArgumentParser(description="Step 2 Comparison Experiment")
    parser.add_argument("config_file", help="Path to YAML configuration")
    args = parser.parse_args()

    try:
        config = PipelineConfig(args.config_file)
    except Exception as e:
        logger.error(f"Config Error: {e}")
        sys.exit(1)

    server = LlamaServer(config)
    client = LLMClient(config)

    def signal_handler(sig, frame):
        server.stop()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        run_experiment(config, server, client)
        logger.info("All Experiments Complete.")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server.stop()

if __name__ == "__main__":
    main()
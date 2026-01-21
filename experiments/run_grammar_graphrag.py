#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative Narrative Logic Extraction Script
Role: Loads a book, splits by chapter, and iteratively builds a JSON database 
      of characters, objects, and locations using a local LLM.

Features:
- Robust JSON parsing (handles Markdown & conversational filler)
- Iterative Context: Output of Chapter N becomes input for Chapter N+1
- Retry Logic: Re-runs generation if JSON cannot be parsed (up to max_retries)
- Backend: Supports Llama.cpp and KoboldCpp

Usage:
    python run_narrative_extraction.py config.yaml
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import re
import datetime
import json
import threading
import requests
import yaml
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

# --- Configuration Class ---

class Config:
    """Configuration container loaded from YAML."""
    
    def __init__(self, config_path: str):
        self.start_time = datetime.datetime.now()
        self._load_config(config_path)
    
    def _load_config(self, config_path: str):
        if not os.path.exists(config_path):
            print(f"Error: Config file '{config_path}' not found.")
            sys.exit(1)
        
        with open(config_path, 'r') as file:
            try:
                data = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")
                sys.exit(1)
        
        # Prompts
        self.prompts = data.get('prompts', {})
        if 'grammar' not in self.prompts:
            print("Error: config is missing the 'grammar' prompt.")
            sys.exit(1)

        # Server settings
        server = data.get('server', {})
        self.host = server.get('host', '127.0.0.1')
        self.port = server.get('port', 5000)
        self.startup_wait = server.get('startup_wait', 420)
        self.cooldown_wait = server.get('cooldown_wait', 5)
        self.api_timeout = server.get('api_timeout', 800)
        self.max_retries = server.get('max_retries', 2)
        
        # Backend settings
        backend = data.get('backend', {})
        self.backend_type = backend.get('type', 'llamacpp')
        self.llama_cpp_server_path = Path(backend.get('llama_cpp_server_path', '')).expanduser().resolve()
        self.koboldcpp_script_path = Path(backend.get('koboldcpp_script_path', '')).expanduser().resolve()
        self.llama_cpp_args = backend.get('llama_cpp_args', [])
        self.koboldcpp_args = backend.get('koboldcpp_args', [])
        
        # Paths
        paths = data.get('paths', {})
        self.model_dir = Path(paths.get('model_dir', '.')).expanduser().resolve()
        self.results_dir = Path(paths.get('results_dir', './results')).expanduser().resolve()
        
        # Model filtering
        model_filter = data.get('model_filter', {})
        min_gb = model_filter.get('min_size_gb')
        max_gb = model_filter.get('max_size_gb')
        self.min_size_bytes = int(min_gb * (1024**3)) if min_gb else None
        self.max_size_bytes = int(max_gb * (1024**3)) if max_gb else None
        
        # Processing settings
        processing = data.get('processing', {})
        self.txt_file = processing.get('txt_file', '')
        self.chapter_split_string = processing.get('chapter_split_string', '=== section break ===')
        self.force_regenerate = processing.get('force_regenerate', False)
        self.file_prefix = processing.get('prefix', '')
        
        # Generation settings
        generation = data.get('generation', {})
        self.max_tokens = generation.get('max_tokens', 8192)
        self.temperature = generation.get('temperature', 0.1) # Low temp for better JSON stability
        self.top_k = generation.get('top_k', 40)
        self.top_p = generation.get('top_p', 0.95)
        self.seed = generation.get('seed', -1)
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# --- Backend Logic ---

class LlamaCppBackend:
    """Manages the server process with robust error handling and streaming."""
    
    def __init__(self, config: Config):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        
        # API Endpoints
        self._api_chat_url = f"{self.config.base_url}/v1/chat/completions"
        self._api_health_url = f"{self.config.base_url}/health"
        
    def start_server(self, model_path: Path) -> bool:
        if self.process is not None:
            self.stop_server()
        
        cmd = []
        if self.config.backend_type == 'llamacpp':
            cmd = [str(self.config.llama_cpp_server_path)]
            cmd.extend(self.config.llama_cpp_args)
            cmd.extend([
                '--model', str(model_path),
                '--host', self.config.host,
                '--port', str(self.config.port),
            ])
        elif self.config.backend_type == 'koboldcpp':
            cmd = ['python3', str(self.config.koboldcpp_script_path)]
            cmd.extend(self.config.koboldcpp_args)
            cmd.extend([
                '--model', str(model_path),
                '--host', self.config.host,
                '--port', str(self.config.port),
            ])
        else:
            raise ValueError(f"Unknown backend type: {self.config.backend_type}")

        print(f"  Starting server with model: {model_path.name}")
        try:
            self.process = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
            )
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to start server: {e}")
            return False
    
    def stop_server(self):
        if self.process is not None:
            print("  Stopping server...")
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                pass
            finally:
                if self.process.stderr: self.process.stderr.close()
                self.process = None

    def wait_for_ready(self) -> bool:
        print(f"  Waiting up to {self.config.startup_wait}s for server...")
        start_time = time.time()
        while time.time() - start_time < self.config.startup_wait:
            try:
                resp = requests.get(self._api_health_url, timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("status") in ["ok", "loading"]:
                        print(f"  Server ready after {time.time() - start_time:.1f}s")
                        return True
            except:
                pass
            
            if self.process and self.process.poll() is not None:
                print(f"  [ERROR] Server process died prematurely.")
                return False
            time.sleep(2)
        return False

    def _stream_capture(self, response) -> Tuple[str, int, int]:
        """Reads the stream and accumulates content + token usage."""
        captured = ""
        p_tok, c_tok = 0, 0
        try:
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: ') and line_str.strip() != 'data: [DONE]':
                        try:
                            chunk = json.loads(line_str[6:])
                            if 'usage' in chunk and chunk['usage']:
                                p_tok = chunk['usage'].get('prompt_tokens', 0)
                                c_tok = chunk['usage'].get('completion_tokens', 0)
                            if 'choices' in chunk and chunk['choices']:
                                content = chunk['choices'][0].get('delta', {}).get('content', '')
                                if content: captured += content
                        except: continue
        except Exception: pass
        return captured, p_tok, c_tok

    def generate(self, messages: List[Dict]) -> Tuple[Optional[str], float, int, int]:
        payload = {
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "seed": self.config.seed,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        
        start = time.time()
        try:
            resp = requests.post(self._api_chat_url, json=payload, timeout=self.config.api_timeout, stream=True)
            if resp.status_code != 200: 
                print(f"  [API Error] Status Code: {resp.status_code}")
                return None, 0, 0, 0
            
            text, p_tok, c_tok = self._stream_capture(resp)
            return text, time.time() - start, p_tok, c_tok
        except Exception as e:
            print(f"  [Gen Error] {e}")
            return None, 0, 0, 0


# --- Data Utilities ---

def load_book_chapters(config: Config) -> List[str]:
    if not os.path.exists(config.txt_file):
        print(f"[ERROR] Book file not found: {config.txt_file}")
        sys.exit(1)
    
    try:
        with open(config.txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"[ERROR] Could not read file: {e}")
        sys.exit(1)
    
    # Split using the configured string
    chapters = content.split(config.chapter_split_string)
    
    # Clean empty chapters and strip whitespace
    chapters = [c.strip() for c in chapters if c.strip()]
    
    print(f"Loaded '{config.txt_file}'")
    print(f"Found {len(chapters)} chapters using split string: '{config.chapter_split_string}'")
    return chapters

def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Robustly extracts JSON from text, handling markdown blocks and conversational filler.
    Returns None if extraction fails.
    """
    if not text: return None

    # Strategy 1: Markdown code block regex
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass # Continue to next strategy

    # Strategy 2: Largest outer bracket matching
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = text[start_idx : end_idx + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return None

def get_empty_state() -> Dict:
    """Returns the initial empty structure."""
    return {
        "objects": [],
        "locations": [],
        "characters": []
    }

def discover_models(config: Config) -> List[Path]:
    if not config.model_dir.exists(): return []
    models = []
    for path in config.model_dir.glob('*.gguf'):
        if not path.is_file(): continue
        # Filter logic
        try:
            size = path.stat().st_size
            if config.max_size_bytes and size > config.max_size_bytes: continue
            if config.min_size_bytes and size < config.min_size_bytes: continue
            models.append(path)
        except: continue
    return sorted(models)

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Run iterative book analysis extraction.")
    parser.add_argument("config_file", help="Path to YAML configuration file")
    args = parser.parse_args()
    
    config = Config(args.config_file)
    models = discover_models(config)
    chapters = load_book_chapters(config)
    
    if not models:
        print("No models found matching criteria.")
        sys.exit(1)
    if not chapters:
        print("No chapters found. Check file path or split string.")
        sys.exit(1)

    backend = LlamaCppBackend(config)
    
    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n[INTERRUPT] Stopping server...")
        backend.stop_server()
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    for model_path in models:
        model_name = model_path.stem
        output_dir = config.results_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Running Model: {model_name}")
        print(f"{'='*60}")

        if not backend.start_server(model_path): continue
        if not backend.wait_for_ready(): 
            backend.stop_server()
            continue
        
        # --- Iteration State ---
        # We hold the JSON data in memory to pass to the next chapter.
        # Initialize with empty structure.
        current_data = get_empty_state()
        
        for i, chapter_text in enumerate(chapters):
            chapter_num = i + 1
            filename = f"{config.file_prefix}chapter_{chapter_num:03d}.json"
            file_path = output_dir / filename
            
            print(f"\n--- Processing Chapter {chapter_num}/{len(chapters)} ---")
            
            # Check if file exists to skip
            if file_path.exists() and not config.force_regenerate:
                print("  File exists. Loading state and skipping generation.")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        current_data = json.load(f)
                    continue
                except Exception as e:
                    print(f"  [WARN] Failed to load existing JSON: {e}. Regenerating.")

            # Prepare Context
            if i == 0:
                replacement_text = "No previous data. This is the first chapter."
            else:
                replacement_text = json.dumps(current_data, indent=2)

            # Construct Prompt
            prompt_text = config.prompts['grammar'] \
                .replace("{{replacement}}", replacement_text) \
                .replace("{{chapter}}", chapter_text)
            
            messages = [{"role": "user", "content": prompt_text}]
            
            # --- Retry Loop ---
            success = False
            
            # range is 0 to max_retries, so (max_retries + 1) attempts total
            for attempt in range(config.max_retries + 1):
                attempt_label = f"Attempt {attempt + 1}"
                print(f"  Generating ({attempt_label})... ", end="", flush=True)
                
                raw_response, duration, p_tok, c_tok = backend.generate(messages)
                
                if raw_response:
                    extracted_json = extract_json_from_text(raw_response)
                    
                    if extracted_json:
                        # Success: Valid JSON found
                        current_data = extracted_json
                        
                        # Save valid output
                        with open(file_path, 'w', encoding='utf-8') as f:
                            json.dump(current_data, f, indent=2)
                        
                        # Save raw debug output
                        debug_path = output_dir / f"{config.file_prefix}chapter_{chapter_num:03d}_RAW.txt"
                        with open(debug_path, 'w', encoding='utf-8') as f:
                            header = f"Time: {duration:.2f}s | In: {p_tok} | Out: {c_tok}\n{'-'*20}\n"
                            f.write(header + raw_response)
                            
                        print(f"Success ({duration:.2f}s). Stats: {len(current_data.get('objects', []))} objs, "
                              f"{len(current_data.get('locations', []))} locs, "
                              f"{len(current_data.get('characters', []))} chars.")
                        success = True
                        break # Break the retry loop
                    else:
                        # Failure: LLM replied, but JSON was invalid
                        print(f"Invalid JSON. ", end="")
                        
                        # Save the failed attempt for debugging
                        fail_log = output_dir / f"{config.file_prefix}chapter_{chapter_num:03d}_FAIL_{attempt}.txt"
                        with open(fail_log, 'w', encoding='utf-8') as f:
                            f.write(f"Reason: Invalid JSON extraction\n{'-'*20}\n{raw_response}")
                            
                        # If retries remain, loop will continue
                        if attempt < config.max_retries:
                            print("Retrying...")
                            time.sleep(2)
                        else:
                            print("Max retries reached.")
                else:
                    # Failure: Network/Backend error
                    print(f"Network/Gen Error. ", end="")
                    if attempt < config.max_retries:
                        print("Retrying...")
                        time.sleep(2)

            if not success:
                print(f"  [ERROR] Failed to extract valid JSON for Chapter {chapter_num} after {config.max_retries + 1} attempts.")
                print("  ! IMPORTANT: Moving to next chapter using OLD data state to preserve pipeline continuity.")
                
                # Create an error placeholder so we don't try to regenerate indefinitely on next run if force_regenerate is False
                with open(file_path.with_suffix('.json.error'), 'w', encoding='utf-8') as f:
                    f.write("Failed to generate valid JSON.")

        # Cleanup current model before moving to next (if multiple models)
        backend.stop_server()
        if model_path != models[-1]:
            time.sleep(config.cooldown_wait)

    print("\nProcessing Complete.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Model Book Inference Script with Fallback & Recovery

Runs prompts iteratively across chapters. 
Features streaming capture to save partial progress on timeouts/failures.
PRE-CHECK: Checks if results exist before loading model to save time.

Usage:
    python run_benchmark_bookchapters.py config.booksummary.yaml

Notes
This script is meant to help with model choice, yiou can view how it handles the summarisation task and judge accuracy    
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
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import yaml

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
        
        # Server settings
        server = data.get('server', {})
        self.host = server.get('host', '127.0.0.1')
        self.port = server.get('port', 5000)
        self.startup_wait = server.get('startup_wait', 420)
        self.cooldown_wait = server.get('cooldown_wait', 5)
        self.api_timeout = server.get('api_timeout', 800)
        self.max_retries = server.get('max_retries', 2) # New config option
        
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
        self.max_tokens = generation.get('max_tokens', 2048)
        self.temperature = generation.get('temperature', 1.0)
        self.top_k = generation.get('top_k', 64)
        self.top_p = generation.get('top_p', 0.95)
        self.seed = generation.get('seed', -1)
        self.prompt_template = generation.get('prompt_template', '{{chapter}}')
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# --- Backend Logic (Robust with Streaming) ---

class LlamaCppBackend:
    """Manages the llama.cpp server process with robust error handling and streaming."""
    
    def __init__(self, config: Config):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.current_model: Optional[Path] = None
        
        # API Endpoints
        self._api_chat_url = f"{self.config.base_url}/v1/chat/completions"
        self._api_health_url = f"{self.config.base_url}/health"
        
        # Threading vars for streaming
        self._partial_response = ""
        self._response_lock = threading.Lock()
    
    def _build_command(self, model_path: Path) -> List[str]:
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
        return cmd
    
    def start_server(self, model_path: Path) -> bool:
        if self.process is not None:
            self.stop_server()
        
        cmd = self._build_command(model_path)
        print(f"  Starting server: {' '.join(cmd[:3])}...")
        
        try:
            # Capture stderr to debug startup failures
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL, # We don't need stdout clutter
                stderr=subprocess.PIPE,
                text=True
            )
            self.current_model = model_path
            return True
        except Exception as e:
            print(f"  [ERROR] Failed to start server: {e}")
            return False
    
    def stop_server(self):
        if self.process is not None:
            print("  Stopping server...")
            try:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("  Server hung, killing...")
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                print(f"  [WARN] Error stopping server: {e}")
            finally:
                # Close stderr pipe
                if self.process.stderr:
                    self.process.stderr.close()
                self.process = None
                self.current_model = None
    
    def get_process_stderr(self) -> str:
        """Reads recent stderr from the process (non-blocking attempt)."""
        if self.process and self.process.stderr:
            try:
                return self.process.stderr.read() 
            except:
                return "[Could not read stderr]"
        return ""

    def is_ready(self) -> bool:
        try:
            response = requests.get(self._api_health_url, timeout=2)
            if response.status_code == 200:
                data = response.json()
                # Llama.cpp health returns {"status": "ok"} or similar
                if data.get("status", "").lower() == "ok" or data.get("status") == "loading":
                    # If loading, strictly speaking it's not ready for inference, 
                    # but the loop handles waiting.
                    return True
            return False
        except:
            return False
    
    def wait_for_ready(self) -> bool:
        print(f"  Waiting up to {self.config.startup_wait}s for server...")
        start_time = time.time()
        
        while time.time() - start_time < self.config.startup_wait:
            if self.is_ready():
                # Double check to ensure model is loaded, give it 2 more seconds
                time.sleep(2) 
                print(f"  Server ready after {time.time() - start_time:.1f}s")
                return True
            
            if self.process and self.process.poll() is not None:
                print(f"  [ERROR] Server process died.")
                print(f"  STDERR: {self.get_process_stderr()[:500]}")
                return False
            
            time.sleep(2)
        
        print(f"\n  [ERROR] Server startup timed out.")
        return False

    def _stream_response_capture(self, response, timeout_seconds):
        """Captures streaming response chunks."""
        start_time = time.time()
        captured_content = ""
        
        try:
            for line in response.iter_lines():
                if time.time() - start_time > timeout_seconds:
                    raise requests.exceptions.Timeout("Internal stream timer exceeded")
                
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]
                        if data_str.strip() == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(data_str)
                            choices = chunk_data.get('choices', [])
                            if choices:
                                content = choices[0].get('delta', {}).get('content', '')
                                if content:
                                    captured_content += content
                                    with self._response_lock:
                                        self._partial_response = captured_content
                        except:
                            continue
        except Exception:
            pass # Caller handles partial return via self._partial_response
        
        return captured_content

    def generate(self, prompt: str) -> Tuple[Optional[str], float, bool, bool, str]:
        """
        Generate response with fallback handling.
        Returns: (text, time, success, used_fallback, failure_reason)
        """
        payload = {
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "stream": True  # Enable streaming for fallback
        }
        
        if self.config.seed >= 0:
            payload["seed"] = self.config.seed

        start_time = time.time()
        
        # Reset partial
        with self._response_lock:
            self._partial_response = ""
            
        try:
            response = requests.post(
                self._api_chat_url,
                json=payload,
                timeout=self.config.api_timeout,
                stream=True
            )
            
            if response.status_code != 200:
                return None, time.time()-start_time, False, False, f"HTTP {response.status_code}"

            # Capture stream
            final_text = self._stream_response_capture(response, self.config.api_timeout)
            
            # If we exited the stream loop naturally, check if we got anything
            if not final_text:
                # Double check the partial buffer
                with self._response_lock:
                    final_text = self._partial_response
            
            if final_text:
                return final_text, time.time()-start_time, True, False, ""
            else:
                return None, time.time()-start_time, False, False, "Empty response"

        except requests.exceptions.Timeout:
            # Timeout happened - Grab Partial
            elapsed = time.time() - start_time
            with self._response_lock:
                partial = self._partial_response
            
            if partial:
                return partial, elapsed, True, True, "Timeout (Saved Partial)"
            return None, elapsed, False, False, "Timeout (No Data)"

        except Exception as e:
            # Other connection error - Grab Partial
            elapsed = time.time() - start_time
            with self._response_lock:
                partial = self._partial_response
            
            if partial:
                return partial, elapsed, True, True, f"Error: {str(e)[:50]} (Saved Partial)"
            return None, elapsed, False, False, f"Error: {str(e)[:100]}"


# --- Helper Functions ---

def discover_models(config: Config) -> List[Path]:
    print(f"Scanning for models in: {config.model_dir}")
    if not config.model_dir.exists():
        return []
    
    models = []
    for path in config.model_dir.glob('*.gguf'):
        if not path.is_file() or path.name.startswith('.'): continue
        
        # Skip multi-part files > part 1
        if re.search(r'-(\d+)-of-(\d+)\.gguf$', path.name):
            if int(re.search(r'-(\d+)-of-(\d+)\.gguf$', path.name).group(1)) > 1:
                continue
                
        try:
            size = path.stat().st_size
            if config.max_size_bytes and size > config.max_size_bytes: continue
            if config.min_size_bytes and size < config.min_size_bytes: continue
        except: continue
        
        models.append(path)
    return sorted(models)

def load_book_chapters(config: Config) -> List[str]:
    if not os.path.exists(config.txt_file):
        print(f"[ERROR] Book file not found: {config.txt_file}")
        sys.exit(1)
    
    try:
        with open(config.txt_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"[ERROR] Failed to read book file: {e}")
        sys.exit(1)
    
    chapters = content.split(config.chapter_split_string)
    chapters = [c.strip() for c in chapters if c.strip()]
    print(f"Loaded book '{config.txt_file}' - found {len(chapters)} chapters.")
    return chapters

def read_previous_output(path: Path) -> str:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        parts = content.split("======\n", 1)
        return parts[1].strip() if len(parts) > 1 else content.strip()
    except:
        return ""

def check_all_chapters_exist(config: Config, model_stem: str, num_chapters: int) -> bool:
    """
    Checks if all expected output files for a model already exist.
    Used to skip model loading entirely if work is done.
    """
    output_dir = config.results_dir / model_stem
    if not output_dir.exists():
        return False
        
    for i in range(num_chapters):
        chapter_num = i + 1
        basename = f"Chapter_{chapter_num:03d}"
        filename = f"{config.file_prefix}{basename}.txt"
        path = output_dir / filename
        if not path.exists():
            return False
            
    return True

def save_output(config: Config, model_stem: str, basename: str, content: str, time_taken: float, model_name: str, fallback_note: str = ""):
    output_dir = config.results_dir / model_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{config.file_prefix}{basename}.txt"
    path = output_dir / filename
    
    header = f"model: {model_name}\n"
    header += f"time: {time_taken:.2f}s\n"
    if fallback_note:
        header += f"status: FALLBACK ({fallback_note})\n"
    header += "======\n"
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + content)

def update_history(history: str, idx: int, content: str) -> str:
    entry = f"CHAPTER {idx + 1} ANALYSIS\n{content}"
    return entry if history == "No previous analysis." else f"{history}\n\n{entry}"

def format_time(start_time: datetime.datetime) -> str:
    return str(datetime.datetime.now() - start_time).split('.')[0]

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to YAML configuration file")
    args = parser.parse_args()
    
    config = Config(args.config_file)
    
    print("=" * 60)
    print("Robust Book Summarization Script")
    print(f"Start Time: {config.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    models = discover_models(config)
    chapters = load_book_chapters(config)
    if not models or not chapters:
        print("No models or chapters found.")
        sys.exit(1)
        
    backend = LlamaCppBackend(config)
    
    # Graceful exit
    def signal_handler(sig, frame):
        print("\n[INTERRUPT] Stopping server and exiting...")
        backend.stop_server()
        sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Stats
    failed_runs = [] # (Model, Chapter, Reason)
    total_processed = 0
    total_skipped = 0
    
    for model_idx, model_path in enumerate(models):
        model_name = model_path.name
        model_stem = model_path.stem

        # --- PRE-CHECK: Skip model load if all outputs exist ---
        if not config.force_regenerate:
            if check_all_chapters_exist(config, model_stem, len(chapters)):
                print(f"[{format_time(config.start_time)}] Skipping {model_name} (All {len(chapters)} chapters exist)")
                total_skipped += len(chapters)
                continue
        # -------------------------------------------------------
        
        print(f"\n[{format_time(config.start_time)}] === Model {model_idx+1}/{len(models)}: {model_name} ===")
        
        if not backend.start_server(model_path):
            failed_runs.append((model_name, "ALL", "Server Start Failed"))
            continue
            
        if not backend.wait_for_ready():
            failed_runs.append((model_name, "ALL", "Server Ready Timeout"))
            backend.stop_server()
            continue
            
        history = "No previous chapter data."
        
        for i, chapter_text in enumerate(chapters):
            chapter_num = i + 1
            basename = f"Chapter_{chapter_num:03d}"
            output_path = config.results_dir / model_stem / f"{config.file_prefix}{basename}.txt"
            
            # Check exist
            if output_path.exists() and not config.force_regenerate:
                print(f"  [Ch {chapter_num}] Exists. Loading context...", end='\r')
                loaded_content = read_previous_output(output_path)
                if loaded_content:
                    history = update_history(history, i, loaded_content)
                    total_skipped += 1
                    continue
            
            # Generate with Retries
            prompt = config.prompt_template.replace('{{chapter}}', chapter_text)
            prompt = prompt.replace('{{previous}}', history)
            prompt = prompt.replace('{{chapter_number}}', str(chapter_num))
            
            success = False
            for attempt in range(config.max_retries + 1):
                attempt_label = f"(Retry {attempt})" if attempt > 0 else ""
                print(f"  [Ch {chapter_num}] Generating {attempt_label}...", end='', flush=True)
                
                content, duration, ok, used_fallback, reason = backend.generate(prompt)
                
                if ok:
                    # Save (even if fallback)
                    note = reason if used_fallback else ""
                    save_output(config, model_stem, basename, content, duration, model_name, note)
                    
                    history = update_history(history, i, content)
                    print(f" Done ({duration:.1f}s) {f'[{reason}]' if used_fallback else ''}")
                    success = True
                    total_processed += 1
                    break
                else:
                    print(f" Failed: {reason}")
                    if attempt == config.max_retries:
                        failed_runs.append((model_name, basename, reason))
                    else:
                        time.sleep(2) # Short pause before retry
            
            if not success:
                print("  [WARN] Chain broken due to failure. Next chapter will lack context.")
                # We do NOT add to history, so next chapter sees old history
        
        backend.stop_server()
        if model_idx < len(models) - 1:
            time.sleep(config.cooldown_wait)

    # Summary
    print("\n" + "="*60)
    print(f"Completed in: {format_time(config.start_time)}")
    print(f"Models: {len(models)} | Chapters: {len(chapters)}")
    print(f"Processed: {total_processed} | Skipped: {total_skipped} | Failed: {len(failed_runs)}")
    
    if failed_runs:
        print("\n--- Failures ---")
        for m, c, r in failed_runs:
            print(f"  {m} :: {c} -> {r}")
    print("="*60)

if __name__ == "__main__":
    main()
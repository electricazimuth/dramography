#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GraphRAG Narrative Entity Extraction
Role: Extracts entities from book chapters, resolves aliases via embeddings, 
      and uses Graph Centrality to determine the authoritative "Core" list.

Prerequisites:
  pip install openai networkx scikit-learn numpy pyyaml

Usage:
  python run_graph_extraction.py config.yaml

Notes:
  current output isn't any better than long context, chapter by chapter standard prompting - still lots on unusual results
"""

import os
import sys
import json
import time
import datetime
import argparse
import yaml
import re
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration Class ---

class Config:
    def __init__(self, config_path: str):
        self._load_config(config_path)

    def _load_config(self, config_path: str):
        if not os.path.exists(config_path):
            print(f"Error: Config file '{config_path}' not found.")
            sys.exit(1)

        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)

        # Server Settings
        server = data.get('server', {})
        self.host = server.get('host', '127.0.0.1')
        self.llm_port = server.get('llm_port', 5000)
        self.emb_port = server.get('embedding_port', 5001)
        self.api_key = server.get('api_key', 'not-needed')
        self.max_retries = server.get('max_retries', 2)

        # Processing Settings
        proc = data.get('processing', {})
        self.txt_file = proc.get('txt_file', '')
        self.chapter_split_string = proc.get('chapter_split_string', '=== section break ===')
        self.results_dir = Path(data.get('paths', {}).get('results_dir', './results')).expanduser().resolve()
        self.force_regenerate = proc.get('force_regenerate', False)

        # Graph/Logic Settings
        logic = data.get('logic', {})
        self.similarity_threshold = logic.get('resolution_threshold', 0.85)
        self.centrality_threshold = logic.get('centrality_threshold', 0.05)

        # Prompts
        self.extract_prompt = data.get('prompts', {}).get('extract', '')

        # Generation Params
        gen = data.get('generation', {})
        self.gen_params = {
            "max_tokens": gen.get('max_tokens', 4096),
            "temperature": gen.get('temperature', 0.1),
            "top_p": gen.get('top_p', 0.95),
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        }

# --- Helper Functions ---

def load_chapters(config: Config) -> List[str]:
    if not os.path.exists(config.txt_file):
        print(f"[ERROR] File not found: {config.txt_file}")
        sys.exit(1)
    with open(config.txt_file, 'r', encoding='utf-8') as f:
        content = f.read()
    chapters = [c.strip() for c in content.split(config.chapter_split_string) if c.strip()]
    print(f"Loaded {len(chapters)} chapters from {config.txt_file}")
    return chapters

def extract_json_from_text(text: str) -> Dict:
    """Robust JSON extraction handling markdown and chatter."""
    # 1. Try Markdown block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except: pass
    
    # 2. Try identifying outer brackets
    try:
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1:
            return json.loads(text[start_idx : end_idx + 1])
    except: pass
    
    return None

def save_debug_file(filepath: Path, duration: float, attempt: int, prompt: str, response_content: str, error: str = None):
    """Saves raw interaction to text file for prompt engineering analysis."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "="*80 + "\n"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Duration: {duration:.4f} seconds\n")
        f.write(f"Attempt: {attempt}\n")
        if error:
            f.write(f"Status: FAILED ({error})\n")
        else:
            f.write(f"Status: SUCCESS\n")
        f.write(separator)
        f.write("RAW RESPONSE OUTPUT:\n")
        f.write(separator)
        f.write(response_content + "\n\n")
        f.write(separator)
        f.write("INPUT PROMPT:\n")
        f.write(separator)
        f.write(prompt)

def load_from_debug_file(filepath: Path) -> Dict:
    """Parses JSON out of an existing debug file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Markers defined in save_debug_file
        start_marker = "RAW RESPONSE OUTPUT:\n" + "="*80 + "\n"
        end_marker = "="*80 + "\nINPUT PROMPT:"
        
        start_idx = content.find(start_marker)
        end_idx = content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            raw_text = content[start_idx + len(start_marker) : end_idx]
            return extract_json_from_text(raw_text)
    except Exception as e:
        print(f"Error parsing debug file {filepath}: {e}")
    return None

# --- Pipeline Class ---

class NarrativePipeline:
    def __init__(self, config: Config):
        self.config = config
        
        # Setup separate clients for LLM and Embeddings
        self.llm_client = OpenAI(
            base_url=f"http://{config.host}:{config.llm_port}/v1",
            api_key=config.api_key
        )
        self.emb_client = OpenAI(
            base_url=f"http://{config.host}:{config.emb_port}/v1",
            api_key=config.api_key
        )
        
        # State
        self.raw_extractions = []
        self.resolved_entities = {} # Name -> Data
        self.graph = nx.Graph()

    def run_extraction(self, chapters: List[str]):
        """Stage 1: LLM Extraction per chapter"""
        self.config.results_dir.mkdir(parents=True, exist_ok=True)
        raw_output_path = self.config.results_dir / "raw_extractions.json"

        # Check for cached run
        if raw_output_path.exists() and not self.config.force_regenerate:
            print("Loading cached raw extractions from JSON...")
            with open(raw_output_path, 'r') as f:
                self.raw_extractions = json.load(f)
            return

        # 2. Try to hydrate from Debug Text Files (The requested feature)
        # We only do this if we are NOT forcing regeneration
        if not self.config.force_regenerate:
            print("Checking for existing debug files to hydrate state...")
            hydrated_data = []
            all_files_exist = True
            
            for i in range(len(chapters)):
                chapter_num = i + 1
                debug_file = self.config.results_dir / f"chapter-{chapter_num:03d}.txt"
                
                if debug_file.exists():
                    data = load_from_debug_file(debug_file)
                    if data and "entities" in data:
                        for ent in data['entities']:
                            ent['source_chapter'] = i
                        hydrated_data.extend(data['entities'])
                    else:
                        print(f"  [WARN] Corrupt or invalid JSON in {debug_file.name}")
                        all_files_exist = False
                        break
                else:
                    all_files_exist = False
                    break
            
            if all_files_exist and hydrated_data:
                print(f"  Success! Hydrated {len(hydrated_data)} entities from {len(chapters)} debug files.")
                print("  Skipping LLM generation.")
                self.raw_extractions = hydrated_data
                # Optionally save the consolidated JSON now for next time
                with open(raw_output_path, 'w') as f:
                    json.dump(self.raw_extractions, f, indent=2)
                return
            else:
                print("  Debug files incomplete or missing. Starting LLM generation...")

        # 3. If no cache, run LLM Loop
        print("\n--- Phase 1: Entity Extraction ---")
        for i, chapter_text in enumerate(chapters):
            chapter_num = i + 1
            print(f"Processing Chapter {chapter_num}/{len(chapters)}...", end="", flush=True)
            
            prompt = self.config.extract_prompt.replace("{{chapter}}", chapter_text)
            debug_filename = self.config.results_dir / f"chapter-{chapter_num:03d}.txt"
            
            success = False
            for attempt in range(self.config.max_retries + 1):
                start_time = time.time()
                try:
                    response = self.llm_client.chat.completions.create(
                        model="default-model", # Llama.cpp usually ignores this, but requires it
                        messages=[{"role": "user", "content": prompt}],
                        **self.config.gen_params
                    )
                    end_time = time.time()
                    duration = end_time - start_time
                    content = response.choices[0].message.content
                    data = extract_json_from_text(content)
                    
                    if data and "entities" in data:
                        # Save Success Debug Log
                        save_debug_file(debug_filename, duration, attempt+1, prompt, content)
                        
                        for ent in data['entities']:
                            ent['source_chapter'] = i
                        self.raw_extractions.extend(data['entities'])
                        print(f" Done ({duration:.2f}s). Found {len(data['entities'])} entities.")
                        success = True
                        break
                    else:
                        # Save Failure Debug Log (Invalid JSON)
                        save_debug_file(debug_filename, duration, attempt+1, prompt, content, error="Invalid JSON")
                        print(f" (Retry {attempt+1}: Invalid JSON) ", end="")
                except Exception as e:
                    end_time = time.time()
                    # Save Failure Debug Log (API Error)
                    error_msg = str(e)
                    # We create a dummy file even on crash so you see what happened
                    save_debug_file(debug_filename, end_time-start_time, attempt+1, prompt, "NO RESPONSE", error=error_msg)
                    print(f" (Retry {attempt+1}: API Error) ", end="")
                    time.sleep(2)

            if not success:
                print(" [FAILED]")

        # Save Raw State
        with open(raw_output_path, 'w') as f:
            json.dump(self.raw_extractions, f, indent=2)

    def run_vectorization_and_resolution(self):
        """Stage 2: Embed descriptions and resolve aliases"""
        print("\n--- Phase 2: Embedding & Resolution ---")
        print(f"Configuration: Resolution Threshold = {self.config.similarity_threshold}")
        
        # 1. Embed Descriptions
        descriptions = []
        valid_indices = []
        
        print("Generating embeddings...", end="", flush=True)
        for idx, ent in enumerate(self.raw_extractions):
            # Check if vector already exists (re-using in-memory or loaded JSON)
            if 'vector' in ent and ent['vector']:
                descriptions.append(ent['vector'])
                valid_indices.append(idx)
                continue

            text_to_embed = f"{ent['name']}: {ent.get('description', '')}"
            try:
                resp = self.emb_client.embeddings.create(
                    input=text_to_embed,
                    model="default-embedding-model"
                )
                ent['vector'] = resp.data[0].embedding
                descriptions.append(ent['vector'])
                valid_indices.append(idx)
            except Exception as e:
                print(f"[Warn] Embedding failed for {ent['name']}: {e}")

        if not descriptions:
            print("No embeddings generated. Exiting.")
            return

        embeddings = np.array(descriptions)
        print(f" Done. ({len(embeddings)} vectors)")

        # 2. Similarity Matrix
        sim_matrix = cosine_similarity(embeddings)

        # 3. Clustering / Merging Logic
        # We map every raw index to a "Canonical Name"
        # We iterate through entities; if it matches an existing canonical group, add it.
        # If not, create a new group.
        
        mapped_entities = {} # {CanonicalName: {merged_data}}
        processed_indices = set()

        for i in valid_indices:
            if i in processed_indices: continue
            
            # Start a new cluster with this entity
            current_ent = self.raw_extractions[i]
            canonical_name = current_ent['name']
            
            cluster_interactions = set(current_ent.get('interactions', []))
            cluster_aliases = set(current_ent.get('aliases', []))
            cluster_chapters = [current_ent['source_chapter']]
            cluster_type = current_ent['type']

            processed_indices.add(i)

            # Look for matches in the rest of the list
            for j in valid_indices:
                if j in processed_indices: continue
                
                # Check similarity
                score = sim_matrix[valid_indices.index(i)][valid_indices.index(j)]
                
                if score > self.config.similarity_threshold:
                    # Match found! Merge logic.
                    match_ent = self.raw_extractions[j]
                    
                    # Update cluster data
                    cluster_interactions.update(match_ent.get('interactions', []))
                    cluster_aliases.update(match_ent.get('aliases', []))
                    cluster_aliases.add(match_ent['name']) # Add the variant name as alias
                    cluster_chapters.append(match_ent['source_chapter'])
                    
                    processed_indices.add(j)

            # Store Resolved Entity
            mapped_entities[canonical_name] = {
                "type": cluster_type,
                "aliases": list(cluster_aliases),
                "interactions": list(cluster_interactions),
                "chapters_appeared": sorted(list(set(cluster_chapters))),
                "mention_count": len(cluster_chapters)
            }

        self.resolved_entities = mapped_entities
        print(f"Resolved {len(self.raw_extractions)} mentions into {len(self.resolved_entities)} unique entities.")

    def build_graph_and_filter(self):
        """Stage 3: Graph Construction and Centrality Filtering"""
        print("\n--- Phase 3: Graph Centrality Analysis ---")
        print(f"Configuration: Centrality Threshold = {self.config.centrality_threshold}")
        G = nx.Graph()

        # Add Nodes
        for name, data in self.resolved_entities.items():
            G.add_node(name, **data)

        # Add Edges
        # We need to map the interaction STRINGS to actual canonical NODES
        # This requires a quick lookup map of all aliases -> canonical name
        alias_lookup = {}
        for canonical, data in self.resolved_entities.items():
            alias_lookup[canonical.lower()] = canonical
            for alias in data['aliases']:
                alias_lookup[alias.lower()] = canonical

        print("Building edges...", end="")
        edge_count = 0
        for name, data in self.resolved_entities.items():
            for target_raw in data['interactions']:
                target_clean = target_raw.lower()
                target_canonical = alias_lookup.get(target_clean)

                if target_canonical and target_canonical != name:
                    if not G.has_edge(name, target_canonical):
                        G.add_edge(name, target_canonical, weight=1)
                        edge_count += 1
                    else:
                        G[name][target_canonical]['weight'] += 1
        print(f" Created {edge_count} interactions.")

        # Calculate Centrality
        if len(G.nodes) == 0:
            print("Graph empty.")
            return

        degree_centrality = nx.degree_centrality(G)
        
        # Filter for "Core" Elements
        core_list = []
        for node, score in degree_centrality.items():
            if score >= self.config.centrality_threshold:
                entity_data = G.nodes[node]
                core_list.append({
                    "name": node,
                    "type": entity_data['type'],
                    "score": round(score, 4),
                    "aliases": entity_data['aliases'],
                    "chapter_count": entity_data['mention_count']
                })

        # Sort by importance
        core_list.sort(key=lambda x: x['score'], reverse=True)

        # Output
        final_path = self.config.results_dir / "authoritative_entities.json"
        with open(final_path, 'w') as f:
            json.dump({
                "meta": {
                    "total_nodes": len(G.nodes), 
                    "core_nodes": len(core_list),
                    "resolution_threshold": self.config.similarity_threshold,
                    "centrality_threshold": self.config.centrality_threshold
                },
                "core_entities": core_list
            }, f, indent=2)
            
        print(f"\nSUCCESS. Authoritative list generated.")
        print(f"Core Entities: {len(core_list)} (out of {len(G.nodes)} total resolved)")
        print(f"Saved to: {final_path}")
        
        # Print top 5 for user verification
        print("\nTop 5 Core Elements:")
        for e in core_list[:5]:
            print(f" - {e['name']} ({e['type']}) [Score: {e['score']}]")


# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = Config(args.config_file)
    pipeline = NarrativePipeline(cfg)
    
    # 1. Load Text
    chapters = load_chapters(cfg)
    
    # 2. Extract
    pipeline.run_extraction(chapters)
    
    # 3. Vectorize & Resolve
    pipeline.run_vectorization_and_resolution()
    
    # 4. Filter & Save
    pipeline.build_graph_and_filter()
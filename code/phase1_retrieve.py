"""
Phase 1: Retrieval Only
Run this first to cache all retrieved contexts to a file.
Then run Phase 2 (inference) separately.
"""
import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from time_aware_retriever import TimeAwareRetriever

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Retrieve and cache contexts")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--encoder_path", type=str, default="./checkpoints/MedQA_encoder")
    parser.add_argument("--output_cache", type=str, default="./cache/retrieved_contexts.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()

def normalize_medmcqa(item: Dict[str, Any]) -> Dict[str, Any]:
    if 'options' in item and 'answer' in item:
        return item
    normalized = {'question': item.get('question', ''), 'options': {}, 'answer': ''}
    idx_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    for v_idx, key in enumerate(['opa', 'opb', 'opc', 'opd']):
        if key in item:
            normalized['options'][idx_map[v_idx+1]] = item[key]
    try:
        normalized['answer'] = idx_map.get(int(item.get('cop')), '')
    except:
        pass
    return normalized

def load_data(filepath: str) -> List[Dict]:
    items = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if 'cop' in item:
                    items.append(normalize_medmcqa(item))
                else:
                    items.append(item)
    return items

def main():
    args = parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    items = load_data(args.data_path)
    if args.limit:
        items = items[:args.limit]
        print(f"Limiting to {args.limit} samples.")
    print(f"Total samples: {len(items)}")
    
    # Check for existing progress (resume support)
    start_idx = 0
    os.makedirs(os.path.dirname(args.output_cache), exist_ok=True)
    
    if os.path.exists(args.output_cache):
        with open(args.output_cache, 'r', encoding='utf-8') as f:
            start_idx = sum(1 for _ in f)
        print(f"Found existing cache with {start_idx} items. Resuming from index {start_idx}...")
    
    if start_idx >= len(items):
        print("All items already processed! Nothing to do.")
        return
    
    # Initialize retriever
    print("Initializing Time-Aware Retriever...")
    retriever = TimeAwareRetriever(encoder_path=args.encoder_path)
    
    # Process and save incrementally (append mode for resume)
    print(f"Retrieving and caching to {args.output_cache}...")
    mode = 'a' if start_idx > 0 else 'w'
    with open(args.output_cache, mode, encoding='utf-8') as f_out:
        for i, item in tqdm(enumerate(items[start_idx:], start=start_idx), 
                           total=len(items) - start_idx, 
                           initial=0,
                           desc="Retrieving"):
            question = item.get('question', '')
            
            # Extract entities
            entities = retriever.extract_entities(question)
            context_str = ""
            
            if entities:
                cuis = list(entities.values())
                top_k_neighbors = retriever.get_reranked_neighbors(
                    query_text=question,
                    cuis=cuis,
                    top_k=5,
                    alpha=0.3
                )
                
                context_str = f"Related Concepts: {', '.join(entities.keys())}\n"
                if top_k_neighbors:
                    neighbor_names = [n['name'] for n in top_k_neighbors]
                    context_str += f"Relevant Details (Time-Aligned): {', '.join(neighbor_names)}"
            
            # Save each item with its context
            cache_item = {
                "index": i,
                "question": question,
                "options": item.get('options', {}),
                "answer": item.get('answer', ''),
                "retrieved_context": context_str
            }
            f_out.write(json.dumps(cache_item, ensure_ascii=False) + '\n')
            f_out.flush()  # Flush after each item to prevent data loss on crash
    
    retriever.close()
    print(f"Done! Cached {len(items)} items to {args.output_cache}")

if __name__ == "__main__":
    main()

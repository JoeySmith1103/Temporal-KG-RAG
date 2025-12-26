"""
Phase 1: MRAG Baseline Retrieval
Run this first to cache all retrieved contexts using MRAG method.
Then run Phase 2 (inference) with the cached results.
"""
import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from mrag_baseline import MRAGBaseline


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: MRAG Baseline Retrieval")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_cache", type=str, default="./cache/mrag_retrieved_contexts.jsonl")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top candidates to retrieve")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for semantic score (0-1)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_password", type=str, default="admin")
    return parser.parse_args()


def normalize_medmcqa(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize MedMCQA format to standard format."""
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
    """Load data from JSONL file."""
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
    
    # Initialize MRAG baseline
    print("Initializing MRAG Baseline...")
    mrag = MRAGBaseline(
        neo4j_uri=args.neo4j_uri,
        neo4j_auth=(args.neo4j_user, args.neo4j_password)
    )
    
    # Prepare output
    os.makedirs(os.path.dirname(args.output_cache), exist_ok=True)
    
    # Process and save incrementally
    print(f"Retrieving with MRAG and caching to {args.output_cache}...")
    print(f"Parameters: top_k={args.top_k}, alpha={args.alpha}")
    
    with open(args.output_cache, 'w', encoding='utf-8') as f_out:
        for i, item in tqdm(enumerate(items), total=len(items), desc="MRAG Retrieval"):
            question = item.get('question', '')
            
            # Run MRAG pipeline
            ranked_candidates, temporal_info = mrag.retrieve(
                question,
                top_k=args.top_k,
                alpha=args.alpha
            )
            
            # Format context string
            context_str = mrag.format_context(ranked_candidates, temporal_info)
            
            # Save cache item
            cache_item = {
                "index": i,
                "question": question,
                "options": item.get('options', {}),
                "answer": item.get('answer', ''),
                "retrieved_context": context_str,
                # MRAG-specific metadata for analysis
                "mrag_metadata": {
                    "temporal_info": temporal_info,
                    "num_candidates": len(ranked_candidates),
                    "top_candidate": ranked_candidates[0] if ranked_candidates else None
                }
            }
            f_out.write(json.dumps(cache_item, ensure_ascii=False) + '\n')
            f_out.flush()
    
    mrag.close()
    print(f"Done! Cached {len(items)} items to {args.output_cache}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
import re
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
import pandas as pd
from knowledge_retriever import KnowledgeRetriever

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA dataset using vllm with RAG")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model name or path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to json/jsonl file")
    parser.add_argument("--file_type", type=str, choices=["json", "jsonl"], help="File type (optional, inferred from extension if not provided)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results_rag.json", help="Output file for detailed results")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for generation")
    parser.add_argument("--enable_rag", action="store_true", help="Enable Retrieval Augmented Generation")
    return parser.parse_args()

def normalize_medmcqa(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize MedMCQA item to standard format."""
    if 'options' in item and 'answer' in item:
        return item
        
    normalized = {
        'question': item.get('question', ''),
        'options': {},
        'answer': ''
    }
    
    idx_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H'}
    vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    
    for v_idx, v in enumerate(vocab):
        key = f"op{v}"
        if key in item:
             normalized['options'][idx_map[v_idx+1]] = item[key]
    
    try:
        cop = int(item.get('cop'))
        normalized['answer'] = idx_map.get(cop, '')
    except:
        normalized['answer'] = ''
        
    return normalized

def normalize_medqa(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize MedQA item to standard format."""
    return item

def load_data(filepath: str, file_type: str = None) -> List[Dict[str, Any]]:
    if file_type is None:
        if filepath.endswith('.jsonl'):
            file_type = 'jsonl'
        else:
            if 'train.json' in filepath:
                 file_type = 'jsonl'
            else:
                 file_type = 'json'

    items = []
    print(f"Loading {filepath} as {file_type}...")
    try:
        raw_items = []
        if file_type == 'json':
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
                if isinstance(content, list):
                    raw_items = content
                else:
                    raise ValueError("JSON content is not a list")
        elif file_type == 'jsonl':
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        raw_items.append(json.loads(line))
        
        for item in raw_items:
            if 'cop' in item:
                items.append(normalize_medmcqa(item))
            else:
                items.append(normalize_medqa(item))
                
    except Exception as e:
        print(f"Error loading file: {e}")
        return []
    
    return items

def format_prompt(item: Dict[str, Any], retrieved_context: str = "") -> str:
    question = item.get('question', '')
    options = item.get('options', {})
    
    option_str = ""
    sorted_keys = sorted(options.keys())
    for key in sorted_keys:
        option_str += f"{key}: {options[key]}\n"
    
    context_section = ""
    if retrieved_context:
        context_section = f"Context Information:\n{retrieved_context}\n\n"

    prompt = f"""{context_section}Answer the following multiple-choice question.
Question: {question}

Options:
{option_str}

Please think step by step to derive the answer.
After your explanation, provide the final answer letter wrapped in <a> and </a> tags, for example: <a>A</a>. Do not write the full answer text inside the tags.
"""
    return prompt

def extract_answer(generated_text: str) -> str:
    match = re.search(r'<a>\s*([A-Za-z])', generated_text)
    if match:
        return match.group(1).upper()
    return ""

def evaluate(args):
    # Load data
    items = load_data(args.data_path, args.file_type)
    if not items:
        print("No data found.")
        return

    if args.limit:
        items = items[:args.limit]
        print(f"Limiting to first {args.limit} samples.")

    # RAG Retrieval
    retrieved_contexts = [""] * len(items)
    if args.enable_rag:
        print("Initializing Knowledge Retriever...")
        retriever = KnowledgeRetriever()
        
        print("Retrieving context for samples...")
        for i, item in enumerate(items):
            question = item.get('question', '')
            # 1. Extract Entities
            entities = retriever.extract_entities(question)
            if not entities:
                 continue
            
            cuis = list(entities.values())
            
            # 2. Get Neighbors
            neighbors = retriever.get_one_hop_neighbors(cuis)
            
            # 3. Format Context
            # Format: Entity: Name (CUI) -> [Neighbor Name (Relationship), ...]
            # Collecting all neighbor names for simplicity as user requested "input these keywords as prompt"
            # But user said "One-hop retrieval to get some neighbor, take these keywords as prompt"
            # So I should include the neighbors in the prompt.
            
            context_parts = []
            all_neighbor_names = set()
            for source_cui, rel_triples in neighbors.items():
                # rel_triples is a set of (rela, neighbor_name, neighbor_cui)
                for rela, n_name, n_cui in rel_triples:
                    all_neighbor_names.add(n_name)
            
            # Add original entities names too? QuickUMLS returns CUI.
            # entities dict is {ngram: cui}
            found_concepts = list(entities.keys())
            
            context_str = f"Related Concepts: {', '.join(found_concepts)}\n"
            if all_neighbor_names:
                context_str += f"Related Neighbors: {', '.join(list(all_neighbor_names))}"
            
            retrieved_contexts[i] = context_str
        
        retriever.close()

    prompts = [format_prompt(item, context) for item, context in zip(items, retrieved_contexts)]
    
    # Initialize vllm
    print(f"Initializing model {args.model}...")
    tokenizer_mode = "mistral" if "mistral" in args.model else "auto"
    llm = LLM(
        model=args.model,
        dtype="bfloat16", 
        gpu_memory_utilization=0.6,
        tokenizer_mode=tokenizer_mode,
        max_model_len=4096,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        repetition_penalty=1.1
    )

    # Generate
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    # Process results
    correct_count = 0
    results = []
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        predicted_answer = extract_answer(generated_text)
        ground_truth = items[i].get('answer', '').strip().upper()
        
        is_correct = (predicted_answer == ground_truth)
        if is_correct:
            correct_count += 1
            
        results.append({
            "question": items[i].get('question'),
            "retrieved_context": retrieved_contexts[i],
            "ground_truth": ground_truth,
            "predicted": predicted_answer,
            "raw_output": generated_text,
            "is_correct": is_correct
        })

    accuracy = correct_count / len(items) if items else 0
    print(f"\nResults for {args.data_path}:")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(items)})")

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

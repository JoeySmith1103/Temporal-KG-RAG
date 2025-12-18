import argparse
import json
import os
import re
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA dataset using vllm")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model name or path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to json/jsonl file")
    parser.add_argument("--file_type", type=str, choices=["json", "jsonl"], help="File type (optional, inferred from extension if not provided)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file for detailed results")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for generation")
    return parser.parse_args()

def normalize_medmcqa(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize MedMCQA item to standard format."""
    # Standard: question, options={A:..., B:...}, answer='A'
    # MedMCQA: question, opa, opb, opc, opd, cop (1-based index)
    
    if 'options' in item and 'answer' in item:
        return item  # Already normalized or close to it? Or maybe mixed?
        
    normalized = {
        'question': item.get('question', ''),
        'options': {},
        'answer': ''
    }
    
    # Map 1->A, 2->B, 3->C, 4->D
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
    # MedQA US_qbank is already in the target format:
    # {"question": "...", "options": {"A": "...", "B": "..."}, "answer": "A"}
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
            # Dispatch based on content heuristics
            if 'cop' in item:
                items.append(normalize_medmcqa(item))
            else:
                items.append(normalize_medqa(item))
                
    except Exception as e:
        print(f"Error loading file: {e}")
        return []
    
    return items

def format_prompt(item: Dict[str, Any]) -> str:
    question = item.get('question', '')
    options = item.get('options', {})
    
    # Format options
    option_str = ""
    sorted_keys = sorted(options.keys())
    for key in sorted_keys:
        option_str += f"{key}: {options[key]}\n"
    
    # CoT + Tag instruction prompt
    prompt = f"""Answer the following multiple-choice question.
Question: {question}

Options:
{option_str}

Please think step by step to derive the answer.
After your explanation, provide the final answer letter wrapped in <a> and </a> tags, for example: <a>A</a>. Do not write the full answer text inside the tags.
"""
    return prompt

def extract_answer(generated_text: str) -> str:
    # Match content inside <a>...</a>
    # We look for the first letter character inside the tags
    match = re.search(r'<a>\s*([A-Za-z])', generated_text)
    if match:
        return match.group(1).upper()
    
    # Fallback: look for "The answer is X" or similar if tags are missing
    # But for now, sticking to tags is safer to avoid false positives from the CoT
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

    prompts = [format_prompt(item) for item in items]
    
    # Initialize vllm with user specified params
    print(f"Initializing model {args.model}...")
    tokenizer_mode = "mistral" if "mistral" in args.model else "slow"
    llm = LLM(
        model=args.model,
        dtype="bfloat16", 
        gpu_memory_utilization=0.6,
        tokenizer_mode=tokenizer_mode,
        max_model_len=4096,
        trust_remote_code=True,
        # tensor_parallel_size=2
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
            "ground_truth": ground_truth,
            "predicted": predicted_answer,
            "raw_output": generated_text,
            "is_correct": is_correct
        })

    accuracy = correct_count / len(items) if items else 0
    print(f"\nResults for {args.data_path}:")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(items)})")

    # Save detailed results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

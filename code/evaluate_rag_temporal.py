import argparse
import json
import os
import re
from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
import pandas as pd
from time_aware_retriever import TimeAwareRetriever
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate QA dataset using vllm with Time-Aware RAG")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Model name or path")
    parser.add_argument("--data_path", type=str, required=True, help="Path to json/jsonl file")
    parser.add_argument("--file_type", type=str, choices=["json", "jsonl"], help="File type")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate")
    parser.add_argument("--output", type=str, default="evaluation_results_temporal.json", help="Output file")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for generation")
    parser.add_argument("--enable_rag", action="store_true", help="Enable Retrieval Augmented Generation")
    parser.add_argument("--encoder_path", type=str, default="./checkpoints/MedQA_encoder", help="Path to Time-Aware Encoder")
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
        context_section = f"Context Information (Prioritized by Time-alignment):\n{retrieved_context}\n\n"

    prompt = f"""You are an expert medical AI specializing in differential diagnosis involves complex timelines.

{context_section}Instruction:
1. Analyze the patient's clinical timeline (e.g., Acute, Chronic, Recurrent) from the Question.
2. Use the Context above. Concepts have been re-ranked: those matching the patient's timeline are listed first.
3. Differentiate between options that may look similar but differ in onset or duration.

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
        print("Initializing Time-Aware Retriever...")
        retriever = TimeAwareRetriever(encoder_path=args.encoder_path)
        
        print("Retrieving context for samples...")
        for i, item in tqdm(enumerate(items), total=len(items), desc="Retrieving"):
            question = item.get('question', '')
            # 1. Extract Entities
            entities = retriever.extract_entities(question)
            if not entities:
                 continue
            
            cuis = list(entities.values())
            
            # 2. Get Re-ranked Neighbors (NEW)
            top_k_neighbors = retriever.get_reranked_neighbors(
                query_text=question,
                cuis=cuis,
                top_k=5, # Top 5 most temporally relevant
                alpha=0.3 # 30% Temporal Weight (Implicitly handled in demo logic, here we just use the function)
            )
            
            # 3. Format Context
            context_str = f"Related Concepts: {', '.join(entities.keys())}\n"
            if top_k_neighbors:
                neighbor_names = [n['name'] for n in top_k_neighbors]
                context_str += f"Relevant Details (Time-Aligned): {', '.join(neighbor_names)}"
            
            retrieved_contexts[i] = context_str
        
        retriever.close()

    prompts = [format_prompt(item, context) for item, context in zip(items, retrieved_contexts)]
    
    # Initialize vllm
    print(f"Initializing model {args.model}...")
    tokenizer_mode = "mistral" if "mistral" in args.model else "auto"
    llm = LLM(
        model=args.model,
        dtype="bfloat16", 
        gpu_memory_utilization=0.7,
        tokenizer_mode=tokenizer_mode,
        max_model_len=4096,
        trust_remote_code=True,
        tensor_parallel_size=2
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

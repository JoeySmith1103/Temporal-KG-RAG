"""
Phase 2: LLM Inference from Cached Contexts
Run this after Phase 1 has completed.
"""
import os
import json
import argparse
import re
from typing import List, Dict, Any
from vllm import LLM, SamplingParams

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: LLM inference from cached contexts")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--cache_path", type=str, default="./cache/retrieved_contexts.jsonl")
    parser.add_argument("--output", type=str, default="results/temporal_eval_result.json")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()

def format_prompt(item: Dict[str, Any]) -> str:
    question = item.get('question', '')
    options = item.get('options', {})
    retrieved_context = item.get('retrieved_context', '')
    
    option_str = ""
    for key in sorted(options.keys()):
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

def main():
    args = parse_args()
    
    # Load cached data
    print(f"Loading cached contexts from {args.cache_path}...")
    items = []
    with open(args.cache_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    if args.limit:
        items = items[:args.limit]
        print(f"Limiting to {args.limit} samples.")
    print(f"Total samples: {len(items)}")
    
    # Prepare prompts
    prompts = [format_prompt(item) for item in items]
    
    # Initialize LLM
    print(f"Initializing model {args.model}...")
    tokenizer_mode = "mistral" if "mistral" in args.model.lower() else "auto"
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
            "retrieved_context": items[i].get('retrieved_context'),
            "ground_truth": ground_truth,
            "predicted": predicted_answer,
            "raw_output": generated_text,
            "is_correct": is_correct
        })
    
    accuracy = correct_count / len(items) if items else 0
    print(f"\nResults:")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(items)})")
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Detailed results saved to {args.output}")

if __name__ == "__main__":
    main()

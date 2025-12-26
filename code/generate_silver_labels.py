
import os
import json
import argparse
from typing import List, Dict
from vllm import LLM, SamplingParams
from tqdm import tqdm

def format_prompt_teacher(question: str) -> str:
    """
    Constructs a prompt for the Teacher model to classify the temporal pattern.
    """
    return f"""You are an expert medical AI. Analyze the following clinical question and determine the 'Temporal Pattern' of the symptoms or condition described.

Classify it into exactly one of these 5 categories:
1. Acute: Sudden onset, short duration (minutes to days), urgent.
2. Chronic: Long-standing, persisting for months or years.
3. Recurrent: Repeated episodes, intermittent symptoms, relapsing-remitting.
4. Progressive: Worsening over time, gradual deterioration.
5. None: No specific temporal pattern mentioned.

Input Question: "{question}"

Instructions:
- Think step-by-step about the time duration and evolution mentioned.
- Output ONLY the category name as the final answer.
- Format: "Category: [Your Label]"

Response:"""

def generate_silver_labels(args):
    # 1. Load Data
    print(f"Loading data from {args.input_path}...")
    data = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if args.limit:
        data = data[:args.limit]
        print(f"Limiting to first {args.limit} samples.")

    prompts = [format_prompt_teacher(item['question']) for item in data]
    print(f"Prepared {len(prompts)} prompts.")

    # 2. Initialize VLLM
    print(f"Initializing Teacher Model: {args.model_name}")
    llm = LLM(
        model=args.model_name,
        trust_remote_code=True,
        tensor_parallel_size=args.tp_size,
        dtype="float16",
        gpu_memory_utilization=0.6
    )
    
    sampling_params = SamplingParams(
        temperature=0.5, 
        max_tokens=256,
        stop=["\n\n"]
    )

    # 3. Generate
    print("Generating labels...")
    outputs = llm.generate(prompts, sampling_params)

    # 4. Parse and Save
    valid_labels = {"Acute", "Chronic", "Recurrent", "Progressive", "None"}
    label_counts = {l: 0 for l in valid_labels}
    label_counts["Unknown"] = 0

    labeled_data = []
    for item, output in zip(data, outputs):
        generated_text = output.outputs[0].text.strip()
        
        # Simple parsing logic
        parsed_label = "None" # Default
        found = False
        
        # Check for explicit "Category: X" format first
        if "Category:" in generated_text:
            part = generated_text.split("Category:")[-1].strip()
            # Clean up punctuation
            if part:
                parts = part.split()
                if parts:
                    part = parts[0].rstrip(".,")
                    if part in valid_labels:
                        parsed_label = part
                        found = True
        
        # Fallback: Search for keywords in the whole response if strict format fails
        if not found:
            for v in valid_labels:
                if v.lower() in generated_text.lower():
                    # Pick the last mentioned or simple match? 
                    # Let's prefer the one that stands out. 
                    # For safety, if multiple appear, we might need logic.
                    # For now, let's trust the greedy model followed instructions mostly.
                    parsed_label = v
                    break
        
        item['teacher_label'] = parsed_label
        item['teacher_reasoning'] = generated_text # Save raw output for debugging
        labeled_data.append(item)
        
        if parsed_label in valid_labels:
            label_counts[parsed_label] += 1
        else:
            label_counts["Unknown"] += 1

    # Save
    print("Labeling complete. Distribution:")
    print(json.dumps(label_counts, indent=2))
    
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in labeled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Saved silver-labeled data to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Teacher model ID")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input jsonl")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save labeled jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor Parallelism size")
    
    args = parser.parse_args()
    generate_silver_labels(args)

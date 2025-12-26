
import os
import json
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm

def format_prompt_teacher(question: str) -> str:
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

def process_dataset(llm, input_path, output_path, limit=None):
    print(f"Processing {input_path}...")
    
    # Load Data
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        if input_path.endswith('.jsonl'):
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        else: # valid json list
            data = json.load(f)
            
    if limit:
        data = data[:limit]
        print(f"Limiting to first {limit} samples.")
        
    # Prepare Prompts
    prompts = [format_prompt_teacher(item['question']) for item in data]
    
    # Generate
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64) # Greedy for classification
    outputs = llm.generate(prompts, sampling_params)
    
    # Parse & Filter
    filtered_data = []
    stats = {"Acute": 0, "Chronic": 0, "Recurrent": 0, "Progressive": 0, "None": 0}
    
    valid_labels = ["Acute", "Chronic", "Recurrent", "Progressive", "None"]
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        
        # Parse logic (reused from generate_silver_labels.py)
        parsed_label = "None"
        if "Category:" in generated_text:
            part = generated_text.split("Category:")[-1].strip().split()[0].rstrip(".,")
            if part in valid_labels:
                parsed_label = part
        else:
             # Fallback
             for label in valid_labels:
                 if label in generated_text:
                     parsed_label = label
                     break
        
        stats[parsed_label] = stats.get(parsed_label, 0) + 1
        
        if parsed_label != "None":
            item = data[i]
            item['temporal_label'] = parsed_label
            filtered_data.append(item)
            
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in filtered_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Finished {input_path}")
    print(f"Original: {len(data)} -> Filtered: {len(filtered_data)}")
    print(f"Distribution: {json.dumps(stats, indent=2)}")
    return len(data), len(filtered_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tp_size", type=int, default=2)
    args = parser.parse_args()
    
    # Load Model once
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
        dtype="float16"
    )
    
    datasets = [
        ("/mnt/NAS/home/tommylee/TimeRAG/datasets/origin/MedQA.jsonl", "/mnt/NAS/home/tommylee/TimeRAG/datasets/clean_time_only/MedQA.jsonl"),
        ("/mnt/NAS/home/tommylee/TimeRAG/datasets/origin/MedMCQA.json", "/mnt/NAS/home/tommylee/TimeRAG/datasets/clean_time_only/MedMCQA.jsonl")
    ]
    
    total_orig = 0
    total_clean = 0
    
    for in_path, out_path in datasets:
        orig, clean = process_dataset(llm, in_path, out_path, args.limit)
        total_orig += orig
        total_clean += clean
        
    print(f"\nTOTAL: {total_orig} -> {total_clean} (Time-Only)")

if __name__ == "__main__":
    main()

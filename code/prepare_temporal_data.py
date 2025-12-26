
import json
import random
import os
import re
from typing import List, Dict, Any

def classify_temporal_heuristic(text: str) -> str:
    """
    (Placeholder) Heuristic-based Silver Labeling.
    Ideally, this should be replaced by an LLM call (Teacher Model) 
    to robustly classify the timeline (e.g., handling negations).
    """
    text = text.lower()
    
    # Refined Heuristics to handle simple negations
    # Step 1: Check for "Progressive" cues
    if any(k in text for k in ["worsening", "progressive", "deteriorating", "gradual onset"]):
        return "Progressive"
        
    # Step 2: Check for "Recurrent" cues
    if any(k in text for k in ["recurrent", "episodes of", "intermittent", "comes and goes", "history of similar"]):
        if "no history of similar" not in text:
            return "Recurrent"
            
    # Step 3: Chronic vs Acute
    # We look for explicit duration markers
    # "years", "months" -> Chronic
    # "minutes", "hours", "days" -> Acute
    
    chronic_markers = ["years", "months", "chronic", "long-standing", "history of"]
    acute_markers = ["minutes", "hours", "days", "sudden", "acute", "emergency", "abrupt"]
    
    score_chronic = sum(1 for k in chronic_markers if k in text)
    score_acute = sum(1 for k in acute_markers if k in text)
    
    if score_chronic > score_acute:
        return "Chronic"
    elif score_acute > score_chronic:
        return "Acute"
        
    return "None"

def process_single_dataset(file_path: str, output_base_dir: str, dataset_name: str, train_ratio: float = 0.8):
    print(f"\nProcessing {dataset_name} from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            
            # Labeling
            # In the future: Replace this line with `label = llm_classify(item['question'])`
            label = classify_temporal_heuristic(item.get('question', ''))
            item['temporal_label'] = label
            data.append(item)
            
    # Shuffle
    random.seed(42)
    random.shuffle(data)
    
    # Split
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Output Dir
    dataset_out_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(dataset_out_dir, exist_ok=True)
    
    # Save
    with open(os.path.join(dataset_out_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(os.path.join(dataset_out_dir, "test.jsonl"), 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Saved {dataset_name}: {len(train_data)} Train, {len(test_data)} Test")
    
    # Stats
    print(f"Stats for {dataset_name} Train:")
    stats = {}
    for x in train_data:
        l = x['temporal_label']
        stats[l] = stats.get(l, 0) + 1
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    # Define your datasets here
    datasets_config = {
        "MedQA": "./datasets/US_qbank_time_only_MedQA.jsonl",
        "MedMCQA": "./datasets/train_time_only_MedMCQA.jsonl"
    }
    
    output_dir = "./datasets/split_individual"
    
    for name, path in datasets_config.items():
        process_single_dataset(path, output_dir, name)

"""
Multi-label Silver Label Generator for Temporal Classification

Generates structured temporal labels including:
- Multiple temporal cues per question
- Span-level annotations
- Role identification (ChiefComplaint vs Background)
- Primary temporal type for the question
"""
import os
import json
import argparse
import re
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from tqdm import tqdm


def format_prompt_multilabel(question: str) -> str:
    """Create prompt for multi-label temporal extraction."""
    return f"""You are an expert medical AI. Analyze the following clinical question and identify ALL temporal patterns present.

Clinical Question:
"{question}"

Instructions:
1. Find all mentions of time duration, onset, or progression
2. For each temporal cue, identify:
   - span: the exact text mentioning time
   - type: Acute (sudden/short), Chronic (long-term), Recurrent (episodic), Progressive (worsening), or None
   - role: ChiefComplaint (main symptom being asked about) or Background (patient history)
3. Determine the primary_type that is most relevant to answering the question (usually the ChiefComplaint)

Output ONLY valid JSON in this exact format:
{{
  "temporal_cues": [
    {{"span": "exact text", "type": "Acute|Chronic|Recurrent|Progressive", "role": "ChiefComplaint|Background"}}
  ],
  "primary_type": "Acute|Chronic|Recurrent|Progressive|None"
}}

JSON Output:"""


def parse_llm_output(text: str) -> Dict[str, Any]:
    """Parse LLM JSON output with error handling."""
    # Try to extract JSON from the text
    text = text.strip()
    
    # Find JSON block
    json_match = re.search(r'\{[\s\S]*\}', text)
    if not json_match:
        return {"temporal_cues": [], "primary_type": "None", "parse_error": True}
    
    json_str = json_match.group()
    
    try:
        result = json.loads(json_str)
        # Validate structure
        if "temporal_cues" not in result:
            result["temporal_cues"] = []
        if "primary_type" not in result:
            result["primary_type"] = "None"
        return result
    except json.JSONDecodeError:
        return {"temporal_cues": [], "primary_type": "None", "parse_error": True}


def generate_multilabel_labels(args):
    """Main function to generate multi-label temporal labels."""
    
    # Load data
    print(f"Loading data from {args.input_path}...")
    data = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if args.limit:
        data = data[:args.limit]
    print(f"Processing {len(data)} samples...")
    
    # Prepare prompts
    prompts = [format_prompt_multilabel(item.get('question', '')) for item in data]
    
    # Initialize LLM
    print(f"Initializing LLM: {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=0.6,
        trust_remote_code=True,
        dtype="float16"
    )
    
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for consistent JSON
        max_tokens=512,
        skip_special_tokens=True
    )
    
    # Generate
    print("Generating labels...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Parse and save
    labeled_data = []
    stats = {
        "Acute": 0, "Chronic": 0, "Recurrent": 0, 
        "Progressive": 0, "None": 0, "parse_errors": 0,
        "multi_type_questions": 0
    }
    
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        parsed = parse_llm_output(generated_text)
        
        # Add to original item
        item = data[i].copy()
        item["temporal_labels"] = parsed
        item["primary_type"] = parsed.get("primary_type", "None")
        
        # Count statistics
        if parsed.get("parse_error"):
            stats["parse_errors"] += 1
        else:
            # Validate primary_type
            primary = parsed.get("primary_type", "None")
            valid_types = ["Acute", "Chronic", "Recurrent", "Progressive", "None"]
            if primary not in valid_types:
                primary = "None"  # Default to None if invalid
                parsed["primary_type"] = primary
            stats[primary] += 1
            
            # Count questions with multiple temporal types
            types_found = set(cue.get("type") for cue in parsed.get("temporal_cues", []))
            if len(types_found) > 1:
                stats["multi_type_questions"] += 1
        
        labeled_data.append(item)
    
    # Save
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        for item in labeled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\nSaved {len(labeled_data)} labeled samples to {args.output_path}")
    print("\nStatistics:")
    print(json.dumps(stats, indent=2))
    
    # Show some examples
    print("\n=== Sample Outputs ===")
    for item in labeled_data[:3]:
        print(f"\nQ: {item['question'][:100]}...")
        print(f"Labels: {json.dumps(item['temporal_labels'], indent=2)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multi-label temporal annotations")
    parser.add_argument("--input_path", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--model", type=str, required=True, help="LLM model path")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples")
    parser.add_argument("--tp_size", type=int, default=2, help="Tensor parallel size")
    args = parser.parse_args()
    
    generate_multilabel_labels(args)

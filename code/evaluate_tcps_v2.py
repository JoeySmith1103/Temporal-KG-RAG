"""
Evaluation Pipeline for TCPS v2 (Temporal-Augmented Query-Path Matching)
"""
import os
import json
import argparse
import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple
from neo4j import GraphDatabase
from quickumls import get_quickumls_client
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Import TCPS v2 components - use training version for matching architecture
from train_tcps_v2 import TemporalQueryAugmenter
from time_aware_encoder import TimeAwareEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate TCPS v2")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tcps_path", type=str, default="./checkpoints/tcps_v2/tcps_v2_model.pt")
    parser.add_argument("--temporal_encoder_path", type=str, default="./checkpoints/MedQA_encoder")
    parser.add_argument("--llm_model", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/tcps_v2_eval.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


class TCPSv2Retriever:
    """Retriever using TCPS v2 (Temporal-Augmented Query-Path Matching)."""
    
    def __init__(
        self,
        tcps_path: str,
        temporal_encoder_path: str,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_auth: Tuple[str, str] = ("neo4j", "admin"),
        encoder_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load SapBERT encoder
        print("Loading SapBERT encoder...")
        self.encoder = AutoModel.from_pretrained(encoder_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder.eval()
        
        # Load TCPS v2 temporal augmenter
        print(f"Loading TCPS v2 from {tcps_path}...")
        checkpoint = torch.load(tcps_path, map_location=self.device)
        # Initialize with default parameters (matching training)
        self.temporal_augmenter = TemporalQueryAugmenter().to(self.device)
        self.temporal_augmenter.load_state_dict(checkpoint['temporal_augmenter'])
        self.temporal_augmenter.eval()
        
        # Load temporal encoder for classification
        print(f"Loading temporal classifier from {temporal_encoder_path}...")
        self.temporal_classifier = TimeAwareEncoder.from_pretrained(
            temporal_encoder_path,
            model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        ).to(self.device)
        self.temporal_classifier.eval()
        self.temporal_tokenizer = AutoTokenizer.from_pretrained(temporal_encoder_path)
        
        # Neo4j
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        
        # QuickUMLS
        self.matcher = get_quickumls_client()
        
        self.temporal_names = ["Acute", "Chronic", "Recurrent", "Progressive", "None"]
    
    def get_temporal_type(self, query: str) -> int:
        """Classify temporal type of query."""
        inputs = self.temporal_tokenizer(
            query, return_tensors="pt", 
            padding=True, truncation=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits, _ = self.temporal_classifier(inputs['input_ids'], inputs['attention_mask'])
            return torch.argmax(logits, dim=-1).item()
    
    def extract_cuis(self, text: str) -> List[str]:
        """Extract CUIs from text."""
        cuis = []
        matches = self.matcher.match(text, best_match=True, ignore_syntax=False)
        for group in matches:
            if group:
                cuis.append(group[0]['cui'])
        return cuis[:10]
    
    def get_paths(self, seed_cuis: List[str], max_paths: int = 50) -> List[List[str]]:
        """Get paths from Neo4j."""
        if not seed_cuis:
            return []
        
        paths = []
        with self.driver.session() as session:
            query = """
            MATCH path = (start:Concept)-[*1..2]->(end:Concept)
            WHERE start.CUI IN $seed_cuis
            RETURN [n IN nodes(path) | n.name] as node_names
            LIMIT $max_paths
            """
            result = session.run(query, seed_cuis=seed_cuis, max_paths=max_paths)
            for record in result:
                nodes = record["node_names"]
                if nodes and len(nodes) >= 2:
                    paths.append(nodes)
        
        return paths
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text with SapBERT."""
        inputs = self.tokenizer(
            text, return_tensors="pt",
            padding=True, truncation=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            return outputs.last_hidden_state[:, 0, :]
    
    def score_paths(self, query_emb: torch.Tensor, temporal_type: int, paths: List[List[str]]) -> List[Tuple[List[str], float]]:
        """Score paths using temporal-augmented query."""
        if not paths:
            return []
        
        # Augment query with temporal
        temporal_tensor = torch.tensor([temporal_type], device=self.device)
        query_aug = self.temporal_augmenter(query_emb, temporal_tensor)
        query_aug = F.normalize(query_aug, dim=-1)
        
        scored = []
        for path in paths:
            path_text = " ".join(path)
            path_emb = self.encode_text(path_text)
            path_emb = F.normalize(path_emb, dim=-1)
            
            score = (query_aug * path_emb).sum().item()
            scored.append((path, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[int, List[List[str]]]:
        """Main retrieval method."""
        # Get temporal type
        temporal_type = self.get_temporal_type(query)
        
        # Encode query
        query_emb = self.encode_text(query)
        
        # Extract CUIs and get paths
        cuis = self.extract_cuis(query)
        paths = self.get_paths(cuis)
        
        # Score and rank
        scored_paths = self.score_paths(query_emb, temporal_type, paths)
        
        return temporal_type, [p for p, s in scored_paths[:top_k]]
    
    def close(self):
        self.driver.close()


def format_context(paths: List[List[str]], temporal_type: int) -> str:
    """Format context for LLM."""
    temporal_names = ["Acute", "Chronic", "Recurrent", "Progressive", "None"]
    
    if not paths:
        return f"Temporal pattern: {temporal_names[temporal_type]}"
    
    context = f"Temporal pattern: {temporal_names[temporal_type]}\n"
    context += "Related medical knowledge:\n"
    
    for i, path in enumerate(paths[:5], 1):
        context += f"  {i}. {' → '.join(path)}\n"
    
    return context


def format_prompt(question: str, options: Dict, context: str) -> str:
    """Format prompt for LLM."""
    option_str = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
    
    return f"""Answer the following medical question using the provided context.

{context}

Question: {question}

Options:
{option_str}

Provide your answer as a single letter wrapped in <a> tags, e.g., <a>A</a>.

Answer:"""


def extract_answer(text: str) -> str:
    """Extract answer from LLM output."""
    match = re.search(r'<a>\s*([A-Za-z])', text)
    return match.group(1).upper() if match else ""


def evaluate(args):
    # Load data
    print(f"Loading data from {args.data_path}...")
    items = []
    with open(args.data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    
    if args.limit:
        items = items[:args.limit]
    print(f"Evaluating {len(items)} samples...")
    
    # Initialize retriever
    retriever = TCPSv2Retriever(args.tcps_path, args.temporal_encoder_path)
    
    # Retrieve
    print("Retrieving with TCPS v2...")
    contexts = []
    temporal_types = []
    for item in tqdm(items, desc="Retrieving"):
        temporal_type, paths = retriever.retrieve(item['question'], top_k=args.top_k)
        context = format_context(paths, temporal_type)
        contexts.append(context)
        temporal_types.append(temporal_type)
    
    retriever.close()
    
    # Prepare prompts
    prompts = [format_prompt(item['question'], item['options'], ctx) for item, ctx in zip(items, contexts)]
    
    # Initialize LLM
    print(f"Initializing LLM: {args.llm_model}...")
    llm = LLM(
        model=args.llm_model,
        dtype="bfloat16",
        gpu_memory_utilization=0.7,
        max_model_len=4096,
        trust_remote_code=True,
        tensor_parallel_size=2
    )
    
    sampling_params = SamplingParams(temperature=0.5, max_tokens=512, skip_special_tokens=True)
    
    # Generate
    print("Generating...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Evaluate
    correct = 0
    results = []
    
    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        predicted = extract_answer(generated)
        gt = items[i].get('answer', '').strip().upper()
        
        is_correct = predicted == gt
        if is_correct:
            correct += 1
        
        results.append({
            "question": items[i]['question'],
            "temporal_type": temporal_types[i],
            "context": contexts[i],
            "ground_truth": gt,
            "predicted": predicted,
            "is_correct": is_correct
        })
    
    accuracy = correct / len(items) if items else 0
    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(items)})")
    
    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

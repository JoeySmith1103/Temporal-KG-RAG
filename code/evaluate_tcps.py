"""
Evaluation Pipeline with TCPS (Temporal-Conditioned Path Scorer)

This script:
1. Loads trained TCPS model
2. For each question: extracts temporal type, retrieves paths, scores them
3. Uses top-k paths as context for LLM
4. Compares accuracy vs baseline
"""
import os
import json
import argparse
import re
import torch
from typing import List, Dict, Any, Tuple
from neo4j import GraphDatabase
from improved_entity_extractor import ImprovedEntityExtractor
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer

# Import TCPS components
from temporal_path_scorer import (
    RelationEmbedding,
    TemporalConditioner,
    PathScorer
)
from time_aware_encoder import TimeAwareEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate with TCPS-based retrieval")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tcps_path", type=str, default="./checkpoints/tcps/tcps_model.pt")
    parser.add_argument("--temporal_encoder_path", type=str, default="./checkpoints/MedQA_encoder")
    parser.add_argument("--llm_model", type=str, required=True)
    parser.add_argument("--output", type=str, default="results/tcps_eval.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    return parser.parse_args()


class TCPSRetriever:
    """Retriever using Temporal-Conditioned Path Scorer."""
    
    def __init__(
        self,
        tcps_path: str,
        temporal_encoder_path: str,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_auth: Tuple[str, str] = ("neo4j", "admin")
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load TCPS model
        print(f"Loading TCPS from {tcps_path}...")
        checkpoint = torch.load(tcps_path, map_location=self.device)
        self.relation2id = checkpoint['relation2id']
        
        # Initialize components
        self.relation_emb = RelationEmbedding(len(self.relation2id), 64).to(self.device)
        self.temporal_cond = TemporalConditioner(5, 64).to(self.device)
        self.path_scorer = PathScorer(64, 64, 128).to(self.device)
        
        # Load weights
        self.relation_emb.load_state_dict(checkpoint['relation_emb'])
        self.temporal_cond.load_state_dict(checkpoint['temporal_cond'])
        self.path_scorer.load_state_dict(checkpoint['path_scorer'])
        
        self.relation_emb.eval()
        self.temporal_cond.eval()
        self.path_scorer.eval()
        
        # Load temporal encoder for query classification
        print(f"Loading Temporal Encoder from {temporal_encoder_path}...")
        self.temporal_encoder = TimeAwareEncoder.from_pretrained(
            temporal_encoder_path, 
            model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        ).to(self.device)
        self.temporal_encoder.eval()
        self.temporal_tokenizer = AutoTokenizer.from_pretrained(temporal_encoder_path)
        
        # Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        
        # Improved Entity Extractor (with stopword filtering)
        self.entity_extractor = ImprovedEntityExtractor(min_similarity=0.70)
        
        self.temporal_map = {"Acute": 0, "Chronic": 1, "Recurrent": 2, "Progressive": 3, "None": 4}
    
    def get_temporal_type(self, query: str) -> int:
        """Get temporal type from query using trained encoder."""
        inputs = self.temporal_tokenizer(
            query, return_tensors="pt", 
            padding=True, truncation=True, max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits, _ = self.temporal_encoder(inputs['input_ids'], inputs['attention_mask'])
            return torch.argmax(logits, dim=-1).item()
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract CUIs using improved entity extractor."""
        return self.entity_extractor.extract_cuis(text)
    
    def get_candidate_paths(self, seed_cuis: List[str], max_paths: int = 50) -> List[Dict]:
        """Retrieve candidate paths from Neo4j."""
        if not seed_cuis:
            return []
        
        paths = []
        with self.driver.session() as session:
            query = """
            MATCH path = (start:Concept)-[*1..2]->(end:Concept)
            WHERE start.CUI IN $seed_cuis
            RETURN [r IN relationships(path) | r.RELA] as relations,
                   [n IN nodes(path) | n.name] as node_names
            LIMIT $max_paths
            """
            result = session.run(query, seed_cuis=seed_cuis, max_paths=max_paths)
            
            for record in result:
                relations = [r for r in record["relations"] if r is not None]
                if relations:
                    paths.append({
                        "relations": relations,
                        "nodes": record["node_names"]
                    })
        
        return paths
    
    def score_paths(self, paths: List[Dict], temporal_type: int) -> List[Tuple[Dict, float]]:
        """Score paths conditioned on temporal type."""
        if not paths:
            return []
        
        temporal_cond = self.temporal_cond(torch.tensor([temporal_type]).to(self.device))
        
        scored = []
        for path in paths:
            rel_ids = [self.relation2id.get(r, self.relation2id['UNK']) for r in path['relations']]
            
            # Pad to length 5
            while len(rel_ids) < 5:
                rel_ids.append(self.relation2id['PAD'])
            rel_ids = rel_ids[:5]
            
            rel_tensor = torch.tensor([rel_ids]).to(self.device)
            rel_emb = self.relation_emb(rel_tensor)
            
            with torch.no_grad():
                score = self.path_scorer(rel_emb, temporal_cond)
            
            scored.append((path, score.item()))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
    
    def retrieve(self, query: str, top_k: int = 5) -> Tuple[int, List[Dict]]:
        """Main retrieval method."""
        # Get temporal type
        temporal_type = self.get_temporal_type(query)
        
        # Extract entities
        seed_cuis = self.extract_entities(query)
        
        # Get candidate paths
        candidate_paths = self.get_candidate_paths(seed_cuis)
        
        # Score and rank
        scored_paths = self.score_paths(candidate_paths, temporal_type)
        
        # Return top-k
        return temporal_type, [p for p, s in scored_paths[:top_k]]
    
    def close(self):
        self.driver.close()


def format_path_context(paths: List[Dict], temporal_type: int) -> str:
    """Format retrieved paths as natural language context for LLM."""
    temporal_names = ["Acute", "Chronic", "Recurrent", "Progressive", "None"]
    temporal_descriptions = {
        "Acute": "sudden onset, short duration",
        "Chronic": "long-term, persistent",
        "Recurrent": "episodic, comes and goes",
        "Progressive": "worsening over time",
        "None": "general"
    }
    
    if not paths:
        return f"This appears to be a {temporal_names[temporal_type]} ({temporal_descriptions[temporal_names[temporal_type]]}) clinical scenario. No additional knowledge paths were retrieved."
    
    # Convert paths to natural language statements
    statements = []
    for path in paths:
        nodes = path.get('nodes', [])
        relations = path.get('relations', [])
        
        if len(nodes) >= 2 and relations:
            # Convert relation names to readable format
            rel = relations[0].replace('_', ' ')
            statement = f"{nodes[0]} {rel} {nodes[1]}"
            if len(nodes) > 2 and len(relations) > 1:
                rel2 = relations[1].replace('_', ' ')
                statement += f", which {rel2} {nodes[2]}"
            statements.append(statement)
    
    if not statements:
        return f"This appears to be a {temporal_names[temporal_type]} clinical scenario."
    
    context = f"""Clinical Context Analysis:
- Temporal pattern: {temporal_names[temporal_type]} ({temporal_descriptions[temporal_names[temporal_type]]})
- Relevant medical knowledge:
"""
    for i, stmt in enumerate(statements[:5], 1):
        context += f"  {i}. {stmt}\n"
    
    return context


def format_prompt(question: str, options: Dict, context: str) -> str:
    """Format prompt for LLM with improved instructions."""
    option_str = "\n".join(f"{k}: {v}" for k, v in sorted(options.items()))
    
    prompt = f"""You are an expert clinical reasoning AI. Answer the following medical question using the provided clinical context.

{context}

Question: {question}

Options:
{option_str}

Instructions:
1. Consider the temporal pattern (acute/chronic/progressive) in your reasoning
2. Use the relevant medical knowledge provided
3. Select the most appropriate answer based on clinical reasoning
4. Provide your final answer as a single letter wrapped in <a> tags, e.g., <a>A</a>

Answer:"""
    return prompt


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
    retriever = TCPSRetriever(
        args.tcps_path,
        args.temporal_encoder_path
    )
    
    # Retrieve contexts
    print("Retrieving with TCPS...")
    contexts = []
    temporal_types = []
    for item in tqdm(items, desc="Retrieving"):
        temporal_type, paths = retriever.retrieve(item.get('question', ''), top_k=args.top_k)
        context = format_path_context(paths, temporal_type)
        contexts.append(context)
        temporal_types.append(temporal_type)
    
    retriever.close()
    
    # Prepare prompts
    prompts = []
    for item, context in zip(items, contexts):
        prompt = format_prompt(item.get('question', ''), item.get('options', {}), context)
        prompts.append(prompt)
    
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
    
    sampling_params = SamplingParams(
        temperature=0.5,
        max_tokens=1024,
        skip_special_tokens=True
    )
    
    # Generate
    print("Generating answers...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Evaluate
    correct = 0
    results = []
    
    for i, output in enumerate(outputs):
        generated = output.outputs[0].text
        predicted = extract_answer(generated)
        ground_truth = items[i].get('answer', '').strip().upper()
        
        is_correct = predicted == ground_truth
        if is_correct:
            correct += 1
        
        results.append({
            "question": items[i].get('question'),
            "temporal_type": temporal_types[i],
            "context": contexts[i],
            "ground_truth": ground_truth,
            "predicted": predicted,
            "is_correct": is_correct
        })
    
    accuracy = correct / len(items) if items else 0
    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{len(items)})")
    
    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)

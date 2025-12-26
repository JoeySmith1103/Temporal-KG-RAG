"""
Preprocess data for TCPS v2 training.

This script:
1. Takes multi-label annotated data
2. Extracts entities using QuickUMLS
3. Retrieves paths from Neo4j
4. Identifies positive paths (related to answer) and negative paths
5. Saves preprocessed data for fast training
"""
import os
import json
import argparse
from neo4j import GraphDatabase
from quickumls import get_quickumls_client
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import random


class PathPreprocessor:
    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_auth: Tuple[str, str] = ("neo4j", "admin")
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.matcher = get_quickumls_client()
    
    def extract_cuis(self, text: str) -> List[str]:
        """Extract CUIs from text using QuickUMLS."""
        cuis = []
        matches = self.matcher.match(text, best_match=True, ignore_syntax=False)
        for group in matches:
            if group:
                cuis.append(group[0]['cui'])
        return cuis[:10]  # Limit to avoid too many
    
    def get_paths(self, seed_cuis: List[str], max_paths: int = 100) -> List[List[str]]:
        """Get paths from Neo4j starting from seed CUIs."""
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
    
    def is_positive(self, path: List[str], answer_text: str) -> bool:
        """Check if path is related to the answer."""
        if not answer_text:
            return False
        
        path_text = " ".join(path).lower()
        answer_words = [w for w in answer_text.lower().split() if len(w) > 3]
        
        # Path is positive if any answer word appears
        for word in answer_words:
            if word in path_text:
                return True
        return False
    
    def process_item(self, item: Dict) -> Optional[Dict]:
        """Process a single data item."""
        question = item.get('question', '')
        labels = item.get('temporal_labels', {})
        
        if labels.get('parse_error'):
            return None
        
        primary_type = labels.get('primary_type', 'None')
        answer_key = item.get('answer', '')
        options = item.get('options', {})
        answer_text = options.get(answer_key, '')
        
        # Extract CUIs
        cuis = self.extract_cuis(question)
        if not cuis:
            return None
        
        # Get paths
        paths = self.get_paths(cuis)
        if not paths:
            return None
        
        # Split into positive and negative
        positive_paths = []
        negative_paths = []
        
        for path in paths:
            if self.is_positive(path, answer_text):
                positive_paths.append(path)
            else:
                negative_paths.append(path)
        
        # Need at least 1 positive and 1 negative
        if not positive_paths:
            # Use first path as pseudo-positive
            if paths:
                positive_paths = paths[:1]
                negative_paths = paths[1:20]
            else:
                return None
        
        if not negative_paths:
            return None
        
        # Format paths as "|" separated strings
        pos_path = " | ".join(positive_paths[0])
        neg_paths = [" | ".join(p) for p in negative_paths[:10]]
        
        return {
            'question': question,
            'primary_type': primary_type,
            'temporal_cues': labels.get('temporal_cues', []),
            'positive_path': pos_path,
            'negative_paths': neg_paths,
            'answer': answer_key,
            'answer_text': answer_text
        }
    
    def close(self):
        self.driver.close()


def preprocess(args):
    print(f"Loading data from {args.input_path}...")
    data = []
    with open(args.input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if args.limit:
        data = data[:args.limit]
    print(f"Processing {len(data)} samples...")
    
    preprocessor = PathPreprocessor()
    
    os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
    
    valid_count = 0
    with open(args.output_path, 'w', encoding='utf-8') as f_out:
        for item in tqdm(data, desc="Preprocessing"):
            processed = preprocessor.process_item(item)
            if processed:
                f_out.write(json.dumps(processed, ensure_ascii=False) + '\n')
                valid_count += 1
    
    preprocessor.close()
    
    print(f"Done! Saved {valid_count} valid samples to {args.output_path}")
    print(f"Skip rate: {100 * (1 - valid_count / len(data)):.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Multi-label annotated data")
    parser.add_argument("--output_path", type=str, required=True, help="Output preprocessed data")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    preprocess(args)

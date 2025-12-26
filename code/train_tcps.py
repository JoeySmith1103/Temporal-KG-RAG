"""
Training script for Temporal-Conditioned Path Scorer (TCPS)

Training objective:
- Given a query with temporal type and its correct answer entity
- Learn to score paths that lead to the answer higher
- Conditioned on the temporal type

Loss: Contrastive loss between paths that reach answer vs paths that don't
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from neo4j import GraphDatabase
from quickumls import get_quickumls_client
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple

from temporal_path_scorer import (
    TemporalConditionedRetriever, 
    build_relation2id,
    RelationEmbedding,
    TemporalConditioner,
    PathScorer
)


class PathDataset(Dataset):
    """
    Dataset that generates (query, temporal_type, positive_paths, negative_paths) tuples.
    Positive paths: paths that lead to or are related to the correct answer
    Negative paths: random paths from the subgraph
    """
    def __init__(
        self, 
        data_path: str,
        relation2id: Dict[str, int],
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_auth: Tuple[str, str] = ("neo4j", "admin"),
        max_samples: int = None
    ):
        self.relation2id = relation2id
        self.driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
        self.matcher = get_quickumls_client()
        
        # Load data
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Support both old format (teacher_label) and new multi-label format
                    if 'temporal_labels' in item:
                        # Multi-label format
                        labels = item['temporal_labels']
                        if labels.get('parse_error'):
                            continue  # Skip items with parse errors
                        temporal = labels.get('primary_type', 'None')
                        temporal_cues = labels.get('temporal_cues', [])
                    else:
                        # Old single-label format
                        temporal = item.get('teacher_label') or item.get('temporal_label', 'None')
                        temporal_cues = []
                    
                    self.data.append({
                        'question': item['question'],
                        'temporal_label': temporal,
                        'temporal_cues': temporal_cues,  # Store for potential future use
                        'answer': item.get('answer', ''),
                        'options': item.get('options', {})
                    })
        
        if max_samples:
            self.data = self.data[:max_samples]
            
        self.temporal_map = {"Acute": 0, "Chronic": 1, "Recurrent": 2, "Progressive": 3, "None": 4}
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract CUIs from text using QuickUMLS."""
        cuis = []
        matches = self.matcher.match(text, best_match=True, ignore_syntax=False)
        for group in matches:
            if group:
                cuis.append(group[0]['cui'])
        return cuis
    
    def get_paths_to_target(self, seed_cuis: List[str], target_text: str) -> List[List[str]]:
        """Get paths from seeds that mention concepts similar to target."""
        paths = []
        with self.driver.session() as session:
            query = """
            MATCH path = (start:Concept)-[*1..2]->(end:Concept)
            WHERE start.CUI IN $seed_cuis
              AND (toLower(end.name) CONTAINS toLower($target) 
                   OR toLower($target) CONTAINS toLower(end.name))
            RETURN [r IN relationships(path) | r.RELA] as relations
            LIMIT 20
            """
            result = session.run(query, seed_cuis=seed_cuis, target=target_text[:50])
            for record in result:
                rels = [r for r in record["relations"] if r]
                if rels:
                    paths.append(rels)
        return paths
    
    def get_random_paths(self, seed_cuis: List[str], exclude_paths: List[List[str]]) -> List[List[str]]:
        """Get random paths as negatives."""
        paths = []
        with self.driver.session() as session:
            query = """
            MATCH path = (start:Concept)-[*1..2]->(end:Concept)
            WHERE start.CUI IN $seed_cuis
            RETURN [r IN relationships(path) | r.RELA] as relations
            LIMIT 50
            """
            result = session.run(query, seed_cuis=seed_cuis)
            for record in result:
                rels = [r for r in record["relations"] if r]
                if rels and rels not in exclude_paths:
                    paths.append(rels)
        return paths[:20]  # Limit negatives
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract entities from question
        seed_cuis = self.extract_entities(item['question'])
        
        # Get answer text
        answer_key = item['answer']
        answer_text = item['options'].get(answer_key, '')
        
        # Get positive paths (paths related to answer)
        positive_paths = self.get_paths_to_target(seed_cuis, answer_text)
        
        # Get negative paths
        negative_paths = self.get_random_paths(seed_cuis, positive_paths)
        
        # Convert to IDs
        temporal_id = self.temporal_map.get(item['temporal_label'], 4)
        
        return {
            'question': item['question'],
            'temporal_id': temporal_id,
            'positive_paths': positive_paths,
            'negative_paths': negative_paths
        }
    
    def close(self):
        self.driver.close()


def path_to_tensor(path: List[str], relation2id: Dict[str, int], max_len: int = 5) -> torch.Tensor:
    """Convert a path (list of relation names) to tensor of IDs."""
    ids = []
    for rel in path[:max_len]:
        ids.append(relation2id.get(rel, relation2id['UNK']))
    # Pad
    while len(ids) < max_len:
        ids.append(relation2id['PAD'])
    return torch.tensor(ids)


def train(args):
    print("Building relation2id mapping...")
    relation2id = build_relation2id(args.neo4j_uri, (args.neo4j_user, args.neo4j_pass))
    print(f"Found {len(relation2id)} relation types")
    
    print(f"Loading dataset from {args.data_path}...")
    dataset = PathDataset(
        args.data_path,
        relation2id,
        args.neo4j_uri,
        (args.neo4j_user, args.neo4j_pass),
        max_samples=args.max_samples
    )
    
    # Initialize model components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    relation_emb = RelationEmbedding(len(relation2id), args.embed_dim).to(device)
    temporal_cond = TemporalConditioner(5, args.embed_dim).to(device)
    path_scorer = PathScorer(args.embed_dim, args.embed_dim, args.hidden_dim).to(device)
    
    # Optimizer
    params = list(relation_emb.parameters()) + \
             list(temporal_cond.parameters()) + \
             list(path_scorer.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    
    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0
        valid_samples = 0
        
        for i in tqdm(range(len(dataset)), desc=f"Epoch {epoch+1}"):
            try:
                sample = dataset[i]
            except Exception as e:
                continue
            
            pos_paths = sample['positive_paths']
            neg_paths = sample['negative_paths']
            temporal_id = sample['temporal_id']
            
            if not pos_paths or not neg_paths:
                continue
            
            # Get temporal conditioning
            temp_cond = temporal_cond(torch.tensor([temporal_id]).to(device))
            
            # Score positive paths
            pos_scores = []
            for path in pos_paths[:5]:  # Limit
                path_tensor = path_to_tensor(path, relation2id).unsqueeze(0).to(device)
                path_emb = relation_emb(path_tensor)
                score = path_scorer(path_emb, temp_cond)
                pos_scores.append(score)
            
            # Score negative paths
            neg_scores = []
            for path in neg_paths[:5]:  # Limit
                path_tensor = path_to_tensor(path, relation2id).unsqueeze(0).to(device)
                path_emb = relation_emb(path_tensor)
                score = path_scorer(path_emb, temp_cond)
                neg_scores.append(score)
            
            if not pos_scores or not neg_scores:
                continue
            
            # Contrastive loss: positive should score higher than negatives
            pos_score = torch.mean(torch.stack(pos_scores))
            neg_score = torch.mean(torch.stack(neg_scores))
            
            # Margin ranking loss
            loss = F.relu(args.margin - pos_score + neg_score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            valid_samples += 1
        
        if valid_samples > 0:
            avg_loss = total_loss / valid_samples
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Valid samples = {valid_samples}")
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        'relation_emb': relation_emb.state_dict(),
        'temporal_cond': temporal_cond.state_dict(),
        'path_scorer': path_scorer.state_dict(),
        'relation2id': relation2id
    }, os.path.join(args.output_dir, 'tcps_model.pt'))
    print(f"Model saved to {args.output_dir}/tcps_model.pt")
    
    dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints/tcps")
    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687")
    parser.add_argument("--neo4j_user", type=str, default="neo4j")
    parser.add_argument("--neo4j_pass", type=str, default="admin")
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    train(args)

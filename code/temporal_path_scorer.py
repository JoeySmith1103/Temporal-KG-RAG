"""
Temporal-Conditioned Path Scorer (TCPS)

Core idea: Learn to score KG paths conditioned on query's temporal type.
The temporal signal comes from the QUERY, not the KG.
The model learns which path patterns are useful for each temporal type.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional
from neo4j import GraphDatabase
from collections import defaultdict

class RelationEmbedding(nn.Module):
    """Learnable embeddings for UMLS relation types."""
    def __init__(self, num_relations: int, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_relations, embed_dim)
        
    def forward(self, relation_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(relation_ids)


class TemporalConditioner(nn.Module):
    """
    Encodes query temporal type into a conditioning vector.
    This vector will modulate path scoring.
    """
    def __init__(self, num_temporal_types: int = 5, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_temporal_types, embed_dim)
        # Temporal types: 0=Acute, 1=Chronic, 2=Recurrent, 3=Progressive, 4=None
        
    def forward(self, temporal_type_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(temporal_type_ids)


class PathScorer(nn.Module):
    """
    Scores a path based on:
    1. Path's relation sequence embedding
    2. Temporal conditioning from query
    
    Score = f(path_embedding, temporal_condition)
    """
    def __init__(self, relation_dim: int = 64, temporal_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.path_encoder = nn.LSTM(
            input_size=relation_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        # Combine path encoding with temporal condition
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + temporal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, path_relations: torch.Tensor, temporal_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            path_relations: [batch, path_len, relation_dim] - embedded relations along path
            temporal_cond: [batch, temporal_dim] - temporal conditioning vector
        Returns:
            scores: [batch, 1] - path relevance scores
        """
        # Encode path with LSTM
        _, (h_n, _) = self.path_encoder(path_relations)
        # Concatenate forward and backward hidden states
        path_emb = torch.cat([h_n[0], h_n[1]], dim=-1)  # [batch, hidden*2]
        
        # Combine with temporal condition
        combined = torch.cat([path_emb, temporal_cond], dim=-1)
        
        # Score
        scores = self.scorer(combined)
        return scores


class TemporalConditionedRetriever(nn.Module):
    """
    Main retrieval module that:
    1. Extracts temporal type from query using pre-trained encoder
    2. Retrieves candidate paths from KG
    3. Scores paths conditioned on temporal type
    4. Returns top-k paths as context
    """
    def __init__(
        self,
        temporal_encoder_path: str,
        relation2id: Dict[str, int],
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_auth: Tuple[str, str] = ("neo4j", "admin"),
        embed_dim: int = 64,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        # Pre-trained temporal encoder (SapBERT-based)
        self.temporal_encoder = AutoModel.from_pretrained(temporal_encoder_path)
        self.temporal_tokenizer = AutoTokenizer.from_pretrained(temporal_encoder_path)
        self.temporal_classifier = nn.Linear(self.temporal_encoder.config.hidden_size, 5)
        
        # Learnable components
        self.relation_embedding = RelationEmbedding(len(relation2id), embed_dim)
        self.temporal_conditioner = TemporalConditioner(5, embed_dim)
        self.path_scorer = PathScorer(embed_dim, embed_dim, hidden_dim)
        
        # Relation mapping
        self.relation2id = relation2id
        self.id2relation = {v: k for k, v in relation2id.items()}
        
        # Neo4j connection
        self.neo4j_uri = neo4j_uri
        self.neo4j_auth = neo4j_auth
        
    def get_temporal_type(self, query: str) -> Tuple[int, torch.Tensor]:
        """Extract temporal type and embedding from query."""
        inputs = self.temporal_tokenizer(
            query, return_tensors="pt", 
            padding=True, truncation=True, max_length=128
        )
        with torch.no_grad():
            outputs = self.temporal_encoder(**inputs)
            pooled = outputs.last_hidden_state[:, 0, :]
        
        logits = self.temporal_classifier(pooled)
        temporal_type = torch.argmax(logits, dim=-1)
        return temporal_type.item(), pooled
    
    def retrieve_candidate_paths(self, seed_cuis: List[str], max_depth: int = 2) -> List[Dict]:
        """
        Retrieve candidate paths from Neo4j starting from seed CUIs.
        Returns paths as list of (relations, nodes) tuples.
        """
        driver = GraphDatabase.driver(self.neo4j_uri, auth=self.neo4j_auth)
        paths = []
        
        with driver.session() as session:
            # Query for paths up to max_depth hops
            query = """
            MATCH path = (start:Concept)-[*1..2]->(end:Concept)
            WHERE start.CUI IN $seed_cuis
            RETURN [r IN relationships(path) | r.RELA] as relations,
                   [n IN nodes(path) | n.name] as node_names,
                   [n IN nodes(path) | n.CUI] as node_cuis
            LIMIT 100
            """
            result = session.run(query, seed_cuis=seed_cuis)
            
            for record in result:
                relations = [r for r in record["relations"] if r is not None]
                if relations:  # Only keep paths with valid relations
                    paths.append({
                        "relations": relations,
                        "nodes": record["node_names"],
                        "cuis": record["node_cuis"]
                    })
        
        driver.close()
        return paths
    
    def score_paths(
        self, 
        paths: List[Dict], 
        temporal_type: int
    ) -> List[Tuple[Dict, float]]:
        """Score paths based on temporal conditioning."""
        if not paths:
            return []
        
        # Get temporal conditioning
        temporal_cond = self.temporal_conditioner(
            torch.tensor([temporal_type])
        )  # [1, embed_dim]
        
        scored_paths = []
        for path in paths:
            # Convert relations to IDs
            rel_ids = []
            for rel in path["relations"]:
                if rel in self.relation2id:
                    rel_ids.append(self.relation2id[rel])
                else:
                    rel_ids.append(self.relation2id.get("UNK", 0))
            
            if not rel_ids:
                continue
                
            # Embed relations
            rel_tensor = torch.tensor([rel_ids])  # [1, path_len]
            rel_emb = self.relation_embedding(rel_tensor)  # [1, path_len, embed_dim]
            
            # Score
            with torch.no_grad():
                score = self.path_scorer(rel_emb, temporal_cond)
            
            scored_paths.append((path, score.item()))
        
        # Sort by score descending
        scored_paths.sort(key=lambda x: x[1], reverse=True)
        return scored_paths
    
    def retrieve(
        self, 
        query: str, 
        seed_cuis: List[str], 
        top_k: int = 5
    ) -> List[Dict]:
        """
        Main retrieval method:
        1. Get temporal type from query
        2. Retrieve candidate paths
        3. Score and rank paths
        4. Return top-k
        """
        # Get temporal type
        temporal_type, _ = self.get_temporal_type(query)
        
        # Retrieve candidates
        candidate_paths = self.retrieve_candidate_paths(seed_cuis)
        
        # Score paths
        scored_paths = self.score_paths(candidate_paths, temporal_type)
        
        # Return top-k
        return [path for path, score in scored_paths[:top_k]]


# Relation to ID mapping (will be populated from Neo4j)
def build_relation2id(neo4j_uri: str, neo4j_auth: Tuple[str, str]) -> Dict[str, int]:
    """Build relation to ID mapping from Neo4j."""
    driver = GraphDatabase.driver(neo4j_uri, auth=neo4j_auth)
    relations = set()
    
    with driver.session() as session:
        result = session.run("""
            MATCH ()-[r]->()
            WHERE r.RELA IS NOT NULL
            RETURN DISTINCT r.RELA as rela
        """)
        for record in result:
            relations.add(record["rela"])
    
    driver.close()
    
    relation2id = {"UNK": 0, "PAD": 1}
    for i, rel in enumerate(sorted(relations)):
        relation2id[rel] = i + 2
    
    return relation2id


if __name__ == "__main__":
    # Test building relation2id
    print("Building relation2id mapping...")
    relation2id = build_relation2id("bolt://localhost:7687", ("neo4j", "admin"))
    print(f"Found {len(relation2id)} relation types")
    print("Sample relations:", list(relation2id.items())[:10])

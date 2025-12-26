"""
TCPS v2: Temporal-Augmented Query-Path Matching

Key idea:
1. Encode query with SapBERT → semantic embedding
2. Encode temporal type → temporal embedding  
3. Augment: query_aug = query_emb + α * W(temporal_emb)
4. Encode path nodes with SapBERT → path embedding
5. Score = cosine_sim(query_aug, path_emb)

Training: Contrastive learning where positive paths are semantically related to correct answers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional


class TemporalQueryAugmenter(nn.Module):
    """
    Augments query embedding with temporal information.
    query_aug = query_emb + α * projection(temporal_emb)
    """
    def __init__(self, query_dim: int = 768, temporal_classes: int = 5, temporal_dim: int = 64):
        super().__init__()
        self.temporal_embedding = nn.Embedding(temporal_classes, temporal_dim)
        self.projection = nn.Linear(temporal_dim, query_dim)  # Project to same dim as query
        self.alpha = nn.Parameter(torch.tensor(0.1))  # Learnable scaling factor
        
    def forward(self, query_emb: torch.Tensor, temporal_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb: [batch, query_dim] - SapBERT embedding of query
            temporal_type: [batch] - temporal class (0-4)
        Returns:
            augmented_query: [batch, query_dim]
        """
        temp_emb = self.temporal_embedding(temporal_type)  # [batch, temporal_dim]
        temp_projected = self.projection(temp_emb)  # [batch, query_dim]
        augmented = query_emb + self.alpha * temp_projected
        return augmented


class SemanticPathEncoder(nn.Module):
    """
    Encodes a path (sequence of node names) into a single embedding.
    Uses SapBERT to encode each node, then aggregates.
    """
    def __init__(self, encoder_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        # Freeze encoder for efficiency (can be fine-tuned later)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Learnable aggregation
        self.node_attention = nn.MultiheadAttention(
            embed_dim=768, num_heads=4, batch_first=True
        )
        self.output_proj = nn.Linear(768, 768)
    
    def encode_nodes(self, node_names: List[str], device: torch.device) -> torch.Tensor:
        """Encode a list of node names into embeddings."""
        if not node_names:
            return torch.zeros(1, 768, device=device)
        
        inputs = self.tokenizer(
            node_names, return_tensors="pt", 
            padding=True, truncation=True, max_length=64
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # Use [CLS] token
            node_embs = outputs.last_hidden_state[:, 0, :]  # [num_nodes, 768]
        
        return node_embs
    
    def forward(self, paths: List[List[str]], device: torch.device) -> torch.Tensor:
        """
        Encode multiple paths into embeddings.
        
        Args:
            paths: List of paths, each path is a list of node names
            device: torch device
        Returns:
            path_embs: [num_paths, 768]
        """
        path_embs = []
        for path_nodes in paths:
            if not path_nodes:
                path_embs.append(torch.zeros(768, device=device))
                continue
            
            node_embs = self.encode_nodes(path_nodes, device)  # [num_nodes, 768]
            
            # Simple aggregation: mean pooling
            # (Can use attention for more sophisticated aggregation)
            path_emb = node_embs.mean(dim=0)
            path_embs.append(path_emb)
        
        return torch.stack(path_embs)  # [num_paths, 768]


class TemporalAugmentedRetriever(nn.Module):
    """
    Main model for temporal-augmented query-path matching.
    """
    def __init__(
        self,
        encoder_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        temporal_classes: int = 5
    ):
        super().__init__()
        
        # Query encoder (SapBERT)
        self.query_encoder = AutoModel.from_pretrained(encoder_name)
        self.query_tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        # Freeze query encoder
        for param in self.query_encoder.parameters():
            param.requires_grad = False
        
        # Temporal augmenter (trainable)
        self.temporal_augmenter = TemporalQueryAugmenter(
            query_dim=768, 
            temporal_classes=temporal_classes,
            temporal_dim=128
        )
        
        # Path encoder
        self.path_encoder = SemanticPathEncoder(encoder_name)
        
    def encode_query(self, query: str, device: torch.device) -> torch.Tensor:
        """Encode a query string."""
        inputs = self.query_tokenizer(
            query, return_tensors="pt",
            padding=True, truncation=True, max_length=256
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.query_encoder(**inputs)
            query_emb = outputs.last_hidden_state[:, 0, :]  # [1, 768]
        
        return query_emb
    
    def score_paths(
        self,
        query: str,
        temporal_type: int,
        paths: List[List[str]],
        device: torch.device
    ) -> torch.Tensor:
        """
        Score paths based on temporal-augmented query.
        
        Args:
            query: question text
            temporal_type: 0-4 (Acute/Chronic/Recurrent/Progressive/None)
            paths: list of paths, each path is list of node names
            device: torch device
        Returns:
            scores: [num_paths] similarity scores
        """
        # Encode query
        query_emb = self.encode_query(query, device)  # [1, 768]
        
        # Augment with temporal
        temporal_tensor = torch.tensor([temporal_type], device=device)
        query_aug = self.temporal_augmenter(query_emb, temporal_tensor)  # [1, 768]
        
        # Encode paths
        path_embs = self.path_encoder(paths, device)  # [num_paths, 768]
        
        # Compute cosine similarity
        query_aug_norm = F.normalize(query_aug, dim=-1)  # [1, 768]
        path_embs_norm = F.normalize(path_embs, dim=-1)  # [num_paths, 768]
        
        scores = torch.mm(query_aug_norm, path_embs_norm.t()).squeeze(0)  # [num_paths]
        
        return scores
    
    def get_trainable_params(self):
        """Get only trainable parameters."""
        return [p for p in self.temporal_augmenter.parameters() if p.requires_grad]


def test_model():
    """Quick test of the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model = TemporalAugmentedRetriever().to(device)
    
    query = "A 45-year-old man presents with sudden chest pain for 30 minutes."
    temporal_type = 0  # Acute
    paths = [
        ["Chest Pain", "Myocardial Infarction"],
        ["Chest Pain", "Angina", "Coronary Artery Disease"],
        ["Headache", "Migraine"],
    ]
    
    scores = model.score_paths(query, temporal_type, paths, device)
    print(f"Query: {query[:50]}...")
    print(f"Temporal: Acute")
    print(f"Path scores:")
    for i, (path, score) in enumerate(zip(paths, scores)):
        print(f"  {i+1}. {' -> '.join(path)}: {score.item():.4f}")


if __name__ == "__main__":
    test_model()

"""
TCPS v2 Batched Training with Full GPU Utilization

Key optimizations:
1. Pre-compute path data offline (no Neo4j during training)
2. Use DataLoader with proper batching
3. Batch SapBERT encoding
4. Use gradient accumulation if needed
"""
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import argparse
from typing import List, Dict, Tuple, Optional
import random


class TemporalQueryAugmenter(nn.Module):
    """Augments query embedding with temporal information."""
    
    def __init__(self, query_dim: int = 768, temporal_classes: int = 5, temporal_dim: int = 128):
        super().__init__()
        self.temporal_embedding = nn.Embedding(temporal_classes, temporal_dim)
        self.projection = nn.Sequential(
            nn.Linear(temporal_dim, query_dim),
            nn.LayerNorm(query_dim),
            nn.Tanh()
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, query_emb: torch.Tensor, temporal_type: torch.Tensor) -> torch.Tensor:
        temp_emb = self.temporal_embedding(temporal_type)
        temp_projected = self.projection(temp_emb)
        augmented = query_emb + self.alpha * temp_projected
        return augmented


class BatchedTCPSModel(nn.Module):
    """
    Batched TCPS model for efficient GPU training.
    """
    def __init__(self, encoder_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
        super().__init__()
        
        # Shared encoder for query and paths
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        
        # Freeze encoder (only train temporal augmenter)
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Temporal augmenter (trainable)
        self.temporal_augmenter = TemporalQueryAugmenter()
        
    def encode_batch(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """Encode a batch of texts."""
        if not texts:
            return torch.zeros(0, 768, device=device)
        
        inputs = self.tokenizer(
            texts, return_tensors="pt",
            padding=True, truncation=True, max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        
        return embeddings
    
    def forward_batch(
        self,
        query_embs: torch.Tensor,      # [batch, 768]
        temporal_types: torch.Tensor,   # [batch]
        pos_path_embs: torch.Tensor,    # [batch, 768] - one positive per query
        neg_path_embs: torch.Tensor     # [batch, num_neg, 768] - multiple negatives
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a batch.
        Returns positive and negative scores.
        """
        # Augment queries with temporal
        query_aug = self.temporal_augmenter(query_embs, temporal_types)
        
        # Normalize
        query_aug = F.normalize(query_aug, dim=-1)
        pos_path_embs = F.normalize(pos_path_embs, dim=-1)
        neg_path_embs = F.normalize(neg_path_embs, dim=-1)
        
        # Positive scores: [batch]
        pos_scores = (query_aug * pos_path_embs).sum(dim=-1)
        
        # Negative scores: [batch, num_neg]
        neg_scores = torch.bmm(neg_path_embs, query_aug.unsqueeze(-1)).squeeze(-1)
        
        return pos_scores, neg_scores


class PreprocessedDataset(Dataset):
    """
    Dataset with pre-computed embeddings for fast training.
    Load pre-processed data (see preprocessing script below).
    """
    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        self.samples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))
        
        if max_samples:
            self.samples = self.samples[:max_samples]
        
        self.temporal_map = {"Acute": 0, "Chronic": 1, "Recurrent": 2, "Progressive": 3, "None": 4}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            'question': item['question'],
            'temporal_type': self.temporal_map.get(item.get('primary_type', 'None'), 4),
            'positive_path': item.get('positive_path', ''),  # String: "node1 | node2 | node3"
            'negative_paths': item.get('negative_paths', [])  # List of strings
        }


def collate_fn(batch):
    """Custom collate function for variable-length data."""
    questions = [item['question'] for item in batch]
    temporal_types = torch.tensor([item['temporal_type'] for item in batch])
    positive_paths = [item['positive_path'] for item in batch]
    
    # Get minimum number of negative paths across batch
    min_neg = min(len(item['negative_paths']) for item in batch) if batch else 0
    min_neg = max(min_neg, 1)  # At least 1
    
    negative_paths = [item['negative_paths'][:min_neg] for item in batch]
    
    return {
        'questions': questions,
        'temporal_types': temporal_types,
        'positive_paths': positive_paths,
        'negative_paths': negative_paths  # List of lists
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = PreprocessedDataset(args.data_path, args.max_samples)
    print(f"Loaded {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Initialize model
    model = BatchedTCPSModel().to(device)
    
    # Only train temporal augmenter
    trainable_params = list(model.temporal_augmenter.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    print(f"Starting training for {args.epochs} epochs...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        for epoch in range(args.epochs):
            total_loss = 0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
            for batch in pbar:
                try:
                    # Encode queries
                    query_embs = model.encode_batch(batch['questions'], device)
                    temporal_types = batch['temporal_types'].to(device)
                    
                    # Encode positive paths (convert node lists to text)
                    pos_path_texts = [p.replace(' | ', ' ') for p in batch['positive_paths']]
                    pos_path_embs = model.encode_batch(pos_path_texts, device)
                    
                    # Encode negative paths
                    # Flatten, encode, reshape
                    neg_paths_flat = []
                    num_neg = len(batch['negative_paths'][0]) if batch['negative_paths'] else 0
                    for negs in batch['negative_paths']:
                        neg_paths_flat.extend([n.replace(' | ', ' ') for n in negs])
                    
                    if neg_paths_flat:
                        neg_path_embs = model.encode_batch(neg_paths_flat, device)
                        neg_path_embs = neg_path_embs.view(len(batch['questions']), num_neg, -1)
                    else:
                        continue
                    
                    # Forward
                    pos_scores, neg_scores = model.forward_batch(
                        query_embs, temporal_types, pos_path_embs, neg_path_embs
                    )
                    
                    # InfoNCE-style contrastive loss
                    # Combine pos and neg scores, use cross-entropy
                    # Positive is always at index 0
                    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [batch, 1+num_neg]
                    labels = torch.zeros(len(batch['questions']), dtype=torch.long, device=device)  # All 0s
                    
                    loss = F.cross_entropy(all_scores / args.temperature, labels)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'alpha': f'{model.temporal_augmenter.alpha.item():.4f}'
                    })
                    
                except Exception as e:
                    print(f"Error in batch: {e}")
                    continue
            
            avg_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}, Alpha = {model.temporal_augmenter.alpha.item():.4f}")
            
            # Save checkpoint after each epoch
            save_path = os.path.join(args.output_dir, f'tcps_v2_epoch{epoch+1}.pt')
            torch.save({
                'temporal_augmenter': model.temporal_augmenter.state_dict(),
                'epoch': epoch + 1,
                'loss': avg_loss,
            }, save_path)
            print(f"Checkpoint saved to {save_path}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model...")
    except Exception as e:
        print(f"\nError: {e}. Saving current model...")
    
    # Final save
    save_path = os.path.join(args.output_dir, 'tcps_v2_model.pt')
    torch.save({
        'temporal_augmenter': model.temporal_augmenter.state_dict(),
    }, save_path)
    print(f"Final model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Pre-processed data path")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/tcps_v2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    
    train(args)

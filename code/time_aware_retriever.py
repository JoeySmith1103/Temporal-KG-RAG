
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from time_aware_encoder import TimeAwareEncoder
from knowledge_retriever import KnowledgeRetriever
from collections import defaultdict

class TimeAwareRetriever(KnowledgeRetriever):
    def __init__(self, encoder_path="./checkpoints/MedQA_encoder", neo4j_uri="bolt://localhost:7687", neo4j_auth=("neo4j", "admin")):
        super().__init__(neo4j_uri, neo4j_auth)
        
        print(f"Loading Time-Aware Encoder from {encoder_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = TimeAwareEncoder.from_pretrained(encoder_path, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_path)
        self.encoder.to(self.device)
        self.encoder.eval()
        
    def _get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            _, embedding = self.encoder(inputs["input_ids"], inputs["attention_mask"])
        return embedding

    def get_reranked_neighbors(self, query_text, cuis, top_k=10, alpha=0.5):
        """
        Retrieves 1-hop neighbors and re-ranks them based on temporal compatibility with the query.
        
        Args:
            query_text (str): The original question text.
            cuis (list): List of extracted entity CUIs.
            top_k (int): Number of neighbors to return after re-ranking.
            alpha (float): Weight for Temporal Score (0.0 to 1.0). 
                           Final Score = (1-alpha)*Rel + alpha*Temporal.
                           Note: 'Rel' here is binary (1.0) since it's a graph hop, 
                           so we mainly use Temporal score to sort.
        """
        # 1. Standard Retrieval (Recall)
        # neighbors = {source_cui: set( (rela, name, neighbor_cui) )}
        raw_neighbors_dict = self.get_one_hop_neighbors(cuis)
        
        if not raw_neighbors_dict:
            return []

        # Flatten neighbors for re-ranking
        # Candidate format: (source_cui, rela, neighbor_name, neighbor_cui)
        candidates = []
        for src_cui, triplets in raw_neighbors_dict.items():
            for rela, n_name, n_cui in triplets:
                candidates.append({
                    "source_cui": src_cui,
                    "rela": rela,
                    "name": n_name,
                    "cui": n_cui
                })
        
        if not candidates:
            return []

        # 2. Encode Query
        q_emb = self._get_embedding(query_text)

        # 3. Encode Candidates (Batching is better for speed, but doing simple loop for MVP)
        # For very large lists, we should batch.
        candidate_names = [c["name"] for c in candidates]
        
        # Batch processing
        batch_size = 32
        all_c_embs = []
        for i in range(0, len(candidate_names), batch_size):
            batch_texts = candidate_names[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                _, batch_embs = self.encoder(inputs["input_ids"], inputs["attention_mask"])
            all_c_embs.append(batch_embs)
            
        c_embs = torch.cat(all_c_embs, dim=0)

        # 4. Calculate Similarity
        # q_emb: [1, dim], c_embs: [N, dim]
        scores = F.cosine_similarity(q_emb, c_embs) # [N]
        
        # 5. Assign Scores
        ranked_candidates = []
        for i, cand in enumerate(candidates):
            temporal_score = scores[i].item()
            cand["temporal_score"] = temporal_score
            ranked_candidates.append(cand)
            
        # 6. Sort
        # We sort primarily by Temporal Score here because the Graph Retrieval already ensures Semantic Relevance (it's a valid link).
        ranked_candidates.sort(key=lambda x: x["temporal_score"], reverse=True)
        
        return ranked_candidates[:top_k]

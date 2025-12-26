"""
MRAG Baseline (Adapted for UMLS KG)

Implementation of "MRAG: A Modular Retrieval Framework for Time-Sensitive Question Answering"
adapted to use UMLS Knowledge Graph as the knowledge source.

Modules:
1. Question Processing: LLM decomposes question into main content + temporal constraint
2. Retrieval & Summarization: UMLS retrieval + LLM summarization
3. Semantic-Temporal Hybrid Ranking: Combines embedding similarity + keyword matching
"""

import os
import json
import re
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModel
from knowledge_retriever import KnowledgeRetriever


class MRAGBaseline:
    """
    MRAG (Modular Retrieval) baseline adapted for UMLS KG.
    
    Key Adaptation:
    - Original MRAG retrieves text passages with timestamps
    - We use UMLS KG nodes (no explicit timestamps)
    - Time_Match is computed by checking if temporal keywords appear in node descriptions
    """
    
    def __init__(
        self,
        embedding_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_auth: Tuple[str, str] = ("neo4j", "admin"),
        llm_client = None,  # Optional: external LLM client for decomposition
        device: str = None
    ):
        # Device setup
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Module 2: Knowledge Retriever (UMLS via Neo4j)
        self.retriever = KnowledgeRetriever(neo4j_uri, neo4j_auth)
        
        # Module 3: Embedding model for semantic similarity
        print(f"Loading embedding model: {embedding_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.embed_model = AutoModel.from_pretrained(embedding_model)
        self.embed_model.to(self.device)
        self.embed_model.eval()
        
        # Optional LLM client for Module 1 (if None, use rule-based extraction)
        self.llm_client = llm_client
        
        # Temporal keywords for rule-based extraction (fallback)
        self.temporal_keywords = {
            "acute": ["acute", "sudden", "abrupt", "rapid onset", "hours", "hour", "minutes", "minute"],
            "chronic": ["chronic", "long-standing", "prolonged", "years", "year", "months", "month", "persistent"],
            "recurrent": ["recurrent", "recurring", "episodic", "intermittent", "comes and goes"],
            "progressive": ["progressive", "worsening", "deteriorating", "gradual onset", "slowly"],
            "subacute": ["subacute", "days", "day", "weeks", "week"]
        }
    
    # =========================================================================
    # Module 1: Question Processing
    # =========================================================================
    
    def process_question(self, question: str) -> Dict[str, str]:
        """
        Module 1: Decompose question into main content and temporal constraint.
        
        Returns:
            dict with keys:
            - "main": Main content of the question (without temporal info)
            - "time": Extracted temporal constraint (e.g., "acute", "2 days")
            - "time_category": Normalized category (acute/chronic/recurrent/progressive/none)
        """
        if self.llm_client:
            return self._process_question_llm(question)
        else:
            return self._process_question_rule(question)
    
    def _process_question_rule(self, question: str) -> Dict[str, str]:
        """Rule-based temporal extraction using comprehensive patterns."""
        question_lower = question.lower()
        
        detected_category = "none"
        detected_keywords = []
        
        # Comprehensive time patterns (from create_time_datasets.py)
        ENG_TIME_PATTERNS = [
            # durations
            r"\bfor (the )?(past|last) \d+ (min|minute|hour|hr|day|d|week|wk|month|mo|year|yr)s?\b",
            r"\bover the past \d+ (day|week|month|year)s?\b",
            r"\b\d+\s*(min|minute|hour|hr|day|d|week|wk|month|mo|year|yr)s?\s*(ago|prior)\b",
            r"\b(in|within) the (past|last) \d+ (day|week|month|year)s?\b",
            r"\b\d+\s*(h|hr|hrs|d|wk|mo|yr)s?\b",
            # ranges
            r"\b\d+\s*(?:–|-|~)\s*\d+\s*(min|minute|hour|hr|day|d|week|wk|month|mo|year|yr)s?\b",
            r"\b\d+\s*(?:to|–|-|~)\s*\d+\s*(h|hr|hrs|d|wk|mo|yr)s?\b",
            # relative mentions
            r"\b(this (morning|afternoon|evening|week|month|year)|today|yesterday|last night|last (week|month|year))\b",
            r"\bsince\s+([A-Za-z]+\s+\d{1,2}(,\s*\d{4})?|yesterday|last (night|week|month|year)|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
            # clock & date
            r"\b\d{1,2}:\d{2}\s*(am|pm)?\b",
            r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Aug|Sept|Oct|Nov|Dec)\b\s+\d{1,2}(?:,\s*\d{4})?",
            # clinical phrases
            r"\b(x|\d+)[-\s]?(day|week|month|year) history\b",
            r"\bpost[- ]?op(?:erative)? day \d+\b",
            r"\b(followed by|after|prior to|subsequent to|preceded by)\b"
        ]
        
        # Extract all time expressions using comprehensive patterns
        for pattern in ENG_TIME_PATTERNS:
            matches = re.findall(pattern, question, re.IGNORECASE)
            if matches:
                # Flatten tuple matches and filter empty strings
                for m in matches:
                    if isinstance(m, tuple):
                        detected_keywords.extend([s for s in m if s and len(s) > 1])
                    elif m:
                        detected_keywords.append(m)
        
        # Also find the full matched strings for better context
        for pattern in ENG_TIME_PATTERNS:
            for match in re.finditer(pattern, question, re.IGNORECASE):
                full_match = match.group(0).strip()
                if full_match and full_match not in detected_keywords:
                    detected_keywords.append(full_match)
        
        # Deduplicate
        detected_keywords = list(dict.fromkeys(detected_keywords))
        
        # Determine category based on keywords
        for category, keywords in self.temporal_keywords.items():
            for kw in keywords:
                if kw in question_lower:
                    detected_category = category
                    break
            if detected_category != "none":
                break
        
        # Infer category from extracted time expressions if not found
        if detected_category == "none" and detected_keywords:
            time_str = " ".join(detected_keywords).lower()
            if any(x in time_str for x in ["hour", "minute", "hr", "min"]):
                detected_category = "acute"
            elif any(x in time_str for x in ["year", "yr"]):
                detected_category = "chronic"
            elif any(x in time_str for x in ["day", "week", "wk"]):
                detected_category = "subacute"
            elif any(x in time_str for x in ["month", "mo"]):
                detected_category = "chronic"
        
        return {
            "main": question,  # Keep full question for now
            "time": ", ".join(detected_keywords) if detected_keywords else "none",
            "time_category": detected_category
        }
    
    def _process_question_llm(self, question: str) -> Dict[str, str]:
        """LLM-based temporal extraction (more accurate but slower)."""
        prompt = f"""Extract the temporal information from this medical question.

Question: {question}

Return a JSON object with:
- "main": The main clinical content of the question
- "time": The specific temporal expression (e.g., "2 days", "acute onset")
- "time_category": One of [acute, chronic, recurrent, progressive, subacute, none]

JSON:"""
        
        try:
            response = self.llm_client.generate(prompt)
            # Parse JSON from response
            json_match = re.search(r'\{[^}]+\}', response)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"LLM extraction failed: {e}, falling back to rule-based")
        
        return self._process_question_rule(question)
    
    # =========================================================================
    # Module 2: Retrieval & Summarization
    # =========================================================================
    
    def retrieve_and_summarize(
        self,
        question: str,
        cuis: List[str],
        max_neighbors: int = 20
    ) -> List[Dict]:
        """
        Module 2: Retrieve UMLS neighbors and prepare summaries.
        
        Note: In original MRAG, LLM summarizes each passage.
        For UMLS, we use node names and relations as "summaries" since
        UMLS concepts are already concise.
        """
        if not cuis:
            return []
        
        # Get 1-hop neighbors from Neo4j
        raw_neighbors = self.retriever.get_one_hop_neighbors(cuis)
        
        # Flatten and format
        candidates = []
        for src_cui, triplets in raw_neighbors.items():
            for rela, neighbor_name, neighbor_cui in triplets:
                # Create "summary" from relation + name
                summary = f"{rela}: {neighbor_name}" if rela else neighbor_name
                candidates.append({
                    "source_cui": src_cui,
                    "cui": neighbor_cui,
                    "name": neighbor_name,
                    "relation": rela or "related_to",
                    "summary": summary
                })
        
        # Deduplicate by CUI
        seen_cuis = set()
        unique_candidates = []
        for c in candidates:
            if c["cui"] not in seen_cuis:
                seen_cuis.add(c["cui"])
                unique_candidates.append(c)
        
        return unique_candidates[:max_neighbors]
    
    # =========================================================================
    # Module 3: Semantic-Temporal Hybrid Ranking
    # =========================================================================
    
    def _get_embedding(self, texts: List[str]) -> torch.Tensor:
        """Get embeddings for a list of texts."""
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
            # Use CLS token or mean pooling
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings
    
    def hybrid_rank(
        self,
        question: str,
        candidates: List[Dict],
        temporal_info: Dict[str, str],
        alpha: float = 0.5,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Module 3: Semantic-Temporal Hybrid Ranking.
        
        Score = alpha * Semantic_Similarity + (1 - alpha) * Time_Match
        
        Args:
            question: Original question
            candidates: List of candidate nodes from Module 2
            temporal_info: Output from Module 1 (contains time_category and time)
            alpha: Weight for semantic score (0-1)
            top_k: Number of top candidates to return
        """
        if not candidates:
            return []
        
        # Get query embedding
        query_emb = self._get_embedding([question])  # [1, dim]
        
        # Get candidate embeddings (batch)
        candidate_texts = [c["summary"] for c in candidates]
        candidate_embs = self._get_embedding(candidate_texts)  # [N, dim]
        
        # Compute semantic similarity
        semantic_scores = F.cosine_similarity(query_emb, candidate_embs)  # [N]
        semantic_scores = semantic_scores.cpu().numpy()
        
        # Compute temporal match (symbolic)
        time_category = temporal_info.get("time_category", "none")
        time_keywords = temporal_info.get("time", "").lower().split(", ")
        
        temporal_scores = []
        for c in candidates:
            text_to_check = (c["summary"] + " " + c["name"]).lower()
            
            # Check if any temporal keyword appears
            time_match = 0.0
            
            # Check category keywords
            if time_category in self.temporal_keywords:
                for kw in self.temporal_keywords[time_category]:
                    if kw in text_to_check:
                        time_match = 1.0
                        break
            
            # Also check extracted time expressions
            if time_match == 0.0:
                for kw in time_keywords:
                    if kw and kw != "none" and kw in text_to_check:
                        time_match = 0.5  # Partial match
                        break
            
            temporal_scores.append(time_match)
        
        # Compute hybrid score
        for i, c in enumerate(candidates):
            c["semantic_score"] = float(semantic_scores[i])
            c["temporal_score"] = temporal_scores[i]
            c["hybrid_score"] = alpha * c["semantic_score"] + (1 - alpha) * c["temporal_score"]
        
        # Sort by hybrid score
        ranked = sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)
        
        return ranked[:top_k]
    
    # =========================================================================
    # Full Pipeline
    # =========================================================================
    
    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        alpha: float = 0.5
    ) -> Tuple[List[Dict], Dict]:
        """
        Full MRAG pipeline: Question Processing -> Retrieval -> Hybrid Ranking
        
        Returns:
            Tuple of (ranked_candidates, temporal_info)
        """
        # Module 1: Question Processing
        temporal_info = self.process_question(question)
        
        # Extract entities for retrieval
        entities = self.retriever.extract_entities(question)
        cuis = list(entities.values()) if entities else []
        
        # Module 2: Retrieval & Summarization
        candidates = self.retrieve_and_summarize(question, cuis)
        
        # Module 3: Hybrid Ranking
        ranked = self.hybrid_rank(question, candidates, temporal_info, alpha=alpha, top_k=top_k)
        
        return ranked, temporal_info
    
    def format_context(self, ranked_candidates: List[Dict], temporal_info: Dict) -> str:
        """Format retrieved results as context string for LLM."""
        if not ranked_candidates:
            return ""
        
        context_parts = [f"Detected Temporal Context: {temporal_info.get('time_category', 'none')} ({temporal_info.get('time', 'none')})"]
        context_parts.append("Related Medical Concepts (ranked by semantic-temporal relevance):")
        
        for i, c in enumerate(ranked_candidates, 1):
            score_info = f"[S:{c.get('semantic_score', 0):.2f}, T:{c.get('temporal_score', 0):.1f}]"
            context_parts.append(f"{i}. {c['name']} ({c['relation']}) {score_info}")
        
        return "\n".join(context_parts)
    
    def close(self):
        """Clean up resources."""
        self.retriever.close()


if __name__ == "__main__":
    # Quick test
    mrag = MRAGBaseline()
    
    test_question = "A 45-year-old man presents with acute chest pain for the past 2 hours. What is the most likely diagnosis?"
    
    print("Testing MRAG Baseline...")
    print(f"Question: {test_question}\n")
    
    # Module 1 test
    temporal_info = mrag.process_question(test_question)
    print(f"Module 1 (Question Processing):")
    print(f"  Time Category: {temporal_info['time_category']}")
    print(f"  Time Keywords: {temporal_info['time']}")
    print()
    
    # Full pipeline test
    ranked, _ = mrag.retrieve(test_question, top_k=5)
    print(f"Full Pipeline Result ({len(ranked)} candidates):")
    for i, c in enumerate(ranked, 1):
        print(f"  {i}. {c['name']} (hybrid={c['hybrid_score']:.3f}, sem={c['semantic_score']:.3f}, time={c['temporal_score']:.1f})")
    
    mrag.close()

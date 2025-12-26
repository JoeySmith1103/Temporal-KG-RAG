
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from time_aware_encoder import TimeAwareEncoder
import numpy as np

def load_model(checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    # We used 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext' during training
    # The config is saved in the checkpoint directory
    model = TimeAwareEncoder.from_pretrained(checkpoint_path, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.eval()
    return model, tokenizer

def get_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        _, embedding = model(inputs["input_ids"], inputs["attention_mask"])
    return embedding

def demo_reranking():
    checkpoint = "./checkpoints/MedQA_encoder"
    model, tokenizer = load_model(checkpoint)
    
    # Scene 1: Acute Case
    query_acute = "Patient presents with sudden onset of severe chest pain starting 30 minutes ago, accompanied by diaphoresis."
    
    # Scene 2: Chronic Case
    query_chronic = "Patient with a 10-year history of Type 2 Diabetes, reporting gradual worsening of vision over the past year."
    
    # Candidates (Mixed temporal types)
    candidates = [
        "Acute Myocardial Infarction (Heart Attack)",
        "Chronic Heart Failure",
        "Diabetic Retinopathy (Chronic Complication)",
        "Acute Hypoglycemic Shock",
        "Stable Angina (managed for years)",
        "Emergency Coronary Intervention"
    ]
    
    queries = [("Acute Query", query_acute), ("Chronic Query", query_chronic)]
    
    print("\n" + "="*50)
    print("Time-Aware Re-ranking Demo")
    print("="*50)
    
    for q_name, q_text in queries:
        print(f"\nQuery: [{q_name}]")
        print(f"Text: {q_text}")
        
        # Get Query Embedding
        q_emb = get_embedding(q_text, model, tokenizer)
        
        # Rank Candidates
        scores = []
        for cand in candidates:
            c_emb = get_embedding(cand, model, tokenizer)
            # Cosine Similarity
            score = F.cosine_similarity(q_emb, c_emb).item()
            scores.append((cand, score))
            
        # Sort desc
        scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\nRe-ranked Candidates (Top is better):")
        print("-" * 40)
        for cand, score in scores:
            print(f"{score:.4f} | {cand}")
            
if __name__ == "__main__":
    demo_reranking()

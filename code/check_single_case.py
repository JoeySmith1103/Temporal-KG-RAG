
import torch
from transformers import AutoTokenizer
from time_aware_encoder import TimeAwareEncoder
import numpy as np

def check_case():
    checkpoint_path = "./checkpoints/MedQA_encoder"
    print(f"Loading model from {checkpoint_path}...")
    model = TimeAwareEncoder.from_pretrained(checkpoint_path, model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model.eval()
    
    question = "A 25 yr old lady develops brown macular lesions over the bridge of nose and cheeks following exposure to sunlight. What is the most probable diagnosis?"
    
    inputs = tokenizer(question, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits, embedding = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        
    label_map = {0: "Acute", 1: "Chronic", 2: "Recurrent", 3: "Progressive", 4: "None"}
    
    print("\n" + "="*50)
    print(f"Question: {question}")
    print("-" * 50)
    print("Prediction Probabilities:")
    for idx, label in label_map.items():
        print(f"{label}: {probs[0][idx].item():.4f}")
        
    print(f"\nFinal Prediction: {label_map[pred_idx]}")
    print("="*50)

if __name__ == "__main__":
    check_case()

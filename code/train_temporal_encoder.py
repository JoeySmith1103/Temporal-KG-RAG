
import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torch.optim import AdamW
from time_aware_encoder import TimeAwareEncoder
from tqdm import tqdm
import torch.nn as nn

# Default Configuration
DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" 
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
MAX_LEN = 128

class TemporalDataset(Dataset):
    def __init__(self, data_path, tokenizer, label_map, max_len=128):
        self.data = []
        # Support loading from the specific 'teacher_label' if available, otherwise 'temporal_label'
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    # Use teacher label if present (High Quality), else heuristic (Silver)
                    if 'teacher_label' in item:
                        item['label_to_use'] = item['teacher_label']
                    else:
                        item['label_to_use'] = item.get('temporal_label', 'None')
                    self.data.append(item)
        
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['question']
        label_str = item['label_to_use']
        label_id = self.label_map.get(label_str, 4) # Default to None=4

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    dataset_name = os.path.basename(os.path.dirname(args.data_path)) # e.g. MedQA
    output_dir = os.path.join(args.output_dir, f"{dataset_name}_encoder")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Training on {dataset_name}, outputting to {output_dir}")
    
    # Initialize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = TimeAwareEncoder(args.model_name)
    model.to(device)
    
    # Data
    train_dataset = TemporalDataset(args.data_path, tokenizer, model.label_map, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    # Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits, _ = model(input_ids, attention_mask)
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=correct/total)
            
        print(f"Epoch {epoch+1} Results: Loss={total_loss/len(train_loader):.4f}, Acc={correct/total:.4f}")
        
    # Save
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to training .jsonl")
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    train(args)

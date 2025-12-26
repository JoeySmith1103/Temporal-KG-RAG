
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig

class TimeAwareEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", num_labels: int = 5):
        super(TimeAwareEncoder, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        self.num_labels = num_labels
        
        # Mapping for labels (Must match the data prep)
        self.label_map = {
            "Acute": 0, 
            "Chronic": 1, 
            "Recurrent": 2, 
            "Progressive": 3, 
            "None": 4
        }
        self.id2label = {v: k for k, v in self.label_map.items()}

    def forward(self, input_ids, attention_mask):
        # BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # We use the pooled output (CLS token) as the latent representation
        # Some BERT models return 'pooler_output', others we take last_hidden_state[:, 0, :]
        if hasattr(outputs, 'pooler_output'):
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        temporal_embedding = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(temporal_embedding)
        
        return logits, temporal_embedding

    def save_pretrained(self, path):
        """Custom save method to save both the BERT model and the classifier."""
        torch.save(self.state_dict(), path + "/pytorch_model.bin")
        self.config.save_pretrained(path)
        # We also need to save the tokenizer usually, but that's handled outside

    @classmethod
    def from_pretrained(cls, path, model_name="bert-base-uncased"):
        model = cls(model_name)
        model.load_state_dict(torch.load(path + "/pytorch_model.bin", map_location='cpu'))
        return model

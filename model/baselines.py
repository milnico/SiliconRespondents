import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BaselineMLP(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.hidden_dim = self.config.hidden_size
        
        # Concatenate 5 latent traits and the sentence embedding
        self.mlp = nn.Sequential(
            nn.Linear(5 + self.hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 5)
        )

    def forward(self, input_ids, attention_mask, latent_traits):
        # Extract features from frozen backbone
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
            
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
            
        # Concat pooled embedding and traits
        x = torch.cat([pooled_output, latent_traits], dim=-1)
        logits = self.mlp(x)
        return logits

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from .latent_encoder import LatentEncoder
from .film_layers import FiLMLayer

class LatentConditionedModel(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", cond_dim=256, num_film_layers=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.hidden_dim = self.config.hidden_size
        self.latent_encoder = LatentEncoder(in_dim=5, out_dim=cond_dim)
        
        # We will inject FiLM conditioning in the last `num_film_layers` transformer blocks
        self.film_layers = nn.ModuleList([
            FiLMLayer(cond_dim, self.hidden_dim) for _ in range(num_film_layers)
        ])
        
        self.num_film_layers = num_film_layers
        
        # Classification head for 1 to 5 Likert response (5 classes)
        self.classifier = nn.Linear(self.hidden_dim, 5)

    def forward(self, input_ids, attention_mask, latent_traits):
        cond = self.latent_encoder(latent_traits)
        
        # Extract features from backbone and inject FiLM during the forward pass.
        hidden_states = self.backbone.embeddings(input_ids=input_ids)
        
        num_layers = len(self.backbone.encoder.layer)
        extended_attention_mask = self.backbone.get_extended_attention_mask(attention_mask, input_ids.size(), hidden_states.device)
        
        for i, layer_module in enumerate(self.backbone.encoder.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask
            )
            hidden_states = layer_outputs[0]
            
            # Inject FiLM in the last `num_film_layers`
            if i >= num_layers - self.num_film_layers:
                film_idx = i - (num_layers - self.num_film_layers)
                hidden_states = self.film_layers[film_idx](hidden_states, cond)
                
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentGatedLoRALinear(nn.Module):
    def __init__(self, base_layer, trait_dim=256, r=16, alpha=32, dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        
        self.r = r
        self.alpha = alpha
        self.scaling = self.alpha / self.r if self.r > 0 else 1.0
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Gating Network (output dimension = rank)
        self.gate_mlp = nn.Sequential(
            nn.Linear(trait_dim, 64),
            nn.SiLU(),
            nn.Linear(64, r)
        )
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        # Base forward is frozen
        base_output = self.base_layer(x)
        
        use_lora = getattr(self, 'use_lora', True)
        use_gate = getattr(self, 'use_gate', True)
        
        if not use_lora:
            return base_output
            
        intermediate = self.lora_A(self.dropout(x))
        
        trait_embedding = getattr(self, 'current_trait_embedding', None)
        if use_gate and trait_embedding is not None:
            gate = torch.sigmoid(self.gate_mlp(trait_embedding))
            gate = gate.unsqueeze(1)
            intermediate = intermediate * gate
            
        delta = self.lora_B(intermediate) * self.scaling
        
        return base_output + delta

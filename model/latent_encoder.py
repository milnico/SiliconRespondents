import torch
import torch.nn as nn

class LatentEncoder(nn.Module):
    def __init__(self, in_dim=5, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, x):
        return self.mlp(x)

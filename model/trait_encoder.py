import torch
import torch.nn as nn

class TraitEncoder(nn.Module):
    def __init__(self, in_dim=5, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, out_dim)
        )
        
    def forward(self, z):
        """
        Args:
            z: Tensor of shape (batch_size, 5) representing [O, C, E, A, N]
        Returns:
            trait_embedding of shape (batch_size, 256)
        """
        return self.mlp(z)

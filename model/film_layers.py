import torch
import torch.nn as nn

class FiLMLayer(nn.Module):
    def __init__(self, cond_dim, hidden_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)
        
        # Initialize gamma to 1 and beta to 0
        nn.init.constant_(self.gamma.weight, 0)
        nn.init.constant_(self.gamma.bias, 1)
        nn.init.constant_(self.beta.weight, 0)
        nn.init.constant_(self.beta.bias, 0)
        
    def forward(self, x, cond):
        # x shape: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        # cond shape: (batch, cond_dim)
        gamma = self.gamma(cond).unsqueeze(1) # (batch, 1, hidden_dim)
        beta = self.beta(cond).unsqueeze(1)   # (batch, 1, hidden_dim)
        return gamma * x + beta

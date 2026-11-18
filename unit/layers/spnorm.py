import torch
import torch.nn as nn


class SPNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.projection = nn.Linear(normalized_shape, normalized_shape)

    def forward(self, x):
        s, u = torch.std_mean(x, dim=-1, keepdim=True)
        n_x = (x - u) / (s + self.eps)
        n_x = self.projection(n_x)
        x = x * n_x
        return x

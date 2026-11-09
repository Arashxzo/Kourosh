"""
Kourosh - Model Architectures
==============================
Real PyTorch neural network modules. This file will grow over time:
currently holds an MLP; later it will host the Transformer that powers Kourosh.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """A configurable Multi-Layer Perceptron.

    Architecture:
        Linear(input_dim -> hidden_dim) -> ReLU -> Linear(hidden_dim -> output_dim) -> Sigmoid

    The hidden layer + non-linear activation (ReLU) is what allows this network
    to solve non-linearly-separable problems such as XOR.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
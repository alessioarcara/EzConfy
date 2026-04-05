import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_dims: list[int], dropout: float, num_classes: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 28 * 28  # MNIST images are 28x28 pixels
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))

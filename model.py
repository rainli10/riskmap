from __future__ import annotations

import torch
from torch import nn


class SimpleRiskCNN(nn.Module):
    """A minimal fully convolutional baseline for dense risk prediction."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: tuple[int, ...] = (32, 64, 64, 32),
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        current_channels = in_channels

        for out_channels in hidden_channels:
            layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            current_channels = out_channels

        layers.append(nn.Conv2d(current_channels, 1, kernel_size=1))

        self.network = nn.Sequential(*layers)
        self.output_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.network(x))

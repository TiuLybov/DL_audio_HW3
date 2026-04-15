"""
PostNet для FastSpeech2.

5-слойная Conv1D сеть, которая рефайнит mel-спектрограмму.
Используется как в Tacotron2 — добавляет residual к выходу decoder.
Помогает убрать размытость и улучшить детализацию.
"""

import torch
import torch.nn as nn


class PostNet(nn.Module):
    """
    PostNet: 5 × Conv1D(512, kernel=5) + BatchNorm + Tanh (+ last linear conv).
    Input и output: (batch, mel_len, n_mel)
    """

    def __init__(self, n_mel_channels: int = 80, postnet_dim: int = 512,
                 postnet_kernel: int = 5, postnet_layers: int = 5,
                 postnet_dropout: float = 0.5):
        super().__init__()

        layers = []

        # First layer: mel_channels → postnet_dim
        layers.append(nn.Sequential(
            nn.Conv1d(n_mel_channels, postnet_dim,
                      kernel_size=postnet_kernel,
                      padding=(postnet_kernel - 1) // 2),
            nn.BatchNorm1d(postnet_dim),
            nn.Tanh(),
            nn.Dropout(postnet_dropout),
        ))

        # Middle layers: postnet_dim → postnet_dim
        for _ in range(postnet_layers - 2):
            layers.append(nn.Sequential(
                nn.Conv1d(postnet_dim, postnet_dim,
                          kernel_size=postnet_kernel,
                          padding=(postnet_kernel - 1) // 2),
                nn.BatchNorm1d(postnet_dim),
                nn.Tanh(),
                nn.Dropout(postnet_dropout),
            ))

        # Last layer: postnet_dim → mel_channels (без активации)
        layers.append(nn.Sequential(
            nn.Conv1d(postnet_dim, n_mel_channels,
                      kernel_size=postnet_kernel,
                      padding=(postnet_kernel - 1) // 2),
            nn.BatchNorm1d(n_mel_channels),
            nn.Dropout(postnet_dropout),
        ))

        self.layers = nn.ModuleList(layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, mel_len, n_mel)
        Returns:
            mel_residual: (batch, mel_len, n_mel) — добавить к исходному mel
        """
        # Conv1d expects (batch, channels, seq_len)
        x = mel.transpose(1, 2)

        for layer in self.layers:
            x = layer(x)

        return x.transpose(1, 2)

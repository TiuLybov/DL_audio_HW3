"""
Feed-Forward Transformer (FFT) Block — основной строительный блок FastSpeech2.
Содержит Multi-Head Attention + Position-wise Feed-Forward Network.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """
    Two-layer conv1d with ReLU, как описано в FastSpeech.
    Используем Conv1D вместо Linear, т.к. это позволяет
    учитывать локальный контекст (kernel_size > 1).
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        kernel_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            d_model, d_inner,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.conv2 = nn.Conv1d(
            d_inner, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x

        # Conv1d ожидает (batch, channels, seq_len)
        output = x.transpose(1, 2)
        output = self.conv1(output)
        output = torch.relu(output)
        output = self.conv2(output)
        output = output.transpose(1, 2)

        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output


class FFTBlock(nn.Module):
    """
    Feed-Forward Transformer Block.
    Self-Attention → Add & Norm → FFN → Add & Norm
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_inner: int,
        kernel_size: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            d_model=d_model,
            n_head=n_head,
            dropout=dropout,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            d_inner=d_inner,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len)
        Returns:
            output: (batch, seq_len, d_model)
        """
        output = self.self_attention(x, mask=mask)
        output = self.feed_forward(output)

        # Зануляем padding позиции
        if mask is not None:
            output = output.masked_fill(mask.transpose(1, 2) == 0, 0.0)

        return output

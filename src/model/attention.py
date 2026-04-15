"""
Multi-Head Attention реализованный с нуля.
Не используем nn.Transformer / nn.TransformerEncoder / nn.TransformerEncoderLayer.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention из статьи 'Attention Is All You Need'."""

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: (batch, n_head, seq_len_q, d_k)
            k: (batch, n_head, seq_len_k, d_k)
            v: (batch, n_head, seq_len_v, d_v)
            mask: (batch, 1, 1, seq_len_k) or (batch, 1, seq_len_q, seq_len_k)
        Returns:
            output: (batch, n_head, seq_len_q, d_v)
            attn_weights: (batch, n_head, seq_len_q, seq_len_k)
        """
        # (batch, n_head, seq_len_q, seq_len_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn, dim=-1)
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention с нуля."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(
            temperature=math.sqrt(self.d_k)
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Self-attention с residual connection и layer norm.

        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len) — padding mask
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.size()
        residual = x

        # Linear projections и reshape для multi-head
        q = self.w_q(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_head, self.d_v).transpose(1, 2)

        # Расширяем mask для broadcast по головам
        if mask is not None:
            mask = mask.unsqueeze(1)  # (batch, 1, 1, seq_len)

        output, _ = self.attention(q, k, v, mask=mask)

        # Конкатенация голов
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_out(output)
        output = self.dropout(output)

        # Residual + LayerNorm
        output = self.layer_norm(output + residual)

        return output

"""
Encoder для FastSpeech2.
Phoneme Embedding → Positional Encoding → N x FFT Block
"""

import torch
import torch.nn as nn

from .fft_block import FFTBlock
from .positional_encoding import PositionalEncoding


class Encoder(nn.Module):
    """Phoneme Encoder: embedding + positional encoding + FFT blocks."""

    def __init__(self, config):
        super().__init__()

        self.phoneme_embedding = nn.Embedding(
            config.vocab_size,
            config.encoder_hidden,
            padding_idx=config.padding_idx,
        )
        self.positional_encoding = PositionalEncoding(
            config.encoder_hidden, max_len=config.max_seq_len
        )

        self.fft_blocks = nn.ModuleList([
            FFTBlock(
                d_model=config.encoder_hidden,
                n_head=config.encoder_head,
                d_inner=config.encoder_conv_filter_size,
                kernel_size=config.encoder_conv_kernel_size,
                dropout=config.encoder_dropout,
            )
            for _ in range(config.encoder_layers)
        ])

    def forward(
        self,
        phonemes: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            phonemes: (batch, src_len) — phoneme ids
            src_mask: (batch, 1, src_len) — attention mask
        Returns:
            output: (batch, src_len, d_model)
        """
        output = self.phoneme_embedding(phonemes)
        output = self.positional_encoding(output)

        for block in self.fft_blocks:
            output = block(output, mask=src_mask)

        return output

"""
Decoder для FastSpeech2.
Positional Encoding → N x FFT Block → Linear projection to mel-spectrogram
"""

import torch
import torch.nn as nn

from .fft_block import FFTBlock
from .positional_encoding import PositionalEncoding


class Decoder(nn.Module):
    """Mel Decoder: positional encoding + FFT blocks + linear projection."""

    def __init__(self, config):
        super().__init__()

        self.positional_encoding = PositionalEncoding(
            config.decoder_hidden, max_len=config.max_seq_len
        )

        self.fft_blocks = nn.ModuleList([
            FFTBlock(
                d_model=config.decoder_hidden,
                n_head=config.decoder_head,
                d_inner=config.decoder_conv_filter_size,
                kernel_size=config.decoder_conv_kernel_size,
                dropout=config.decoder_dropout,
            )
            for _ in range(config.decoder_layers)
        ])

        self.mel_linear = nn.Linear(config.decoder_hidden, config.n_mel_channels)

    def forward(
        self,
        x: torch.Tensor,
        mel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, mel_len, d_model) — output from variance adaptor
            mel_mask: (batch, mel_len) — True = valid position
        Returns:
            mel_output: (batch, mel_len, n_mel_channels)
        """
        output = self.positional_encoding(x)

        # Подготовка mask для attention: (batch, 1, mel_len)
        attn_mask = None
        if mel_mask is not None:
            attn_mask = mel_mask.unsqueeze(1)

        for block in self.fft_blocks:
            output = block(output, mask=attn_mask)

        mel_output = self.mel_linear(output)

        return mel_output

"""
FastSpeech2 — полная модель.

Архитектура:
    Encoder → Variance Adaptor → Decoder → Mel-spectrogram

При обучении используем GT duration/pitch/energy.
При инференсе — предсказанные значения + контроль параметров.
"""

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .variance_adaptor import VarianceAdaptor
from .postnet import PostNet


class FastSpeech2(nn.Module):
    """FastSpeech2 TTS model."""

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = Decoder(config)
        self.postnet = PostNet(n_mel_channels=config.n_mel_channels)

        self.config = config

    def forward(
        self,
        phonemes: torch.Tensor,
        src_lengths: torch.Tensor,
        mel_targets: torch.Tensor | None = None,
        mel_lengths: torch.Tensor | None = None,
        duration_targets: torch.Tensor | None = None,
        pitch_targets: torch.Tensor | None = None,
        energy_targets: torch.Tensor | None = None,
        duration_control: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0,
    ):
        """
        Args:
            phonemes: (batch, src_len)
            src_lengths: (batch,)
            mel_targets: (batch, mel_len, n_mel) — only training
            mel_lengths: (batch,) — only training
            duration_targets: (batch, src_len) — only training
            pitch_targets: (batch, src_len) — only training
            energy_targets: (batch, src_len) — only training
            duration_control: inference speed scale
            pitch_control: inference pitch scale
            energy_control: inference energy scale
        Returns:
            dict with mel_output, duration_pred, pitch_pred, energy_pred, etc.
        """
        # Source mask: (batch, src_len)
        src_mask = self._get_mask_from_lengths(src_lengths, phonemes.size(1))

        # Attention mask: (batch, 1, src_len) для broadcast в attention
        src_attn_mask = src_mask.unsqueeze(1)

        # Encode phonemes
        encoder_output = self.encoder(phonemes, src_mask=src_attn_mask)

        # Variance Adaptor
        max_mel_len = mel_targets.size(1) if mel_targets is not None else None

        (
            adaptor_output,
            log_duration_pred,
            pitch_pred,
            energy_pred,
            mel_lens,
            mel_mask,
        ) = self.variance_adaptor(
            encoder_output,
            src_mask=src_mask,
            duration_target=duration_targets,
            pitch_target=pitch_targets,
            energy_target=energy_targets,
            max_mel_len=max_mel_len,
            duration_control=duration_control,
            pitch_control=pitch_control,
            energy_control=energy_control,
        )

        # Decode to mel-spectrogram
        mel_output = self.decoder(adaptor_output, mel_mask=mel_mask)

        # PostNet refinement (residual)
        mel_postnet = mel_output + self.postnet(mel_output)

        return {
            "mel_output": mel_output,
            "mel_postnet": mel_postnet,
            "log_duration_pred": log_duration_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
            "mel_lens": mel_lens,
            "mel_mask": mel_mask,
            "src_mask": src_mask,
        }

    @staticmethod
    def _get_mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return ids < lengths.unsqueeze(1)

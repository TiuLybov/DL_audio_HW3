"""
Loss для FastSpeech2.

Total Loss = mel_loss + duration_loss + pitch_loss + energy_loss

- mel_loss: MSE между predicted и target mel-спектрограммами
- duration_loss: MSE на log-durations
- pitch_loss: MSE на pitch values
- energy_loss: MSE на energy values
"""

import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    """Combined loss for FastSpeech2 training."""

    def __init__(self, config):
        super().__init__()

        self.mel_weight = config.mel_loss_weight
        self.duration_weight = config.duration_loss_weight
        self.pitch_weight = config.pitch_loss_weight
        self.energy_weight = config.energy_loss_weight

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(
        self,
        mel_pred: torch.Tensor,
        mel_postnet: torch.Tensor,
        mel_target: torch.Tensor,
        log_duration_pred: torch.Tensor,
        duration_target: torch.Tensor,
        pitch_pred: torch.Tensor,
        pitch_target: torch.Tensor,
        energy_pred: torch.Tensor,
        energy_target: torch.Tensor,
        src_mask: torch.Tensor,
        mel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            mel_pred: (batch, mel_len, n_mel) — pre-postnet
            mel_postnet: (batch, mel_len, n_mel) — post-postnet
            mel_target: (batch, mel_len, n_mel)
            ...
        """
        mel_mask_expanded = mel_mask.unsqueeze(-1)  # (batch, mel_len, 1)

        # Pre-postnet mel loss
        mel_pred_masked = mel_pred.masked_select(mel_mask_expanded)
        mel_target_masked = mel_target.masked_select(mel_mask_expanded)
        mel_loss = self.mae_loss(mel_pred_masked, mel_target_masked)

        # Post-postnet mel loss
        mel_postnet_masked = mel_postnet.masked_select(mel_mask_expanded)
        mel_postnet_loss = self.mae_loss(mel_postnet_masked, mel_target_masked)

        # Duration loss — log(duration + 1) для GT, предиктор выдаёт log
        log_duration_target = torch.log(duration_target.float() + 1)
        duration_pred_masked = log_duration_pred.masked_select(src_mask)
        duration_target_masked = log_duration_target.masked_select(src_mask)
        duration_loss = self.mse_loss(duration_pred_masked, duration_target_masked)

        # Pitch loss
        pitch_pred_masked = pitch_pred.masked_select(src_mask)
        pitch_target_masked = pitch_target.masked_select(src_mask)
        pitch_loss = self.mse_loss(pitch_pred_masked, pitch_target_masked)

        # Energy loss
        energy_pred_masked = energy_pred.masked_select(src_mask)
        energy_target_masked = energy_target.masked_select(src_mask)
        energy_loss = self.mse_loss(energy_pred_masked, energy_target_masked)

        total_loss = (
            self.mel_weight * mel_loss
            + self.mel_weight * mel_postnet_loss
            + self.duration_weight * duration_loss
            + self.pitch_weight * pitch_loss
            + self.energy_weight * energy_loss
        )

        return {
            "total_loss": total_loss,
            "mel_loss": mel_loss.detach(),
            "mel_postnet_loss": mel_postnet_loss.detach(),
            "duration_loss": duration_loss.detach(),
            "pitch_loss": pitch_loss.detach(),
            "energy_loss": energy_loss.detach(),
        }

"""
Variance Adaptor для FastSpeech2.

Содержит:
- Duration Predictor
- Pitch Predictor
- Energy Predictor
- Length Regulator (расширение последовательности по durations)
"""

import torch
import torch.nn as nn


class VariancePredictor(nn.Module):
    """
    Предиктор для duration / pitch / energy.
    Архитектура: 2 слоя Conv1D + ReLU + LayerNorm + Dropout → Linear.
    Одинаковая архитектура для всех трёх предикторов (как в статье).
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        kernel_size: int,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(
                d_model, d_inner,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.LayerNorm(d_inner),
            nn.Dropout(dropout),
            # Second conv block
            nn.Conv1d(
                d_inner, d_inner,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.LayerNorm(d_inner),
            nn.Dropout(dropout),
        )

        self.linear = nn.Linear(d_inner, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) — padding mask (True = valid, False = pad)
        Returns:
            prediction: (batch, seq_len)
        """
        # LayerNorm ожидает last dim, а Conv1d — (batch, channels, seq_len)
        # Поэтому делаем по блокам вручную
        out = x.transpose(1, 2)  # (batch, d_model, seq_len)

        for layer in self.conv_layers:
            if isinstance(layer, nn.LayerNorm):
                out = out.transpose(1, 2)
                out = layer(out)
                out = out.transpose(1, 2)
            else:
                out = layer(out)

        out = out.transpose(1, 2)  # (batch, seq_len, d_inner)
        out = self.linear(out).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            out = out.masked_fill(~mask, 0.0)

        return out


class LengthRegulator(nn.Module):
    """
    Length Regulator расширяет encoder output по предсказанным/GT durations.
    Каждый фрейм фонемы повторяется duration раз.
    """

    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_model)
            durations: (batch, seq_len) — целочисленные длительности
            max_len: максимальная длина выходной последовательности
        Returns:
            output: (batch, max_mel_len, d_model)
            mel_lens: (batch,) — реальные длины mel
        """
        batch_size = x.size(0)
        outputs = []
        mel_lens = []

        for i in range(batch_size):
            # Повторяем каждый элемент duration[i][j] раз
            expanded = x[i].repeat_interleave(durations[i].long(), dim=0)
            outputs.append(expanded)
            mel_lens.append(expanded.size(0))

        # Pad до max_len
        if max_len is None:
            max_len = max(mel_lens)

        padded = torch.zeros(batch_size, max_len, x.size(2), device=x.device)
        for i, output in enumerate(outputs):
            length = min(output.size(0), max_len)
            padded[i, :length] = output[:length]

        mel_lens = torch.tensor(mel_lens, device=x.device)
        return padded, mel_lens


class VarianceAdaptor(nn.Module):
    """
    Variance Adaptor из FastSpeech2.

    При обучении используем GT duration/pitch/energy.
    При инференсе — предсказанные значения.

    pitch и energy квантизуются в бины и проходят через Embedding,
    чтобы добавить информацию к hidden representation.
    """

    def __init__(self, config):
        super().__init__()

        d_model = config.encoder_hidden
        d_inner = config.variance_predictor_filter_size
        kernel_size = config.variance_predictor_kernel_size
        dropout = config.variance_predictor_dropout
        n_bins = config.n_bins

        # Три предиктора с одинаковой архитектурой
        self.duration_predictor = VariancePredictor(
            d_model, d_inner, kernel_size, dropout
        )
        self.pitch_predictor = VariancePredictor(
            d_model, d_inner, kernel_size, dropout
        )
        self.energy_predictor = VariancePredictor(
            d_model, d_inner, kernel_size, dropout
        )

        self.length_regulator = LengthRegulator()

        # Квантизация pitch и energy в бины
        # Бины равномерно распределены в log-пространстве
        self.pitch_bins = nn.Parameter(
            torch.linspace(config.pitch_min, config.pitch_max, n_bins - 1),
            requires_grad=False,
        )
        self.energy_bins = nn.Parameter(
            torch.linspace(config.energy_min, config.energy_max, n_bins - 1),
            requires_grad=False,
        )

        # Embedding для квантизованных pitch / energy
        self.pitch_embedding = nn.Embedding(n_bins, d_model)
        self.energy_embedding = nn.Embedding(n_bins, d_model)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        duration_target: torch.Tensor | None = None,
        pitch_target: torch.Tensor | None = None,
        energy_target: torch.Tensor | None = None,
        max_mel_len: int | None = None,
        duration_control: float = 1.0,
        pitch_control: float = 1.0,
        energy_control: float = 1.0,
    ):
        """
        Args:
            x: encoder output (batch, src_len, d_model)
            src_mask: (batch, src_len) True = valid
            duration_target: GT durations (batch, src_len) — only for training
            pitch_target: GT pitch (batch, src_len) — only for training
            energy_target: GT energy (batch, src_len) — only for training
            max_mel_len: max mel length for training
            duration_control: speed control (inference)
            pitch_control: pitch scale (inference)
            energy_control: energy scale (inference)
        Returns:
            output: (batch, mel_len, d_model)
            duration_pred: (batch, src_len)
            pitch_pred: (batch, src_len)
            energy_pred: (batch, src_len)
            mel_lens: (batch,)
            mel_mask: (batch, mel_len)
        """
        # === Duration ===
        log_duration_pred = self.duration_predictor(x, mask=src_mask)

        if duration_target is not None:
            # Training: используем GT durations
            duration_rounded = duration_target
        else:
            # Inference: используем предсказания
            # log_duration_pred ≈ log(duration + 1), поэтому exp - 1
            duration_pred = torch.exp(log_duration_pred) - 1
            duration_rounded = torch.clamp(
                torch.round(duration_pred * duration_control), min=0
            ).long()
            # Гарантируем хотя бы 1 фрейм на каждую не-padding фонему
            if src_mask is not None:
                duration_rounded = duration_rounded.masked_fill(
                    src_mask & (duration_rounded == 0), 1
                )

        # Length Regulator
        output, mel_lens = self.length_regulator(x, duration_rounded, max_mel_len)

        # === Pitch ===
        pitch_pred = self.pitch_predictor(x, mask=src_mask)

        if pitch_target is not None:
            # Training: квантизация GT pitch
            pitch_embedding = self.pitch_embedding(
                torch.bucketize(pitch_target, self.pitch_bins)
            )
        else:
            # Inference: квантизация предсказанного pitch
            pitch_embedding = self.pitch_embedding(
                torch.bucketize(pitch_pred * pitch_control, self.pitch_bins)
            )

        # Расширяем pitch embedding по durations и добавляем
        pitch_expanded, _ = self.length_regulator(
            pitch_embedding, duration_rounded, max_mel_len
        )
        output = output + pitch_expanded

        # === Energy ===
        energy_pred = self.energy_predictor(x, mask=src_mask)

        if energy_target is not None:
            energy_embedding = self.energy_embedding(
                torch.bucketize(energy_target, self.energy_bins)
            )
        else:
            energy_embedding = self.energy_embedding(
                torch.bucketize(energy_pred * energy_control, self.energy_bins)
            )

        energy_expanded, _ = self.length_regulator(
            energy_embedding, duration_rounded, max_mel_len
        )
        output = output + energy_expanded

        # Создаём mel mask
        mel_mask = self._get_mask_from_lengths(mel_lens, output.size(1))

        return output, log_duration_pred, pitch_pred, energy_pred, mel_lens, mel_mask

    @staticmethod
    def _get_mask_from_lengths(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create boolean mask from lengths. True = valid position."""
        ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        mask = ids < lengths.unsqueeze(1)
        return mask

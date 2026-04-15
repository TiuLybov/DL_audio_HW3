"""
Утилиты для вокодеров.

Поддерживаемые вокодеры:
1. WaveGlow (NVIDIA) — нейросетевой, лучшее качество
2. Griffin-Lim — алгоритмический fallback, не требует весов

Griffin-Lim полезен для быстрой отладки без скачивания WaveGlow.
"""

import numpy as np
import torch
import librosa


class GriffinLimVocoder:
    """
    Griffin-Lim вокодер как fallback.
    Конвертирует log-mel спектрограмму обратно в аудио.
    Качество хуже нейросетевых вокодеров, но не требует весов.
    """

    def __init__(self, sr=22050, n_fft=1024, hop_length=256,
                 win_length=1024, n_mels=80, fmin=0.0, fmax=8000.0,
                 n_iter=60):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_iter = n_iter

        # Mel filterbank для инверсии
        self.mel_basis = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
        )
        # Pseudo-inverse для mel → linear
        self.mel_basis_inv = np.linalg.pinv(self.mel_basis)

    def __call__(self, mel: torch.Tensor) -> np.ndarray:
        """
        Args:
            mel: (1, n_mel, mel_len) — log-mel spectrogram (на GPU или CPU)
        Returns:
            audio: (T,) numpy array
        """
        mel_np = mel.squeeze(0).cpu().numpy()  # (n_mel, mel_len)

        # log-mel → linear mel
        mel_linear = np.exp(mel_np)

        # mel → linear spectrogram (approximate)
        spec = np.maximum(1e-10, self.mel_basis_inv @ mel_linear)

        # Griffin-Lim
        audio = librosa.griffinlim(
            spec,
            n_iter=self.n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-6) * 0.95
        return audio


def load_waveglow(path: str, device: str = "cpu"):
    """
    Загрузка WaveGlow.
    Обрабатываем разные форматы чекпоинтов.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if "model" in checkpoint:
        waveglow = checkpoint["model"]
    else:
        waveglow = checkpoint

    # Remove weight norm если метод есть
    if hasattr(waveglow, "remove_weightnorm"):
        waveglow = waveglow.remove_weightnorm(waveglow)
    else:
        # Пробуем вручную
        for module in waveglow.modules():
            if hasattr(module, "weight_g"):
                torch.nn.utils.remove_weight_norm(module)

    waveglow.eval()
    return waveglow.to(device)


class WaveGlowVocoder:
    """Обёртка над WaveGlow для единого интерфейса."""

    def __init__(self, waveglow_model, sigma=0.666):
        self.model = waveglow_model
        self.sigma = sigma

    def __call__(self, mel: torch.Tensor) -> np.ndarray:
        """
        Args:
            mel: (1, n_mel, mel_len)
        Returns:
            audio: (T,) numpy array
        """
        with torch.no_grad():
            audio = self.model.infer(mel, sigma=self.sigma)
        return audio.cpu().numpy().flatten()


def get_vocoder(vocoder_type: str = "griffin_lim", vocoder_path: str = None,
                device: str = "cpu"):
    """
    Фабрика вокодеров.

    Args:
        vocoder_type: "waveglow" или "griffin_lim"
        vocoder_path: путь к чекпоинту (для waveglow)
        device: устройство
    Returns:
        vocoder callable: mel (1, n_mel, mel_len) → audio (T,) np.array
    """
    if vocoder_type == "waveglow":
        if vocoder_path is None:
            raise ValueError("vocoder_path required for WaveGlow")
        wg = load_waveglow(vocoder_path, device)
        return WaveGlowVocoder(wg)
    elif vocoder_type == "griffin_lim":
        return GriffinLimVocoder()
    else:
        raise ValueError(f"Unknown vocoder: {vocoder_type}")

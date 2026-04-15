"""
Утилиты для обработки аудио:
- Извлечение mel-спектрограмм
- Извлечение pitch (F0) через pyin
- Извлечение energy (RMS)
"""

import numpy as np
import torch
import librosa


def get_mel_spectrogram(
    audio: np.ndarray,
    sr: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    fmin: float = 0.0,
    fmax: float = 8000.0,
) -> np.ndarray:
    """
    Извлечение mel-спектрограммы из аудио.

    Args:
        audio: waveform (T,)
        sr: sample rate
        n_fft, hop_length, win_length: STFT параметры
        n_mels: количество mel-фильтров
        fmin, fmax: частотный диапазон
    Returns:
        mel: (n_mels, n_frames) — в log scale
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    # Log scale (добавляем eps для стабильности)
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    return mel


def get_pitch(
    audio: np.ndarray,
    sr: int = 22050,
    hop_length: int = 256,
    fmin: float = 80.0,
    fmax: float = 800.0,
) -> np.ndarray:
    """
    Извлечение pitch (F0) с помощью librosa pyin.

    pyin работает лучше чем yin для речи, т.к. использует
    вероятностную модель и лучше обрабатывает unvoiced сегменты.

    Args:
        audio: waveform
        sr: sample rate
        hop_length: hop size
        fmin, fmax: диапазон F0
    Returns:
        pitch: (n_frames,) — F0 в Hz, 0 для unvoiced
    """
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        hop_length=hop_length,
    )
    # NaN → 0 (unvoiced segments)
    f0 = np.nan_to_num(f0, nan=0.0)
    return f0


def get_energy(
    audio: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
) -> np.ndarray:
    """
    Извлечение energy (L2-norm спектра) для каждого фрейма.

    Energy = sqrt(sum(|STFT|^2)) для каждого фрейма.
    Это соответствует определению в статье FastSpeech2.

    Args:
        audio: waveform
        n_fft, hop_length, win_length: STFT params
    Returns:
        energy: (n_frames,)
    """
    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    # L2 norm по частотным бинам
    energy = np.linalg.norm(np.abs(stft), axis=0)
    return energy


def interpolate_pitch(pitch: np.ndarray) -> np.ndarray:
    """
    Интерполяция pitch для заполнения unvoiced сегментов.
    Используем линейную интерполяцию между voiced фреймами.

    Args:
        pitch: (n_frames,) — с нулями в unvoiced
    Returns:
        pitch_interp: (n_frames,) — интерполированный
    """
    nonzero = np.where(pitch > 0)[0]
    if len(nonzero) == 0:
        return pitch

    # Интерполируем
    interp_fn = np.interp(
        np.arange(len(pitch)),
        nonzero,
        pitch[nonzero],
    )

    return interp_fn


def normalize_pitch(pitch: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Нормализация pitch к [0, 1] для квантизации.

    Returns:
        normalized_pitch, pitch_min, pitch_max
    """
    p_min = pitch[pitch > 0].min() if (pitch > 0).any() else 0
    p_max = pitch.max()

    if p_max - p_min < 1e-6:
        return np.zeros_like(pitch), p_min, p_max

    normalized = (pitch - p_min) / (p_max - p_min)
    return normalized, float(p_min), float(p_max)

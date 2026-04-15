"""
Скрипт инференса FastSpeech2.

Генерация mel-спектрограмм из текста → аудио через вокодер.
Поддержка контроля pitch, speed (duration), energy.
Поддержка WaveGlow и Griffin-Lim (fallback без весов).
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.fastspeech2 import FastSpeech2
from src.configs.model_config import ModelConfig
from src.utils.text import text_to_sequence, VOCAB_SIZE
from src.utils.vocoder import get_vocoder


def load_model(checkpoint_path: str, stats_path: str = None, device: str = "cpu"):
    """Загрузка обученной модели."""
    config = ModelConfig()
    config.vocab_size = VOCAB_SIZE

    if stats_path and os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        config.pitch_min = stats["pitch_min"]
        config.pitch_max = stats["pitch_max"]
        config.energy_min = stats["energy_min"]
        config.energy_max = stats["energy_max"]

    model = FastSpeech2(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


@torch.no_grad()
def synthesize(model, text, device="cpu", duration_control=1.0,
               pitch_control=1.0, energy_control=1.0):
    """
    Синтез mel-спектрограммы из текста.

    Returns:
        mel_postnet: (1, mel_len, n_mel) — refined mel-spectrogram
    """
    phoneme_ids = text_to_sequence(text)
    phonemes = torch.tensor([phoneme_ids], dtype=torch.long, device=device)
    src_lengths = torch.tensor([len(phoneme_ids)], dtype=torch.long, device=device)

    output = model(
        phonemes=phonemes,
        src_lengths=src_lengths,
        duration_control=duration_control,
        pitch_control=pitch_control,
        energy_control=energy_control,
    )
    # Используем mel_postnet (refined) для лучшего качества
    return output["mel_postnet"]


def main():
    parser = argparse.ArgumentParser(description="FastSpeech2 Inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--stats", type=str, default="data/processed/stats.json")
    parser.add_argument("--vocoder", type=str, default="griffin_lim",
                        choices=["waveglow", "griffin_lim"])
    parser.add_argument("--vocoder_path", type=str, default=None,
                        help="Path to vocoder checkpoint (required for waveglow)")
    parser.add_argument("--output_dir", type=str, default="outputs/audio")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--text", type=str, default=None,
                        help="Single text to synthesize (optional)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Загрузка
    model = load_model(args.checkpoint, args.stats, device)
    vocoder = get_vocoder(args.vocoder, args.vocoder_path, device)

    sr = 22050

    # Если передан один текст
    if args.text:
        mel = synthesize(model, args.text, device)
        mel_voc = mel.transpose(1, 2)  # (1, n_mel, mel_len)
        audio = vocoder(mel_voc)
        out_path = os.path.join(args.output_dir, "output.wav")
        sf.write(out_path, audio, sr)
        print(f"Saved: {out_path}")
        return

    # Тестовые предложения (из задания)
    test_texts = [
        "Текстура поверхности определяется кристаллической структурой материала",
        "Синтез речи является одной из наиболее сложных задач обработки естественного языка",
        "Трансформер это архитектура глубокого обучения основанная на механизме внимания",
    ]

    # Конфигурации синтеза (из задания)
    configs = [
        {"name": "normal",      "dur": 1.0, "pitch": 1.0, "energy": 1.0},
        {"name": "pitch_up",    "dur": 1.0, "pitch": 1.2, "energy": 1.0},
        {"name": "pitch_down",  "dur": 1.0, "pitch": 0.8, "energy": 1.0},
        {"name": "speed_up",    "dur": 0.8, "pitch": 1.0, "energy": 1.0},
        {"name": "speed_down",  "dur": 1.2, "pitch": 1.0, "energy": 1.0},
        {"name": "energy_up",   "dur": 1.0, "pitch": 1.0, "energy": 1.2},
        {"name": "energy_down", "dur": 1.0, "pitch": 1.0, "energy": 0.8},
        {"name": "all_up",      "dur": 1.2, "pitch": 1.2, "energy": 1.2},
        {"name": "all_down",    "dur": 0.8, "pitch": 0.8, "energy": 0.8},
    ]

    for i, text in enumerate(test_texts):
        print(f"\nText {i+1}: {text}")
        for cfg in configs:
            mel = synthesize(
                model, text, device,
                duration_control=cfg["dur"],
                pitch_control=cfg["pitch"],
                energy_control=cfg["energy"],
            )
            mel_voc = mel.transpose(1, 2)
            audio = vocoder(mel_voc)

            filename = f"text{i+1}_{cfg['name']}.wav"
            out_path = os.path.join(args.output_dir, filename)
            sf.write(out_path, audio, sr)
            print(f"  {cfg['name']:15s} → {filename}")

    print(f"\nAll audio saved to {args.output_dir}")


if __name__ == "__main__":
    main()

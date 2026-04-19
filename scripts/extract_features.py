"""
Извлечение характеристик из датасета RUSLAN.

Структура RUSLAN:
    ruslan_corpus/
    ├── metadata_RUSLAN_22200.csv   (формат: filename|text)
    └── wavs/
        ├── 000001_RUSLAN.wav
        └── ...

Этапы:
1. Загружаем аудио и текст
2. Извлекаем mel-спектрограмму
3. Извлекаем pitch (F0) через pyin
4. Извлекаем energy (L2-norm STFT)
5. Загружаем durations (из MFA или Tacotron2 alignments)
6. Усредняем pitch и energy по фонемам
7. Сохраняем в processed/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.audio import get_mel_spectrogram, get_pitch, get_energy, interpolate_pitch
from src.utils.text import text_to_sequence


def find_metadata(data_dir: str) -> str:
    """
    Автоопределение файла metadata в датасете RUSLAN.
    RUSLAN использует 'metadata_RUSLAN_22200.csv'.
    """
    candidates = [
        "metadata_RUSLAN_22200.csv",
        "metadata.csv",
        "metadata.txt",
        "transcripts.txt",
        "text.csv",
    ]
    # Также ищем любой csv с 'metadata' в имени
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path

    # Поиск по паттерну
    for f in Path(data_dir).glob("*metadata*.csv"):
        return str(f)
    for f in Path(data_dir).glob("*.csv"):
        return str(f)

    raise FileNotFoundError(
        f"No metadata file found in {data_dir}. "
        f"Expected one of: {candidates}. "
        f"Files in directory: {os.listdir(data_dir)}"
    )


def find_wav(data_dir: str, basename: str) -> str | None:
    """Поиск wav файла в разных поддиректориях."""
    # RUSLAN хранит wav в wavs/
    for subdir in ["wavs", "audio", "wav", ""]:
        d = os.path.join(data_dir, subdir) if subdir else data_dir
        for ext in [".wav", ".WAV", ".flac"]:
            path = os.path.join(d, basename + ext)
            if os.path.exists(path):
                return path
    return None


def average_by_duration(values: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """Усреднение frame-level значений по фонемам."""
    averaged = np.zeros(len(durations))
    pos = 0
    for i, dur in enumerate(durations):
        dur = int(dur)
        if dur > 0 and pos + dur <= len(values):
            averaged[i] = np.mean(values[pos: pos + dur])
        pos += dur
    return averaged


def adjust_durations(durations: np.ndarray, target_length: int) -> np.ndarray:
    """Корректирует durations чтобы сумма = target_length."""
    durations = durations.copy().astype(np.int64)
    diff = target_length - int(durations.sum())
    if diff == 0:
        return durations

    indices = np.argsort(durations)[::-1]
    step = 1 if diff > 0 else -1
    for i in range(abs(diff)):
        idx = indices[i % len(indices)]
        durations[idx] = max(0, durations[idx] + step)
    return durations


def process_utterance(wav_path, text, duration, sr=22050, n_fft=1024,
                      hop_length=256, win_length=1024, n_mels=80,
                      fmin=0.0, fmax=11025.0):
    """Обработка одного utterance."""
    audio, _ = librosa.load(wav_path, sr=sr)

    mel = get_mel_spectrogram(audio, sr, n_fft, hop_length, win_length,
                              n_mels, fmin, fmax)
    pitch = get_pitch(audio, sr=sr, hop_length=hop_length)
    energy = get_energy(audio, n_fft=n_fft, hop_length=hop_length,
                        win_length=win_length)

    n_frames = mel.shape[1]
    total_dur = int(duration.sum())
    min_frames = min(n_frames, total_dur, len(pitch), len(energy))

    if min_frames < 10:
        return None

    mel = mel[:, :min_frames]
    pitch = pitch[:min_frames]
    energy = energy[:min_frames]
    duration = adjust_durations(duration, min_frames)

    pitch = interpolate_pitch(pitch)
    pitch_phoneme = average_by_duration(pitch, duration)
    energy_phoneme = average_by_duration(energy, duration)

    phoneme_ids = text_to_sequence(text)

    # Подгоняем длины
    if len(phoneme_ids) != len(duration):
        min_len = min(len(phoneme_ids), len(duration))
        phoneme_ids = phoneme_ids[:min_len]
        duration = duration[:min_len]
        pitch_phoneme = pitch_phoneme[:min_len]
        energy_phoneme = energy_phoneme[:min_len]

    return {
        "mel": mel,
        "pitch": pitch_phoneme,
        "energy": energy_phoneme,
        "duration": duration,
        "phonemes": phoneme_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract features from RUSLAN")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--alignments_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--n_mels", type=int, default=80)
    parser.add_argument("--fmax", type=float, default=11025.0)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    for subdir in ["mel", "pitch", "energy", "duration"]:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # Находим metadata
    metadata_path = find_metadata(args.data_dir)
    print(f"Metadata: {metadata_path}")

    # Парсим metadata
    # Формат RUSLAN: filename|text (filename может быть с/без .wav)
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                basename = parts[0].strip().replace(".wav", "")
                text = parts[1].strip()
                entries.append({"basename": basename, "text": text})

    print(f"Found {len(entries)} entries in metadata")

    if len(entries) == 0:
        # Может быть другой разделитель
        with open(metadata_path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        print(f"First line: {first_line}")
        print("Trying tab separator...")
        entries = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    basename = parts[0].strip().replace(".wav", "")
                    text = parts[1].strip()
                    entries.append({"basename": basename, "text": text})
        print(f"Found {len(entries)} entries with tab separator")

    all_pitch = []
    all_energy = []
    train_metadata = []
    val_metadata = []

    n_val = max(1, int(len(entries) * args.val_ratio))
    val_basenames = set(e["basename"] for e in entries[-n_val:])

    skipped = 0

    for entry in tqdm(entries, desc="Processing"):
        basename = entry["basename"]
        text = entry["text"]

        wav_path = find_wav(args.data_dir, basename)
        if wav_path is None:
            skipped += 1
            continue

        # Загружаем durations
        dur_path = os.path.join(args.alignments_dir, f"{basename}.npy")
        if not os.path.exists(dur_path):
            dur_txt_path = os.path.join(args.alignments_dir, f"{basename}.txt")
            if os.path.exists(dur_txt_path):
                duration = np.loadtxt(dur_txt_path).astype(np.int64)
            else:
                skipped += 1
                continue
        else:
            duration = np.load(dur_path).astype(np.int64)

        result = process_utterance(
            wav_path, text, duration,
            sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
            n_mels=args.n_mels, fmax=args.fmax,
        )

        if result is None:
            skipped += 1
            continue

        np.save(os.path.join(args.output_dir, "mel", f"{basename}.npy"), result["mel"])
        np.save(os.path.join(args.output_dir, "pitch", f"{basename}.npy"), result["pitch"])
        np.save(os.path.join(args.output_dir, "energy", f"{basename}.npy"), result["energy"])
        np.save(os.path.join(args.output_dir, "duration", f"{basename}.npy"), result["duration"])

        all_pitch.append(result["pitch"])
        all_energy.append(result["energy"])

        item = {
            "basename": basename,
            "text": text,
            "phonemes": result["phonemes"],
        }

        if basename in val_basenames:
            val_metadata.append(item)
        else:
            train_metadata.append(item)

    print(f"\nProcessed: {len(train_metadata) + len(val_metadata)}, Skipped: {skipped}")
    print(f"Train: {len(train_metadata)}, Val: {len(val_metadata)}")

    all_pitch = np.concatenate(all_pitch)
    all_energy = np.concatenate(all_energy)

    stats = {
        "pitch_min": float(all_pitch[all_pitch > 0].min()) if (all_pitch > 0).any() else 0,
        "pitch_max": float(all_pitch.max()),
        "pitch_mean": float(all_pitch[all_pitch > 0].mean()) if (all_pitch > 0).any() else 0,
        "pitch_std": float(all_pitch[all_pitch > 0].std()) if (all_pitch > 0).any() else 1,
        "energy_min": float(all_energy.min()),
        "energy_max": float(all_energy.max()),
        "energy_mean": float(all_energy.mean()),
        "energy_std": float(all_energy.std()),
    }

    with open(os.path.join(args.output_dir, "train_metadata.json"), "w") as f:
        json.dump(train_metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "val_metadata.json"), "w") as f:
        json.dump(val_metadata, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()

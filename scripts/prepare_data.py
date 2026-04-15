"""
Подготовка датасета RUSLAN.

Скрипт для первичной обработки: создание metadata в нужном формате,
ресемплинг аудио если нужно, и т.д.

RUSLAN dataset: https://ruslan-corpus.github.io/
~31 часов русской речи одного диктора.
"""

import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm

import librosa
import soundfile as sf
import numpy as np


def resample_audio(input_dir: str, output_dir: str, target_sr: int = 22050):
    """
    Ресемплинг всех wav файлов к target_sr.
    RUSLAN может быть записан в разном sample rate.
    """
    os.makedirs(output_dir, exist_ok=True)

    wav_files = sorted(Path(input_dir).glob("*.wav"))
    print(f"Found {len(wav_files)} wav files")

    for wav_path in tqdm(wav_files, desc="Resampling"):
        audio, sr = librosa.load(str(wav_path), sr=target_sr)

        out_path = os.path.join(output_dir, wav_path.name)
        sf.write(out_path, audio, target_sr)


def create_metadata(data_dir: str, transcript_path: str, output_path: str):
    """
    Создание metadata.csv в формате: basename|text
    из транскрипций RUSLAN.
    """
    # RUSLAN может иметь разные форматы транскрипций
    entries = []

    # Пробуем прочитать как tsv/csv
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Пробуем разные разделители
            for sep in ["|", "\t", ","]:
                parts = line.split(sep)
                if len(parts) >= 2:
                    basename = parts[0].strip().replace(".wav", "")
                    text = parts[1].strip()
                    # Проверяем что аудио существует
                    wav_path = os.path.join(data_dir, "wavs", f"{basename}.wav")
                    if not os.path.exists(wav_path):
                        wav_path = os.path.join(data_dir, "audio", f"{basename}.wav")
                    if os.path.exists(wav_path):
                        entries.append(f"{basename}|{text}")
                    break

    print(f"Created metadata with {len(entries)} entries")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))


def check_dataset(data_dir: str):
    """Проверка целостности датасета."""
    print(f"\nChecking dataset at {data_dir}")

    # Ищем аудио
    for subdir in ["wavs", "audio", "wav", ""]:
        audio_dir = os.path.join(data_dir, subdir) if subdir else data_dir
        wavs = list(Path(audio_dir).glob("*.wav"))
        if wavs:
            print(f"  Audio directory: {audio_dir}")
            print(f"  Number of wav files: {len(wavs)}")

            # Проверяем sample rate первого файла
            sr = librosa.get_samplerate(str(wavs[0]))
            print(f"  Sample rate: {sr}")

            # Средняя длительность
            durations = []
            for w in wavs[:100]:  # первые 100
                dur = librosa.get_duration(path=str(w))
                durations.append(dur)
            print(f"  Avg duration (first 100): {np.mean(durations):.1f}s")
            print(f"  Total estimated hours: {len(wavs) * np.mean(durations) / 3600:.1f}h")
            break
    else:
        print("  WARNING: No wav files found!")

    # Ищем транскрипции
    for name in ["metadata.csv", "metadata.txt", "transcripts.txt", "text.csv"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                n_lines = sum(1 for _ in f)
            print(f"  Transcript file: {path} ({n_lines} lines)")
            break
    else:
        print("  WARNING: No transcript file found!")


def main():
    parser = argparse.ArgumentParser(description="Prepare RUSLAN dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to raw RUSLAN dataset")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as data_dir)")
    parser.add_argument("--resample", action="store_true",
                        help="Resample audio to 22050 Hz")
    parser.add_argument("--check", action="store_true",
                        help="Only check dataset integrity")
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    if args.check:
        check_dataset(args.data_dir)
        return

    output_dir = args.output_dir or args.data_dir

    if args.resample:
        # Ищем аудио директорию
        for subdir in ["wavs", "audio"]:
            audio_dir = os.path.join(args.data_dir, subdir)
            if os.path.exists(audio_dir):
                resampled_dir = os.path.join(output_dir, "wavs_22k")
                resample_audio(audio_dir, resampled_dir, args.sr)
                print(f"Resampled audio saved to {resampled_dir}")
                break

    check_dataset(args.data_dir)
    print("\nDataset preparation complete!")
    print("Next step: python scripts/extract_features.py --data_dir ...")


if __name__ == "__main__":
    main()

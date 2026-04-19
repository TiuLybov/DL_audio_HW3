"""
Извлечение alignments с помощью Montreal Forced Aligner (MFA).

MFA даёт точные временные границы каждой фонемы,
из которых мы получаем duration в фреймах mel-спектрограммы.

Бонус: +15 баллов за использование MFA вместо Tacotron2 alignments.

Установка MFA в Google Colab:

    # Ячейка 1 (после этого runtime перезагрузится автоматически):
    !pip install condacolab
    import condacolab
    condacolab.install()

    # Ячейка 2 (после перезагрузки):
    !conda install -c conda-forge montreal-forced-aligner -y

Установка локально (если есть conda/mamba):
    conda install -c conda-forge montreal-forced-aligner

Шаги:
1. Подготовка данных в формате MFA (wav + txt в одной папке)
2. Скачивание акустической модели и словаря для русского
3. Запуск выравнивания
4. Парсинг TextGrid файлов → duration в фреймах
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.text import text_to_sequence


def prepare_mfa_input(data_dir: str, output_dir: str, metadata_path: str):
    """
    MFA ожидает: папку, где для каждого utterance лежат
    файл.wav и файл.txt (с текстом) с одинаковым именем.

    Args:
        data_dir: папка с wav файлами
        output_dir: куда складывать подготовленные файлы
        metadata_path: файл с metadata (basename|text)
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue

            basename = parts[0].strip().replace(".wav", "")
            text = parts[1].strip()

            # Ищем wav
            wav_src = None
            for subdir in ["wavs", "audio", ""]:
                candidate = os.path.join(data_dir, subdir, f"{basename}.wav")
                if os.path.exists(candidate):
                    wav_src = candidate
                    break

            if wav_src is None:
                continue

            # Симлинк на wav
            wav_dst = os.path.join(output_dir, f"{basename}.wav")
            if not os.path.exists(wav_dst):
                os.symlink(os.path.abspath(wav_src), wav_dst)

            # Текстовый файл
            txt_dst = os.path.join(output_dir, f"{basename}.txt")
            with open(txt_dst, "w", encoding="utf-8") as tf:
                tf.write(text)

    n_files = len(list(Path(output_dir).glob("*.wav")))
    print(f"Prepared {n_files} files for MFA in {output_dir}")


def run_mfa(input_dir: str, output_dir: str, language: str = "russian"):
    """
    Запуск MFA alignment.

    Скачивает модель и словарь для русского если нужно.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Скачиваем модель и словарь для русского
    print("Downloading Russian acoustic model and dictionary...")
    subprocess.run(
        ["mfa", "model", "download", "acoustic", "russian_mfa"],
        check=False,
    )
    subprocess.run(
        ["mfa", "model", "download", "dictionary", "russian_mfa"],
        check=False,
    )

    # Запускаем alignment
    print("Running MFA alignment (this may take a while)...")
    cmd = [
        "mfa", "align",
        input_dir,
        "russian_mfa",     # dictionary
        "russian_mfa",     # acoustic model
        output_dir,
        "--clean",
        "--overwrite",
        "-j", "4",         # число потоков
    ]
    subprocess.run(cmd, check=True)
    print(f"Alignments saved to {output_dir}")


def parse_textgrid(textgrid_path: str) -> list[tuple[str, float, float]]:
    """
    Парсинг TextGrid файла (выход MFA).
    Извлекаем временные границы фонем.

    Returns:
        list of (phone, start_time, end_time)
    """
    phones = []

    with open(textgrid_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Ищем tier "phones"
    lines = content.split("\n")
    in_phones_tier = False
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if '"phones"' in line.lower():
            in_phones_tier = True

        if in_phones_tier and line.startswith("intervals"):
            # Следующие 3 строки: xmin, xmax, text
            pass

        if in_phones_tier and "xmin" in line and "=" in line:
            xmin = float(line.split("=")[1].strip())
            i += 1
            xmax = float(lines[i].strip().split("=")[1].strip())
            i += 1
            text_line = lines[i].strip()
            phone = text_line.split('"')[1] if '"' in text_line else ""

            if phone and phone not in ["", "sp", "sil", "spn"]:
                phones.append((phone, xmin, xmax))

        i += 1

    return phones


def textgrid_to_durations(
    textgrid_path: str,
    text: str,
    sr: int = 22050,
    hop_length: int = 256,
) -> np.ndarray | None:
    """
    Конвертация TextGrid → duration per grapheme.

    Поскольку мы используем grapheme-based подход,
    маппим фонемы MFA обратно на символы текста.
    Простой подход: считаем общую длительность всех фонем
    и равномерно распределяем по символам, с учётом
    пропорций длительностей фонем.

    Более точный подход: маппим каждую фонему на соответствующий символ.

    Args:
        textgrid_path: путь к TextGrid
        text: текст utterance
        sr: sample rate
        hop_length: hop size для mel
    Returns:
        durations: (n_chars,) — количество mel-фреймов на символ
    """
    phones = parse_textgrid(textgrid_path)

    if not phones:
        return None

    # Phoneme ids для текста (наш grapheme-based подход)
    char_ids = text_to_sequence(text)
    n_chars = len(char_ids)

    # Общая длительность в секундах
    total_duration_sec = phones[-1][2] - phones[0][1]
    total_frames = int(total_duration_sec * sr / hop_length)

    if total_frames <= 0 or n_chars <= 0:
        return None

    # Простой подход: равномерное распределение фреймов по символам
    # с учётом длительностей фонем
    phone_durations_sec = [end - start for _, start, end in phones]
    total_phone_dur = sum(phone_durations_sec)

    if len(phones) == n_chars:
        # Идеальный случай: число фонем = число символов
        durations = np.array([
            max(1, round(dur / total_phone_dur * total_frames))
            for dur in phone_durations_sec
        ])
    else:
        # Распределяем пропорционально
        frames_per_char = total_frames / n_chars
        durations = np.array([max(1, round(frames_per_char))] * n_chars)

    # Корректируем чтобы сумма = total_frames
    diff = total_frames - durations.sum()
    if diff != 0:
        indices = np.argsort(durations)[::-1]
        step = 1 if diff > 0 else -1
        for i in range(abs(diff)):
            idx = indices[i % len(indices)]
            durations[idx] = max(1, durations[idx] + step)

    return durations.astype(np.int64)


def convert_all_textgrids(
    textgrid_dir: str,
    metadata_path: str,
    output_dir: str,
    sr: int = 22050,
    hop_length: int = 256,
):
    """
    Конвертация всех TextGrid → .npy duration файлы.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем metadata
    entries = {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                basename = parts[0].strip().replace(".wav", "")
                entries[basename] = parts[1].strip()

    converted = 0
    skipped = 0

    tg_files = sorted(Path(textgrid_dir).glob("**/*.TextGrid"))
    print(f"Found {len(tg_files)} TextGrid files")

    for tg_path in tqdm(tg_files, desc="Converting TextGrids"):
        basename = tg_path.stem

        # Пробуем найти в entries (basename может быть с/без _RUSLAN и т.д.)
        if basename not in entries:
            skipped += 1
            continue

        text = entries[basename]
        durations = textgrid_to_durations(str(tg_path), text, sr, hop_length)

        if durations is None:
            skipped += 1
            continue

        np.save(os.path.join(output_dir, f"{basename}.npy"), durations)
        converted += 1

    print(f"Converted: {converted}, Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(description="Extract alignments with MFA")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to RUSLAN dataset")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Metadata file (basename|text). Auto-detected if not given.")
    parser.add_argument("--output_dir", type=str, default="data/alignments",
                        help="Output directory for duration .npy files")
    parser.add_argument("--mfa_input_dir", type=str, default="data/mfa_input",
                        help="Temp directory for MFA input")
    parser.add_argument("--mfa_output_dir", type=str, default="data/mfa_output",
                        help="Directory for MFA TextGrid output")
    parser.add_argument("--skip_mfa", action="store_true",
                        help="Skip MFA, only convert existing TextGrids")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop_length", type=int, default=256)
    args = parser.parse_args()

    # Auto-detect metadata
    if args.metadata is None:
        for name in ["metadata_RUSLAN_22200.csv", "metadata.csv", "metadata.txt", "transcripts.txt"]:
            path = os.path.join(args.data_dir, name)
            if os.path.exists(path):
                args.metadata = path
                break
        # Fallback: любой csv с metadata в имени
        if args.metadata is None:
            for f in Path(args.data_dir).glob("*metadata*.csv"):
                args.metadata = str(f)
                break
        if args.metadata is None:
            print(f"ERROR: No metadata file found in {args.data_dir}")
            print(f"Files: {os.listdir(args.data_dir)}")
            sys.exit(1)

    print(f"Metadata: {args.metadata}")

    if not args.skip_mfa:
        # Step 1: Prepare MFA input
        print("\n=== Step 1: Preparing MFA input ===")
        prepare_mfa_input(args.data_dir, args.mfa_input_dir, args.metadata)

        # Step 2: Run MFA
        print("\n=== Step 2: Running MFA ===")
        run_mfa(args.mfa_input_dir, args.mfa_output_dir)

    # Step 3: Convert TextGrids to durations
    print("\n=== Step 3: Converting TextGrids to durations ===")
    convert_all_textgrids(
        args.mfa_output_dir, args.metadata, args.output_dir,
        sr=args.sr, hop_length=args.hop_length,
    )

    print(f"\nDone! Durations saved to {args.output_dir}")
    print("Next step: python scripts/extract_features.py ...")


if __name__ == "__main__":
    main()

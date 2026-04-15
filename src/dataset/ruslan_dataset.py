"""
Dataset для RUSLAN + предизвлечённые характеристики (duration, pitch, energy).

Структура данных (после предобработки):
    processed/
        mel/          - mel-spectrograms (.npy)
        pitch/        - pitch values (.npy)
        energy/       - energy values (.npy)
        duration/     - duration values (.npy)
        metadata.txt  - filename|text|phoneme_ids
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class RuslanDataset(Dataset):
    """
    Загружаем предобработанные данные:
    - phoneme ids
    - mel-spectrogram
    - duration (per phoneme)
    - pitch (per phoneme, после усреднения по фреймам)
    - energy (per phoneme, после усреднения по фреймам)
    """

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = data_dir
        self.split = split

        # Загружаем метаданные
        metadata_path = os.path.join(data_dir, f"{split}_metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.mel_dir = os.path.join(data_dir, "mel")
        self.pitch_dir = os.path.join(data_dir, "pitch")
        self.energy_dir = os.path.join(data_dir, "energy")
        self.duration_dir = os.path.join(data_dir, "duration")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        item = self.metadata[idx]
        basename = item["basename"]

        # Загружаем данные
        phoneme_ids = np.array(item["phonemes"], dtype=np.int64)
        mel = np.load(os.path.join(self.mel_dir, f"{basename}.npy"))
        pitch = np.load(os.path.join(self.pitch_dir, f"{basename}.npy"))
        energy = np.load(os.path.join(self.energy_dir, f"{basename}.npy"))
        duration = np.load(os.path.join(self.duration_dir, f"{basename}.npy"))

        # mel shape: (n_mel, n_frames) → (n_frames, n_mel)
        if mel.shape[0] == 80:  # n_mel_channels
            mel = mel.T

        return {
            "phonemes": torch.from_numpy(phoneme_ids).long(),
            "mel": torch.from_numpy(mel).float(),
            "pitch": torch.from_numpy(pitch).float(),
            "energy": torch.from_numpy(energy).float(),
            "duration": torch.from_numpy(duration).long(),
            "basename": basename,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: паддим все последовательности до максимальной длины в батче.
    """
    # Длины
    src_lengths = torch.tensor([item["phonemes"].size(0) for item in batch])
    mel_lengths = torch.tensor([item["mel"].size(0) for item in batch])

    # Сортируем по src_length (убывание) для эффективности
    sorted_indices = torch.argsort(src_lengths, descending=True)

    # Pad phonemes
    phonemes = pad_sequence(
        [batch[i]["phonemes"] for i in sorted_indices],
        batch_first=True,
        padding_value=0,
    )

    # Pad mel
    max_mel_len = mel_lengths.max().item()
    n_mel = batch[0]["mel"].size(1)
    mel_padded = torch.zeros(len(batch), max_mel_len, n_mel)
    for j, i in enumerate(sorted_indices):
        mel_len = batch[i]["mel"].size(0)
        mel_padded[j, :mel_len] = batch[i]["mel"]

    # Pad duration, pitch, energy (phoneme-level)
    duration = pad_sequence(
        [batch[i]["duration"] for i in sorted_indices],
        batch_first=True,
        padding_value=0,
    )
    pitch = pad_sequence(
        [batch[i]["pitch"] for i in sorted_indices],
        batch_first=True,
        padding_value=0.0,
    )
    energy = pad_sequence(
        [batch[i]["energy"] for i in sorted_indices],
        batch_first=True,
        padding_value=0.0,
    )

    src_lengths_sorted = src_lengths[sorted_indices]
    mel_lengths_sorted = mel_lengths[sorted_indices]

    return {
        "phonemes": phonemes,
        "src_lengths": src_lengths_sorted,
        "mel": mel_padded,
        "mel_lengths": mel_lengths_sorted,
        "duration": duration,
        "pitch": pitch,
        "energy": energy,
    }


def get_dataloader(
    data_dir: str,
    split: str = "train",
    batch_size: int = 48,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Создать DataLoader для указанного split."""
    dataset = RuslanDataset(data_dir, split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True if split == "train" else False,
    )

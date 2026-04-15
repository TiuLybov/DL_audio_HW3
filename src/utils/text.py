"""
Конвертация русского текста в последовательность фонем.

Используем простой подход: буква → фонема (grapheme-based).
Для лучшего качества можно использовать russian_g2p, но для базовой
реализации достаточно символьного подхода.

Словарь фонем основан на кириллическом алфавите + спец символы.
"""

import re
from typing import Optional

# Базовый набор символов (grapheme-based подход)
PAD = "_"
BOS = "^"
EOS = "$"
SPACE = " "

# Русские буквы
RUSSIAN_CHARS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

# Все символы
SYMBOLS = [PAD, BOS, EOS, SPACE] + list(RUSSIAN_CHARS)

# Symbol to id mapping
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}

VOCAB_SIZE = len(SYMBOLS)


def text_to_sequence(text: str) -> list[int]:
    """
    Конвертируем русский текст в последовательность id.

    Шаги:
    1. Приводим к нижнему регистру
    2. Убираем всё кроме русских букв и пробелов
    3. Добавляем BOS/EOS токены
    4. Конвертируем каждый символ в id

    Args:
        text: входной текст на русском
    Returns:
        list of int — id символов
    """
    text = text.lower().strip()

    # Оставляем только русские буквы и пробелы
    text = re.sub(r"[^а-яёА-ЯЁ\s]", "", text)

    # Убираем множественные пробелы
    text = re.sub(r"\s+", " ", text).strip()

    # BOS + text + EOS
    sequence = [SYMBOL_TO_ID[BOS]]
    for char in text:
        if char in SYMBOL_TO_ID:
            sequence.append(SYMBOL_TO_ID[char])
    sequence.append(SYMBOL_TO_ID[EOS])

    return sequence


def sequence_to_text(sequence: list[int]) -> str:
    """Обратное преобразование id → текст."""
    return "".join(ID_TO_SYMBOL.get(idx, "") for idx in sequence)


def get_vocab_size() -> int:
    """Размер словаря."""
    return VOCAB_SIZE

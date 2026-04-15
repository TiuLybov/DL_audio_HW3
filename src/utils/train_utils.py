"""
Утилиты для обучения:
- Noam LR Scheduler
- WandB logging helpers
- Seed everything
"""

import os
import random
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int = 42):
    """Фиксация всех random seeds для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic mode (может замедлить)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_noam_scheduler(optimizer, d_model: int, warmup_steps: int):
    """
    Noam scheduler из 'Attention Is All You Need'.

    lr(step) = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Warmup фаза: lr линейно растёт от 0 до peak.
    После warmup: lr убывает как 1/sqrt(step).
    """
    def lr_lambda(step):
        step = max(step, 1)
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    return LambdaLR(optimizer, lr_lambda)


def count_parameters(model) -> int:
    """Подсчёт обучаемых параметров."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_mel_to_wandb(mel_pred, mel_target, step):
    """Логирование mel-спектрограмм как изображений в WandB."""
    try:
        import wandb
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        mel_gt_np = mel_target[0].cpu().numpy().T  # (n_mel, mel_len)
        mel_pred_np = mel_pred[0].cpu().numpy().T

        axes[0].imshow(mel_gt_np, aspect="auto", origin="lower")
        axes[0].set_title("Ground Truth")
        axes[0].set_ylabel("Mel Channel")

        axes[1].imshow(mel_pred_np, aspect="auto", origin="lower")
        axes[1].set_title("Predicted")
        axes[1].set_ylabel("Mel Channel")
        axes[1].set_xlabel("Frame")

        plt.tight_layout()
        wandb.log({"mel_comparison": wandb.Image(fig)}, step=step)
        plt.close(fig)
    except Exception as e:
        print(f"Warning: Could not log mel to wandb: {e}")


def log_audio_to_wandb(audio: np.ndarray, sr: int, name: str, step: int):
    """Логирование аудио в WandB."""
    try:
        import wandb
        wandb.log({name: wandb.Audio(audio, sample_rate=sr)}, step=step)
    except Exception as e:
        print(f"Warning: Could not log audio to wandb: {e}")

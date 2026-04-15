"""
Скрипт обучения FastSpeech2.

Логирование через WandB:
- loss (total, mel, duration, pitch, energy)
- mel-спектрограммы (predicted vs GT)
- learning rate

Scheduler: Noam (warmup + inverse sqrt decay), как в оригинальном Transformer.
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.fastspeech2 import FastSpeech2
from src.model.loss import FastSpeech2Loss
from src.dataset.ruslan_dataset import get_dataloader
from src.configs.model_config import ModelConfig, TrainConfig
from src.utils.text import VOCAB_SIZE
from src.utils.train_utils import set_seed, get_noam_scheduler, count_parameters, log_mel_to_wandb


def train_epoch(model, dataloader, loss_fn, optimizer, scheduler, device, epoch, config):
    """Одна эпоха обучения."""
    model.train()
    total_losses = {
        "total_loss": 0, "mel_loss": 0, "mel_postnet_loss": 0,
        "duration_loss": 0, "pitch_loss": 0, "energy_loss": 0,
    }
    n_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        phonemes = batch["phonemes"].to(device)
        src_lengths = batch["src_lengths"].to(device)
        mel = batch["mel"].to(device)
        mel_lengths = batch["mel_lengths"].to(device)
        duration = batch["duration"].to(device)
        pitch = batch["pitch"].to(device)
        energy = batch["energy"].to(device)

        output = model(
            phonemes=phonemes,
            src_lengths=src_lengths,
            mel_targets=mel,
            mel_lengths=mel_lengths,
            duration_targets=duration,
            pitch_targets=pitch,
            energy_targets=energy,
        )

        losses = loss_fn(
            mel_pred=output["mel_output"],
            mel_postnet=output["mel_postnet"],
            mel_target=mel,
            log_duration_pred=output["log_duration_pred"],
            duration_target=duration,
            pitch_pred=output["pitch_pred"],
            pitch_target=pitch,
            energy_pred=output["energy_pred"],
            energy_target=energy,
            src_mask=output["src_mask"],
            mel_mask=output["mel_mask"],
        )

        optimizer.zero_grad()
        losses["total_loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{losses['total_loss'].item():.4f}",
            "mel": f"{losses['mel_loss'].item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    avg_losses = {k: v / max(n_batches, 1) for k, v in total_losses.items()}
    return avg_losses


@torch.no_grad()
def validate(model, dataloader, loss_fn, device):
    """Валидация."""
    model.eval()
    total_losses = {
        "total_loss": 0, "mel_loss": 0, "mel_postnet_loss": 0,
        "duration_loss": 0, "pitch_loss": 0, "energy_loss": 0,
    }
    n_batches = 0

    for batch in dataloader:
        phonemes = batch["phonemes"].to(device)
        src_lengths = batch["src_lengths"].to(device)
        mel = batch["mel"].to(device)
        mel_lengths = batch["mel_lengths"].to(device)
        duration = batch["duration"].to(device)
        pitch = batch["pitch"].to(device)
        energy = batch["energy"].to(device)

        output = model(
            phonemes=phonemes,
            src_lengths=src_lengths,
            mel_targets=mel,
            mel_lengths=mel_lengths,
            duration_targets=duration,
            pitch_targets=pitch,
            energy_targets=energy,
        )

        losses = loss_fn(
            mel_pred=output["mel_output"],
            mel_postnet=output["mel_postnet"],
            mel_target=mel,
            log_duration_pred=output["log_duration_pred"],
            duration_target=duration,
            pitch_pred=output["pitch_pred"],
            pitch_target=pitch,
            energy_pred=output["energy_pred"],
            energy_target=energy,
            src_mask=output["src_mask"],
            mel_mask=output["mel_mask"],
        )

        for k in total_losses:
            total_losses[k] += losses[k].item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in total_losses.items()}


def log_mel_images(model, batch, device, step):
    """Логируем mel-спектрограммы в WandB."""
    model.eval()
    with torch.no_grad():
        phonemes = batch["phonemes"][:1].to(device)
        src_lengths = batch["src_lengths"][:1].to(device)
        mel = batch["mel"][:1].to(device)
        mel_lengths = batch["mel_lengths"][:1].to(device)
        duration = batch["duration"][:1].to(device)
        pitch = batch["pitch"][:1].to(device)
        energy = batch["energy"][:1].to(device)

        output = model(
            phonemes=phonemes, src_lengths=src_lengths,
            mel_targets=mel, mel_lengths=mel_lengths,
            duration_targets=duration, pitch_targets=pitch, energy_targets=energy,
        )

        log_mel_to_wandb(output["mel_postnet"], mel, step)

    model.train()


def save_checkpoint(model, optimizer, scheduler, epoch, step, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
    }, path)
    print(f"Saved checkpoint: {path}")


def main():
    parser = argparse.ArgumentParser(description="Train FastSpeech2")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb_project", type=str, default="fastspeech2-ruslan")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    model_config = ModelConfig()
    train_config = TrainConfig()
    model_config.vocab_size = VOCAB_SIZE

    # Воспроизводимость
    set_seed(train_config.seed)

    # Загружаем статистику pitch/energy
    stats_path = os.path.join(args.data_dir, "stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        model_config.pitch_min = stats["pitch_min"]
        model_config.pitch_max = stats["pitch_max"]
        model_config.energy_min = stats["energy_min"]
        model_config.energy_max = stats["energy_max"]
        print(f"Loaded stats: pitch [{stats['pitch_min']:.2f}, {stats['pitch_max']:.2f}], "
              f"energy [{stats['energy_min']:.2f}, {stats['energy_max']:.2f}]")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={"model": model_config.__dict__, "train": train_config.__dict__},
    )

    device = torch.device(args.device)
    model = FastSpeech2(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    loss_fn = FastSpeech2Loss(train_config)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9,
                     weight_decay=train_config.weight_decay)
    scheduler = get_noam_scheduler(optimizer, model_config.encoder_hidden, args.warmup_steps)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    train_loader = get_dataloader(args.data_dir, "train", args.batch_size,
                                  train_config.num_workers, shuffle=True)
    val_loader = get_dataloader(args.data_dir, "val", args.batch_size,
                                train_config.num_workers, shuffle=False)

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.num_epochs):
        train_losses = train_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, device, epoch, train_config,
        )
        global_step += len(train_loader)

        wandb.log({f"train/{k}": v for k, v in train_losses.items()}, step=global_step)
        wandb.log({"lr": scheduler.get_last_lr()[0]}, step=global_step)
        print(f"Epoch {epoch} Train: " + ", ".join(f"{k}={v:.4f}" for k, v in train_losses.items()))

        val_losses = validate(model, val_loader, loss_fn, device)
        wandb.log({f"val/{k}": v for k, v in val_losses.items()}, step=global_step)
        print(f"Epoch {epoch} Val: " + ", ".join(f"{k}={v:.4f}" for k, v in val_losses.items()))

        # Log mel images every 5 epochs
        if (epoch + 1) % 5 == 0:
            sample_batch = next(iter(val_loader))
            log_mel_images(model, sample_batch, device, global_step)

        if val_losses["total_loss"] < best_val_loss:
            best_val_loss = val_losses["total_loss"]
            save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                            os.path.join(args.checkpoint_dir, "best_model.pth"))

        if (epoch + 1) % 20 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                            os.path.join(args.checkpoint_dir, f"checkpoint_epoch{epoch}.pth"))

    save_checkpoint(model, optimizer, scheduler, args.num_epochs - 1, global_step,
                    os.path.join(args.checkpoint_dir, "last_model.pth"))
    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()

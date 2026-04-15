from dataclasses import dataclass


@dataclass
class ModelConfig:
    """FastSpeech2 model configuration."""

    # Phoneme vocabulary
    vocab_size: int = 300  # будет уточнено после создания словаря
    padding_idx: int = 0

    # Encoder / Decoder
    encoder_hidden: int = 256
    encoder_head: int = 2
    encoder_layers: int = 4
    encoder_conv_filter_size: int = 1024
    encoder_conv_kernel_size: int = 9

    decoder_hidden: int = 256
    decoder_head: int = 2
    decoder_layers: int = 4
    decoder_conv_filter_size: int = 1024
    decoder_conv_kernel_size: int = 9

    # Variance Adaptor
    variance_predictor_filter_size: int = 256
    variance_predictor_kernel_size: int = 3
    variance_predictor_dropout: float = 0.5

    # Duration / Pitch / Energy bins
    n_bins: int = 256
    pitch_min: float = 0.0
    pitch_max: float = 1.0  # будет пересчитано из данных
    energy_min: float = 0.0
    energy_max: float = 1.0  # аналогично

    # Mel-spectrogram
    n_mel_channels: int = 80

    # Dropout
    encoder_dropout: float = 0.2
    decoder_dropout: float = 0.2

    # Max sequence length
    max_seq_len: int = 3000


@dataclass
class TrainConfig:
    """Training configuration."""

    # Paths
    data_dir: str = "data/ruslan"
    alignments_dir: str = "data/alignments"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Training
    batch_size: int = 48
    num_epochs: int = 200
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    warmup_steps: int = 4000
    grad_clip: float = 1.0

    # Loss weights
    mel_loss_weight: float = 1.0
    duration_loss_weight: float = 1.0
    pitch_loss_weight: float = 1.0
    energy_loss_weight: float = 1.0

    # Logging
    log_step: int = 100
    save_step: int = 5000
    val_step: int = 1000

    # Data
    num_workers: int = 4
    seed: int = 42

    # WandB
    wandb_project: str = "fastspeech2-ruslan"


@dataclass
class MelConfig:
    """Mel-spectrogram extraction config."""

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0

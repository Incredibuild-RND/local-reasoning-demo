"""Configuration dataclasses for the reasoning training pipeline."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "float32"
    attn_implementation: str = "eager"

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )


@dataclass
class DataConfig:
    """Data generation configuration."""
    num_samples: int = 500
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Number ranges for arithmetic problems
    min_val: int = 0
    max_val: int = 50
    min_val_c: int = 0
    max_val_c: int = 20

    # Seed for reproducibility
    seed: int = 42


@dataclass
class SFTConfig:
    """SFT training configuration."""
    output_dir: str = "./sft_output"
    num_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    learning_rate: float = 2e-4
    max_length: int = 512
    save_strategy: str = "epoch"


@dataclass
class GRPOConfig:
    """GRPO training configuration."""
    output_dir: str = "./grpo_output"
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    num_generations: int = 2
    max_completion_length: int = 512
    temperature: float = 0.6
    save_strategy: str = "epoch"

    # Reward weights
    correctness_weight: float = 1.0
    format_weight: float = 0.5


@dataclass
class TrainingConfig:
    """Main training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)

    # Output directories
    output_dir: str = "./output"
    sft_checkpoint_dir: str = "./sft_final"
    grpo_checkpoint_dir: str = "./reasoning_final"

    # Wandb configuration
    wandb_project: str = "reasoning-training"
    wandb_run_name: Optional[str] = None
    use_wandb: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Training mode: "sft", "grpo", or "full"
    mode: str = "full"


def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()

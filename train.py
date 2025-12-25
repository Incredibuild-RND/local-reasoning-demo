#!/usr/bin/env python3
"""Main training script for reasoning model with SFT + GRPO pipeline.

Usage:
    python train.py --help
    python train.py --mode full --num-samples 500 --sft-epochs 2
    python train.py --mode sft --output-dir ./my_experiment
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig as TRLGRPOConfig
from trl import GRPOTrainer, SFTConfig as TRLSFTConfig, SFTTrainer

from config import DataConfig, GRPOConfig, ModelConfig, SFTConfig, TrainingConfig
from data import generate_data, generate_test_problems, prepare_sft_dataset, split_data
from evaluate import (
    EvaluationResult,
    evaluate_model,
    log_to_wandb,
    print_sample_outputs,
    save_results,
)
from rewards import correctness_reward_func, format_reward_func
from utils import (
    ensure_dir,
    free_memory,
    get_device,
    log_memory_usage,
    sanitize_model_dtype,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a reasoning model with SFT + GRPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Training mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sft", "grpo", "full"],
        default="full",
        help="Training mode: sft only, grpo only, or full pipeline",
    )

    # Data configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of training samples to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name or path",
    )

    # SFT configuration
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=2,
        help="Number of SFT training epochs",
    )
    parser.add_argument(
        "--sft-batch-size",
        type=int,
        default=4,
        help="SFT batch size per device",
    )
    parser.add_argument(
        "--sft-lr",
        type=float,
        default=2e-4,
        help="SFT learning rate",
    )

    # GRPO configuration
    parser.add_argument(
        "--grpo-epochs",
        type=int,
        default=1,
        help="Number of GRPO training epochs",
    )
    parser.add_argument(
        "--grpo-batch-size",
        type=int,
        default=1,
        help="GRPO batch size per device",
    )
    parser.add_argument(
        "--grpo-lr",
        type=float,
        default=1e-5,
        help="GRPO learning rate",
    )
    parser.add_argument(
        "--grpo-samples",
        type=int,
        default=50,
        help="Number of samples to use for GRPO (subset of training data)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for checkpoints and logs",
    )

    # Wandb configuration
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="reasoning-training",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Weights & Biases run name (auto-generated if not specified)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    # Evaluation
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=50,
        help="Number of samples for final evaluation",
    )

    # Resume from checkpoint
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        default=None,
        help="Path to SFT checkpoint to resume from (for GRPO mode)",
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """Create TrainingConfig from command line arguments."""
    return TrainingConfig(
        model=ModelConfig(name=args.model_name),
        data=DataConfig(num_samples=args.num_samples, seed=args.seed),
        sft=SFTConfig(
            num_epochs=args.sft_epochs,
            batch_size=args.sft_batch_size,
            learning_rate=args.sft_lr,
        ),
        grpo=GRPOConfig(
            num_epochs=args.grpo_epochs,
            batch_size=args.grpo_batch_size,
            learning_rate=args.grpo_lr,
        ),
        output_dir=args.output_dir,
        sft_checkpoint_dir=str(Path(args.output_dir) / "sft_final"),
        grpo_checkpoint_dir=str(Path(args.output_dir) / "reasoning_final"),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        use_wandb=not args.no_wandb,
        log_level=args.log_level,
        mode=args.mode,
    )


def train_sft(
    config: TrainingConfig,
    train_dataset,
    tokenizer,
    device: str,
    logger,
) -> str:
    """Run SFT training phase.

    Returns:
        Path to saved checkpoint
    """
    logger.info("=" * 50)
    logger.info("PHASE: SFT (Teaching the <think> format)")
    logger.info("=" * 50)

    # Load model
    logger.info(f"Loading model: {config.model.name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        device_map=device,
        attn_implementation=config.model.attn_implementation,
        torch_dtype=torch.float32,
    )
    model.config.torch_dtype = torch.float32
    model.config.use_cache = False

    # Apply LoRA
    logger.info("Applying LoRA adapters...")
    peft_config = LoraConfig(
        r=config.model.lora_r,
        lora_alpha=config.model.lora_alpha,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.model.lora_target_modules,
    )
    model = get_peft_model(model, peft_config)
    sanitize_model_dtype(model)

    # Prepare dataset
    sft_dataset = prepare_sft_dataset(train_dataset)
    logger.info(f"Training on {len(sft_dataset)} samples")

    # Configure trainer
    sft_args = TRLSFTConfig(
        output_dir=str(Path(config.output_dir) / "sft_checkpoints"),
        max_length=512,
        num_train_epochs=config.sft.num_epochs,
        per_device_train_batch_size=config.sft.batch_size,
        gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
        learning_rate=config.sft.learning_rate,
        fp16=False,
        bf16=False,
        packing=False,
        report_to="wandb" if config.use_wandb else "none",
        dataset_text_field="text",
        save_strategy=config.sft.save_strategy,
        logging_steps=10,
        use_mps_device=device == "mps",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=sft_dataset,
        args=sft_args,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting SFT training...")
    trainer.train()

    # Save checkpoint
    checkpoint_dir = config.sft_checkpoint_dir
    ensure_dir(checkpoint_dir)
    trainer.model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    logger.info(f"SFT checkpoint saved to: {checkpoint_dir}")

    # Cleanup
    del trainer, model
    free_memory()
    logger.info("SFT training complete.")

    return checkpoint_dir


def train_grpo(
    config: TrainingConfig,
    train_dataset,
    tokenizer,
    sft_checkpoint: str,
    device: str,
    logger,
    grpo_samples: int = 50,
) -> str:
    """Run GRPO training phase.

    Returns:
        Path to saved checkpoint
    """
    logger.info("=" * 50)
    logger.info("PHASE: GRPO (Refining Logic via Rewards)")
    logger.info("=" * 50)

    # Load base model + SFT adapters
    logger.info(f"Loading base model: {config.model.name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        device_map=device,
        attn_implementation=config.model.attn_implementation,
        torch_dtype=torch.float32,
    )
    model.config.torch_dtype = torch.float32

    logger.info(f"Loading SFT adapters from: {sft_checkpoint}")
    model = PeftModel.from_pretrained(model, sft_checkpoint, is_trainable=True)
    sanitize_model_dtype(model)

    # Use subset for GRPO
    grpo_dataset = train_dataset.select(range(min(grpo_samples, len(train_dataset))))
    logger.info(f"GRPO training on {len(grpo_dataset)} samples")

    # Configure trainer
    grpo_args = TRLGRPOConfig(
        output_dir=str(Path(config.output_dir) / "grpo_checkpoints"),
        learning_rate=config.grpo.learning_rate,
        num_train_epochs=config.grpo.num_epochs,
        per_device_train_batch_size=config.grpo.batch_size,
        gradient_accumulation_steps=config.grpo.gradient_accumulation_steps,
        num_generations=config.grpo.num_generations,
        max_completion_length=config.grpo.max_completion_length,
        fp16=False,
        bf16=False,
        report_to="wandb" if config.use_wandb else "none",
        save_strategy=config.grpo.save_strategy,
        temperature=config.grpo.temperature,
        logging_steps=5,
        use_mps_device=device == "mps",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward_func, format_reward_func],
        args=grpo_args,
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting GRPO training...")
    trainer.train()

    # Save checkpoint
    checkpoint_dir = config.grpo_checkpoint_dir
    ensure_dir(checkpoint_dir)
    trainer.save_model(checkpoint_dir)
    logger.info(f"GRPO checkpoint saved to: {checkpoint_dir}")

    # Cleanup
    del trainer, model
    free_memory()
    logger.info("GRPO training complete.")

    return checkpoint_dir


def run_evaluation(
    checkpoint_dir: str,
    model_name: str,
    tokenizer,
    eval_samples: int,
    output_dir: str,
    logger,
    prefix: str = "final",
) -> EvaluationResult:
    """Run final evaluation on the trained model."""
    logger.info("=" * 50)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 50)

    # Force CPU for inference stability
    inference_device = "cpu"
    logger.info(f"Running inference on {inference_device.upper()}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=inference_device,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model.eval()

    # Generate test data
    test_data = generate_test_problems(num_samples=eval_samples)
    logger.info(f"Evaluating on {len(test_data)} test problems")

    # Evaluate
    result, samples = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        device=inference_device,
        logger=logger,
    )

    # Save results
    results_file = save_results(result, samples, output_dir, prefix=prefix)
    logger.info(f"Results saved to: {results_file}")

    # Log to wandb
    log_to_wandb(result, samples, prefix=prefix)

    # Print samples
    print_sample_outputs(samples, num_samples=3, logger=logger)

    # Cleanup
    del model
    free_memory()

    return result


def main():
    """Main entry point."""
    args = parse_args()
    config = create_config_from_args(args)

    # Setup output directory
    ensure_dir(config.output_dir)

    # Setup logging
    logger = setup_logging(
        log_level=config.log_level,
        output_dir=config.output_dir,
    )

    logger.info("=" * 60)
    logger.info("REASONING MODEL TRAINING")
    logger.info("=" * 60)

    # Log configuration
    logger.info(f"Mode: {config.mode}")
    logger.info(f"Model: {config.model.name}")
    logger.info(f"Samples: {config.data.num_samples}")
    logger.info(f"Output: {config.output_dir}")

    # Detect device
    device = get_device()
    logger.info(f"Device: {device.upper()}")

    # Initialize wandb
    if config.use_wandb:
        run_name = config.wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "mode": config.mode,
                "model": config.model.name,
                "num_samples": config.data.num_samples,
                "sft_epochs": config.sft.num_epochs,
                "grpo_epochs": config.grpo.num_epochs,
                "sft_lr": config.sft.learning_rate,
                "grpo_lr": config.grpo.learning_rate,
            },
        )
        logger.info(f"Wandb initialized: {run_name}")

    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.name, trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token

        # Generate data
        logger.info("Generating training data...")
        data = generate_data(config.data)
        train_dataset, val_dataset, test_dataset = split_data(data, config.data)
        logger.info(
            f"Data split: {len(train_dataset)} train, "
            f"{len(val_dataset)} val, {len(test_dataset)} test"
        )

        # Run training based on mode
        sft_checkpoint = args.sft_checkpoint

        if config.mode in ["sft", "full"]:
            sft_checkpoint = train_sft(
                config=config,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                device=device,
                logger=logger,
            )

        if config.mode in ["grpo", "full"]:
            if sft_checkpoint is None:
                logger.error("GRPO mode requires --sft-checkpoint or running in 'full' mode")
                sys.exit(1)

            grpo_checkpoint = train_grpo(
                config=config,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
                sft_checkpoint=sft_checkpoint,
                device=device,
                logger=logger,
                grpo_samples=args.grpo_samples,
            )
        else:
            grpo_checkpoint = sft_checkpoint

        # Final evaluation
        final_checkpoint = grpo_checkpoint if config.mode != "sft" else sft_checkpoint
        result = run_evaluation(
            checkpoint_dir=final_checkpoint,
            model_name=config.model.name,
            tokenizer=tokenizer,
            eval_samples=args.eval_samples,
            output_dir=config.output_dir,
            logger=logger,
            prefix=f"{config.mode}_eval",
        )

        # Summary
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final Accuracy: {result.accuracy:.1%}")
        logger.info(f"Format Compliance: {result.format_compliance:.1%}")
        logger.info(f"Checkpoint: {final_checkpoint}")

        if config.use_wandb:
            wandb.log({
                "final/accuracy": result.accuracy,
                "final/format_compliance": result.format_compliance,
            })
            wandb.finish()
            logger.info("Wandb run finished.")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if config.use_wandb:
            wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    main()

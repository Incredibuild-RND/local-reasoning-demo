"""Evaluation framework with metrics and wandb integration."""

import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rewards import (
    check_format,
    count_reasoning_steps,
    extract_answer_after_think,
    has_verification,
)


@dataclass
class EvaluationResult:
    """Results from model evaluation."""

    accuracy: float
    format_compliance: float
    avg_reasoning_steps: float
    verification_rate: float
    total_samples: int
    correct_samples: int
    timestamp: str


@dataclass
class SampleOutput:
    """Single sample output for logging."""

    prompt: str
    expected: str
    predicted: str
    is_correct: bool
    has_format: bool
    reasoning_steps: int
    full_output: str


def evaluate_model(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    test_data: List[Dict[str, str]],
    device: str = "cpu",
    max_new_tokens: int = 200,
    temperature: float = 0.6,
    logger: Optional[logging.Logger] = None,
) -> tuple[EvaluationResult, List[SampleOutput]]:
    """Evaluate a model on test data.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        test_data: List of test problems with 'prompt' and 'answer'
        device: Device to run inference on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        logger: Optional logger

    Returns:
        Tuple of (EvaluationResult, list of SampleOutputs)
    """
    if logger:
        logger.info(f"Evaluating on {len(test_data)} samples...")

    model.eval()
    samples = []
    stats = {
        "total": 0,
        "correct": 0,
        "format_ok": 0,
        "total_steps": 0,
        "verification_count": 0,
    }

    for item in tqdm(test_data, desc="Evaluating", disable=logger is None):
        prompt = f"<|user|>\n{item['prompt']}<|assistant|>\n<think>"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract prediction
        predicted = extract_answer_after_think(output_text)
        predicted_str = str(predicted) if predicted is not None else "None"

        # Check format
        has_format = check_format(output_text)
        steps = count_reasoning_steps(output_text)
        has_verif = has_verification(output_text)

        # Check correctness
        is_correct = predicted_str == item["answer"]

        # Update stats
        stats["total"] += 1
        if is_correct:
            stats["correct"] += 1
        if has_format:
            stats["format_ok"] += 1
        stats["total_steps"] += steps
        if has_verif:
            stats["verification_count"] += 1

        # Store sample
        samples.append(
            SampleOutput(
                prompt=item["prompt"],
                expected=item["answer"],
                predicted=predicted_str,
                is_correct=is_correct,
                has_format=has_format,
                reasoning_steps=steps,
                full_output=output_text,
            )
        )

    # Calculate metrics
    result = EvaluationResult(
        accuracy=stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0,
        format_compliance=stats["format_ok"] / stats["total"] if stats["total"] > 0 else 0.0,
        avg_reasoning_steps=stats["total_steps"] / stats["total"] if stats["total"] > 0 else 0.0,
        verification_rate=stats["verification_count"] / stats["total"] if stats["total"] > 0 else 0.0,
        total_samples=stats["total"],
        correct_samples=stats["correct"],
        timestamp=datetime.now().isoformat(),
    )

    if logger:
        logger.info(f"Accuracy: {result.accuracy:.1%}")
        logger.info(f"Format compliance: {result.format_compliance:.1%}")
        logger.info(f"Avg reasoning steps: {result.avg_reasoning_steps:.1f}")
        logger.info(f"Verification rate: {result.verification_rate:.1%}")

    return result, samples


def save_results(
    result: EvaluationResult,
    samples: List[SampleOutput],
    output_dir: str,
    prefix: str = "eval",
) -> Path:
    """Save evaluation results to JSON file.

    Args:
        result: Evaluation results
        samples: List of sample outputs
        output_dir: Output directory
        prefix: Filename prefix

    Returns:
        Path to saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"{prefix}_{timestamp}.json"

    data = {
        "metrics": asdict(result),
        "samples": [asdict(s) for s in samples],
    }

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    return filename


def log_to_wandb(
    result: EvaluationResult,
    samples: List[SampleOutput],
    prefix: str = "eval",
    log_samples: int = 10,
) -> None:
    """Log evaluation results to Weights & Biases.

    Args:
        result: Evaluation results
        samples: List of sample outputs
        prefix: Metric prefix (e.g., 'sft_eval', 'grpo_eval')
        log_samples: Number of sample outputs to log
    """
    try:
        import wandb

        if wandb.run is None:
            return

        # Log metrics
        metrics = {
            f"{prefix}/accuracy": result.accuracy,
            f"{prefix}/format_compliance": result.format_compliance,
            f"{prefix}/avg_reasoning_steps": result.avg_reasoning_steps,
            f"{prefix}/verification_rate": result.verification_rate,
        }
        wandb.log(metrics)

        # Log sample outputs as table
        if samples and log_samples > 0:
            table_data = []
            for s in samples[:log_samples]:
                table_data.append([
                    s.prompt,
                    s.expected,
                    s.predicted,
                    s.is_correct,
                    s.has_format,
                    s.reasoning_steps,
                    s.full_output[:500],  # Truncate long outputs
                ])

            table = wandb.Table(
                columns=[
                    "Prompt",
                    "Expected",
                    "Predicted",
                    "Correct",
                    "Has Format",
                    "Steps",
                    "Full Output",
                ],
                data=table_data,
            )
            wandb.log({f"{prefix}/samples": table})

    except ImportError:
        pass  # wandb not installed
    except Exception:
        pass  # wandb not configured


def print_sample_outputs(
    samples: List[SampleOutput],
    num_samples: int = 3,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Print sample outputs for inspection.

    Args:
        samples: List of sample outputs
        num_samples: Number of samples to print
        logger: Optional logger
    """
    output_fn = logger.info if logger else print

    output_fn("\n" + "=" * 50)
    output_fn("SAMPLE OUTPUTS")
    output_fn("=" * 50)

    for i, s in enumerate(samples[:num_samples]):
        output_fn(f"\n--- Sample {i + 1} ---")
        output_fn(f"Prompt: {s.prompt}")
        output_fn(f"Expected: {s.expected} | Predicted: {s.predicted} | Correct: {s.is_correct}")
        output_fn(f"Format OK: {s.has_format} | Steps: {s.reasoning_steps}")
        output_fn(f"Full output:\n{s.full_output}")
        output_fn("-" * 40)

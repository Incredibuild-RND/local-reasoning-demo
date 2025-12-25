#!/usr/bin/env python3
"""Interactive inference script for testing trained reasoning models.

Usage:
    python inference.py --checkpoint ./output/reasoning_final
    python inference.py --checkpoint ./output/reasoning_final --interactive
    python inference.py --checkpoint ./output/reasoning_final --batch --num-samples 20
"""

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import generate_test_problems
from evaluate import evaluate_model, print_sample_outputs
from utils import free_memory, setup_logging


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with a trained reasoning model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Base model name",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch evaluation on generated problems",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples for batch evaluation",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for inference",
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    model_name: str,
    device: str,
    logger,
):
    """Load model and tokenizer."""
    logger.info(f"Loading tokenizer from: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)

    logger.info(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )

    logger.info(f"Loading adapters from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    logger.info("Model loaded successfully.")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_tokens: int = 200,
    temperature: float = 0.6,
) -> str:
    """Generate a response for a given prompt."""
    formatted_prompt = f"<|user|>\n{prompt}<|assistant|>\n<think>"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def run_interactive(model, tokenizer, device: str, args, logger):
    """Run interactive inference mode."""
    logger.info("Interactive mode. Type 'quit' to exit.")
    logger.info("Enter arithmetic problems like: Calculate 23 + 15 + 8")
    print()

    while True:
        try:
            prompt = input("You: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                logger.info("Exiting...")
                break

            if not prompt:
                continue

            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )

            print(f"\nModel:\n{response}\n")
            print("-" * 40)

        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")


def run_batch(model, tokenizer, device: str, args, logger):
    """Run batch evaluation mode."""
    logger.info(f"Generating {args.num_samples} test problems...")
    test_data = generate_test_problems(num_samples=args.num_samples)

    result, samples = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        test_data=test_data,
        device=device,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        logger=logger,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:           {result.accuracy:.1%}")
    print(f"Format Compliance:  {result.format_compliance:.1%}")
    print(f"Avg Reasoning Steps: {result.avg_reasoning_steps:.1f}")
    print(f"Verification Rate:  {result.verification_rate:.1%}")
    print("=" * 50)

    print_sample_outputs(samples, num_samples=5)


def run_single_demo(model, tokenizer, device: str, args, logger):
    """Run a single demo with example problems."""
    examples = [
        "Calculate 23 + 15 + 8",
        "Calculate 42 + 17 + 5",
        "Calculate 35 + 28 + 12",
    ]

    print("\n" + "=" * 50)
    print("DEMO: Example Problems")
    print("=" * 50)

    for prompt in examples:
        print(f"\nProblem: {prompt}")
        print("-" * 40)

        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        print(response)
        print("-" * 40)


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging(log_level="INFO")

    logger.info("=" * 50)
    logger.info("REASONING MODEL INFERENCE")
    logger.info("=" * 50)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device}")

    try:
        model, tokenizer = load_model(
            checkpoint_path=args.checkpoint,
            model_name=args.model_name,
            device=args.device,
            logger=logger,
        )

        if args.interactive:
            run_interactive(model, tokenizer, args.device, args, logger)
        elif args.batch:
            run_batch(model, tokenizer, args.device, args, logger)
        else:
            run_single_demo(model, tokenizer, args.device, args, logger)

    except FileNotFoundError as e:
        logger.error(f"Checkpoint not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        sys.exit(1)
    finally:
        free_memory()


if __name__ == "__main__":
    main()

"""Data generation utilities for arithmetic reasoning problems."""

import random
from typing import Dict, List, Tuple

from datasets import Dataset

from config import DataConfig


def generate_reasoning_trace(a: int, b: int, c: int) -> str:
    """Generate a step-by-step reasoning trace for a + b + c.

    Args:
        a: First number
        b: Second number
        c: Third number

    Returns:
        Reasoning trace string
    """
    sum_ab = a + b
    answer = sum_ab + c

    return (
        f"I need to calculate the sum of {a}, {b}, and {c}.\n"
        f"1. Adding the first two: {a} + {b} = {sum_ab}\n"
        f"2. Adding the third: {sum_ab} + {c} = {answer}\n"
        f"3. Verification: {a} + {b} + {c} is indeed {answer}."
    )


def generate_sample(config: DataConfig) -> Dict[str, str]:
    """Generate a single training sample.

    Args:
        config: Data configuration

    Returns:
        Dictionary with prompt, answer, and formatted text
    """
    a = random.randint(config.min_val, config.max_val)
    b = random.randint(config.min_val, config.max_val)
    c = random.randint(config.min_val_c, config.max_val_c)
    answer = a + b + c

    prompt = f"Calculate {a} + {b} + {c}"
    thought = generate_reasoning_trace(a, b, c)

    # Format for SFT training
    text = (
        f"<|user|>\n{prompt}<|assistant|>\n"
        f"<think>\n{thought}\n</think>\n{answer}"
    )

    return {
        "prompt": prompt,
        "answer": str(answer),
        "text": text,
    }


def generate_data(config: DataConfig) -> List[Dict[str, str]]:
    """Generate training data.

    Args:
        config: Data configuration

    Returns:
        List of training samples
    """
    random.seed(config.seed)
    return [generate_sample(config) for _ in range(config.num_samples)]


def split_data(
    data: List[Dict[str, str]], config: DataConfig
) -> Tuple[Dataset, Dataset, Dataset]:
    """Split data into train/val/test sets.

    Args:
        data: List of samples
        config: Data configuration

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    n = len(data)
    train_end = int(n * config.train_ratio)
    val_end = train_end + int(n * config.val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return (
        Dataset.from_list(train_data),
        Dataset.from_list(val_data),
        Dataset.from_list(test_data),
    )


def prepare_sft_dataset(dataset: Dataset) -> Dataset:
    """Prepare dataset for SFT training.

    Args:
        dataset: Raw dataset with prompt, answer, text fields

    Returns:
        Dataset with only text field for SFT
    """
    return dataset.map(
        lambda x: {"text": x["text"]},
        remove_columns=["prompt", "answer"],
    )


def generate_test_problems(
    num_samples: int = 50,
    min_val: int = 10,
    max_val: int = 30,
    seed: int = 123,
) -> List[Dict[str, str]]:
    """Generate test problems with a different distribution.

    Args:
        num_samples: Number of test samples
        min_val: Minimum value for numbers
        max_val: Maximum value for numbers
        seed: Random seed

    Returns:
        List of test problems with 'prompt' and 'answer' keys
    """
    random.seed(seed)
    problems = []

    for _ in range(num_samples):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        c = random.randint(0, 10)
        answer = a + b + c

        problems.append({
            "prompt": f"Calculate {a} + {b} + {c}",
            "answer": str(answer),
        })

    return problems

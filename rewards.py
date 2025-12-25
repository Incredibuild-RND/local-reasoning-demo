"""Reward functions for GRPO training with semantic evaluation."""

import re
from typing import List, Optional


def extract_final_number(text: str) -> Optional[int]:
    """Extract the final number from a text string.

    Args:
        text: Text to extract number from

    Returns:
        The last number found, or None if no numbers
    """
    numbers = re.findall(r"\d+", text)
    if numbers:
        return int(numbers[-1])
    return None


def extract_answer_after_think(text: str) -> Optional[int]:
    """Extract the answer that appears after </think> tag.

    Args:
        text: Full model output

    Returns:
        The extracted answer, or None if not found
    """
    if "</think>" in text:
        after_think = text.split("</think>")[-1]
        return extract_final_number(after_think)
    return extract_final_number(text)


def check_format(text: str) -> bool:
    """Check if the output has proper <think>...</think> format.

    Args:
        text: Model output text

    Returns:
        True if format is correct
    """
    has_close = "</think>" in text
    # We prompt with <think>, so we mainly check for closing tag
    return has_close


def count_reasoning_steps(text: str) -> int:
    """Count the number of reasoning steps in the output.

    Args:
        text: Model output text

    Returns:
        Number of numbered steps (e.g., "1.", "2.", etc.)
    """
    steps = re.findall(r"\d+\.", text)
    return len(steps)


def has_verification(text: str) -> bool:
    """Check if the reasoning includes a verification step.

    Args:
        text: Model output text

    Returns:
        True if verification is present
    """
    verification_patterns = [
        r"verification",
        r"verify",
        r"check",
        r"indeed",
        r"confirms?",
    ]
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in verification_patterns)


def correctness_reward_func(
    completions: List[str],
    answer: List[str],
    **kwargs,
) -> List[float]:
    """Reward function for correctness of final answer.

    Args:
        completions: List of model completions
        answer: List of expected answers

    Returns:
        List of rewards (1.0 for correct, 0.0 for incorrect)
    """
    rewards = []
    for completion, expected in zip(completions, answer):
        predicted = extract_answer_after_think(completion)
        try:
            expected_int = int(expected)
            if predicted is not None and predicted == expected_int:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except (ValueError, TypeError):
            rewards.append(0.0)
    return rewards


def format_reward_func(
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Reward function for proper thinking format.

    Args:
        completions: List of model completions

    Returns:
        List of rewards (0.5 for correct format, 0.0 otherwise)
    """
    return [0.5 if check_format(c) else 0.0 for c in completions]


def reasoning_quality_reward_func(
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Reward function for reasoning quality.

    Considers:
    - Number of reasoning steps (more is better, up to a point)
    - Presence of verification

    Args:
        completions: List of model completions

    Returns:
        List of rewards (0.0 to 0.3)
    """
    rewards = []
    for completion in completions:
        reward = 0.0

        # Reward for reasoning steps (max 0.2 for 3+ steps)
        steps = count_reasoning_steps(completion)
        if steps >= 3:
            reward += 0.2
        elif steps >= 2:
            reward += 0.1
        elif steps >= 1:
            reward += 0.05

        # Reward for verification (0.1)
        if has_verification(completion):
            reward += 0.1

        rewards.append(reward)
    return rewards


def combined_reward_func(
    completions: List[str],
    answer: List[str],
    correctness_weight: float = 1.0,
    format_weight: float = 0.5,
    quality_weight: float = 0.3,
    **kwargs,
) -> List[float]:
    """Combined reward function with weighted components.

    Args:
        completions: List of model completions
        answer: List of expected answers
        correctness_weight: Weight for correctness reward
        format_weight: Weight for format reward
        quality_weight: Weight for quality reward

    Returns:
        List of combined rewards
    """
    correctness = correctness_reward_func(completions, answer)
    format_rewards = format_reward_func(completions)
    quality = reasoning_quality_reward_func(completions)

    combined = []
    for c, f, q in zip(correctness, format_rewards, quality):
        total = (
            c * correctness_weight +
            f * format_weight +
            q * quality_weight
        )
        combined.append(total)

    return combined

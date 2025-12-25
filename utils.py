"""Utility functions for logging, memory management, and device handling."""

import gc
import logging
import sys
from pathlib import Path
from typing import Optional

import torch


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> logging.Logger:
    """Set up logging with console and optional file output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        output_dir: Output directory for auto-generated log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("reasoning_training")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_file or output_dir:
        if log_file:
            file_path = Path(log_file)
        else:
            file_path = Path(output_dir) / "train.log"

        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {file_path}")

    return logger


def get_device() -> str:
    """Detect and return the best available device.

    Returns:
        Device string: 'mps', 'cuda', or 'cpu'
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def free_memory() -> None:
    """Free GPU/MPS memory by running garbage collection and clearing caches."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_memory_usage() -> dict:
    """Get current memory usage statistics.

    Returns:
        Dictionary with memory usage information
    """
    info = {"device": get_device()}

    if torch.cuda.is_available():
        info["allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        info["reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        info["max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    elif torch.backends.mps.is_available():
        # MPS doesn't have detailed memory stats
        info["allocated_gb"] = torch.mps.current_allocated_memory() / 1e9

    return info


def log_memory_usage(logger: logging.Logger) -> None:
    """Log current memory usage."""
    info = get_memory_usage()
    if "allocated_gb" in info:
        logger.debug(f"Memory usage: {info['allocated_gb']:.2f} GB allocated")


def sanitize_model_dtype(model: torch.nn.Module) -> None:
    """Ensure all trainable parameters are in float32 for MPS compatibility.

    Args:
        model: The model to sanitize
    """
    for name, param in model.named_parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)


def ensure_dir(path: str) -> Path:
    """Ensure directory exists and return Path object.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

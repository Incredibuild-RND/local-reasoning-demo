# Local Reasoning Demo

Train a small language model (Qwen2.5-0.5B) to perform step-by-step reasoning using SFT + GRPO.

## How It Works

### The Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Raw LLM    │───▶│  Phase 1:    │───▶│  Phase 2:    │───▶│  Reasoning│ │
│  │  (Qwen 0.5B) │    │    SFT       │    │    GRPO      │    │   Model   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                             │                   │                           │
│                             ▼                   ▼                           │
│                      Learns FORMAT        Learns CORRECTNESS                │
│                      "<think>...</think>"  via trial & reward               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Phase 1: SFT (Supervised Fine-Tuning)

**What happens:** We show the model examples of the reasoning format we want.

```
Input:  "Calculate 12 + 15 + 3"
Target: "<think>I need to calculate the sum of 12, 15, and 3.</think>
         1. Adding the first two: 12 + 15 = 27
         2. Adding the third: 27 + 3 = 30
         3. Verification: 12 + 15 + 3 is indeed 30.
         </think>
         30"
```

**What emerges:** The model learns to *structure* its output with `<think>` tags and step-by-step format. It doesn't reliably get correct answers yet - it's just mimicking the pattern.

### Phase 2: GRPO (Group Relative Policy Optimization)

**What happens:** The model generates multiple answers, and we reward the ones that:
1. Get the **correct answer** (highest weight)
2. Use proper **`<think>` format**
3. Show **reasoning steps** (numbered steps)
4. Include **verification** ("is indeed X")

```
┌─────────────────────────────────────────────────────────────┐
│  GRPO Reward Loop                                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Prompt: "Calculate 11 + 22 + 8"                          │
│                         │                                   │
│                         ▼                                   │
│            ┌───────────────────────┐                       │
│            │  Generate N responses │                       │
│            └───────────────────────┘                       │
│                         │                                   │
│         ┌───────────────┼───────────────┐                  │
│         ▼               ▼               ▼                  │
│    Response A      Response B      Response C              │
│    Answer: 41 ✓    Answer: 39 ✗    Answer: 41 ✓           │
│    Format: ✓       Format: ✓       Format: ✗              │
│    Steps: 3        Steps: 1        Steps: 0               │
│         │               │               │                  │
│         ▼               ▼               ▼                  │
│    Reward: 0.9     Reward: 0.2     Reward: 0.4            │
│         │               │               │                  │
│         └───────────────┼───────────────┘                  │
│                         ▼                                   │
│            ┌───────────────────────┐                       │
│            │ Update model weights  │                       │
│            │ (favor high rewards)  │                       │
│            └───────────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**What emerges:** Through trial and error, the model learns that:
- Correct math → high reward → do more of this
- Wrong math → low reward → do less of this
- The reasoning steps actually *help* it get correct answers

### The Key Insight

**SFT teaches the model HOW to express reasoning.**
**GRPO teaches the model WHAT reasoning leads to correct answers.**

Together, they produce a model that:
1. Shows its work in a structured format
2. Actually performs correct calculations
3. Verifies its own answers

### Results

| Training Data | Accuracy | What We Learned |
|---------------|----------|-----------------|
| 100 samples   | 10%      | Not enough examples |
| 500 samples   | 30%      | Getting better |
| 1000 samples  | **100%** | Model reliably reasons |

## Overview

This project demonstrates a two-phase training pipeline:

1. **SFT (Supervised Fine-Tuning)**: Teaches the model to use `<think>...</think>` format for reasoning
2. **GRPO (Group Relative Policy Optimization)**: Reinforces correct reasoning and answers via reward functions

The model learns to solve 3-number addition problems with explicit reasoning traces.

## Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

## Quick Start

### Full Training Pipeline

```bash
uv run python train.py --mode full --num-samples 500 --sft-epochs 2 --grpo-epochs 1
```

### Training with Custom Settings

```bash
uv run python train.py \
    --mode full \
    --num-samples 1000 \
    --sft-epochs 3 \
    --grpo-epochs 2 \
    --output-dir ./my_experiment \
    --wandb-project my-reasoning-project
```

### SFT Only

```bash
uv run python train.py --mode sft --num-samples 500 --sft-epochs 2
```

### GRPO Only (from existing SFT checkpoint)

```bash
uv run python train.py --mode grpo --sft-checkpoint ./output/sft_final --grpo-epochs 2
```

## CLI Options

```
python train.py --help

Training options:
  --mode {sft,grpo,full}    Training mode (default: full)
  --num-samples INT         Number of training samples (default: 500)
  --seed INT                Random seed (default: 42)

SFT options:
  --sft-epochs INT          Number of SFT epochs (default: 2)
  --sft-batch-size INT      Batch size (default: 4)
  --sft-lr FLOAT            Learning rate (default: 2e-4)

GRPO options:
  --grpo-epochs INT         Number of GRPO epochs (default: 1)
  --grpo-samples INT        Samples for GRPO (default: 50)
  --grpo-lr FLOAT           Learning rate (default: 1e-5)

Output:
  --output-dir PATH         Output directory (default: ./output)
  --wandb-project STR       W&B project name
  --no-wandb                Disable W&B logging

Logging:
  --log-level {DEBUG,INFO,WARNING,ERROR}
```

## Inference

### Interactive Mode

```bash
uv run python inference.py --checkpoint ./output/reasoning_final --interactive
```

### Batch Evaluation

```bash
uv run python inference.py --checkpoint ./output/reasoning_final --batch --num-samples 50
```

### Demo Mode

```bash
uv run python inference.py --checkpoint ./output/reasoning_final
```

## Project Structure

```
local_reasoning_demo/
├── config.py        # Configuration dataclasses
├── data.py          # Data generation utilities
├── rewards.py       # Reward functions for GRPO
├── evaluate.py      # Evaluation framework
├── train.py         # Main training script with CLI
├── inference.py     # Inference and testing
├── utils.py         # Logging, memory, device handling
└── pyproject.toml   # Dependencies
```

## Experiment Tracking

Training metrics are logged to Weights & Biases:

- Loss curves for SFT and GRPO phases
- Evaluation accuracy and format compliance
- Sample outputs in W&B Tables
- Hyperparameters and configuration

View your runs at: https://wandb.ai

## Example Output

```
Problem: Calculate 23 + 15 + 8

<think>
I need to calculate the sum of 23, 15, and 8.
1. Adding the first two: 23 + 15 = 38
2. Adding the third: 38 + 8 = 46
3. Verification: 23 + 15 + 8 is indeed 46.
</think>
46
```

## Requirements

- Python 3.14+
- Apple Silicon (MPS) or NVIDIA GPU (CUDA)
- ~2GB disk space for model checkpoints

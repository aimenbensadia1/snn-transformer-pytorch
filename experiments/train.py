#!/usr/bin/env python3
"""
Training script for the SNN Transformer.

Usage:
    python experiments/train.py --data data/english.txt --steps 10000

This script trains the LUT-based SNN Transformer on text data and logs
training progress.
"""

import argparse
import random
import time
import os

import torch

from snn_transformer.config import CONTEXT_SIZE, SEED
from snn_transformer.models.snn_gpu import FastModel, compute_lr
from snn_transformer.utils.data import load_data, create_english_like_data, get_random_batch


def train(
    data_file: str,
    num_steps: int = 10000,
    log_interval: int = 100,
    seed: int = SEED,
    device: str = 'auto'
):
    """
    Train the SNN Transformer.

    Args:
        data_file: Path to training data file
        num_steps: Number of training steps
        log_interval: Steps between logging
        seed: Random seed
        device: Device to use ('auto', 'cuda', or 'cpu')
    """
    # Setup
    random.seed(seed)
    torch.manual_seed(seed)

    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    print("=" * 60)
    print("SNN Transformer Training")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load or create data
    if not os.path.exists(data_file):
        print(f"\nCreating dataset: {data_file}")
        create_english_like_data(data_file, 500000)

    data = load_data(data_file)
    print(f"Data: {len(data):,} bytes")

    # Build model
    print("\nBuilding model...")
    model = FastModel(device)
    print(f"Parameters: {model.count_parameters():,}")

    # Training loop
    print(f"\nTraining for {num_steps:,} steps...")
    print("-" * 60)

    start_time = time.time()
    running_loss = 0
    log_count = 0

    for t in range(num_steps):
        tokens = get_random_batch(data, device)
        lr = compute_lr(t)

        loss = model.training_step(tokens, lr)
        running_loss += loss
        log_count += 1

        if (t + 1) % log_interval == 0:
            avg_loss = running_loss / log_count
            elapsed = time.time() - start_time
            speed = (t + 1) / elapsed

            print(f"Step {t+1:6d} | Loss: {avg_loss:.4f} | "
                  f"LR: {lr:.6f} | {speed:.1f} steps/s")

            running_loss = 0
            log_count = 0

    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training complete in {total_time:.1f}s ({num_steps/total_time:.1f} steps/s)")

    # Generate sample
    print("\nGenerating sample text...")
    prompt_text = "The quick brown fox jumps ov"
    prompt = torch.tensor([ord(c) for c in prompt_text], dtype=torch.long, device=device)

    if len(prompt) < CONTEXT_SIZE:
        padding = torch.tensor([ord(' ')] * (CONTEXT_SIZE - len(prompt)),
                              dtype=torch.long, device=device)
        prompt = torch.cat([padding, prompt])

    generated = model.generate(prompt, 50, temperature=0.5)
    gen_text = ''.join(chr(c) for c in generated.cpu().tolist() if 32 <= c < 127)
    print(f"Prompt: '{prompt_text}'")
    print(f"Generated: '{gen_text}'")


def main():
    parser = argparse.ArgumentParser(description="Train SNN Transformer")
    parser.add_argument("--data", type=str, default="data/english.txt",
                       help="Path to training data")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Number of training steps")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Steps between logging")
    parser.add_argument("--seed", type=int, default=SEED,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")

    args = parser.parse_args()

    train(
        data_file=args.data,
        num_steps=args.steps,
        log_interval=args.log_interval,
        seed=args.seed,
        device=args.device
    )


if __name__ == "__main__":
    main()

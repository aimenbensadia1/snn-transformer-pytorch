#!/usr/bin/env python3
"""
Comprehensive comparison between SNN Transformer and Standard Transformer.

This script trains both architectures on the same data and produces:
- Training curves comparison
- Speed benchmarks
- Memory usage
- Generated text samples
- Summary statistics

Usage:
    python experiments/compare.py --data data/english.txt --steps 15000
"""

import argparse
import json
import random
import time
import os
from dataclasses import dataclass, asdict
from typing import List

import torch
import torch.nn.functional as F
import numpy as np

from snn_transformer.config import CONTEXT_SIZE, SEED
from snn_transformer.models.snn_gpu import FastModel, compute_lr
from snn_transformer.models.transformer import StandardTransformer, create_transformer
from snn_transformer.utils.data import load_data, create_english_like_data, get_random_batch


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    model_name: str
    params: int
    train_losses: List[float]
    val_losses: List[float]
    steps: List[int]
    train_time: float
    steps_per_sec: float
    final_val_loss: float
    generated_samples: List[str]
    memory_mb: float


class SNNWrapper:
    """Wrapper for SNN model to provide common interface."""

    def __init__(self, device):
        self.model = FastModel(device)
        self.device = device

    def train_step(self, tokens, lr):
        return self.model.training_step(tokens, lr)

    def val_loss(self, tokens):
        with torch.no_grad():
            z = self.model.token_embedder[tokens[:CONTEXT_SIZE]]
            output, _ = self.model.forward(z)
            probs = F.softmax(output, dim=-1)
            targets = tokens[1:CONTEXT_SIZE + 1]
            loss = 0
            for pos in range(CONTEXT_SIZE):
                loss += -torch.log(probs[pos, targets[pos]] + 1e-10).item()
            return loss / CONTEXT_SIZE

    def generate(self, prompt_tokens, length, temperature=0.4):
        return self.model.generate(prompt_tokens, length, temperature)

    def count_parameters(self):
        return self.model.count_parameters()


class TransformerWrapper:
    """Wrapper for standard transformer to provide common interface."""

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, betas=(0.9, 0.95)
        )

    def train_step(self, tokens, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.model.train()
        self.optimizer.zero_grad()

        x = tokens[:CONTEXT_SIZE].unsqueeze(0)
        y = tokens[1:CONTEXT_SIZE + 1].unsqueeze(0)

        logits, loss = self.model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def val_loss(self, tokens):
        self.model.eval()
        with torch.no_grad():
            x = tokens[:CONTEXT_SIZE].unsqueeze(0)
            y = tokens[1:CONTEXT_SIZE + 1].unsqueeze(0)
            _, loss = self.model(x, y)
            return loss.item()

    def generate(self, prompt_tokens, length, temperature=0.4):
        self.model.eval()
        with torch.no_grad():
            idx = prompt_tokens.unsqueeze(0)
            output = self.model.generate(idx, length, temperature)
            return output[0, CONTEXT_SIZE:].cpu()

    def count_parameters(self):
        return self.model.count_parameters()


def run_experiment(
    model_wrapper,
    model_name: str,
    data: bytes,
    device: torch.device,
    num_steps: int = 10000,
    log_interval: int = 1000,
    val_samples: int = 100
) -> ExperimentResult:
    """Run training experiment on a model."""

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"Parameters: {model_wrapper.count_parameters():,}")
    print(f"{'='*60}")

    val_indices = [
        random.randint(0, len(data) - CONTEXT_SIZE - 2)
        for _ in range(val_samples)
    ]

    train_losses = []
    val_losses = []
    steps = []

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start_time = time.time()

    running_loss = 0
    log_count = 0

    for t in range(num_steps):
        tokens = get_random_batch(data, device)
        lr = compute_lr(t)

        loss = model_wrapper.train_step(tokens, lr)
        running_loss += loss
        log_count += 1

        if t % log_interval == 0 or t == num_steps - 1:
            # Validation
            val_loss = 0
            for val_idx in val_indices:
                val_tokens = torch.tensor(
                    [data[val_idx + i] for i in range(CONTEXT_SIZE + 1)],
                    dtype=torch.long, device=device
                )
                val_loss += model_wrapper.val_loss(val_tokens)
            val_loss /= len(val_indices)

            avg_train = running_loss / log_count

            train_losses.append(avg_train)
            val_losses.append(val_loss)
            steps.append(t)

            elapsed = time.time() - start_time
            speed = (t + 1) / elapsed if elapsed > 0 else 0

            print(f"Step {t:6d} | Train: {avg_train:.3f} | Val: {val_loss:.3f} | "
                  f"{speed:.1f} steps/s")

            running_loss = 0
            log_count = 0

    total_time = time.time() - start_time
    memory_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
                 if torch.cuda.is_available() else 0)

    # Generate samples
    prompt = torch.tensor(
        [ord(c) for c in "The quick brown fox jumps ov"],
        dtype=torch.long, device=device
    )
    if len(prompt) < CONTEXT_SIZE:
        padding = torch.tensor(
            [ord(' ')] * (CONTEXT_SIZE - len(prompt)),
            dtype=torch.long, device=device
        )
        prompt = torch.cat([padding, prompt])

    generated_samples = []
    for temp in [0.3, 0.5, 0.8]:
        gen = model_wrapper.generate(prompt, 50, temperature=temp)
        text = ''.join(chr(c) for c in gen.cpu().tolist() if 32 <= c < 127)
        generated_samples.append(f"T={temp}: {text}")

    return ExperimentResult(
        model_name=model_name,
        params=model_wrapper.count_parameters(),
        train_losses=train_losses,
        val_losses=val_losses,
        steps=steps,
        train_time=total_time,
        steps_per_sec=num_steps / total_time,
        final_val_loss=val_losses[-1],
        generated_samples=generated_samples,
        memory_mb=memory_mb
    )


def plot_comparison(results: List[ExperimentResult], filename: str):
    """Plot comparison of all experiments."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('SNN Transformer vs Standard Transformer Comparison', fontsize=14)

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))

    # Training loss
    ax = axes[0, 0]
    for i, r in enumerate(results):
        ax.plot(r.steps, r.train_losses, color=colors[i], label=r.model_name, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Validation loss
    ax = axes[0, 1]
    for i, r in enumerate(results):
        ax.plot(r.steps, r.val_losses, color=colors[i], label=r.model_name, linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Speed comparison
    ax = axes[0, 2]
    names = [r.model_name for r in results]
    speeds = [r.steps_per_sec for r in results]
    bars = ax.bar(names, speeds, color=colors[:len(results)])
    ax.set_ylabel('Steps/second')
    ax.set_title('Training Speed')
    ax.tick_params(axis='x', rotation=45)

    # Parameter count
    ax = axes[1, 0]
    params = [r.params / 1e6 for r in results]
    bars = ax.bar(names, params, color=colors[:len(results)])
    ax.set_ylabel('Parameters (M)')
    ax.set_title('Model Size')
    ax.set_yscale('log')
    ax.tick_params(axis='x', rotation=45)

    # Memory usage
    ax = axes[1, 1]
    memory = [r.memory_mb for r in results]
    bars = ax.bar(names, memory, color=colors[:len(results)])
    ax.set_ylabel('GPU Memory (MB)')
    ax.set_title('Memory Usage')
    ax.tick_params(axis='x', rotation=45)

    # Final loss comparison
    ax = axes[1, 2]
    final_losses = [r.final_val_loss for r in results]
    bars = ax.bar(names, final_losses, color=colors[:len(results)])
    ax.set_ylabel('Final Validation Loss')
    ax.set_title('Final Performance')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\nSaved comparison plot: {filename}")


def run_comparison(
    data_file: str = "data/english.txt",
    num_steps: int = 10000,
    log_interval: int = 1000,
    seed: int = SEED,
    output_dir: str = "outputs"
):
    """Run comprehensive comparison between models."""

    print("=" * 70)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("SNN Transformer vs Standard Transformer")
    print("=" * 70)

    # Setup
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load or create data
    if not os.path.exists(data_file):
        os.makedirs(os.path.dirname(data_file) or '.', exist_ok=True)
        create_english_like_data(data_file, 500000)

    data = load_data(data_file)
    print(f"Data: {len(data):,} bytes")

    results = []

    # 1. SNN Transformer
    print("\n" + "=" * 70)
    print("MODEL 1: SNN Transformer (LUT-based)")
    print("=" * 70)
    random.seed(seed)
    torch.manual_seed(seed)
    snn = SNNWrapper(device)
    result = run_experiment(snn, "SNN-LUT", data, device, num_steps, log_interval)
    results.append(result)
    del snn
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 2. Matched Transformer
    print("\n" + "=" * 70)
    print("MODEL 2: Standard Transformer (matched architecture)")
    print("=" * 70)
    random.seed(seed)
    torch.manual_seed(seed)
    transformer = create_transformer('matched')
    wrapper = TransformerWrapper(transformer, device)
    result = run_experiment(wrapper, "Transformer-Matched", data, device, num_steps, log_interval)
    results.append(result)
    del wrapper, transformer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. Medium Transformer
    print("\n" + "=" * 70)
    print("MODEL 3: Standard Transformer (medium)")
    print("=" * 70)
    random.seed(seed)
    torch.manual_seed(seed)
    transformer = create_transformer('medium')
    wrapper = TransformerWrapper(transformer, device)
    result = run_experiment(wrapper, "Transformer-Medium", data, device, num_steps, log_interval)
    results.append(result)
    del wrapper, transformer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Plot comparison
    plot_comparison(results, os.path.join(output_dir, "model_comparison.png"))

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<25} {'Params':<15} {'Speed':<15} {'Final Loss':<15} {'Memory':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r.model_name:<25} {r.params/1e6:>10.2f}M    {r.steps_per_sec:>10.1f}/s    "
              f"{r.final_val_loss:>10.3f}      {r.memory_mb:>6.0f}MB")

    print("\n" + "=" * 70)
    print("GENERATED SAMPLES")
    print("=" * 70)
    for r in results:
        print(f"\n{r.model_name}:")
        for sample in r.generated_samples:
            print(f"  {sample[:80]}")

    # Save results
    results_dict = [asdict(r) for r in results]
    results_file = os.path.join(output_dir, "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare SNN and Standard Transformers")
    parser.add_argument("--data", type=str, default="data/english.txt")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default="outputs")

    args = parser.parse_args()

    run_comparison(
        data_file=args.data,
        num_steps=args.steps,
        log_interval=args.log_interval,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

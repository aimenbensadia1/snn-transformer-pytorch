#!/usr/bin/env python3
"""
Deep Analysis of LUT Behavior in the SNN Transformer.

This script provides comprehensive analysis of how the Look-Up Table
mechanism works during training and inference:

1. Index Distribution Analysis
   - Which of the 64 possible indices are used?
   - How uniform is the distribution?
   - How does it vary across layers?

2. Uncertainty Analysis (u_min, r_min)
   - How often are decisions "close calls"?
   - Which comparisons are most uncertain?

3. Table Values Analysis
   - Distribution of learned table values
   - Sparsity patterns

4. Anchor Pattern Analysis
   - Which dimensions are compared most often?
   - Are there preferred comparison pairs?

5. Training Dynamics
   - How do these metrics change during training?

Usage:
    python analysis/analyze_lut.py --steps 3000

Outputs saved to analysis/figures/
"""

import argparse
import os
import random
import sys
import time
from collections import defaultdict

import torch
import numpy as np

from snn_transformer.config import CONTEXT_SIZE, NUM_LAYERS, N_C, N_T, EMBEDDING_DIM, SEED
from snn_transformer.models.snn_gpu import FastModel, compute_lr
from snn_transformer.utils.data import load_data, create_toy_dataset


def analyze_index_distribution(model, data, num_samples=1000, device='cuda', output_dir='analysis/figures'):
    """Analyze which LUT indices are being used."""
    print("\n" + "=" * 60)
    print("1. INDEX DISTRIBUTION ANALYSIS")
    print("=" * 60)

    ffn_indices = {layer: [] for layer in range(NUM_LAYERS)}
    unembedder_indices = []

    print(f"Running {num_samples} forward passes...")

    for i in range(num_samples):
        idx = random.randint(0, len(data) - CONTEXT_SIZE - 2)
        tokens = torch.tensor(
            [data[idx + j] for j in range(CONTEXT_SIZE + 1)],
            dtype=torch.long, device=device
        )

        z = model.token_embedder[tokens[:CONTEXT_SIZE]]

        for layer in range(NUM_LAYERS):
            j, r_min, u_min = model.ffn[layer].compute_indices(z)
            ffn_indices[layer].extend(j.cpu().numpy().flatten().tolist())
            z = z + model.ffn[layer].forward(j)

        j, _, _ = model.unembedder.compute_indices(z)
        unembedder_indices.extend(j.cpu().numpy().flatten().tolist())

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{num_samples}")

    max_index = 2 ** N_C

    # Compute statistics
    print("\nIndex Distribution Statistics:")
    print("-" * 40)
    for layer in range(NUM_LAYERS):
        indices = ffn_indices[layer]
        hist, _ = np.histogram(indices, bins=max_index, range=(0, max_index))
        unique_used = np.sum(hist > 0)
        prob = hist / hist.sum() + 1e-10
        entropy = -np.sum(prob * np.log2(prob))
        print(f"  Layer {layer}: {unique_used}/{max_index} indices used, entropy={entropy:.2f} bits")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('LUT Index Distribution Analysis', fontsize=14)

        for plot_idx, layer in enumerate([0, 2, 5]):
            ax = axes[0, plot_idx]
            indices = ffn_indices[layer]
            hist, bins = np.histogram(indices, bins=max_index, range=(0, max_index))
            ax.bar(range(max_index), hist, width=1.0, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Index')
            ax.set_ylabel('Frequency')
            ax.set_title(f'FFN Layer {layer}')

            unique_used = np.sum(hist > 0)
            prob = hist / hist.sum() + 1e-10
            entropy = -np.sum(prob * np.log2(prob))
            ax.text(0.95, 0.95, f'Used: {unique_used}/{max_index}\nEntropy: {entropy:.2f} bits',
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Unembedder
        ax = axes[1, 0]
        hist, _ = np.histogram(unembedder_indices, bins=max_index, range=(0, max_index))
        ax.bar(range(max_index), hist, width=1.0, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Index')
        ax.set_ylabel('Frequency')
        ax.set_title('Unembedder')

        # Entropy across layers
        ax = axes[1, 1]
        entropies = []
        for layer in range(NUM_LAYERS):
            hist, _ = np.histogram(ffn_indices[layer], bins=max_index, range=(0, max_index))
            prob = hist / hist.sum() + 1e-10
            entropy = -np.sum(prob * np.log2(prob))
            entropies.append(entropy)
        ax.bar(range(NUM_LAYERS), entropies, color='orange', edgecolor='black')
        ax.axhline(y=np.log2(max_index), color='red', linestyle='--', label=f'Max ({np.log2(max_index):.1f} bits)')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Entropy (bits)')
        ax.set_title('Index Entropy per Layer')
        ax.legend()

        # Usage heatmap
        ax = axes[1, 2]
        usage_matrix = np.zeros((NUM_LAYERS, max_index))
        for layer in range(NUM_LAYERS):
            hist, _ = np.histogram(ffn_indices[layer], bins=max_index, range=(0, max_index))
            usage_matrix[layer] = hist / (hist.max() + 1e-10)
        im = ax.imshow(usage_matrix, aspect='auto', cmap='hot')
        ax.set_xlabel('Index')
        ax.set_ylabel('Layer')
        ax.set_title('Index Usage Heatmap')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'lut_indices.png'), dpi=150)
        plt.close()
        print(f"\nSaved: {output_dir}/lut_indices.png")

    except ImportError:
        print("matplotlib not available, skipping plots")

    return ffn_indices, unembedder_indices


def analyze_uncertainty(model, data, num_samples=1000, device='cuda', output_dir='analysis/figures'):
    """Analyze u_min values - the uncertainty at decision boundaries."""
    print("\n" + "=" * 60)
    print("2. UNCERTAINTY ANALYSIS (u_min, r_min)")
    print("=" * 60)

    u_min_values = {layer: [] for layer in range(NUM_LAYERS)}
    r_min_counts = {layer: defaultdict(int) for layer in range(NUM_LAYERS)}

    print(f"Collecting uncertainty data from {num_samples} samples...")

    for i in range(num_samples):
        idx = random.randint(0, len(data) - CONTEXT_SIZE - 2)
        tokens = torch.tensor(
            [data[idx + j] for j in range(CONTEXT_SIZE + 1)],
            dtype=torch.long, device=device
        )

        z = model.token_embedder[tokens[:CONTEXT_SIZE]]

        for layer in range(NUM_LAYERS):
            j, r_min, u_min = model.ffn[layer].compute_indices(z)

            u_min_np = u_min.cpu().numpy().flatten()
            r_min_np = r_min.cpu().numpy().flatten()

            u_min_values[layer].extend(u_min_np.tolist())
            for r in r_min_np:
                r_min_counts[layer][int(r)] += 1

            z = z + model.ffn[layer].forward(j)

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{num_samples}")

    # Statistics
    print("\nUncertainty Statistics:")
    print("-" * 40)
    for layer in range(NUM_LAYERS):
        values = np.abs(u_min_values[layer])
        near_zero = np.mean(values < 0.1) * 100
        print(f"  Layer {layer}: mean|u_min|={np.mean(values):.4f}, "
              f"near zero (<0.1): {near_zero:.1f}%")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Uncertainty Analysis (Decision Boundaries)', fontsize=14)

        for plot_idx, layer in enumerate([0, 2, 5]):
            ax = axes[0, plot_idx]
            values = u_min_values[layer]
            ax.hist(values, bins=100, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('u_min value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Layer {layer}: u_min Distribution')

        # |u_min| distribution
        ax = axes[1, 0]
        all_abs_u = []
        for layer in range(NUM_LAYERS):
            all_abs_u.extend(np.abs(u_min_values[layer]))
        ax.hist(all_abs_u, bins=100, edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('|u_min|')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall Uncertainty Magnitude')

        # r_min frequency
        ax = axes[1, 1]
        layer_colors = plt.cm.viridis(np.linspace(0.2, 0.8, NUM_LAYERS))
        x = np.arange(N_C)
        width = 0.12
        for layer in range(NUM_LAYERS):
            counts = [r_min_counts[layer][r] for r in range(N_C)]
            total = sum(counts)
            probs = [c / total for c in counts]
            ax.bar(x + layer * width, probs, width, label=f'Layer {layer}', color=layer_colors[layer])
        ax.set_xlabel('r_min')
        ax.set_ylabel('Probability')
        ax.set_title('Most Uncertain Bit by Layer')
        ax.legend(loc='upper right', fontsize=8)

        # Mean |u_min| per layer
        ax = axes[1, 2]
        mean_abs_u = [np.mean(np.abs(u_min_values[layer])) for layer in range(NUM_LAYERS)]
        ax.bar(range(NUM_LAYERS), mean_abs_u, color='steelblue', edgecolor='black')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Mean |u_min|')
        ax.set_title('Average Uncertainty per Layer')

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'lut_uncertainty.png'), dpi=150)
        plt.close()
        print(f"\nSaved: {output_dir}/lut_uncertainty.png")

    except ImportError:
        print("matplotlib not available, skipping plots")

    return u_min_values, r_min_counts


def analyze_table_values(model, output_dir='analysis/figures'):
    """Analyze the learned table values."""
    print("\n" + "=" * 60)
    print("3. TABLE VALUES ANALYSIS")
    print("=" * 60)

    print("\nTable Statistics:")
    print("-" * 40)
    for layer in range(NUM_LAYERS):
        values = model.ffn[layer].S.cpu().numpy().flatten()
        print(f"  Layer {layer}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
              f"sparsity={np.mean(np.abs(values) < 0.01)*100:.1f}%")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('LUT Table Values Analysis', fontsize=14)

        for plot_idx, layer in enumerate([0, 2, 5]):
            ax = axes[0, plot_idx]
            all_values = model.ffn[layer].S.cpu().numpy().flatten()
            ax.hist(all_values, bins=100, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Table Value')
            ax.set_ylabel('Frequency')
            ax.set_title(f'FFN Layer {layer}')

        # Magnitude per layer
        ax = axes[1, 0]
        magnitudes = [torch.norm(model.ffn[layer].S).item() for layer in range(NUM_LAYERS)]
        ax.bar(range(NUM_LAYERS), magnitudes, color='purple', edgecolor='black')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Frobenius Norm')
        ax.set_title('Table Magnitude per Layer')

        # Sparsity
        ax = axes[1, 1]
        sparsity = []
        for layer in range(NUM_LAYERS):
            values = model.ffn[layer].S.cpu().numpy().flatten()
            sparsity.append(np.mean(np.abs(values) < 0.01) * 100)
        ax.bar(range(NUM_LAYERS), sparsity, color='green', edgecolor='black')
        ax.set_xlabel('Layer')
        ax.set_ylabel('% values < 0.01')
        ax.set_title('Table Sparsity per Layer')

        # Heatmap of one table
        ax = axes[1, 2]
        table_0 = model.ffn[0].S[0].cpu().numpy()
        im = ax.imshow(table_0[:, :32].T, aspect='auto', cmap='RdBu', vmin=-0.1, vmax=0.1)
        ax.set_xlabel('Table Index')
        ax.set_ylabel('Output Dimension')
        ax.set_title('Layer 0, Table 0 Values')
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'lut_tables.png'), dpi=150)
        plt.close()
        print(f"\nSaved: {output_dir}/lut_tables.png")

    except ImportError:
        print("matplotlib not available, skipping plots")


def analyze_training_dynamics(data, device='cuda', num_steps=2000, log_every=100, output_dir='analysis/figures'):
    """Track how LUT behavior changes during training."""
    print("\n" + "=" * 60)
    print("4. TRAINING DYNAMICS ANALYSIS")
    print("=" * 60)

    model = FastModel(device)

    entropy_history = []
    magnitude_history = []
    uncertainty_history = []
    loss_history = []

    print(f"Training for {num_steps} steps, logging every {log_every}...")

    for t in range(num_steps):
        idx = random.randint(0, len(data) - CONTEXT_SIZE - 2)
        tokens = torch.tensor(
            [data[idx + j] for j in range(CONTEXT_SIZE + 1)],
            dtype=torch.long, device=device
        )
        lr = compute_lr(t)
        loss = model.training_step(tokens, lr)

        if t % log_every == 0:
            z = model.token_embedder[tokens[:CONTEXT_SIZE]]
            j, r_min, u_min = model.ffn[0].compute_indices(z)

            # Index entropy
            indices = j.cpu().numpy().flatten()
            hist, _ = np.histogram(indices, bins=64, range=(0, 64))
            prob = hist / hist.sum() + 1e-10
            entropy = -np.sum(prob * np.log2(prob))

            # Table magnitude
            mag = torch.norm(model.ffn[0].S).item()

            # Average uncertainty
            avg_uncertainty = torch.abs(u_min).mean().item()

            entropy_history.append((t, entropy))
            magnitude_history.append((t, mag))
            uncertainty_history.append((t, avg_uncertainty))
            loss_history.append((t, loss))

            print(f"  Step {t}: loss={loss:.3f}, entropy={entropy:.2f}, "
                  f"mag={mag:.3f}, |u_min|={avg_uncertainty:.4f}")

    # Plot
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Dynamics', fontsize=14)

        steps = [x[0] for x in loss_history]

        ax = axes[0, 0]
        ax.plot(steps, [x[1] for x in loss_history], 'b-', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(steps, [x[1] for x in entropy_history], 'g-', linewidth=2)
        ax.axhline(y=6, color='red', linestyle='--', label='Max (6 bits)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy (bits)')
        ax.set_title('Index Entropy (Layer 0)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(steps, [x[1] for x in magnitude_history], 'purple', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Frobenius Norm')
        ax.set_title('Table Magnitude (Layer 0)')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(steps, [x[1] for x in uncertainty_history], 'orange', linewidth=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean |u_min|')
        ax.set_title('Average Uncertainty (Layer 0)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'lut_dynamics.png'), dpi=150)
        plt.close()
        print(f"\nSaved: {output_dir}/lut_dynamics.png")

    except ImportError:
        print("matplotlib not available, skipping plots")

    return model


def main():
    parser = argparse.ArgumentParser(description="Analyze LUT behavior in SNN Transformer")
    parser.add_argument("--steps", type=int, default=3000, help="Training steps for dynamics analysis")
    parser.add_argument("--samples", type=int, default=500, help="Samples for distribution analysis")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default="analysis/figures")
    parser.add_argument("--data", type=str, default="data/toy.txt")

    args = parser.parse_args()

    print("=" * 60)
    print("LUT BEHAVIOR ANALYSIS")
    print("=" * 60)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load or create data
    if not os.path.exists(args.data):
        os.makedirs(os.path.dirname(args.data) or '.', exist_ok=True)
        create_toy_dataset(args.data)

    data = load_data(args.data)
    print(f"Loaded {len(data):,} bytes of data")

    # Train and analyze
    model = analyze_training_dynamics(data, device, args.steps, output_dir=args.output_dir)
    analyze_index_distribution(model, data, args.samples, device, args.output_dir)
    analyze_uncertainty(model, data, args.samples, device, args.output_dir)
    analyze_table_values(model, args.output_dir)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated figures in {args.output_dir}/")


if __name__ == "__main__":
    main()

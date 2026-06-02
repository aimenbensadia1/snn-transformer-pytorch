<div align="center">

# SNN Transformer

**A PyTorch implementation of the Look-Up Table Spiking Neural Network Transformer**

*Faithful port of Eugene Izhikevich's ["The Spiking Manifesto"](https://izhikevich.org) (2025)*

[![Python](https://img.shields.io/badge/python-3.9%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-research-orange)](.)

</div>

---

Replaces every matrix multiplication with **binary comparisons** and **look-up table (LUT)** reads — the computational model of a spiking neuron. Intended for neuromorphic hardware, ultra-low-power edge devices, and integer-only inference pipelines where floating-point matrix ops are unavailable.

```
Input x  →  Compare pairs (x[a] > x[b]?)  →  6-bit binary index j  →  S[table, j]  →  Sum  →  Output
```

---

## Table of Contents

- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [File Tree](#file-tree)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Python API](#python-api)
- [How It Works](#how-it-works)
- [Testing](#testing)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      SNN TRANSFORMER                         │
│                                                              │
│   Token IDs ──▶  Embedding Table  [256 × 32]                │
│                          │                                   │
│           ╔══════════════╩══════════════╗                    │
│           ║    Transformer Layer × 6    ║                    │
│           ║                            ║                    │
│           ║  ┌──────────────────────┐  ║                    │
│           ║  │   Multi-Head Attn    │  ║  4 heads / layer   │
│           ║  │   (LUT-based)        │  ║  Q|K|PE → 16 bits  │
│           ║  │   65,536 entries/tbl │  ║                    │
│           ║  └──────────┬───────────┘  ║                    │
│           ║  ┌──────────▼───────────┐  ║                    │
│           ║  │       FFN            │  ║  6-bit index       │
│           ║  │   (LUT-based)        │  ║  64 entries/table  │
│           ║  └──────────────────────┘  ║                    │
│           ╚══════════════╦══════════════╝                    │
│                          │                                   │
│          Unembed LUT  ──▶  Logits [256]                     │
└──────────────────────────────────────────────────────────────┘
```

| Hyperparameter | Value | Notes |
|---|---|---|
| Context size | 32 | tokens |
| Vocabulary | 256 | byte-level ASCII |
| Embedding dim | 32 | hidden dimension |
| Layers | 6 | stacked blocks |
| Attention heads | 4 | per layer |
| Tables per LUT (`N_T`) | 16 | |
| Bits per index (`N_C`) | 6 | → 64 entries per FFN table |
| Attention bits | 16 | Q(6) \| K(6) \| PE(4) → 65,536 entries |
| **Total parameters** | **805M** | 99.9% in attention LUTs |
| Effective learning capacity | ~84K | equivalent dense parameters |

---

## Benchmarks

Trained on 500K bytes of synthetic English-like text for 15,000 steps:

| Model | Parameters | Final Val Loss | Speed (steps/s) |
|---|---|---|---|
| **SNN-LUT** | 805.82M | 2.073 | 41.6 |
| Transformer (matched, 84K) | 84K | **1.253** | 79.1 |
| Transformer (medium, 9.5M) | 9.5M | 2.539 | 43.6 |

> The SNN's 805M parameters do not imply 805M degrees of freedom. Effective learning capacity is ~84K — the number of unique index patterns that actually get gradient updates. Full analysis in [`reports/comparison_report.md`](reports/comparison_report.md).

---

## File Tree

```
snn-transformer-pytorch/
├── snn_transformer/               # Core installable package
│   ├── __init__.py
│   ├── config.py                  # All hyperparameters (matches original C)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── snn_gpu.py             # FastLUT + FastModel (GPU-optimized)
│   │   └── transformer.py         # Standard transformer baseline
│   └── utils/
│       ├── __init__.py
│       └── data.py                # Data loading and synthetic generation
├── experiments/
│   ├── train.py                   # Train the SNN Transformer
│   └── compare.py                 # Head-to-head SNN vs Transformer benchmark
├── analysis/
│   ├── analyze_lut.py             # Visualize LUT internals
│   └── figures/                   # Pre-generated analysis plots (committed)
│       ├── lut_analysis_anchors.png
│       ├── lut_analysis_dynamics.png
│       ├── lut_analysis_indices.png
│       ├── lut_analysis_tables.png
│       ├── lut_analysis_uncertainty.png
│       └── model_comparison.png
├── reports/
│   ├── lut_analysis_report.md     # LUT behavior findings
│   └── comparison_report.md       # Model comparison findings
├── tests/
│   ├── __init__.py
│   └── test_model.py              # pytest unit tests
├── data/                          # Training data — gitignored
├── outputs/                       # Experiment outputs — gitignored
├── .gitignore
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.9+ | |
| PyTorch | 2.0+ | [Installation guide](https://pytorch.org/get-started/locally/) |
| NumPy | 1.21+ | |
| Matplotlib | 3.5+ | Optional — only for analysis plots |
| CUDA GPU | any | Strongly recommended; tested on AMD MI210 |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/aimenbensadia1/snn-transformer-pytorch.git
cd snn-transformer-pytorch

# 2. Install as an editable package (recommended)
pip install -e ".[all]"

# --- OR --- install only runtime dependencies
pip install -r requirements.txt
```

**Extras available:**

| Extra | Installs |
|---|---|
| `pip install -e .` | torch, numpy only |
| `pip install -e ".[viz]"` | + matplotlib |
| `pip install -e ".[dev]"` | + pytest, pytest-cov |
| `pip install -e ".[all]"` | everything above |

---

## Quick Start

### Train the SNN Transformer

```bash
# Synthetic data is auto-generated on first run if the file doesn't exist
python experiments/train.py --data data/english.txt --steps 10000

# With explicit settings
python experiments/train.py \
    --data data/english.txt \
    --steps 50000           \
    --log-interval 500      \
    --device cuda
```

Expected output:
```
============================================================
SNN Transformer Training
============================================================
Device: cuda  |  GPU: AMD Instinct MI210
Data: 500,000 bytes  |  Parameters: 805,830,656

Step    100 | Loss: 5.4312 | LR: 0.000025 | 41.6 steps/s
Step    200 | Loss: 4.7891 | LR: 0.000050 | 41.9 steps/s
...
```

### Compare SNN vs Standard Transformer

```bash
python experiments/compare.py \
    --data data/english.txt \
    --steps 15000           \
    --output-dir outputs/
# Saves: outputs/comparison_results.json + outputs/model_comparison.png
```

### Analyze LUT Behavior

```bash
python analysis/analyze_lut.py --steps 3000
# Saves figures to analysis/figures/
```

---

## Python API

```python
import torch
from snn_transformer.models.snn_gpu import FastModel, compute_lr
from snn_transformer.config import CONTEXT_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FastModel(device)
print(f"Parameters: {model.count_parameters():,}")  # 805,830,656

# --- Training step ---
tokens = torch.randint(0, 256, (CONTEXT_SIZE + 1,), device=device)
loss = model.training_step(tokens, lr=compute_lr(step=1000))
print(f"Loss: {loss:.4f}")

# --- Text generation ---
prompt_str = "The quick brown fox"
prompt = torch.tensor(
    [ord(c) for c in prompt_str.ljust(CONTEXT_SIZE)[:CONTEXT_SIZE]],
    dtype=torch.long, device=device
)
generated = model.generate(prompt, length=64, temperature=0.5)
text = "".join(chr(t) for t in generated.cpu().tolist() if 32 <= t < 127)
print(f"Generated: {text}")
```

---

## How It Works

### Binary Indexing (the "Spiking" Aspect)

For each LUT layer with `N_C = 6` comparisons and `N_T = 16` tables:

```
1. For each table t, comparison r:
      u[r] = x[ a[r] ] - x[ b[r] ]      # difference of two input dimensions
      bit[r] = 1 if u[r] > 0 else 0      # "spike" if positive

2. Form binary index:
      j = Σ_r  bit[r] × 2^r              # 6-bit integer in [0, 63]

3. Lookup and sum:
      y = Σ_t  S[t, j]                   # sum N_T table entries
```

Anchors `a[r]` and `b[r]` are fixed random index pairs, initialized once and never updated during training (only the table values `S` are learned).

### Surrogate Gradient

The Heaviside step function has zero gradient almost everywhere. The backward pass routes gradient through `r_min` — the comparison with the smallest `|u|` (maximum uncertainty):

```
Up(u) = −0.5 × sign(u) / (1 + |u|)²
```

This is maximally sensitive near `u = 0` and decays for confident comparisons — a biologically plausible approximation to spike-timing-dependent plasticity.

### Attention Mechanism

Attention uses concatenated indices across query, key, and position:

```
j_attn = (j_Q << 10) | (j_K << 4) | j_PE       # 6 | 6 | 4 = 16 bits
                                                  # → 65,536 entries per table
```

The quadratic growth from 64 (FFN) to 65,536 (attention) entries per table is why 99.9% of parameters live in attention LUTs.

### LUT Analysis Findings

1. **Full utilization**: All 64 FFN indices are used with entropy ~5.8 bits (theoretical max: 6 bits)
2. **Active gradient signal**: 15–25% of comparisons have `|u_min| < 0.1`, providing dense gradient flow
3. **Table sparsity**: 60–80% of table values stay near zero — substantial compression potential for hardware deployment

Full findings: [`reports/lut_analysis_report.md`](reports/lut_analysis_report.md)

---

## Testing

```bash
# Run all tests
pytest

# With coverage report
pytest --cov=snn_transformer --cov-report=html

# Single test class
pytest tests/test_model.py::TestFastLUT -v
```

Test coverage includes:
- Configuration correctness (exact match to original C implementation)
- LUT index range validity and binary comparison rules
- Forward pass output shapes
- Surrogate gradient formula properties
- Training convergence (loss decreases)
- Generation output validity

---

## Use Cases

| Target | Suitable |
|---|---|
| Ultra-low-power IoT / microcontrollers | Target use case |
| Neuromorphic hardware (Intel Loihi, IBM TrueNorth) | Integer-only inference |
| Always-on sensing (wake word, gesture) | Continuous low-energy operation |
| Research into binary / spike-based learning | Full gradient support |
| Competing with dense transformers on GPU | Out of scope by design |

---

## Contributing

Contributions are welcome. Please:

1. Fork the repo and create a feature branch: `git checkout -b feature/your-feature`
2. Write tests for new functionality in `tests/`
3. Verify tests pass: `pytest`
4. Follow PEP 8; keep existing code style
5. Open a pull request with a clear description

For significant changes, open an issue first to discuss the design.

---

## Citation

If you use this implementation in your research, please cite the original work:

```bibtex
@article{izhikevich2025spiking,
  title  = {The Spiking Manifesto},
  author = {Izhikevich, Eugene},
  year   = {2025}
}
```

---

<div align="center">
<sub>Built with PyTorch &nbsp;·&nbsp; Targeting neuromorphic hardware &nbsp;·&nbsp; MIT licensed</sub>
</div>

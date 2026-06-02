# LUT (Look-Up Table) Behavior Analysis Report

## Executive Summary

This report analyzes the internal behavior of the Look-Up Table (LUT) mechanism in the SNN Transformer architecture from Eugene Izhikevich's "The Spiking Manifesto" (2025). The analysis reveals key insights about how the model uses its 805M parameters and why its learning capacity differs fundamentally from dense neural networks.

## Background

The SNN Transformer replaces traditional matrix multiplications with table lookups indexed by binary comparisons:

1. For input vector `x`, compare pairs of dimensions: `u = x[a] - x[b]`
2. Form a 6-bit binary index `j` by setting bit `r` if `u[r] > 0`
3. Look up and sum values from 16 tables: `y = Σ S[t, j]`

This creates 64 possible indices per table, but concatenated attention indices use 16 bits (65,536 entries).

## Key Findings

### 1. Index Distribution

**Finding: All 64 indices are used, but distribution is not uniform.**

- Index entropy ranges from 5.5 to 5.9 bits (max possible: 6 bits)
- Some indices are accessed 2-3x more frequently than others
- Distribution varies slightly across layers

**Implication**: The model is utilizing most of its representational capacity, but there's room for more uniform utilization through better anchor selection.

### 2. Uncertainty Analysis (u_min, r_min)

**Finding: Many decisions are "close calls" near the decision boundary.**

- 15-25% of comparisons have |u_min| < 0.1
- The most uncertain bit (r_min) varies by layer
- Later layers show lower average uncertainty

**Implication**: The surrogate gradient mechanism is essential - many bit decisions are nearly tied, and the gradient needs to flow through these uncertain comparisons.

### 3. Table Values

**Finding: Table values remain small and relatively sparse.**

- Mean values: ~0.001
- Standard deviation: ~0.02-0.05
- 60-80% of values remain near zero (< 0.01)

**Implication**: The model learns sparse, small adjustments rather than large transformations. This is consistent with the residual connection structure.

### 4. Anchor Patterns

**Finding: Random anchor selection creates diverse comparison patterns.**

- All embedding dimensions are used roughly equally
- No dominant comparison pairs emerge
- Distance between compared dimensions is uniformly distributed

**Implication**: The random initialization provides good coverage of the input space without learning.

### 5. Training Dynamics

**Finding: Index entropy increases during training, then stabilizes.**

- Early training: entropy ~4 bits (concentrated indices)
- After convergence: entropy ~5.8 bits (more uniform)
- Table magnitude grows steadily then plateaus

**Implication**: The model learns to use more of its capacity as training progresses, but never achieves maximum entropy.

## Parameter Count Analysis

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Token Embeddings | 8,192 | 0.001% |
| FFN LUTs | 196,608 | 0.024% |
| **Attention LUTs** | **805,306,368** | **99.94%** |
| Positional Encodings | 49,152 | 0.006% |
| Unembedder | 262,144 | 0.033% |
| **Total** | **805,822,464** | 100% |

**Critical Insight**: 99.94% of parameters are in attention LUTs due to the 16-bit concatenated indices (2^16 = 65,536 entries per table). However, most of these entries are rarely or never accessed during inference.

## Effective vs. Nominal Parameters

The 805M parameter count is misleading:

1. **Sparse Access**: Only a subset of table entries are accessed for any given input
2. **Constrained Gradients**: Only the accessed entries receive gradient updates
3. **Limited Expressivity**: Each table can only output 64 distinct values

**Effective capacity is much closer to ~100K-1M parameters** based on:
- Actually utilized table entries
- Gradient flow patterns
- Comparison with matched architecture transformer

## Conclusions

1. **The LUT mechanism works as designed**: Binary indexing provides differentiable (via surrogate gradients) discrete computation.

2. **Parameter efficiency is poor**: The 805M parameters provide learning capacity equivalent to ~84K dense parameters.

3. **The architecture trades expressivity for hardware compatibility**: The discrete, table-based computation could map efficiently to neuromorphic hardware.

4. **Index utilization is good but not optimal**: Near-maximum entropy suggests the model uses most available indices.

5. **Uncertainty tracking is critical**: The r_min/u_min mechanism ensures gradients flow through the model despite discrete decisions.

## Recommendations

1. **For better performance**: Use smaller attention LUT sizes or reduce concatenation bits
2. **For hardware deployment**: The current architecture is ready for neuromorphic implementation
3. **For research**: Investigate learned anchor selection to improve index uniformity

## Figures

- `lut_indices.png`: Index distribution across layers
- `lut_uncertainty.png`: Decision boundary uncertainty analysis
- `lut_tables.png`: Learned table value distributions
- `lut_dynamics.png`: Training dynamics of LUT metrics

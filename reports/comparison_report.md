# Model Comparison Report: SNN Transformer vs Standard Transformer

## Executive Summary

This report presents a comprehensive comparison between the LUT-based SNN Transformer (805M parameters) and traditional dense Transformers of various sizes. The key finding is that **a tiny 84K parameter traditional transformer significantly outperforms the 805M parameter SNN model**, revealing fundamental differences in parameter efficiency between these architectures.

## Experimental Setup

### Models Compared

| Model | Architecture | Parameters |
|-------|-------------|------------|
| SNN-LUT | 6 layers, 4 heads, 32 dim, LUT-based | 805.82M |
| Transformer-Matched | 6 layers, 4 heads, 32 dim, dense | 83,776 (84K) |
| Transformer-Medium | 12 layers, 8 heads, 256 dim | 9.52M |
| Transformer-Large | 24 layers, 16 heads, 512 dim | 75.70M |

### Training Configuration

- **Dataset**: 500KB synthetic English-like text
- **Training Steps**: 15,000
- **Batch Size**: 1 (online learning)
- **Context Length**: 32 tokens
- **Learning Rate**: Warmup + inverse sqrt decay
- **Hardware**: AMD MI210 GPU

## Results

### Final Performance

| Model | Final Val Loss | Training Speed | Memory |
|-------|---------------|----------------|--------|
| SNN-LUT | 2.073 | 41.6 steps/s | 3,079 MB |
| **Transformer-Matched** | **1.253** | **79.1 steps/s** | **78 MB** |
| Transformer-Medium | 2.539 | 43.6 steps/s | 222 MB |
| Transformer-Large | 2.566 | 19.6 steps/s | 1,444 MB |

### Key Observations

1. **The tiny matched transformer wins decisively**
   - Best loss: 1.253 (vs 2.073 for SNN)
   - Fastest training: 79 steps/s (vs 42 for SNN)
   - Lowest memory: 78 MB (vs 3,079 MB for SNN)

2. **The SNN model comes second**
   - Better than both medium and large transformers
   - Reasonable loss of 2.073

3. **Larger transformers failed**
   - Both medium (9.5M) and large (75.7M) got worse losses (~2.5)
   - Severe overfitting to the small dataset

### Generated Text Quality

**Transformer-Matched (best):**
```
T=0.3: er. The this these these this these. This will thi
T=0.5: er. This these train these this my thin. The be my
```
*Produces recognizable English words and patterns*

**SNN-LUT (second best):**
```
T=0.3: e Thin hin ar an The thin be an an an the t thin t
T=0.5: e s ime Thir The the ak t d This ay On ald me wo t
```
*Produces word fragments and partial patterns*

**Transformer-Medium/Large (worst):**
```
T=0.3: e m n n os t t won t t we tos n t t n to ten rn te
```
*Produces mostly gibberish*

## Analysis

### Why Does the Tiny Transformer Win?

1. **Dataset Size Mismatch**
   - 500KB of text is insufficient for large models
   - The matched transformer's 84K parameters are appropriate for this data scale
   - Larger models overfit catastrophically

2. **Dense vs. Sparse Parameters**
   - The SNN's 805M parameters are in sparse LUT entries
   - Most entries are rarely accessed
   - Effective capacity is much lower than nominal count

3. **Architecture Parity**
   - The matched transformer has identical architecture (6 layers, 4 heads, 32 dim)
   - Dense computation can express everything the LUT can, and more
   - No architectural advantage for SNN on this task

### Why Did the SNN Beat Larger Transformers?

1. **Implicit Regularization**
   - LUT structure constrains what the model can learn
   - Binary indexing acts as a form of discretization
   - Prevents the severe overfitting seen in larger dense models

2. **Simpler Optimization Landscape**
   - Fewer effective degrees of freedom
   - Surrogate gradient may help avoid local minima

## The True Trade-off

The SNN architecture is **not** designed to compete with dense transformers on accuracy. Instead, it offers:

| Aspect | SNN-LUT | Dense Transformer |
|--------|---------|-------------------|
| **Accuracy** | Lower | Higher |
| **GPU Training** | Slower | Faster |
| **Memory** | Higher | Lower |
| **Neuromorphic HW** | Native | Requires conversion |
| **Power (inference)** | Potentially much lower | Standard |
| **Fixed-point friendly** | Yes | Requires quantization |

### Where SNN Architecture Makes Sense

1. **Edge deployment** on ultra-low power devices
2. **Neuromorphic chips** (Intel Loihi, IBM TrueNorth)
3. **Always-on sensing** (wake word, gesture detection)
4. **Embedded systems** without floating-point units

### Where Dense Transformers Are Better

1. **GPU/TPU training and inference**
2. **Maximum accuracy requirements**
3. **Large-scale language modeling**
4. **Standard cloud deployment**

## Conclusions

1. **Parameter count is misleading**: 805M LUT parameters ≈ 84K dense parameters in learning capacity.

2. **Architecture choice depends on deployment target**: SNN for neuromorphic hardware, dense for GPUs.

3. **Dataset size matters critically**: Match model capacity to data availability.

4. **The comparison is apples to oranges**: These architectures optimize for different objectives.

## Recommendations

1. **For accuracy-critical tasks**: Use dense transformers
2. **For edge deployment**: Consider SNN if targeting neuromorphic hardware
3. **For fair comparison**: Use much larger datasets or compare on neuromorphic hardware
4. **For research**: Investigate hybrid approaches that combine LUT efficiency with dense expressivity

## Figures

- `model_comparison.png`: Complete comparison visualization
  - Training/validation loss curves
  - Speed benchmarks
  - Parameter counts
  - Memory usage
  - Final performance comparison

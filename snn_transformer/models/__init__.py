"""
Model implementations for SNN Transformer.

Available models:
- SNNTransformerCPU: Faithful CPU implementation matching C code exactly
- SNNTransformerGPU: Vectorized GPU implementation for fast training
- StandardTransformer: Traditional transformer baseline for comparison
"""

from .snn_gpu import FastModel as SNNTransformerGPU
from .snn_gpu import FastLUT, compute_lr
from .transformer import StandardTransformer, TransformerConfig, create_transformer

__all__ = [
    'SNNTransformerGPU',
    'FastLUT',
    'compute_lr',
    'StandardTransformer',
    'TransformerConfig',
    'create_transformer',
]

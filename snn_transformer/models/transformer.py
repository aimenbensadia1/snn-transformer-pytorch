"""
Standard Transformer Baseline

A traditional transformer implementation for comparison with the SNN Transformer.
This serves as a baseline to evaluate the trade-offs of the LUT-based approach.

Architectures available:
- matched: Same architecture as SNN (6 layers, 4 heads, 32 dim) = 84K params
- medium: Scaled up (12 layers, 8 heads, 256 dim) = 9.5M params
- large: Further scaled (24 layers, 16 heads, 512 dim) = 75.7M params
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from snn_transformer.config import VOCAB_SIZE, CONTEXT_SIZE, NUM_LAYERS, NUM_HEADS, EMBEDDING_DIM


@dataclass
class TransformerConfig:
    """Configuration for standard transformer."""
    vocab_size: int = VOCAB_SIZE
    context_size: int = CONTEXT_SIZE
    n_layers: int = NUM_LAYERS
    n_heads: int = NUM_HEADS
    d_model: int = EMBEDDING_DIM
    d_ff: int = EMBEDDING_DIM * 4
    dropout: float = 0.0


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention with causal masking."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_heads

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

        self.dropout = nn.Dropout(cfg.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_size, cfg.context_size))
            .view(1, 1, cfg.context_size, cfg.context_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out_proj(out)


class FeedForward(nn.Module):
    """Standard feed-forward network with GELU activation."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class StandardTransformer(nn.Module):
    """
    Standard transformer language model.

    Uses:
    - Learned token and position embeddings
    - Pre-norm transformer blocks
    - Weight tying between embedding and output projection
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            idx: Input token indices [batch, seq_len]
            targets: Target token indices for loss computation

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            loss: Cross-entropy loss if targets provided
        """
        B, T = idx.shape

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = self.dropout(tok_emb + pos_emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.4
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.context_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer(size: str = 'matched') -> StandardTransformer:
    """
    Create transformer of specified size.

    Args:
        size: One of 'matched', 'medium', 'large'
            - matched: Same architecture as SNN (84K params)
            - medium: 12 layers, 256 dim (9.5M params)
            - large: 24 layers, 512 dim (75.7M params)

    Returns:
        Configured StandardTransformer instance
    """
    configs = {
        'matched': TransformerConfig(),  # 6 layers, 4 heads, 32 dim
        'medium': TransformerConfig(n_layers=12, n_heads=8, d_model=256, d_ff=1024),
        'large': TransformerConfig(n_layers=24, n_heads=16, d_model=512, d_ff=2048),
    }

    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")

    return StandardTransformer(configs[size])


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Standard Transformer Configurations")
    print("=" * 50)

    for size in ['matched', 'medium', 'large']:
        model = create_transformer(size).to(device)
        params = model.count_parameters()
        print(f"{size:10s}: {params:>12,} params ({params/1e6:.2f}M)")

        # Quick forward pass test
        x = torch.randint(0, 256, (2, 32), device=device)
        logits, loss = model(x, x)
        print(f"           Output shape: {logits.shape}")

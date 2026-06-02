"""
Configuration parameters for the SNN Transformer.

These hyperparameters exactly match Eugene Izhikevich's original C implementation
from "The Spiking Manifesto" (2025).
"""

# =============================================================================
# Model Architecture
# =============================================================================

CONTEXT_SIZE = 32       # Sequence length / context window
VOCAB_SIZE = 256        # Byte-level vocabulary (ASCII)
EMBEDDING_DIM = 32      # Hidden dimension
POSITIONAL_DIM = 4      # Bits for positional encoding in attention
NUM_LAYERS = 6          # Number of transformer layers
NUM_HEADS = 4           # Number of attention heads per layer

# =============================================================================
# LUT (Look-Up Table) Parameters
# =============================================================================

N_T = 16                # Number of tables per LUT operation
N_C = 6                 # Number of binary comparisons per table (2^6 = 64 entries)

# Derived constants
TABLE_SIZE = 1 << N_C   # 64 entries per table
ATTN_BITS = N_C + N_C + POSITIONAL_DIM  # 16 bits for attention (Q|K|PE)
ATTN_TABLE_SIZE = 1 << ATTN_BITS        # 65,536 entries for attention tables

# =============================================================================
# Training Parameters
# =============================================================================

WARMUP_STEPS = 4000     # Linear warmup steps for learning rate
SEED = 42               # Random seed for reproducibility

# =============================================================================
# Computed Parameter Counts
# =============================================================================

def compute_parameter_count():
    """Compute total parameter count for the model."""
    # Token embeddings: VOCAB_SIZE * EMBEDDING_DIM
    token_emb = VOCAB_SIZE * EMBEDDING_DIM

    # FFN LUTs: NUM_LAYERS * N_T * TABLE_SIZE * EMBEDDING_DIM
    ffn_params = NUM_LAYERS * N_T * TABLE_SIZE * EMBEDDING_DIM

    # Attention LUTs: NUM_LAYERS * NUM_HEADS * N_T * ATTN_TABLE_SIZE * EMBEDDING_DIM
    attn_params = NUM_LAYERS * NUM_HEADS * N_T * ATTN_TABLE_SIZE * EMBEDDING_DIM

    # Positional encodings: NUM_LAYERS * NUM_HEADS * CONTEXT_SIZE * N_T * POSITIONAL_DIM
    pe_params = NUM_LAYERS * NUM_HEADS * CONTEXT_SIZE * N_T * POSITIONAL_DIM

    # Unembedder: N_T * TABLE_SIZE * VOCAB_SIZE
    unembed_params = N_T * TABLE_SIZE * VOCAB_SIZE

    total = token_emb + ffn_params + attn_params + pe_params + unembed_params

    return {
        'token_embeddings': token_emb,
        'ffn_luts': ffn_params,
        'attention_luts': attn_params,
        'positional_encodings': pe_params,
        'unembedder': unembed_params,
        'total': total
    }


if __name__ == "__main__":
    print("SNN Transformer Configuration")
    print("=" * 50)
    print(f"Context Size:     {CONTEXT_SIZE}")
    print(f"Vocab Size:       {VOCAB_SIZE}")
    print(f"Embedding Dim:    {EMBEDDING_DIM}")
    print(f"Num Layers:       {NUM_LAYERS}")
    print(f"Num Heads:        {NUM_HEADS}")
    print(f"N_T (tables):     {N_T}")
    print(f"N_C (bits):       {N_C}")
    print()

    params = compute_parameter_count()
    print("Parameter Count:")
    print("-" * 50)
    for name, count in params.items():
        print(f"  {name:25s}: {count:>15,}")
    print(f"\n  Total: {params['total']/1e6:.2f}M parameters")

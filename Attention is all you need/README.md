# Attention Is All You Need

Reproduction of the Transformer architecture from [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., NeurIPS 2017).

## Key Concepts

- **Scaled Dot-Product Attention**: `softmax(QK^T / sqrt(d_k)) V`
- **Multi-Head Attention**: Parallel attention in h=8 subspaces, then concatenate
- **Positional Encoding**: Sinusoidal encoding injecting sequence order
- **Encoder-Decoder**: N=6 layers each, with residual connections + LayerNorm
- **Weight Sharing**: Input/output embeddings and pre-softmax projection share weights
- **Noam LR Schedule**: Linear warmup (4000 steps) + inverse sqrt decay

## Architecture

```
Input Embedding + Positional Encoding
            |
    ┌───────┴───────┐
    │   ENCODER x6   │
    │  Self-Attn     │
    │  + FFN         │
    └───────┬───────┘
            |
    ┌───────┴───────┐
    │   DECODER x6   │
    │  Masked Self   │
    │  + Cross-Attn  │
    │  + FFN         │
    └───────┬───────┘
            |
    Linear + Softmax
```

## Setup

```bash
cd "Attention is all you need"
pip install -e .
```

## Quick Start

```python
from transformer_mt import build_transformer_base, ExperimentConfig, set_seed

set_seed(42)
config = ExperimentConfig.default()
model = build_transformer_base(vocab_size=37000)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Configuration

| Config | d_model | d_ff | N | h | dropout | steps |
|--------|---------|------|---|---|---------|-------|
| Base   | 512     | 2048 | 6 | 8 | 0.1     | 100K  |
| Big    | 1024    | 4096 | 6 | 16| 0.3     | 300K  |

## Expected Results

| Model | EN-DE BLEU | EN-FR BLEU | Parameters |
|-------|-----------|-----------|------------|
| Base  | 27.3      | 38.1      | ~65M       |
| Big   | 28.4      | 41.8      | ~213M      |

## Paper

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. *NeurIPS 2017*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

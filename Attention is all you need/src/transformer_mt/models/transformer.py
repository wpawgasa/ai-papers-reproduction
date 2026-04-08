"""
The Transformer model (Section 3, Figure 1).
"Attention Is All You Need" -- Vaswani et al., NeurIPS 2017.

Architecture:
- Encoder: N=6 layers of [MultiHeadSelfAttention -> FFN]
- Decoder: N=6 layers of [MaskedSelfAttention -> CrossAttention -> FFN]
- Shared embedding weights (Section 3.4)
"""

import copy
import math

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionalEncoding, PositionWiseFeedForward, SublayerConnection


class EncoderLayer(nn.Module):
    """Single encoder layer (Section 3.1): self-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, src_mask))
        x = self.sublayer2(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    """Single decoder layer (Section 3.1): masked self-attention + cross-attention + FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer2(
            x, lambda x: self.cross_attn(x, encoder_output, encoder_output, src_mask)
        )
        x = self.sublayer3(x, self.ffn)
        return x


class Encoder(nn.Module):
    """Stack of N=6 encoder layers."""

    def __init__(self, layer: EncoderLayer, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N=6 decoder layers."""

    def __init__(self, layer: DecoderLayer, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Full Transformer model (Figure 1).

    Section 3.4: "we share the same weight matrix between the two
    embedding layers and the pre-softmax linear transformation...
    In the embedding layers, we multiply those weights by sqrt(d_model)."
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
        share_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, n_layers)

        decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, n_layers)

        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        if share_embeddings:
            self.output_projection.weight = self.embedding.weight

        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        src_emb = self.positional_encoding(
            self.embedding(src) * math.sqrt(self.d_model)
        )
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_emb = self.positional_encoding(
            self.embedding(tgt) * math.sqrt(self.d_model)
        )
        return self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits: (batch, tgt_len, vocab_size)."""
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        return self.output_projection(decoder_output)

    @staticmethod
    def make_causal_mask(size: int) -> torch.Tensor:
        """Create causal (look-ahead) mask for decoder self-attention."""
        return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def make_padding_mask(seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        """Create padding mask: 1 for real tokens, 0 for padding."""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def build_transformer_base(vocab_size: int) -> Transformer:
    """Build the base Transformer model with paper hyperparameters."""
    return Transformer(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
    )


def build_transformer_big(vocab_size: int) -> Transformer:
    """Build the big Transformer model with paper hyperparameters."""
    return Transformer(
        vocab_size=vocab_size,
        d_model=1024,
        n_layers=6,
        n_heads=16,
        d_ff=4096,
        dropout=0.3,
    )

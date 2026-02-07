"""
FNet: Mixing Tokens with Fourier Transforms
PyTorch reproduction — Lee-Thorp et al. (NAACL 2022)

Architecture Reference (Section 3.2, Figure 1):
    Each encoder block: Fourier Sublayer → Add & Norm → FFN → Add & Norm

Key Equations:
    Eq. 1: X_k = sum_{n=0}^{N-1} x_n * exp(-2*pi*i*n*k / N)
    Eq. 2: W_{nk} = exp(-2*pi*i*n*k / N) / sqrt(N)
    Eq. 3: y = Re(F_seq(F_h(x)))
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


# ============================================================================
# Fourier Mixing Sublayer (replaces Self-Attention)
# ============================================================================

class FourierTransformLayer(nn.Module):
    """
    Fourier mixing sublayer — zero learnable parameters.

    Implements Equation 3 from the paper:
        y = Re(F_seq(F_h(x)))

    Applies 2D DFT to the (seq_len, d_model) embedding input:
        1. F_h:   1D DFT along hidden dimension (dim=-1)
        2. F_seq: 1D DFT along sequence dimension (dim=-2)
        3. Re:    Extract real part of the complex output

    Complexity:
        FFT:    O(n * d_h * log(n) + n * d_h * log(d_h))
        Matrix: O(n^2 * d_h + n * d_h^2)

    The two 1D DFTs commute (footnote 3 in paper), so order is immaterial.
    torch.fft.fft2 applies FFT along last two dimensions simultaneously.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model) real-valued tensor
        Returns:
            (batch_size, seq_len, d_model) real-valued tensor
        """
        # fft2 computes 2D FFT along dimensions (-2, -1) = (seq_len, d_model)
        return torch.fft.fft2(x).real


class DFTMatrixLayer(nn.Module):
    """
    Alternative DFT implementation via matrix multiplication.
    Used on TPUs for sequences <= 4096 (Section 3.3).

    Computes DFT as: X = W @ x, where W is the DFT matrix (Eq. 2):
        W_{nk} = exp(-2*pi*i*n*k / N) / sqrt(N)

    Complexity: O(n^2 * d_h + n * d_h^2) — faster on TPUs for short sequences.
    """

    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        # Pre-compute DFT matrices (not learnable)
        W_seq = self._build_dft_matrix(seq_len)    # (seq_len, seq_len)
        W_hid = self._build_dft_matrix(d_model)    # (d_model, d_model)
        self.register_buffer('W_seq', W_seq)
        self.register_buffer('W_hid', W_hid)

    @staticmethod
    def _build_dft_matrix(n: int) -> torch.Tensor:
        """Build the DFT Vandermonde matrix W (Equation 2)."""
        indices = torch.arange(n, dtype=torch.float32)
        # W_{nk} = exp(-2*pi*i*n*k / N) / sqrt(N)
        exponent = -2.0 * math.pi * torch.outer(indices, indices) / n
        W = torch.complex(torch.cos(exponent), torch.sin(exponent)) / math.sqrt(n)
        return W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) real tensor
        Returns:
            (batch, seq_len, d_model) real tensor
        """
        # Cast to complex, apply DFT matrices, extract real part
        x_complex = x.to(torch.complex64)
        # DFT along hidden dim: x @ W_hid^T
        x_complex = torch.matmul(x_complex, self.W_hid.T)
        # DFT along sequence dim: W_seq @ x
        x_complex = torch.matmul(self.W_seq, x_complex)
        return x_complex.real


# ============================================================================
# Feed-Forward Network (shared by all models)
# ============================================================================

class FeedForwardLayer(nn.Module):
    """
    Position-wise feed-forward network (identical to Transformer/BERT).

    FFN(x) = GELU(x @ W_1 + b_1) @ W_2 + b_2

    Paper uses GELU activation (following BERT), not ReLU.
    Initialization: Normal(0, 0.02) for weights and biases.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)      # (d_model → d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)      # (d_ff → d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Paper Appendix A.7: Normal initialization with std=0.02."""
        nn.init.normal_(self.linear1.weight, std=0.02)
        nn.init.normal_(self.linear1.bias, std=0.02)
        nn.init.normal_(self.linear2.weight, std=0.02)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


# ============================================================================
# FNet Encoder Block
# ============================================================================

class FNetEncoderBlock(nn.Module):
    """
    Single FNet encoder block (Figure 1).

    Architecture:
        h = LayerNorm(x + Re(FFT2D(x)))         # Fourier mixing + residual
        output = LayerNorm(h + FFN(h))           # Feed-forward + residual
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        self.fourier = FourierTransformLayer()
        self.ff = FeedForwardLayer(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fourier sublayer with residual connection
        fourier_output = self.fourier(x)
        x = self.norm1(x + fourier_output)
        # Feed-forward sublayer with residual connection
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x


# ============================================================================
# Embeddings (identical to BERT)
# ============================================================================

class FNetEmbeddings(nn.Module):
    """
    Input embeddings: e = e_word + e_position + e_type

    Position embeddings are technically redundant for FNet (the DFT's
    twiddle factors encode position), but included for fair BERT comparison
    (Section 3.2).
    """

    def __init__(self, vocab_size: int = 32000, d_model: int = 768,
                 max_seq_len: int = 512, type_vocab_size: int = 2,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "position_ids", torch.arange(max_seq_len).unsqueeze(0)
        )

    def forward(self, input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(self.position_ids[:, :seq_len])
            + self.token_type_embeddings(token_type_ids)
        )
        return self.dropout(self.layer_norm(embeddings))


# ============================================================================
# Full FNet Model
# ============================================================================

class FNetModel(nn.Module):
    """
    FNet encoder (Section 3, Table 1).

    Base config:  d_model=768,  num_layers=12, d_ff=3072, params=83M
    Large config: d_model=1024, num_layers=24, d_ff=4096, params=238M
    """

    def __init__(self, vocab_size: int = 32000, d_model: int = 768,
                 num_layers: int = 12, d_ff: int = 3072,
                 max_seq_len: int = 512, type_vocab_size: int = 2,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-12):
        super().__init__()
        self.d_model = d_model
        self.embeddings = FNetEmbeddings(
            vocab_size, d_model, max_seq_len, type_vocab_size,
            dropout, layer_norm_eps
        )
        self.encoder_blocks = nn.ModuleList([
            FNetEncoderBlock(d_model, d_ff, dropout, layer_norm_eps)
            for _ in range(num_layers)
        ])
        # Pooler: Dense + tanh on [CLS] token (same as BERT)
        self.pooler = nn.Linear(d_model, d_model)
        nn.init.normal_(self.pooler.weight, std=0.02)
        nn.init.zeros_(self.pooler.bias)

    def forward(self, input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        hidden_states = self.embeddings(input_ids, token_type_ids)
        for block in self.encoder_blocks:
            hidden_states = block(hidden_states)
        pooled_output = torch.tanh(self.pooler(hidden_states[:, 0]))
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled_output,
        }


# ============================================================================
# Task Heads
# ============================================================================

class FNetForSequenceClassification(nn.Module):
    """FNet + classification head for GLUE tasks."""

    def __init__(self, num_labels: int, **model_kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.fnet = FNetModel(**model_kwargs)
        d_model = model_kwargs.get("d_model", 768)
        self.dropout = nn.Dropout(model_kwargs.get("dropout", 0.1))
        self.classifier = nn.Linear(d_model, num_labels)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict:
        outputs = self.fnet(input_ids, token_type_ids)
        logits = self.classifier(self.dropout(outputs["pooler_output"]))
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss = nn.MSELoss()(logits.squeeze(-1), labels.float())
            else:
                loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


class FNetForMaskedLM(nn.Module):
    """FNet + MLM head for pre-training (Section 4.1, Table 5)."""

    def __init__(self, **model_kwargs):
        super().__init__()
        vocab_size = model_kwargs.get("vocab_size", 32000)
        d_model = model_kwargs.get("d_model", 768)
        self.fnet = FNetModel(**model_kwargs)
        self.mlm_dense = nn.Linear(d_model, d_model)
        self.mlm_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.mlm_decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))
        # Weight tying (standard practice)
        self.mlm_decoder.weight = self.fnet.embeddings.word_embeddings.weight

    def forward(self, input_ids, token_type_ids=None, labels=None):
        outputs = self.fnet(input_ids, token_type_ids)
        h = F.gelu(self.mlm_dense(outputs["last_hidden_state"]))
        logits = self.mlm_decoder(self.mlm_norm(h)) + self.mlm_bias
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        return {"loss": loss, "logits": logits}


# ============================================================================
# Configuration Presets (Table 1 & Table 6)
# ============================================================================

FNET_CONFIGS = {
    "base": dict(vocab_size=32000, d_model=768, num_layers=12, d_ff=3072,
                 max_seq_len=512, dropout=0.1),
    "large": dict(vocab_size=32000, d_model=1024, num_layers=24, d_ff=4096,
                  max_seq_len=512, dropout=0.1),
    "small-512x8": dict(vocab_size=32000, d_model=512, num_layers=8, d_ff=2048,
                        max_seq_len=512, dropout=0.1),
    "small-256x4": dict(vocab_size=32000, d_model=256, num_layers=4, d_ff=1024,
                        max_seq_len=512, dropout=0.1),
    "tiny-128x2": dict(vocab_size=32000, d_model=128, num_layers=2, d_ff=512,
                       max_seq_len=512, dropout=0.1),
}


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("FNet Reproduction — Architecture Verification")
    print("=" * 70)

    batch_size, seq_len = 4, 128

    for name, cfg in FNET_CONFIGS.items():
        model = FNetModel(**cfg)
        n_params = sum(p.numel() for p in model.parameters())
        x = torch.randint(0, cfg["vocab_size"], (batch_size, seq_len))
        with torch.no_grad():
            out = model(x)
        assert out["last_hidden_state"].shape == (batch_size, seq_len, cfg["d_model"])
        assert out["pooler_output"].shape == (batch_size, cfg["d_model"])
        print(f"  [{name:>12s}]  params={n_params/1e6:6.1f}M  "
              f"output={tuple(out['last_hidden_state'].shape)}  ✓")

    # Test classification head
    cls_model = FNetForSequenceClassification(num_labels=2, **FNET_CONFIGS["base"])
    labels = torch.randint(0, 2, (batch_size,))
    x = torch.randint(0, 32000, (batch_size, seq_len))
    with torch.no_grad():
        out = cls_model(x, labels=labels)
    assert out["logits"].shape == (batch_size, 2)
    print(f"\n  Classification head: loss={out['loss'].item():.4f}  ✓")

    # Test MLM head
    mlm_model = FNetForMaskedLM(**FNET_CONFIGS["base"])
    mlm_labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    mlm_labels[:, 1:5] = torch.randint(0, 32000, (batch_size, 4))
    with torch.no_grad():
        out = mlm_model(x, labels=mlm_labels)
    print(f"  MLM head: loss={out['loss'].item():.4f}  ✓")

    # Verify parameter count matches paper (Table 1: FNet-Base = 83M)
    base_params = sum(p.numel() for p in FNetModel(**FNET_CONFIGS["base"]).parameters())
    print(f"\n  FNet-Base parameters: {base_params/1e6:.1f}M (paper reports ~83M)")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)

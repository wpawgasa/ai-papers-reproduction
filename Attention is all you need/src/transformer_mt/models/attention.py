"""
Attention mechanisms from "Attention Is All You Need" (Vaswani et al., 2017).

Implements:
- Equation (1): Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
- Equation (2): MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (Section 3.2.1, Figure 2 left).

    Computes: softmax(QK^T / sqrt(d_k)) V

    The scaling factor 1/sqrt(d_k) prevents dot products from growing
    large in magnitude for large d_k, which would push softmax into
    regions with extremely small gradients.
    """

    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(d_k)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,   # (batch, h, seq_q, d_k)
        key: torch.Tensor,     # (batch, h, seq_k, d_k)
        value: torch.Tensor,   # (batch, h, seq_k, d_v)
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # QK^T / sqrt(d_k)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale

        # Apply mask (causal or padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, value)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Section 3.2.2, Figure 2 right).

    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)

    With h=8 heads and d_k = d_v = d_model/h = 64, total cost is
    similar to single-head attention with full dimensionality.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Combined projection matrices for all heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)

    def forward(
        self,
        query: torch.Tensor,   # (batch, seq_q, d_model)
        key: torch.Tensor,     # (batch, seq_k, d_model)
        value: torch.Tensor,   # (batch, seq_k, d_model)
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = query.size(0)

        # Project and reshape to (batch, h, seq, d_k)
        Q = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, _ = self.attention(Q, K, V, mask)

        # Concat heads: (batch, h, seq_q, d_k) -> (batch, seq_q, d_model)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )

        return self.W_O(attn_output)

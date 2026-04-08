"""
Supporting layers for the Transformer.

- Position-wise FFN: Equation (3) -- FFN(x) = max(0, xW1+b1)W2+b2
- Positional Encoding: Equation (4) -- sinusoidal PE
- Sublayer connection: LayerNorm(x + Sublayer(x))
"""

import math

import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (Section 3.3).

    FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
    d_model = 512, d_ff = 2048 (4x expansion).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Section 3.5).

    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    Wavelengths form geometric progression from 2pi to 10000*2pi.
    For any fixed offset k, PE_{pos+k} is a linear function of PE_{pos}.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SublayerConnection(nn.Module):
    """
    Residual connection + Layer Normalization (Section 3.1).

    Output = LayerNorm(x + Sublayer(x))

    Uses post-norm as described in the paper.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer_fn) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer_fn(x)))

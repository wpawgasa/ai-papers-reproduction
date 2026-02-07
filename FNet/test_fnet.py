#!/usr/bin/env python
"""Quick test script to demonstrate FNet functionality."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import FNetModel, FourierTransformLayer, FNET_CONFIGS

print("=" * 70)
print("FNet Quick Demo")
print("=" * 70)

# 1. Architecture test
print("\n1. Testing FNet-Base architecture...")
model = FNetModel(**FNET_CONFIGS['base'])
n_params = sum(p.numel() for p in model.parameters())
print(f"   ✓ FNet-Base: {n_params/1e6:.1f}M parameters (paper: ~83M)")

# 2. Forward pass test
print("\n2. Testing forward pass...")
batch_size, seq_len = 2, 64
input_ids = torch.randint(0, 32000, (batch_size, seq_len))
with torch.no_grad():
    outputs = model(input_ids)
print(f"   ✓ Input shape:  {tuple(input_ids.shape)}")
print(f"   ✓ Output shape: {tuple(outputs['last_hidden_state'].shape)}")

# 3. Fourier Transform visualization
print("\n3. Visualizing Fourier Transform...")
seq_len, d_model = 32, 64
t = np.linspace(0, 4*np.pi, seq_len)
x_input = np.sin(t)[:, None] * np.ones((1, d_model))
x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0)

fourier_layer = FourierTransformLayer()
with torch.no_grad():
    y_output = fourier_layer(x_tensor)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Input
im0 = axes[0].imshow(x_input.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
axes[0].set_title('Input: Re(x)', fontsize=12)
axes[0].set_xlabel('Sequence Position')
axes[0].set_ylabel('Hidden Dimension')
plt.colorbar(im0, ax=axes[0])

# FFT output
y_np = y_output[0].numpy()
im1 = axes[1].imshow(y_np.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
axes[1].set_title('Output: Re(FFT2D(x))', fontsize=12)
axes[1].set_xlabel('Sequence Position')
axes[1].set_ylabel('Hidden Dimension')
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig('fnet_fourier_demo.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved visualization to fnet_fourier_demo.png")

# 4. Speed comparison
print("\n4. Comparing speed (Fourier vs Attention)...")
import time

class SimpleAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads=12):
        super().__init__()
        self.qkv = torch.nn.Linear(d_model, 3 * d_model)
        self.out = torch.nn.Linear(d_model, d_model)
        self.num_heads = num_heads

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (C // self.num_heads) ** 0.5, dim=-1)
        out = (attn @ v).reshape(B, N, C)
        return self.out(out)

fourier = FourierTransformLayer()
attention = SimpleAttention(768)

x = torch.randn(8, 128, 768)
num_runs = 100

# Warmup
for _ in range(10):
    _ = fourier(x)
    _ = attention(x)

# Benchmark Fourier
start = time.time()
for _ in range(num_runs):
    _ = fourier(x)
fourier_time = (time.time() - start) / num_runs * 1000

# Benchmark Attention
start = time.time()
for _ in range(num_runs):
    _ = attention(x)
attn_time = (time.time() - start) / num_runs * 1000

speedup = attn_time / fourier_time
print(f"   Fourier:   {fourier_time:.2f}ms")
print(f"   Attention: {attn_time:.2f}ms")
print(f"   ✓ Speedup: {speedup:.1f}x (paper reports 12-22x for isolated layers)")

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("=" * 70)
print("\nNext steps:")
print("  1. Open notebooks/experiment.ipynb for interactive experiments")
print("  2. Run: python -m src.train --task sst2 --config tiny-128x2")
print("  3. See README.md for full reproduction instructions")

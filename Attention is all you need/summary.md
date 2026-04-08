# Attention Is All You Need

**Paper**: Attention Is All You Need **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin **Venue**: NeurIPS 2017 (31st Conference on Neural Information Processing Systems) **arXiv**: 1706.03762v7 (Last revised: August 2, 2023) **Affiliation**: Google Brain, Google Research, University of Toronto

---

## 1. Summary

**Problem**: The dominant sequence-to-sequence models in 2017 relied on recurrent neural networks (RNNs) or convolutional neural networks (CNNs) arranged in encoder-decoder configurations. These architectures suffered from inherently sequential computation (preventing parallelization during training) and difficulty in learning long-range dependencies due to the length of the signal paths between distant positions.

**Approach**: The authors propose the **Transformer**, a novel architecture that replaces recurrence and convolution entirely with **self-attention mechanisms**. The model uses stacked multi-head self-attention and position-wise feed-forward layers for both encoding and decoding, with positional encodings injected to provide sequence order information.

**Key Contributions**:

1. The first sequence transduction model based **entirely** on attention — no recurrence, no convolution.
2. Introduction of **Scaled Dot-Product Attention** and **Multi-Head Attention** as core computational primitives.
3. A novel **sinusoidal positional encoding** scheme that encodes absolute position and may generalize to unseen sequence lengths.
4. State-of-the-art BLEU scores on WMT 2014 EN-DE (28.4) and EN-FR (41.8) translation benchmarks.
5. Dramatically reduced training cost — the big model trained in **3.5 days on 8 P100 GPUs**, a small fraction of competing models.
6. Demonstration of generalization to **English constituency parsing**, achieving competitive results with minimal task-specific tuning.

**Results**: The Transformer (big) achieved 28.4 BLEU on EN-DE (surpassing all prior models including ensembles by >2 BLEU) and 41.8 BLEU on EN-FR (new single-model SOTA), while requiring orders of magnitude fewer FLOPs to train than previous state-of-the-art systems.

**Significance**: This paper fundamentally reshaped the landscape of deep learning. The Transformer architecture became the foundation for virtually all subsequent breakthroughs in NLP (BERT, GPT, T5, LLaMA, etc.), and has been successfully extended to vision (ViT), audio, protein folding (AlphaFold2), and multimodal AI. It is arguably the single most impactful deep learning paper of the last decade.

---

## 2. Architecture Deep Dive & Figure Explanations

### 2.1 The Transformer — Full Model Architecture

![[Screenshot 2569-03-14 at 20.50.43.png]]

The Transformer follows an **encoder-decoder** structure, but replaces all recurrent layers with attention and feed-forward sub-layers.

**Left half — Encoder Stack:**

- Composed of $N = 6$ identical layers.
- Each layer contains two sub-layers:
    1. **Multi-Head Self-Attention**: Every position in the input attends to all other positions in the same layer.
    2. **Position-wise Feed-Forward Network (FFN)**: Applied independently to each position.
- Each sub-layer is wrapped with a **residual connection** and **layer normalization**:

$$\text{LayerNorm}(x + \text{Sublayer}(x))$$

- Input tokens are first converted to $d_{\text{model}} = 512$ dimensional embeddings, to which **positional encodings** are added before being fed into the encoder stack.

**Right half — Decoder Stack:**

- Also $N = 6$ identical layers, but with **three** sub-layers per layer:
    1. **Masked Multi-Head Self-Attention**: Self-attention over previously generated output tokens. The masking prevents positions from attending to future positions, preserving the **auto-regressive** property.
    2. **Multi-Head Cross-Attention (Encoder-Decoder Attention)**: Queries come from the decoder, while keys and values come from the encoder output. This is the bridge connecting encoder representations to decoder generation.
    3. **Position-wise Feed-Forward Network (FFN)**: Same architecture as the encoder's FFN.
- The decoder output is projected through a linear layer and softmax to produce next-token probabilities.
- Output embeddings are **offset by one position** (shifted right) to ensure predictions at position $i$ depend only on positions $< i$.

**Weight Sharing**: The same weight matrix is shared between the input embedding layers, the output embedding layer, and the pre-softmax linear transformation. Embedding weights are scaled by $\sqrt{d_{\text{model}}}$.

### 2.2 Scaled Dot-Product Attention & Multi-Head Attention

![[Screenshot 2569-03-14 at 20.51.19.png]]

**Left: Scaled Dot-Product Attention**

This is the core attention primitive. Given queries $Q$, keys $K$ (both of dimension $d_k$), and values $V$ (of dimension $d_v$):

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V \tag{1}$$

The computation flows as:

1. Compute dot products $QK^T$ — this produces a matrix of raw compatibility scores between all query-key pairs.
2. **Scale** by $\frac{1}{\sqrt{d_k}}$ — this is critical because for large $d_k$, dot products grow in magnitude (if $q$ and $k$ have components with mean 0 and variance 1, then $q \cdot k$ has variance $d_k$), pushing the softmax into saturation regions with extremely small gradients.
3. Apply **softmax** row-wise to obtain attention weights (probability distribution over values).
4. Multiply by $V$ to compute a weighted sum of value vectors.

**Optional masking**: In the decoder's self-attention, an additive mask of $-\infty$ is applied to all positions corresponding to future tokens before the softmax, ensuring that illegal leftward information flow is blocked.

**Right: Multi-Head Attention**

Rather than performing a single attention function over $d_{\text{model}}$-dimensional inputs, the model linearly projects $Q$, $K$, $V$ into $h$ different subspaces and computes attention in parallel:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) \tag{2}$$

Where the learned projection matrices are:

$$W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}, \quad W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}, \quad W^O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$$

With $h = 8$ heads and $d_k = d_v = d_{\text{model}} / h = 64$, the total computation is comparable to single-head full-dimensional attention, but each head can attend to different representation subspaces, capturing diverse aspects of the input (e.g., syntactic relations in one head, semantic relations in another).

### 2.3 Figures 3, 4, 5: Attention Visualizations

These figures from the appendix provide interpretability evidence:
![[Screenshot 2569-03-14 at 20.52.02.png]]
- **Figure 3**: Shows encoder self-attention at layer 5, illustrating how the attention mechanism captures **long-distance dependencies**. The word "making" attends to "more difficult" despite several intervening tokens, completing the phrase "making...more difficult."
![[Screenshot 2569-03-14 at 20.53.11.png]]
- **Figure 4**: Demonstrates **anaphora resolution** — two heads in layer 5 show the word "its" sharply attending to the referent noun (e.g., "law" or "application"), with very focused attention distributions.
![[Screenshot 2569-03-14 at 20.53.57.png]]
- **Figure 5**: Different heads at layer 5 appear to learn distinct structural roles — one head captures phrase-level grouping while another follows dependency structures, suggesting that multi-head attention naturally decomposes into specialized sub-functions.

---

## 3. Key Technical Components — Equations & Insights

### 3.1 Position-wise Feed-Forward Network

Applied to each position identically (but with different parameters per layer):

$$\text{FFN}(x) = \max(0, ; xW_1 + b_1) W_2 + b_2 \tag{3}$$

- Inner dimension: $d_{ff} = 2048$ (4× expansion from $d_{\text{model}} = 512$).
- This is equivalent to two 1×1 convolutions with a ReLU activation between them.
- Provides the model's non-linearity and acts as a "memory" layer that stores factual knowledge (as later research has shown).

### 3.2 Positional Encoding

Since the Transformer has no inherent notion of sequence order, positional information is injected via sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right) \tag{4}$$

Where $pos$ is the token position and $i$ is the dimension index. Key properties:

- Each dimension corresponds to a sinusoid with a wavelength forming a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.
- For any fixed offset $k$, $PE_{pos+k}$ can be expressed as a **linear function** of $PE_{pos}$, which the authors hypothesize allows the model to learn relative positional attention.
- Learned positional embeddings produced **nearly identical** results (Table 3, row E), but sinusoidal encodings may allow **extrapolation** to longer sequences than seen during training.

### 3.3 Learning Rate Schedule (Warmup + Inverse Square Root Decay)

$$lrate = d_{\text{model}}^{-0.5} \cdot \min\left(step\_num^{-0.5},  step\_num \cdot warmup\_steps^{-1.5}\right) \tag{5}$$

- Linear warmup for the first $warmup\_steps = 4000$ steps.
- Then decays proportionally to the inverse square root of the step number.
- Optimizer: Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$.

### 3.4 Regularization

- **Residual Dropout** ($P_{drop} = 0.1$ for base, $0.3$ for big EN-DE): Applied to the output of each sub-layer before residual addition, and to the sum of embeddings and positional encodings.
- **Label Smoothing** ($\epsilon_{ls} = 0.1$): Hurts perplexity but improves BLEU, making the model less overconfident.

---

## 4. Experiments & Results Analysis

### 4.1 Machine Translation

|Model|EN-DE BLEU|EN-FR BLEU|Training FLOPs|
|---|---|---|---|
|GNMT + RL|24.6|39.92|$2.3 \times 10^{19}$|
|ConvS2S|25.16|40.46|$9.6 \times 10^{18}$|
|MoE|26.03|40.56|$2.0 \times 10^{19}$|
|Transformer (base)|27.3|38.1|$3.3 \times 10^{18}$|
|**Transformer (big)**|**28.4**|**41.8**|$2.3 \times 10^{19}$|

**Key Findings**:

- The base Transformer already surpasses all previous single models **and ensembles** on EN-DE at a fraction of the cost.
- The big model achieves +2.0 BLEU over the best prior ensemble on EN-DE.
- Training cost is **1–2 orders of magnitude lower** than comparable models. The base model uses only $3.3 \times 10^{18}$ FLOPs.

### 4.2 Ablation Study (Table 3)

Key ablation findings on EN-DE dev set:

- **(A) Number of heads**: Single-head attention drops 0.9 BLEU; too many heads (32) also degrades. Sweet spot at $h = 8$.
- **(B) Key dimension $d_k$**: Reducing $d_k$ hurts quality, suggesting dot-product compatibility needs sufficient capacity.
- **(C) Model size**: Larger $d_{\text{model}}$ and $d_{ff}$ consistently improve quality. Depth (number of layers $N$) matters significantly.
- **(D) Regularization**: Dropout is essential — removing it drops 1.2 BLEU. Label smoothing helps BLEU but hurts perplexity.
- **(E) Positional encoding**: Sinusoidal vs. learned embeddings produce nearly identical results (25.8 vs. 25.7 BLEU).

### 4.3 English Constituency Parsing (Table 4)

A 4-layer Transformer with $d_{\text{model}} = 1024$ achieves:

- **91.3 F1** (WSJ only) — outperforming BerkeleyParser and nearly matching RNN Grammar.
- **92.7 F1** (semi-supervised) — competitive with the best models, demonstrating generalizability beyond machine translation with minimal task-specific tuning.

### 4.4 Computational Complexity Comparison (Table 1)

|Layer Type|Complexity/Layer|Sequential Ops|Max Path Length|
|---|---|---|---|
|Self-Attention|$O(n^2 \cdot d)$|$O(1)$|$O(1)$|
|Recurrent|$O(n \cdot d^2)$|$O(n)$|$O(n)$|
|Convolutional|$O(k \cdot n \cdot d^2)$|$O(1)$|$O(\log_k(n))$|
|Self-Attention (restricted)|$O(r \cdot n \cdot d)$|$O(1)$|$O(n/r)$|

- Self-attention achieves **constant** maximum path length $O(1)$ — critical for learning long-range dependencies.
- Self-attention is faster than recurrent layers when $n < d$ (typical for NLP with subword tokenization).
- The trade-off: $O(n^2)$ in sequence length — which later work addressed (Longformer, Flash Attention, etc.).

---

## 5. Research Gaps & Limitations

### 5.1 Acknowledged Limitations

1. **Quadratic complexity in sequence length**: Self-attention scales as $O(n^2 \cdot d)$, making it computationally expensive for long sequences (images, audio, video). The authors acknowledge this and mention restricted self-attention as future work.
2. **No mechanism for handling very long sequences**: The paper acknowledges that local/restricted attention is needed for large inputs.

### 5.2 Unacknowledged Gaps

1. **No analysis of failure modes**: The paper reports only successes. When does the Transformer fail? What are the error patterns? No qualitative error analysis is provided.
2. **Positional encoding limitations**: While sinusoidal encodings are elegant, the paper does not rigorously test extrapolation to longer sequences. Later work (RoPE, ALiBi) showed significant room for improvement.
3. **Lack of theoretical justification**: Why does self-attention work so well? The paper provides intuitions (constant path length, parallelizability) but no formal convergence or expressivity analysis.
4. **Single-task evaluation bias**: Only machine translation and constituency parsing are tested. No evaluation on language understanding, generation quality, or other NLP tasks.
5. **No analysis of training stability**: The warmup schedule is presented without ablation or analysis of why it is necessary (later understood as related to Adam's initial variance estimation).
6. **Decoder autoregressive bottleneck**: The paper does not discuss the inherent sequential nature of autoregressive decoding at inference time — a significant practical limitation.
7. **No discussion of pre-training paradigm**: The Transformer is trained end-to-end on translation data. The paper does not explore unsupervised pre-training, which became the dominant paradigm within a year (GPT, BERT).

### 5.3 Assumptions to Challenge

1. **Is $O(1)$ path length always better?** Later work on hierarchical/sparse attention showed that structured inductive biases can be beneficial.
2. **Is the FFN layer optimal?** GLU variants (SwiGLU, GeGLU) significantly improved upon ReLU FFN in subsequent work.
3. **Is post-norm the best arrangement?** The paper uses post-LayerNorm, but pre-LayerNorm (used in GPT-2+) was later found to be more stable for training deep models.
4. **Are sinusoidal positional encodings sufficient?** Rotary Position Embeddings (RoPE) and relative position encodings proved more effective, especially for length generalization.

### 5.4 Missing Experiments

1. **Scaling laws**: How does performance change with model size, data size, and compute budget? (Answered by Kaplan et al., 2020)
2. **Cross-lingual transfer**: Does the architecture generalize to low-resource languages?
3. **Decoder-only vs. encoder-decoder**: Which architecture is more effective for which tasks? (Answered by GPT vs. T5)
4. **Attention head pruning**: How many heads are actually necessary? (Answered by Michel et al., 2019 — many heads are redundant)
5. **Layer-by-layer analysis**: What is each layer learning? (Partially addressed by probing studies)

---

## 6. Future Research Directions

### 6.1 Directions Proposed by Authors

1. Application to **non-text modalities** (images, audio, video) — subsequently realized by ViT, Whisper, VideoMAE, etc.
2. **Local/restricted attention** for long sequences — realized by Longformer, BigBird, Sparse Transformers.
3. **Less sequential generation** — realized by non-autoregressive translation, parallel decoding, speculative decoding.

### 6.2 Post-Paper Research Directions (Realized & Open)

1. **Efficient attention**: Flash Attention, linear attention, ring attention — reducing the $O(n^2)$ bottleneck.
2. **Pre-training at scale**: GPT, BERT, T5, PaLM, LLaMA — unsupervised/self-supervised pre-training became the dominant paradigm.
3. **Mixture of Experts (MoE)**: Scaling model capacity without proportional compute increase (Switch Transformer, Mixtral).
4. **Positional encoding advances**: RoPE, ALiBi, NoPE — improving length generalization and relative position handling.
5. **Architecture search**: Is the Transformer optimal? Mamba (state-space models), RWKV, and hybrid architectures explore alternatives.
6. **Multimodal fusion**: Extending the Transformer to handle interleaved text, image, audio, and video (GPT-4V, Gemini).
7. **Interpretability**: Circuit-level understanding of what Transformer components compute (mechanistic interpretability).
8. **Training stability at scale**: Understanding and mitigating loss spikes, gradient issues in very large models.

### 6.3 Extension Ideas for New Research

1. **Adaptive computation per token**: Allowing different numbers of layers or heads per token based on difficulty.
2. **Memory-augmented Transformers**: External memory banks for extremely long contexts without quadratic cost.
3. **Continuous/analog attention**: Moving beyond discrete softmax to learnable, continuous attention kernels.

---

## 7. Reproduction Codebase

### 7.1 Scope

**Target**: Reproduce the Transformer (base model) architecture for EN-DE machine translation as described in the paper. **Success criterion**: Within 1.0 BLEU of the reported 27.3 on newstest2014. **Compute**: 8× GPU (P100-equivalent or better), ~12 hours for base model.

### 7.2 Project Structure

```
reproduction/
├── README.md                # Setup, usage, expected results
├── requirements.txt         # Pinned dependencies
├── config.yaml              # Hyperparameters from paper
├── src/
│   ├── __init__.py
│   ├── model.py             # Core Transformer architecture
│   ├── attention.py         # Scaled Dot-Product & Multi-Head Attention
│   ├── layers.py            # FFN, LayerNorm, Positional Encoding
│   ├── data.py              # WMT data pipeline with BPE
│   ├── train.py             # Training loop with warmup LR schedule
│   ├── evaluate.py          # BLEU evaluation
│   └── utils.py             # Label smoothing, beam search
└── scripts/
    └── run_experiment.sh    # Full training + eval pipeline
```

### 7.3 Configuration (config.yaml)

```yaml
# Transformer Base Model — "Attention Is All You Need" (Vaswani et al., 2017)
model:
  d_model: 512
  d_ff: 2048
  n_layers: 6         # N = 6 for both encoder and decoder
  n_heads: 8           # h = 8
  d_k: 64              # d_model / h
  d_v: 64              # d_model / h
  max_seq_len: 512
  dropout: 0.1         # P_drop
  label_smoothing: 0.1 # epsilon_ls
  share_embeddings: true  # Shared input/output/pre-softmax weights

training:
  optimizer: adam
  beta1: 0.9
  beta2: 0.98
  epsilon: 1.0e-9
  warmup_steps: 4000
  total_steps: 100000
  batch_tokens: 25000  # ~25K source + ~25K target tokens per batch

data:
  dataset: wmt14_en_de
  vocab_size: 37000    # Shared BPE vocabulary
  tokenizer: byte_pair_encoding

inference:
  beam_size: 4
  length_penalty_alpha: 0.6
  max_length_offset: 50  # max_output = input_length + 50
  checkpoint_averaging: 5  # Average last 5 checkpoints (10-min intervals)

hardware:
  gpus: 8              # NVIDIA P100
  step_time_seconds: 0.4
  total_training_hours: 12
```

### 7.4 Core Implementation

#### `src/attention.py` — Scaled Dot-Product & Multi-Head Attention

```python
"""
Attention mechanisms from "Attention Is All You Need" (Vaswani et al., 2017).
Implements Equation (1): Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
and Equation (2): MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (Section 3.2.1, Figure 2 left).
    
    Computes: softmax(QK^T / sqrt(d_k)) V
    
    The scaling factor 1/sqrt(d_k) prevents dot products from growing
    large in magnitude for large d_k, which would push softmax into
    regions with extremely small gradients.
    
    Footnote 1: If q, k components are i.i.d. with mean 0 and variance 1,
    then q·k has mean 0 and variance d_k.
    """
    
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.scale = math.sqrt(d_k)  # sqrt(d_k) for scaling, Eq. (1)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,   # (batch, h, seq_q, d_k)
        key: torch.Tensor,     # (batch, h, seq_k, d_k)
        value: torch.Tensor,   # (batch, h, seq_k, d_v)
        mask: torch.Tensor = None  # (batch, 1, seq_q, seq_k) or broadcastable
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: (batch, h, seq_q, d_v) — weighted sum of values
            attn_weights: (batch, h, seq_q, seq_k) — attention probabilities
        """
        # Step 1: Compute raw attention scores QK^T
        # (batch, h, seq_q, d_k) @ (batch, h, d_k, seq_k) -> (batch, h, seq_q, seq_k)
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: Scale by 1/sqrt(d_k) — Eq. (1)
        attn_scores = attn_scores / self.scale
        
        # Step 3: Apply mask (for decoder self-attention causal masking)
        # Mask sets future positions to -inf so softmax assigns them ~0 weight
        # Section 3.2.3: "masking out (setting to -inf) all values in the input
        # of the softmax which correspond to illegal connections"
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Step 4: Softmax to get attention weights (probability distribution)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Step 5: Weighted sum of values
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (Section 3.2.2, Figure 2 right).
    
    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O
    where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    
    With h=8 heads and d_k = d_v = d_model/h = 64, total cost is
    similar to single-head attention with full dimensionality.
    
    "Multi-head attention allows the model to jointly attend to
    information from different representation subspaces at different
    positions." — Section 3.2.2
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads  # h = 8
        self.d_k = d_model // n_heads  # d_k = 64
        self.d_v = d_model // n_heads  # d_v = 64
        
        # Projection matrices: W_i^Q, W_i^K, W_i^V ∈ R^{d_model × d_k}
        # Implemented as single large matrices for efficiency
        self.W_Q = nn.Linear(d_model, d_model, bias=False)  # All h heads' W_i^Q combined
        self.W_K = nn.Linear(d_model, d_model, bias=False)  # All h heads' W_i^K combined
        self.W_V = nn.Linear(d_model, d_model, bias=False)  # All h heads' W_i^V combined
        
        # Output projection: W^O ∈ R^{h*d_v × d_model}
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
    
    def forward(
        self,
        query: torch.Tensor,   # (batch, seq_q, d_model)
        key: torch.Tensor,     # (batch, seq_k, d_model)
        value: torch.Tensor,   # (batch, seq_k, d_model)
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear projections and reshape to (batch, h, seq, d_k)
        Q = self.W_Q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        
        # Apply attention — Eq. (1) applied h times in parallel
        attn_output, _ = self.attention(Q, K, V, mask)
        
        # Concat heads: (batch, h, seq_q, d_v) -> (batch, seq_q, h*d_v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection W^O — Eq. (2)
        output = self.W_O(attn_output)
        
        return output
```

#### `src/layers.py` — Feed-Forward, LayerNorm, Positional Encoding

```python
"""
Supporting layers for the Transformer.
- Position-wise FFN: Equation (2) — FFN(x) = max(0, xW1+b1)W2+b2
- Positional Encoding: Equation (4) — sinusoidal PE
- Sublayer connection: LayerNorm(x + Sublayer(x))
"""

import torch
import torch.nn as nn
import math


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (Section 3.3).
    
    FFN(x) = max(0, xW_1 + b_1) W_2 + b_2  — Eq. (2)
    
    d_model = 512, d_ff = 2048 (4x expansion).
    "Another way of describing this is as two convolutions with kernel size 1."
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # W_1: 512 -> 2048
        self.linear2 = nn.Linear(d_ff, d_model)    # W_2: 2048 -> 512
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # max(0, xW_1 + b_1) = ReLU(xW_1 + b_1)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Section 3.5).
    
    PE(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
    
    Wavelengths form geometric progression from 2π to 10000·2π.
    
    Key property: For any fixed offset k, PE_{pos+k} is a linear
    function of PE_{pos}, enabling the model to learn relative
    positional attention.
    
    "We chose the sinusoidal version because it may allow the model
    to extrapolate to sequence lengths longer than the ones
    encountered during training." — Section 3.5
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Pre-compute positional encodings for all positions up to max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the denominator: 10000^{2i/d_model}
        # Using log-space for numerical stability:
        # 10000^{2i/d_model} = exp(2i * log(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices: sin
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices: cos
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model) for broadcasting
        self.register_buffer('pe', pe)  # Not a learnable parameter
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — token embeddings
        Returns:
            (batch, seq_len, d_model) — embeddings + positional encoding
        """
        # Section 3.5: "we add positional encodings to the input embeddings"
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)  # Section 5.4: dropout on sum of embeddings + PE


class SublayerConnection(nn.Module):
    """
    Residual connection + Layer Normalization (Section 3.1).
    
    Output = LayerNorm(x + Sublayer(x))
    
    Note: The paper uses post-norm (norm after residual addition).
    Later work (GPT-2, etc.) found pre-norm to be more stable.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)  # Ba et al. [1]
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x: torch.Tensor, sublayer_fn) -> torch.Tensor:
        """
        Args:
            x: input tensor
            sublayer_fn: callable (either attention or FFN)
        """
        # Section 5.4: "We apply dropout to the output of each sub-layer,
        # before it is added to the sub-layer input and normalized."
        return self.norm(x + self.dropout(sublayer_fn(x)))
```

#### `src/model.py` — Full Transformer Architecture

```python
"""
The Transformer model (Section 3, Figure 1).
"Attention Is All You Need" — Vaswani et al., NeurIPS 2017.

Architecture:
- Encoder: N=6 layers of [MultiHeadSelfAttention → FFN], each with
  residual connections and layer normalization.
- Decoder: N=6 layers of [MaskedMultiHeadSelfAttention →
  MultiHeadCrossAttention → FFN], each with residual + LayerNorm.
- Shared embedding weights between input embeddings, output embeddings,
  and pre-softmax linear transformation (Section 3.4).
"""

import torch
import torch.nn as nn
import math
import copy

from .attention import MultiHeadAttention
from .layers import PositionWiseFeedForward, PositionalEncoding, SublayerConnection


class EncoderLayer(nn.Module):
    """
    Single encoder layer (Section 3.1 — Encoder).
    
    Two sub-layers:
    1. Multi-head self-attention
    2. Position-wise FFN
    Each with residual + LayerNorm.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
    
    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Sub-layer 1: Multi-head self-attention
        # "In a self-attention layer all of the keys, values and queries
        # come from the same place" — Section 3.2.3
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, src_mask))
        # Sub-layer 2: Position-wise FFN
        x = self.sublayer2(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    """
    Single decoder layer (Section 3.1 — Decoder).
    
    Three sub-layers:
    1. Masked multi-head self-attention (prevents attending to future)
    2. Multi-head cross-attention (attends to encoder output)
    3. Position-wise FFN
    Each with residual + LayerNorm.
    """
    
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
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        # Sub-layer 1: Masked self-attention
        # "We modify the self-attention sub-layer in the decoder stack
        # to prevent positions from attending to subsequent positions" — Section 3.1
        x = self.sublayer1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        # Sub-layer 2: Encoder-decoder cross-attention
        # "queries come from the previous decoder layer, and the memory
        # keys and values come from the output of the encoder" — Section 3.2.3
        x = self.sublayer2(x, lambda x: self.cross_attn(x, encoder_output,
                                                         encoder_output, src_mask))
        
        # Sub-layer 3: Position-wise FFN
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
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Full Transformer model (Figure 1).
    
    Architecture: Encoder-Decoder with shared embeddings.
    
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
        max_len: int = 5000
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Shared embedding (Section 3.4)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder stack (N=6 layers)
        encoder_layer = EncoderLayer(d_model, n_heads, d_ff, dropout)
        self.encoder = Encoder(encoder_layer, n_layers)
        
        # Decoder stack (N=6 layers)
        decoder_layer = DecoderLayer(d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(decoder_layer, n_layers)
        
        # Output projection (shared with embedding — Section 3.4)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight sharing: embedding.weight == output_projection.weight
        self.output_projection.weight = self.embedding.weight
        
        # Initialize parameters (Xavier uniform)
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Section 3.4: "In the embedding layers, we multiply those
        # weights by sqrt(d_model)"
        src_emb = self.positional_encoding(
            self.embedding(src) * math.sqrt(self.d_model)
        )
        return self.encoder(src_emb, src_mask)
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        tgt_emb = self.positional_encoding(
            self.embedding(tgt) * math.sqrt(self.d_model)
        )
        return self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
    
    def forward(
        self,
        src: torch.Tensor,       # (batch, src_len) — source token IDs
        tgt: torch.Tensor,       # (batch, tgt_len) — target token IDs (shifted right)
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns logits: (batch, tgt_len, vocab_size)
        """
        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        logits = self.output_projection(decoder_output)
        return logits
    
    @staticmethod
    def make_causal_mask(size: int) -> torch.Tensor:
        """
        Create causal (look-ahead) mask for decoder self-attention.
        
        Section 3.2.3: "masking out (setting to -inf) all values in the
        input of the softmax which correspond to illegal connections"
        
        Returns upper-triangular mask where future positions are 0.
        """
        mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, size, size)


def build_transformer_base(vocab_size: int) -> Transformer:
    """Build the base Transformer model with paper hyperparameters."""
    return Transformer(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
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
        max_len=5000
    )
```

#### `src/train.py` — Training Loop with Warmup LR Schedule

```python
"""
Training loop for the Transformer (Section 5).

Key components:
- Adam optimizer: β1=0.9, β2=0.98, ε=10^-9 (Section 5.3)
- Warmup LR schedule: Eq. (3) (Section 5.3)
- Label smoothing: ε_ls = 0.1 (Section 5.4)
- Residual dropout: P_drop = 0.1 (Section 5.4)
- Batch by approximate sequence length: ~25K src + ~25K tgt tokens (Section 5.1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Iterator


class TransformerLRScheduler:
    """
    Noam learning rate schedule (Section 5.3, Equation 3).
    
    lrate = d_model^{-0.5} * min(step_num^{-0.5}, step_num * warmup_steps^{-1.5})
    
    Linear warmup for first warmup_steps, then inverse sqrt decay.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int = 4000
    ):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self._compute_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _compute_lr(self) -> float:
        """Equation (3): lrate = d_model^{-0.5} * min(step^{-0.5}, step * warmup^{-1.5})"""
        return (
            self.d_model ** (-0.5)
            * min(
                self.step_num ** (-0.5),
                self.step_num * self.warmup_steps ** (-1.5)
            )
        )


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing (Section 5.4).
    
    ε_ls = 0.1: distributes 0.1 of probability mass uniformly across
    all tokens, keeping 0.9 on the target token.
    
    "This hurts perplexity, as the model learns to be more unsure,
    but improves accuracy and BLEU score." — Section 5.4
    """
    
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.criterion = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch * seq_len, vocab_size) — raw model output
            target: (batch * seq_len,) — target token IDs
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Create smoothed target distribution
        smooth_target = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        smooth_target.scatter_(1, target.unsqueeze(1), self.confidence)
        smooth_target[:, self.padding_idx] = 0
        
        # Zero out padding positions
        mask = target != self.padding_idx
        smooth_target[~mask] = 0
        
        return self.criterion(log_probs, smooth_target)


def train_step(
    model: nn.Module,
    src: torch.Tensor,
    tgt: torch.Tensor,
    src_mask: torch.Tensor,
    tgt_mask: torch.Tensor,
    criterion: LabelSmoothingLoss,
    optimizer: optim.Optimizer,
    scheduler: TransformerLRScheduler
) -> float:
    """Single training step."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass — decoder input is tgt[:-1], target is tgt[1:]
    tgt_input = tgt[:, :-1]   # Shift right (Section 3.1)
    tgt_output = tgt[:, 1:]   # Predict next token
    
    logits = model(src, tgt_input, src_mask, tgt_mask)
    
    # Compute loss
    loss = criterion(
        logits.contiguous().view(-1, logits.size(-1)),
        tgt_output.contiguous().view(-1)
    )
    
    # Backward pass
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    return loss.item()


def train(
    model: nn.Module,
    train_dataloader: Iterator,
    vocab_size: int,
    d_model: int = 512,
    warmup_steps: int = 4000,
    total_steps: int = 100000,
    padding_idx: int = 0,
    label_smoothing: float = 0.1
):
    """
    Full training loop (Section 5).
    
    Hardware: 8 NVIDIA P100 GPUs
    Base model: ~0.4s/step, 100K steps, ~12 hours
    Big model: ~1.0s/step, 300K steps, ~3.5 days
    """
    # Section 5.3: Adam with β1=0.9, β2=0.98, ε=10^-9
    optimizer = optim.Adam(
        model.parameters(),
        lr=0,  # Will be set by scheduler
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    scheduler = TransformerLRScheduler(optimizer, d_model, warmup_steps)
    criterion = LabelSmoothingLoss(vocab_size, padding_idx, label_smoothing)
    
    for step in range(1, total_steps + 1):
        src, tgt, src_mask, tgt_mask = next(train_dataloader)
        
        loss = train_step(model, src, tgt, src_mask, tgt_mask,
                         criterion, optimizer, scheduler)
        
        if step % 1000 == 0:
            print(f"Step {step}/{total_steps} | Loss: {loss:.4f} | "
                  f"LR: {scheduler._compute_lr():.6e}")
```

#### `src/evaluate.py` — Beam Search & BLEU Evaluation

```python
"""
Evaluation utilities (Section 6.1).

Inference settings:
- Beam search with beam_size=4
- Length penalty α=0.6 (Wu et al., 2016 [38])
- Max output length = input_length + 50
- Checkpoint averaging: last 5 (base) or 20 (big) checkpoints
"""

import torch
import torch.nn.functional as F
from typing import List


def beam_search(
    model,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    bos_token: int,
    eos_token: int,
    beam_size: int = 4,
    max_len: int = 200,
    length_penalty_alpha: float = 0.6
) -> torch.Tensor:
    """
    Beam search decoding (Section 6.1).
    
    "We used beam search with a beam size of 4 and length penalty
    α=0.6. We set the maximum output length during inference to
    input length + 50, but terminate early when possible." — Section 6.1
    """
    device = src.device
    encoder_output = model.encode(src, src_mask)
    
    # Initialize beams: (beam_size, 1) starting with BOS token
    beams = torch.full((beam_size, 1), bos_token, dtype=torch.long, device=device)
    beam_scores = torch.zeros(beam_size, device=device)
    finished = []
    
    for step in range(max_len):
        # Create causal mask for current sequence length
        tgt_mask = model.make_causal_mask(beams.size(1)).to(device)
        
        # Decode all beams
        # Expand encoder output for beam_size
        expanded_enc = encoder_output.expand(beam_size, -1, -1)
        expanded_src_mask = src_mask.expand(beam_size, -1, -1, -1)
        
        decoder_output = model.decode(beams, expanded_enc,
                                       expanded_src_mask, tgt_mask)
        logits = model.output_projection(decoder_output[:, -1, :])
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Score all candidates
        vocab_size = log_probs.size(-1)
        next_scores = beam_scores.unsqueeze(1) + log_probs  # (beam, vocab)
        next_scores = next_scores.view(-1)  # Flatten
        
        # Select top-k
        topk_scores, topk_indices = next_scores.topk(beam_size, dim=0)
        beam_indices = topk_indices // vocab_size
        token_indices = topk_indices % vocab_size
        
        # Update beams
        beams = torch.cat([beams[beam_indices], token_indices.unsqueeze(1)], dim=1)
        beam_scores = topk_scores
        
        # Check for EOS — apply length penalty: score / length^α
        for i in range(beam_size):
            if token_indices[i] == eos_token:
                length = beams[i].size(0)
                # Length penalty (Wu et al., 2016)
                lp = ((5 + length) / 6) ** length_penalty_alpha
                finished.append((beam_scores[i] / lp, beams[i]))
        
        # Early termination
        if len(finished) >= beam_size:
            break
    
    # Return best hypothesis
    if finished:
        finished.sort(key=lambda x: x[0], reverse=True)
        return finished[0][1]
    return beams[0]


def average_checkpoints(checkpoint_paths: List[str]) -> dict:
    """
    Checkpoint averaging (Section 6.1).
    
    "For the base models, we used a single model obtained by averaging
    the last 5 checkpoints, which were written at 10-minute intervals.
    For the big models, we averaged the last 20 checkpoints." — Section 6.1
    """
    avg_state = None
    
    for path in checkpoint_paths:
        state = torch.load(path, map_location='cpu')
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k in avg_state:
                avg_state[k] += state[k].float()
    
    # Average
    n = len(checkpoint_paths)
    for k in avg_state:
        avg_state[k] /= n
    
    return avg_state
```

### 7.5 Expected Reproduction Results

|Metric|Paper (Base)|Expected Reproduction|Δ|
|---|---|---|---|
|EN-DE BLEU (newstest2014)|27.3|26.5–27.5|±0.8|
|EN-DE PPL (newstest2013 dev)|4.92|4.8–5.1|±0.2|
|Training Steps|100K|100K|—|
|Training Time (8× P100)|~12 hours|~12 hours|—|
|Parameters|65M|65M|—|

**Potential sources of variance**: BPE tokenizer implementation differences, batching strategy, random seed, GPU precision (FP16 vs FP32), and checkpoint averaging window.

---

## 8. Impact Assessment & Historical Context

The Transformer's impact cannot be overstated. Within 7 years of publication, it became the **de facto standard architecture** for:

- **Language models**: GPT-1/2/3/4, BERT, T5, PaLM, LLaMA, Gemini, Claude
- **Computer vision**: ViT, DeiT, Swin Transformer, DINO
- **Speech & audio**: Whisper, AudioLM, MusicLM
- **Protein biology**: AlphaFold2, ESMFold
- **Multimodal**: CLIP, DALL-E, Flamingo, GPT-4V
- **Robotics**: RT-2, Gato

The paper has been cited over **140,000 times** (as of 2025), making it one of the most cited papers in computer science history. The title "Attention Is All You Need" became an iconic phrase, spawning countless variations in subsequent paper titles.

The authors themselves went on to found or co-found multiple influential AI companies (Cohere, NEAR Protocol, Inceptive, Essential AI, Sakana AI, Character.AI), demonstrating the direct commercial impact of this foundational work.
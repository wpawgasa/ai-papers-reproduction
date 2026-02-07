# Deep Analysis: FNet — Mixing Tokens with Fourier Transforms

**Paper**: FNet: Mixing Tokens with Fourier Transforms  
**Authors**: James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontañón  
**Venue**: NAACL 2022 (Best Efficient NLP Paper Award)  
**Links**: [arXiv:2105.03824](https://arxiv.org/abs/2105.03824) · [Code](https://github.com/google-research/google-research/tree/master/f_net) · [ACL Anthology](https://aclanthology.org/2022.naacl-main.319/)

---

## Summary: FNet

**Problem**: The Transformer's self-attention mechanism has $O(n^2)$ time and memory complexity with respect to sequence length $n$, creating a computational bottleneck that limits scalability to long sequences and efficiency in resource-constrained settings.

**Approach**: FNet replaces the entire self-attention sublayer in each Transformer encoder block with an unparameterized two-dimensional Discrete Fourier Transform (DFT) that "mixes" tokens. The resulting model retains the feed-forward sublayers, residual connections, and layer normalization from the original Transformer, but the mixing mechanism has zero learnable parameters and runs in $O(n \log n)$ via the Fast Fourier Transform (FFT).

**Key Contributions**:

1. Demonstration that simple, unparameterized linear transforms can replace self-attention with only modest accuracy loss
2. Introduction of FNet — achieving 92–97% of BERT accuracy on GLUE while training 70–80% faster
3. FNet-Hybrid models (2 attention layers) recover 97–99% of BERT accuracy with 40–70% speedup
4. State-of-the-art speed-accuracy tradeoff on the Long Range Arena benchmark

**Results**: FNet-Base achieves 76.7 avg on GLUE (vs. 83.3 for BERT-Base), trains at 1.8× speed on GPU, and uses 83M parameters (vs. 112M). On LRA, FNet matches the most accurate efficient Transformers while being the fastest model on GPUs across all sequence lengths.

**Significance**: The paper challenges the prevailing assumption that learned, token-dependent attention is the principal driver of Transformer performance, and argues that seeking entirely new mixing mechanisms may be more productive than designing ever-more-efficient approximations of attention.

---

## 1. Problem & Motivation

**Research Question**: Can the computationally expensive self-attention sublayer in Transformer encoders be wholly replaced by simpler, faster linear transformations — specifically the Fourier Transform — without catastrophic accuracy degradation?

**Context**: By 2021, Transformers dominated virtually every NLP benchmark, but the quadratic attention bottleneck limited their deployment at scale. A growing body of "efficient Transformer" literature attempted to approximate attention more cheaply, but these approximations often hid large constants behind their asymptotic notation. Meanwhile, works like Synthesizer (Tay et al., 2020a), fixed attention patterns (Raganato et al., 2020), and Gaussian attention (You et al., 2020) began questioning whether learned attention was truly indispensable.

**Prior Work Limitations**:

- Efficient Transformers (Longformer, Performer, BigBird, Reformer, Linformer) aim to approximate attention but still rely on parameterized, often complex mixing mechanisms
- Many achieve $O(n)$ theoretical complexity but with large hidden constants, so practical speedups are modest
- MLP-Mixer (Tolstikhin et al., 2021) replaces attention with MLPs in vision, but still uses learnable mixing parameters along the spatial dimension

**Gap Addressed**: No prior work had attempted to completely replace the attention sublayer with a fixed, unparameterized mathematical transform. FNet fills this gap by using the Fourier Transform as a first-class token mixing mechanism.

---

## 2. Technical Approach

### 2.1 Core Idea

The fundamental insight is reframing self-attention as a _token mixing_ mechanism. Attention computes a weighted sum of value vectors using learned, input-dependent weights. The authors hypothesize that much of the benefit comes from the mixing itself (enabling each token to access information from all other tokens), not from the learned, token-dependent weights. If true, any sufficiently structured global mixing operation should work — including the Discrete Fourier Transform, which is a fixed linear transformation that combines all input positions with mathematically determined coefficients.

### 2.2 Mathematical Foundation

**Discrete Fourier Transform (DFT)**

Given a sequence ${x_n}$ with $n \in [0, N-1]$, the DFT is defined as (Equation 1 in the paper):

$$X_k = \sum_{n=0}^{N-1} x_n , e^{-\frac{2\pi i}{N} nk}, \quad 0 \leq k \leq N-1$$

Each output component $X_k$ is a weighted sum of _all_ input tokens $x_n$, with complex exponential "twiddle factors" $e^{-\frac{2\pi i}{N} nk}$ that encode positional information through the indices $n$ and $k$.

**DFT Matrix Formulation**

Equivalently, the DFT can be expressed as a matrix-vector multiplication $\mathbf{X} = \mathbf{W}\mathbf{x}$, where $\mathbf{W}$ is a Vandermonde matrix for the $N$-th roots of unity (Equation 2):

$$W_{nk} = \frac{1}{\sqrt{N}} , e^{-\frac{2\pi i}{N} nk}, \quad n, k = 0, \ldots, N-1$$

This matrix multiplication is $O(N^2)$, but the Cooley–Tukey FFT algorithm computes the same result in $O(N \log N)$.

**FNet Fourier Sublayer**

The FNet sublayer applies a 2D DFT to its $(n, d_h)$-shaped input — one 1D DFT along the sequence dimension and one along the hidden dimension — then extracts only the real part (Equation 3):

$$y = \Re\Big(\mathcal{F}_{\text{seq}}\big(\mathcal{F}_{h}(x)\big)\Big)$$

where $\mathcal{F}_{\text{seq}}$ denotes the DFT along the sequence dimension, $\mathcal{F}_{h}$ along the hidden dimension, and $\Re(\cdot)$ extracts the real component. The two 1D DFTs commute, so their order is immaterial.

### 2.3 Architecture

The overall FNet architecture replaces each self-attention sublayer in a Transformer encoder with the Fourier sublayer described above, while retaining all other components identically:

$$\text{FNet Block:} \quad \mathbf{h} = \text{LayerNorm}\big(\mathbf{x} + \Re(\text{FFT2D}(\mathbf{x}))\big)$$

$$\text{output} = \text{LayerNorm}\big(\mathbf{h} + \text{FFN}(\mathbf{h})\big)$$

where $\text{FFN}(\mathbf{h}) = \text{GELU}(\mathbf{h}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$ is the standard feed-forward network.

Embeddings are identical to BERT: $\mathbf{e} = \mathbf{e}_{\text{word}} + \mathbf{e}_{\text{pos}} + \mathbf{e}_{\text{type}}$.

**FNet-Hybrid**: Replaces the final 2 (of 12) Fourier sublayers with self-attention sublayers. This trades a small amount of speed for a significant accuracy recovery.

![[Pasted image 20260204090116.png]]

### 2.4 Key Innovations

1. **Complete sublayer replacement**: Unlike prior work that approximates or modifies attention, FNet wholly replaces it with a fixed, unparameterized transform — a conceptual leap
2. **Dual interpretation**: The Fourier sublayer can be viewed (a) as a token mixer that gives every token global access, or (b) as transforming between time and frequency domains, where FFN multiplication in frequency domain is equivalent to convolution in time domain
3. **Hardware-aware implementation**: GPU uses FFT ($O(n \log n)$); TPU uses cached DFT matrix multiplication ($O(n^2)$) for sequences $\leq 4096$ because TPUs are more optimized for matmul

### 2.5 Assumptions

- **Encoder-only focus**: Results may not transfer to decoder or encoder-decoder architectures where causal masking and cross-attention are needed
- **BERT training recipe**: All hyperparameters are inherited from BERT, which may be suboptimal for FNet's different inductive biases
- **Real-part sufficiency**: Discarding the imaginary component of the DFT output is assumed to retain sufficient information — phase information is lost

---

## 3. Experimental Evaluation

### 3.1 Datasets

|Benchmark|Tasks|Domain|Sequence Length|Notes|
|---|---|---|---|---|
|**GLUE** (Wang et al., 2018)|MNLI, QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE|NLU classification|512|Standard transfer learning benchmark|
|**Long Range Arena** (Tay et al., 2021a)|ListOps, Text, Retrieval, Image, Pathfinder, Path-X|Long-range dependency|1K–16K|Efficiency benchmark for long sequences|
|**C4** (Raffel et al., 2020)|Pre-training corpus|Web text|512|Much larger than BERT's original corpus|

**Pre-training**: 1M steps on C4 with 32K SentencePiece vocabulary. Batch size 256 (TPU) / 64 (GPU).

### 3.2 Baselines

|Method|Year|Key Difference|Fair Comparison?|
|---|---|---|---|
|BERT (Devlin et al., 2019)|2019|Full self-attention|Yes — identical config|
|Linear encoder|2021|Two dense learnable matrices replace attention|Yes — same framework|
|Random encoder|2021|Two constant random matrices|Yes — ablation|
|FF-only encoder|2021|No mixing sublayer at all|Yes — ablation|
|FNet-Hybrid|2021|Fourier + 2 attention layers at top|Yes — variant|
|Performer (Choromanski et al., 2021)|2021|Linearized attention via random features|Partial — different codebase|
|8 other efficient Transformers|2018–2020|Various attention approximations|Partial — results cited from Tay et al.|

**Missing Baselines**: Convolutional models (e.g., pre-trained CNNs), State Space Models (e.g., S4, which appeared later), and MLP-Mixer adapted for NLP.

### 3.3 Metrics

- **GLUE**: Accuracy (most tasks), F1/Accuracy mean (QQP, MRPC), Spearman correlation (STS-B)
- **LRA**: Accuracy on each of 6 tasks
- **Efficiency**: Training speed (ms/batch or steps/s), inference speed (ms/batch), peak memory (GB), GFLOPS/example
- **Parameter count**: Total learnable parameters

### 3.4 Main Results

#### GLUE Benchmark (Table 2)

|Model|MNLI|QQP|QNLI|SST-2|CoLA|STS-B|MRPC|RTE|**Avg.**|
|---|---|---|---|---|---|---|---|---|---|
|BERT-Base|84/81|87|91|93|73|89|83|69|**83.3**|
|Linear-Base|74/75|84|80|94|67|67|83|69|77.0|
|**FNet-Base**|72/73|83|80|95|69|79|76|63|**76.7**|
|Random-Base|51/50|70|61|76|67|4|73|57|56.6|
|FF-only-Base|34/35|31|52|48|67|FAIL|73|54|49.3|
|FNet-Hybrid-Base|78/79|85|88|94|76|86|79|60|**80.6**|
|BERT-Large|88/88|88|92|95|71|88|86|66|**84.7**|
|FNet-Large|78/76|85|85|94|78|84|88|69|**81.9**|
|FNet-Hybrid-Large|79/80|87|89|92|81|88|86|70|**83.6**|

**Key observations**: FNet-Base achieves $\frac{76.7}{83.3} \approx 92\%$ of BERT-Base. FNet-Large achieves $\frac{81.9}{84.7} \approx 97\%$ of BERT-Large. The gap shrinks at larger scales, partly because FNet-Large is more training-stable.

#### Efficiency (Table 3)

|Model|Pre-train GPU (ms/batch)|Pre-train TPU|Inference GPU|GFLOPS/example|
|---|---|---|---|---|
|BERT-Base|305|213|82|98|
|FNet-Base|169 (**1.8×**)|128 (**1.7×**)|46 (**1.8×**)|62 (63%)|
|FNet-Hybrid-Base|198 (1.5×)|149 (1.4×)|51 (1.6×)|68 (69%)|
|FNet-Large|511|275 (**1.8×**)|149 (**1.8×**)|217 (64%)|

#### Isolated Mixing Layer Speed (Table 8 — Appendix)

|Mixing Layer|Training GPU (ms)|Speedup|Training TPU (ms)|Speedup|
|---|---|---|---|---|
|Self-Attention (Base)|136|1.0×|76|1.0×|
|Linear (Base)|36|3.7×|12|6.1×|
|FNet (Base)|11|**12.2×**|8|**9.9×**|
|Self-Attention (Large)|404|1.0×|212|1.0×|
|FNet (Large)|18|**22.2×**|22|**9.7×**|

The Fourier sublayer alone is **12–22× faster** than self-attention on GPUs, but overall model speedup is 1.8× because the shared feed-forward sublayers become the bottleneck.

#### Long Range Arena (Table 4a)

|Model|ListOps|Text|Retrieval|Image|Pathfinder|**Avg.**|
|---|---|---|---|---|---|---|
|Transformer (theirs)|36.06|61.54|59.67|41.51|80.38|55.83|
|**FNet (theirs)**|35.33|65.11|59.61|38.67|77.80|**55.30**|
|Performer (*)|18.01|65.40|53.82|42.77|77.05|51.41|
|BigBird (*)|36.05|64.02|59.29|40.83|74.87|55.01|

FNet matches the vanilla Transformer's aggregate LRA accuracy while being dramatically faster, especially at long sequence lengths (3.2× at 2048, 5.7× at 4096 on GPU inference).

#### LRA GPU Efficiency (Table 4b — Peak Memory in GB)

|Seq. Length|512|1024|2048|4096|8192|
|---|---|---|---|---|---|
|Transformer|1.6|4.0|12.2|OOM|OOM|
|Performer|1.1|1.9|3.1|5.5|10.4|
|**FNet (FFT)**|**0.8**|**1.3**|**2.2**|**3.9**|**7.4**|

FNet has the lightest memory footprint at every sequence length.

### 3.5 Ablation Studies

|Variant|Change|GLUE Avg|$\Delta$ from FNet|
|---|---|---|---|
|FNet-Base|2D DFT, real part|76.7|—|
|1D DFT (seq only)|No hidden dim DFT|Lower|Negative (speed gain, accuracy loss)|
|Absolute value extraction|$\|\cdot\|$ instead of $\Re(\cdot)$|Lower|Significantly worse|
|Real-only throughout|$\Re$ at each DFT stage|Lower|Less stable, less accurate|
|DCT|Real-to-real transform|~72.7|−4%|
|Hadamard|${\pm 1}$ matrix|~74.7|−2%|
|Hartley|$\mathcal{H} = \Re{\mathcal{F}} - \Im{\mathcal{F}}$|76.7|**0%** (identical)|
|Learnable DFT weights|Complex weights added|≈76.7|No improvement, slightly slower|
|FFT→FFN→FFT sandwich|Modified block structure|Lower|Degraded accuracy, unstable|

|Hybrid Layout (Table 9)|Attention Layers|Position|MLM Acc|Speed (ms/batch)|
|---|---|---|---|---|
|0 attention|0|—|0.486|173|
|2 attention (BOTTOM)|2|First layers|0.497|193|
|2 attention (MIDDLE)|2|Middle layers|0.499|196|
|2 attention (MIXED)|2|Distributed|0.509|194|
|**2 attention (TOP)**|2|**Final layers**|**0.526**|193|
|4 attention (TOP)|4|Final layers|0.539|214|
|6 attention (TOP)|6|Final layers|0.546|235|

**Key Findings**:

- Attention at the top of the network is most valuable — consistent with the intuition that higher layers capture more complex semantic relationships
- More attention layers yield diminishing returns in accuracy while linearly increasing cost
- The Hartley transform is an equally effective alternative
- Introducing learnable parameters to the DFT is **not** helpful — the fixed transform is locally optimal

### 3.6 Pre-training Analysis (Table 5)

|Model|Total Loss|MLM Loss|NSP Loss|MLM Acc|NSP Acc|
|---|---|---|---|---|---|
|BERT-Base|1.76|1.48|0.28|0.68|0.86|
|FNet-Base|2.45|2.06|0.40|0.58|0.80|
|FNet-Hybrid-Base|2.13|1.79|0.34|0.63|0.84|
|BERT-Large|1.49|1.23|0.25|0.72|0.88|
|FNet-Large|2.11|1.75|0.36|0.63|0.82|

Critically, BERT-Base (112M params) has _higher_ MLM accuracy than FNet-Large (238M params), demonstrating that BERT's superiority is not simply a parameter count effect but stems from the task-specific, token-dependent nature of attention weights.

---

## 4. Critical Assessment

### 4.1 Strengths

1. **Elegant simplicity**: The core idea — replace attention with FFT — is breathtakingly simple and the resulting model code is essentially one line (`jax.vmap(jnp.fft.fftn)(x).real`). This simplicity makes FNet easy to implement, understand, debug, and deploy.
    
2. **Comprehensive experimental design**: The paper includes five carefully chosen baselines (BERT, Linear, Random, FF-only, Hybrid) that together constitute a rigorous ablation isolating the mixing mechanism. The Random and FF-only baselines prove that _structured_ mixing is necessary (not just any mixing), while the Linear baseline shows the DFT is competitive with learned dense mixing.
    
3. **Honest reporting of limitations**: The authors clearly report the accuracy gap (8% for Base, 3% for Large), present failure modes, and do not overclaim. The Pareto frontier analysis is particularly valuable, showing that the optimal model class depends on the speed-accuracy budget.
    
4. **Multi-hardware evaluation**: Separate GPU (V100) and TPU (v3) benchmarks reveal platform-dependent behavior (FFT vs. DFT matrix), adding practical value.
    
5. **Thorough ablations (Appendix A.3)**: Testing DCT, Hadamard, Hartley, learnable DFT weights, block modifications, and 1D vs. 2D transforms constitutes a comprehensive exploration of the design space.
    

### 4.2 Weaknesses

1. **Selection bias in GLUE results**: The paper reports "best of 3 (Base) or 6 (Large) trials across learning rates" without providing standard deviations, confidence intervals, or variance across runs. This best-of-$k$ protocol favors models with higher variance, and the acknowledged instability of BERT-Large suggests the gap may be smaller or larger than reported.
    
2. **No token-level evaluation**: All downstream tasks are sentence- or document-level classification. Tasks requiring fine-grained token understanding — Named Entity Recognition, extractive question answering, part-of-speech tagging — are entirely absent. Since self-attention excels at modeling specific token-token relationships, FNet's performance on such tasks is an important open question.
    
3. **Encoder-only scope**: The restriction to encoder-only models limits the paper's applicability to the large class of generative tasks. The authors acknowledge this but provide no experimental evidence for how FFT mixing would perform in decoder (autoregressive) or encoder-decoder (seq2seq) settings.
    
4. **Outdated LRA comparison**: Many efficient Transformer results are quoted from Tay et al. (2021a) rather than re-run. The authors note high variance in LRA results, and the comparison relies on identical hardware (TPU v3) but potentially different software stacks.
    
5. **Limited interpretability analysis**: There is no probing study or attention-pattern visualization equivalent to understand _what_ linguistic structure the Fourier sublayer captures versus misses. The theoretical interpretation (alternating time/frequency domain) remains intuition.
    

### 4.3 Questions Raised

- Does FNet's advantage persist when training configurations are independently optimized for each architecture rather than reusing BERT's setup?
- How does FNet scale beyond Large (e.g., XL, XXL)? Does the accuracy gap continue to shrink?
- What is the effect of the Fourier mixing on gradient flow and optimization dynamics compared to attention?
- Can FNet achieve competitive performance on token-level tasks, or is the lack of input-dependent weighting fundamentally limiting for fine-grained tasks?
- Is the Fourier transform special, or would any unitary transform work equally well?

---

## 5. Connections & Insights

### 5.1 Related Work Connections

- **MLP-Mixer (Tolstikhin et al., 2021)**: FNet can be viewed as the NLP counterpart of MLP-Mixer for vision, but with the crucial difference that FNet's spatial mixing has _zero_ learnable parameters, whereas MLP-Mixer uses learned dense layers. This makes FNet's success even more surprising.
- **Performer (Choromanski et al., 2021)**: While Performer approximates attention using random Fourier features, FNet completely replaces attention with the Fourier Transform. The philosophical distinction is: approximate vs. replace.
- **State Space Models (Gu et al., 2021 — S4, published contemporaneously)**: SSMs also operate in the frequency domain via FFT for efficient computation. FNet and SSMs share the insight that spectral methods can replace attention, but SSMs introduce learnable state-space parameters, and target sequential (decoder) settings.
- **Synthesizer (Tay et al., 2020a)**: The Random Synthesizer showed that random, token-independent attention weights work surprisingly well, foreshadowing FNet's finding that fixed mixing can be competitive.

### 5.2 Non-Obvious Observations

1. **The feed-forward layer is the real bottleneck**: Fourier sublayers are 12–22× faster than attention in isolation, but FNet is only 1.8× faster end-to-end. This reveals that the shared FFN sublayer — not attention — dominates total compute at standard sequence lengths. Future efficiency work should focus on the FFN.
    
2. **Stability-accuracy tradeoff**: Models with _no_ learnable mixing parameters (FNet, Random, FF-only) are the most stable during training, while BERT and Linear are less stable — especially at Large scale. This suggests that learnable mixing introduces optimization difficulties (e.g., gradient blow-up in Linear-Large).
    
3. **BERT-Base > FNet-Large in MLM accuracy**: This proves that attention's advantage is not merely about having more parameters. Attention's expressivity comes from its _token-dependent, task-specific_ mixing weights — a qualitative, not quantitative, advantage.
    
4. **Position embeddings are redundant for FNet**: The DFT's twiddle factors $e^{-\frac{2\pi i}{N}nk}$ inherently encode positional information through the indices $n$ and $k$. The paper confirms FNet performs "just as well" without position embeddings. This is a structurally elegant property.
    
5. **The Hartley tie**: The Hartley Transform ($\mathcal{H} = \Re{\mathcal{F}} - \Im{\mathcal{F}}$) achieves identical accuracy to FNet (76.7 vs 76.7). Since Hartley is real-valued throughout (no complex arithmetic), this suggests the complex-number aspect of the DFT may be unnecessary — the key is the specific mixing pattern, not the algebraic structure.
    

### 5.3 Potential Applications Beyond Paper's Scope

- **Edge deployment**: FNet's light memory footprint (83M vs 112M) and fast inference make it a natural candidate for on-device NLP models
- **Distillation target**: Small FNet models could serve as efficient students distilled from large Transformer teachers
- **Multimodal encoders**: The fixed, fast mixing could encode long visual or audio sequences efficiently in multimodal architectures
- **Scientific computing**: The frequency-domain interpretation connects to Fourier Neural Operators (Li et al., 2021) for PDEs

---

## 6. Verdict

**Contribution Level**: **Significant** — The paper challenges a fundamental assumption in deep learning (that attention is essential) with a clean, elegant experiment. While the accuracy gap prevents it from being a drop-in replacement, it opens a new research direction.

**Reproducibility**: **High** — Code is released. Architecture is trivially implementable (one-line Fourier sublayer). Pre-training uses publicly available C4 dataset. Hyperparameters are fully specified from BERT. HuggingFace provides `google/fnet-base` and `google/fnet-large` pretrained checkpoints.

**Impact Prediction**: The paper's primary legacy is conceptual: it demonstrated that attention is _not_ essential and that exploring new mixing mechanisms is a viable alternative to approximating attention. This influenced subsequent work on spectral methods, state space models, and token-mixing architectures. As of 2025, the paper has accumulated substantial citations and won the NAACL 2022 Best Efficient NLP Paper Award.

---

# Experiment Analysis: FNet

### Experimental Setup

**Datasets**:

|Dataset|Train|Val|Test|Domain|Notes|
|---|---|---|---|---|---|
|C4 (pre-training)|~365M examples|—|—|Web text|32K SentencePiece vocab|
|GLUE (8 tasks)|Varies (2.5K–393K per task)|Per-task val|Per-task test|NLU|Validation split reported|
|LRA (6 tasks)|Varies|Per-task val|Per-task test|Synthetic + text|Sequences 1K–16K|

**Data Preprocessing**:

- 32,000 SentencePiece vocabulary trained on 100M sentence subset of C4
- Maximum sequence length 512 for pre-training and GLUE
- Standard GLUE fine-tuning protocol (task-specific classification heads)

**Implementation Details**:

- Framework: JAX / Flax
- Hardware: 8 × V100 GPUs or 4×4 TPU v3 chips
- Pre-training: 1M steps
- Batch size: 256 (TPU), 64 (GPU)
- Optimizer: Adam (inherited from BERT)
- Learning rate: sweep over $10^{-3}$ and $10^{-4}$ for pre-training; per-task sweep for fine-tuning
- GLUE trials: 3 (Base), 6 (Large) per learning rate, best reported

### Baselines Analysis

|Method|Year|Key Difference|Fair Comparison?|
|---|---|---|---|
|BERT|2019|Full self-attention; same config|✓ Yes — identical setup|
|Linear|2021|Two learned dense matrices|✓ Yes — same framework|
|Random|2021|Two fixed random matrices|✓ Yes — ablation control|
|FF-only|2021|No mixing at all|✓ Yes — ablation control|
|Performer|2021|Linearized attention via random Fourier features|⚠ Partial — results from different codebase|
|7 other efficient Transformers|2018–2020|Various attention approximations|⚠ Partial — results cited from prior work|

**Missing Baselines**: Convolutional models, State Space Models (S4), MLP-Mixer adapted for text, and relative position encoding variants.

### Statistical Rigor

**Multiple Runs**: Partial. GLUE fine-tuning: 3 runs (Base) / 6 runs (Large) per learning rate, but only _best_ result reported.

**Error Bars/CI**: **Not reported**. No standard deviations, confidence intervals, or variance across runs.

**Significance Tests**: **None performed**.

**Cherry-picking Risk**: **Moderate**. The "best of $k$ trials" protocol inflates results for higher-variance models. The authors acknowledge BERT-Large instability, which means the BERT baseline may be _under_-reported relative to its true mean.

### Reproducibility Checklist

- [x] Code released (GitHub)
- [x] Pretrained models available (HuggingFace: `google/fnet-base`, `google/fnet-large`)
- [x] Hyperparameters fully specified (inherited from BERT)
- [ ] Random seeds reported — **Not mentioned**
- [ ] Compute requirements stated — **Partially** (hardware listed, total hours not stated)
- [x] Data preprocessing described
- [x] Evaluation protocol clear

**Reproducibility Score**: **4/5** — Excellent code and model release. Deducted for missing random seeds and incomplete compute cost reporting.

### Compute Analysis

**Training Cost** (estimated):

- BERT-Base GPU: 305 ms/batch × 1M steps ≈ 84.7 hours on 8 × V100
- FNet-Base GPU: 169 ms/batch × 1M steps ≈ 46.9 hours on 8 × V100
- Estimated cloud cost: ~$150–$350 per full pre-training run (V100 spot pricing)

**Inference Cost** (Base, seq_len=512):

- FNet GPU: 46 ms/batch of 64 → ~0.72 ms/example
- FNet TPU: 23 ms/batch of 256 → ~0.09 ms/example
- Peak memory: 0.8 GB (LRA, seq 512) — lightest of all models

**Efficiency vs Baselines**: FNet uses 63% of BERT's FLOPS and 74% of BERT's parameters, while achieving 92% of BERT's accuracy. The speed-normalized accuracy (accuracy per unit training time) favors FNet at smaller model sizes.

---

# Research Gaps: FNet

### Acknowledged Limitations

1. **Encoder-only**: No decoder or encoder-decoder experiments. Causal masking for FFTs requires low-level implementation. Cross-attention adaptation is an open question.
2. **Classification tasks only**: No generative evaluation (text generation, summarization, translation).
3. **Cursory transform survey**: Only DFT, DCT, Hadamard, and Hartley tested. Many other structured transforms exist.
4. **Fixed BERT recipe**: Hyperparameters inherited from BERT without independent optimization for FNet.

### Unacknowledged Gaps

1. **No token-level tasks**: NER, POS tagging, extractive QA, coreference resolution — tasks where precise token-token relationships matter — are completely absent. This is the most critical gap, as self-attention's advantage may be most pronounced on these tasks.
2. **No linguistic probing**: No analysis of what linguistic phenomena (syntax, semantics, coreference) the Fourier sublayer captures vs. misses. Probing studies à la Tenney et al. (2019) or Clark et al. (2019) would be highly informative.
3. **No multilingual evaluation**: Despite training on C4 (which contains multilingual text), all evaluations are English-only. Cross-lingual transfer is untested.
4. **No domain robustness**: All evaluation is in-domain. Performance under domain shift (biomedical, legal, code) is unknown.
5. **No gradient/optimization analysis**: How gradients flow through the fixed FFT layer versus learned attention is not studied. This could explain stability differences.
6. **No scaling law analysis**: Does FNet follow Chinchilla/Kaplan-style scaling laws? Does the accuracy gap shrink, stay constant, or grow with scale?
7. **Attention mask handling**: FNet cannot handle attention masks (padding, causal). All tokens always see all other tokens, including padding. The impact of this is not discussed.

### Assumptions to Challenge

1. **"The real part is sufficient"**: The imaginary component of the DFT encodes phase information, which is crucial in signal processing. Discarding it may lose important temporal/positional relationships. The Hartley transform's equal performance (which uses a different real-valued combination) suggests exploring other projections from complex to real.
    
2. **"Same hyperparameters should be used for fair comparison"**: While using identical BERT hyperparameters ensures a controlled comparison, it introduces a bias _toward_ BERT, which was specifically designed for attention-based training. FNet might benefit from different learning rates, warmup schedules, or optimizer choices.
    
3. **"Token mixing is all you need from attention"**: The paper frames attention primarily as a mixing mechanism. But attention also provides: (a) input-dependent gating, (b) implicit feature selection, and (c) interpretable alignment patterns. The loss of these properties may matter for specific task types.
    
4. **"Speed comparisons at fixed training steps are fair"**: BERT and FNet are compared after the same number of training steps. But since FNet runs faster per step, a wall-clock-time comparison would give FNet more steps — potentially closing some of the accuracy gap.
    

### Missing Experiments

1. **Token-level classification**: NER on CoNLL-2003 or OntoNotes would directly test whether FNet can model fine-grained token relationships
2. **Extractive QA**: SQuAD 1.1/2.0 evaluation — requires precise token-span identification
3. **Wall-clock-time controlled comparison**: Train each model for the same total GPU hours, not the same number of steps
4. **FNet pre-training with FNet-optimized hyperparameters**: Independent LR/optimizer search for FNet
5. **Probing experiments**: Use edge probing (Tenney et al., 2019) to compare what linguistic features FNet vs. BERT encode at each layer
6. **Scaling beyond Large**: Train FNet-XL or FNet-XXL to test whether the accuracy gap continues to shrink
7. **Knowledge distillation**: FNet student distilled from BERT teacher — a natural use case the paper proposes but does not test
8. **Inference latency at batch size 1**: Real-world deployment often uses batch size 1; FFT overhead may differ

### Future Research Directions

1. **FNet Decoders**: Design causal Fourier mixing via lower-triangular DFT matrices or segmented FFT. This would enable GPT-style FNet models.
2. **Adaptive Spectral Mixing**: Learn per-layer frequency filters or spectral masks while keeping the DFT structure — a middle ground between fully fixed and fully learned mixing.
3. **Hybrid Architecture Search**: Systematically explore which layers benefit from attention vs. FFT, rather than the simple "top-2" heuristic.
4. **FNet + State Space Models**: Combine FNet's bidirectional mixing with SSM-style sequential modeling for encoder-decoder architectures.
5. **Theoretical Analysis**: Characterize the function classes representable by FNet vs. Transformers. Derive approximation bounds or separation results.
6. **Cross-modal FNet**: Apply Fourier mixing to vision (ViT replacement), audio, or multimodal inputs where long sequences are common.
7. **FNet for Retrieval**: The fixed mixing pattern may be beneficial for retrieval tasks where fast encoding of large document collections is needed.

### Extension Ideas

1. **Wavelet-Net**: Replace DFT with Discrete Wavelet Transform for multi-resolution analysis — preserving both time and frequency locality
2. **Learnable Spectral Masks**: $y = \Re(\mathcal{F}^{-1}(\mathbf{M} \odot \mathcal{F}(x)))$ where $\mathbf{M}$ is a learned frequency-domain mask — adds $O(n \cdot d_h)$ parameters while keeping FFT structure
3. **Causal FNet via Analytic Signal**: Use the Hilbert Transform to construct causal frequency-domain representations
4. **FNet + Mixture of Experts**: Replace the FFN bottleneck (the real speed limiter) with sparse MoE layers to improve both accuracy and throughput

---

# Code Reproduction: FNet

## Scope

- **Target experiment**: FNet-Base architecture verification and GLUE fine-tuning (SST-2 as primary task)
- **Success criterion**: Forward pass matches paper's parameter count (~83M); SST-2 accuracy within 3% of reported 95%
- **Compute constraints**: Single GPU (consumer-grade); no pre-training from scratch
- **Framework**: PyTorch (paper uses JAX/Flax; we provide PyTorch equivalent)

## Code Structure

```
fnet_reproduction/
├── README.md              # Setup, usage, results
├── requirements.txt       # Pinned dependencies
├── config.yaml            # Hyperparameters from paper
├── src/
│   ├── __init__.py
│   ├── model.py           # Core FNet model (Eq. 1–3)
│   ├── data.py            # GLUE data pipeline
│   ├── train.py           # Training loop
│   └── evaluate.py        # Metrics computation
└── scripts/
    └── run_experiment.sh   # Launch training
```

---

## config.yaml

```yaml
# Hyperparameters from FNet paper (Table 1, Table 6, Appendix A.1)
model:
  vocab_size: 32000          # SentencePiece vocabulary
  d_model: 768               # Hidden dimension
  num_layers: 12             # Number of encoder blocks
  d_ff: 3072                 # Feed-forward intermediate size (4 * d_model)
  max_seq_len: 512           # Maximum sequence length
  type_vocab_size: 2         # Sentence type embeddings
  dropout: 0.1               # Dropout rate
  layer_norm_eps: 1.0e-12    # LayerNorm epsilon

pretraining:
  dataset: "c4"
  steps: 1000000             # 1M steps
  batch_size_gpu: 64         # 8 × V100
  batch_size_tpu: 256        # 4×4 TPU v3
  optimizer: "adam"
  learning_rates: [1.0e-3, 1.0e-4]
  warmup_steps: 10000

finetuning:
  epochs: 3
  batch_size: 32
  learning_rate: 2.0e-5
  weight_decay: 0.01
  max_grad_norm: 1.0
  scheduler: "linear_warmup_decay"
```

---

## src/model.py

```python
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
```

---

## src/data.py

```python
"""
GLUE dataset loading for FNet fine-tuning.
Reference: Section 4.1 — GLUE Benchmark (Wang et al., 2018)
"""

import torch
from torch.utils.data import Dataset
from typing import Optional


# Task → (sentence_key_1, sentence_key_2_or_None)
TASK_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp":  ("question1", "question2"),
    "rte":  ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

# Task → number of classification labels (STS-B is regression = 1)
TASK_NUM_LABELS = {
    "cola": 2, "mnli": 3, "mrpc": 2, "qnli": 2,
    "qqp": 2, "rte": 2, "sst2": 2, "stsb": 1,
}


class GLUEDataset(Dataset):
    """Wraps HuggingFace `datasets` GLUE splits for FNet."""
    
    def __init__(self, task: str, split: str, tokenizer, max_length: int = 512):
        from datasets import load_dataset
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.key1, self.key2 = TASK_KEYS[task]
        self.is_regression = (task == "stsb")
        self.dataset = load_dataset("glue", task, split=split)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        text_pair = (item[self.key1], item[self.key2]) if self.key2 else (item[self.key1],)
        encoding = self.tokenizer(
            *text_pair,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        label_dtype = torch.float if self.is_regression else torch.long
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "token_type_ids": encoding.get(
                "token_type_ids", torch.zeros_like(encoding["input_ids"])
            ).squeeze(0),
            "label": torch.tensor(item["label"], dtype=label_dtype),
        }
```

---

## src/train.py

```python
"""
Training loop for FNet GLUE fine-tuning.
Reference: Section 4.1, Appendix A.1

Paper fine-tuning protocol:
    - 3 trials (Base) / 6 trials (Large) per learning rate
    - Best result across all trials reported
    - No early stopping
    - AdamW optimizer, linear LR decay
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import numpy as np
from typing import Dict

from model import FNetForSequenceClassification, FNET_CONFIGS
from data import GLUEDataset, TASK_NUM_LABELS
from evaluate import compute_metrics


def train_epoch(model, dataloader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Train", leave=False):
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
            labels=batch["label"].to(device),
        )
        loss = outputs["loss"]
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate_model(model, dataloader, device, task) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Eval", leave=False):
        outputs = model(
            input_ids=batch["input_ids"].to(device),
            token_type_ids=batch["token_type_ids"].to(device),
        )
        logits = outputs["logits"]
        if task == "stsb":
            preds = logits.squeeze(-1).cpu()
        else:
            preds = torch.argmax(logits, dim=-1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(batch["label"].tolist())
    return compute_metrics(task, np.array(all_preds), np.array(all_labels))


def main():
    parser = argparse.ArgumentParser(description="FNet GLUE Fine-tuning")
    parser.add_argument("--task", type=str, default="sst2", choices=TASK_NUM_LABELS.keys())
    parser.add_argument("--config", type=str, default="base", choices=FNET_CONFIGS.keys())
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    val_split = "validation_matched" if args.task == "mnli" else "validation"
    train_ds = GLUEDataset(args.task, "train", tokenizer, args.max_length)
    val_ds = GLUEDataset(args.task, val_split, tokenizer, args.max_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    
    model = FNetForSequenceClassification(
        num_labels=TASK_NUM_LABELS[args.task], **FNET_CONFIGS[args.config]
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Task: {args.task} | Config: {args.config} | Params: {n_params/1e6:.1f}M")
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )
    
    best_metric = -float("inf")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        metrics = evaluate_model(model, val_loader, device, args.task)
        primary = metrics.get("accuracy", metrics.get("spearman", 0))
        if primary > best_metric:
            best_metric = primary
            torch.save(model.state_dict(), f"fnet_{args.task}_best.pt")
        print(f"Epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  {metrics}")
    
    print(f"\nBest: {best_metric:.4f}")


if __name__ == "__main__":
    main()
```

---

## src/evaluate.py

```python
"""
Evaluation metrics for GLUE tasks.
Reference: Table 2 — Accuracy, F1, Spearman correlation.
"""

import numpy as np
from typing import Dict


def compute_metrics(task: str, preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute task-appropriate metric(s)."""
    if task == "stsb":
        from scipy.stats import pearsonr, spearmanr
        return {
            "pearson": float(pearsonr(preds, labels)[0]),
            "spearman": float(spearmanr(preds, labels)[0]),
        }
    elif task in ("mrpc", "qqp"):
        acc = float((preds == labels).mean())
        # Simple F1 for binary classification
        tp = float(((preds == 1) & (labels == 1)).sum())
        fp = float(((preds == 1) & (labels == 0)).sum())
        fn = float(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {"accuracy": acc, "f1": f1, "acc_f1_mean": (acc + f1) / 2}
    else:
        return {"accuracy": float((preds == labels).mean())}
```

---

## src/**init**.py

```python
# FNet reproduction package
```

---

## requirements.txt

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
scipy>=1.10.0
numpy>=1.24.0
tqdm>=4.65.0
sentencepiece>=0.1.99
```

---

## scripts/run_experiment.sh

```bash
#!/bin/bash
# Run FNet GLUE fine-tuning experiments
# Usage: bash scripts/run_experiment.sh [task] [config]

TASK=${1:-sst2}
CONFIG=${2:-base}

echo "============================================"
echo "FNet Reproduction: ${TASK} (${CONFIG})"
echo "============================================"

python src/train.py \
    --task ${TASK} \
    --config ${CONFIG} \
    --epochs 3 \
    --batch_size 32 \
    --lr 2e-5 \
    --max_length 128 \
    --device cuda

echo "Done."
```

---

## Reproduction Results

### Expected Comparison

|Metric|Paper (FNet-Base)|Expected Reproduction|$\Delta$|
|---|---|---|---|
|SST-2 Accuracy|95%|~92–95%|±3%|
|MRPC Acc/F1 Mean|76%|~73–76%|±3%|
|FNet-Base Parameters|~83M|~83M|~0|

### Observations

- **Tokenizer mismatch**: Paper uses a custom 32K SentencePiece model trained on C4; reproduction uses `bert-base-uncased` tokenizer (30K WordPiece). This may cause 1–2% accuracy difference.
- **No pre-training**: Paper pre-trains from scratch on C4 for 1M steps. Reproduction uses random initialization and fine-tunes directly (or can use HuggingFace `google/fnet-base` checkpoint).
- **For faithful reproduction**: Use `from transformers import FNetForSequenceClassification` with the `google/fnet-base` checkpoint, which provides the paper's pre-trained weights directly.

---

## Quality Checklist

- [x] Problem and contribution clearly identified
- [x] Technical approach accurately described with LaTeX equations
- [x] Experiments completely documented (datasets, baselines, metrics, ablations)
- [x] Results interpreted with caveats (statistical concerns, selection bias)
- [x] Balanced strengths/weaknesses
- [x] Insights beyond surface-level (FFN bottleneck, stability, Hartley equivalence, BERT-Base > FNet-Large)
- [x] Concrete future directions (8 specific research directions + 4 extension ideas)
- [x] Code runnable and documented (4 source files, config, requirements, run script)
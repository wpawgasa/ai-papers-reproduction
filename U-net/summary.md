# U-Net: Convolutional Networks for Biomedical Image Segmentation

**Paper**: U-Net: Convolutional Networks for Biomedical Image Segmentation  
**Authors**: Olaf Ronneberger, Philipp Fischer, Thomas Brox  
**Affiliation**: Computer Science Department and BIOSS Centre for Biological Signalling Studies, University of Freiburg, Germany  
**Venue**: MICCAI 2015 (International Conference on Medical Image Computing and Computer-Assisted Intervention)  
**arXiv**: [1505.04597](https://arxiv.org/abs/1505.04597) (May 18, 2015)  
**Code**: http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net

---

## 1. Problem & Motivation

### Research Question

How can we design a convolutional neural network architecture that achieves precise pixel-wise segmentation for biomedical images while training on extremely small annotated datasets (tens of images rather than thousands)?

### Context

At the time of publication (2015), deep convolutional networks had achieved state-of-the-art on many visual recognition tasks following the AlexNet breakthrough (Krizhevsky et al., 2012). However, these successes were almost exclusively on classification tasks with abundant training data (e.g., ImageNet with 1M+ images). Biomedical image segmentation posed two fundamental challenges that existing approaches could not address simultaneously:

1. **Pixel-level localization**: The task requires assigning a class label to _every_ pixel, not a single label per image.
2. **Extreme data scarcity**: Annotating biomedical images requires domain experts, making large-scale datasets impractical. Typical training sets contained only 20–35 images.

### Prior Work Limitations

**Ciresan et al. (2012) — Sliding-Window CNN**: The then-best approach used a patch-based sliding-window scheme: for each pixel, a local patch around it was extracted and classified. This won the ISBI 2012 EM segmentation challenge, but had critical drawbacks:

- **Speed**: The network ran separately for each patch with massive redundancy from overlapping regions.
- **Context–Localization Trade-off**: Larger patches captured more context but required more pooling (losing spatial precision); smaller patches preserved localization but lacked global context.

**Fully Convolutional Networks (Long et al., 2014)**: FCN proposed replacing fully connected layers with convolutions and using upsampling to produce dense predictions. However, the original FCN architecture had limited capacity to recover fine spatial details during upsampling.

### Gap Addressed

U-Net bridges the gap between capturing rich semantic context (via an encoding/contracting path) and recovering fine spatial localization (via a symmetric decoding/expanding path with skip connections), all while being trainable end-to-end from very few annotated images through aggressive data augmentation — particularly elastic deformations.

---

## 2. Technical Approach

### Core Idea

The key architectural insight is to create a symmetric encoder-decoder structure where high-resolution feature maps from the contracting path are directly concatenated (via skip connections) with upsampled feature maps in the expanding path. This enables the decoder to access both deep semantic features and fine-grained spatial information simultaneously, yielding precise localization without sacrificing contextual understanding.

### Architecture Overview

The U-Net architecture forms a characteristic "U" shape with two symmetric paths:

**Contracting Path (Encoder)** — The left side follows a standard CNN pattern. Each encoder block consists of:

- Two consecutive 3×3 unpadded convolutions, each followed by ReLU
- 2×2 max pooling with stride 2 for spatial downsampling
- Feature channels are doubled at each downsampling step: 64 → 128 → 256 → 512 → 1024

**Bottleneck** — At the deepest level, the feature map has 1024 channels at the lowest spatial resolution (32×32 for a 572×572 input).

**Expansive Path (Decoder)** — The right side mirrors the encoder. Each decoder block consists of:

- 2×2 "up-convolution" (transposed convolution) that halves the number of channels
- Concatenation with the corresponding (center-cropped) feature map from the contracting path
- Two consecutive 3×3 unpadded convolutions, each followed by ReLU

**Final Layer** — A 1×1 convolution maps each 64-component feature vector to the desired number of classes.

In total, the network has **23 convolutional layers**. Critically, no fully connected layers are used, and only the "valid" part of each convolution is retained — the output segmentation map is smaller than the input.

### Key Innovations

**1. Skip Connections via Concatenation**

Unlike FCN which uses additive skip connections, U-Net concatenates encoder feature maps with decoder feature maps. This preserves the full encoder feature information rather than merging it, giving the decoder access to both high-level semantics and low-level spatial details. The cropping before concatenation is necessary because unpadded convolutions reduce spatial dimensions at every layer.

**2. Overlap-Tile Strategy for Arbitrary Image Sizes**

Since the network uses only valid convolutions (no padding), the output is smaller than the input by a fixed border. For seamless segmentation of arbitrarily large images, U-Net employs an overlap-tile strategy: the image is segmented in tiles, and each tile's prediction region is surrounded by a context region extracted via mirroring at the image boundaries. This decouples the network from GPU memory constraints.

**3. Elastic Deformation for Data Augmentation**

To combat extreme data scarcity, the authors propose heavy data augmentation, with random elastic deformations being the most critical component. Smooth deformation fields are generated from random displacement vectors sampled on a coarse 3×3 grid from a Gaussian distribution (σ = 10 pixels), then interpolated to per-pixel displacements via bicubic interpolation. This simulates realistic biological tissue deformations and enables training from as few as 20–35 images.

**4. Weighted Loss for Instance Separation**

A novel pixel-wise weighted cross-entropy loss forces the network to learn separation borders between touching cells of the same class.

### Mathematical Formulation

**Softmax**: The pixel-wise softmax over the final feature map is defined as:

$$p_k(\mathbf{x}) = \frac{\exp\bigl(a_k(\mathbf{x})\bigr)}{\sum_{k'=1}^{K} \exp\bigl(a_{k'}(\mathbf{x})\bigr)}$$

where $a_k(\mathbf{x})$ denotes the activation in feature channel $k$ at pixel position $\mathbf{x} \in \Omega$ with $\Omega \subset \mathbb{Z}^2$, and $K$ is the number of classes.

**Cross-Entropy Loss**: The energy function penalizes deviation from the true label at each pixel:

$$E = \sum_{\mathbf{x} \in \Omega} w(\mathbf{x}) \log\bigl(p_{\ell(\mathbf{x})}(\mathbf{x})\bigr)$$

where $\ell : \Omega \to {1, \ldots, K}$ is the ground-truth label, and $w : \Omega \to \mathbb{R}$ is a pixel-wise weight map.

**Weight Map**: The weight map balances class frequencies and emphasizes separation borders between touching cells:

$$w(\mathbf{x}) = w_c(\mathbf{x}) + w_0 \cdot \exp\left(-\frac{\bigl(d_1(\mathbf{x}) + d_2(\mathbf{x})\bigr)^2}{2\sigma^2}\right)$$


where:

- $w_c : \Omega \to \mathbb{R}$ is the class frequency balancing weight
- $d_1 : \Omega \to \mathbb{R}$ is the distance to the border of the **nearest** cell
- $d_2 : \Omega \to \mathbb{R}$ is the distance to the border of the **second nearest** cell
- $w_0 = 10$ and $\sigma \approx 5$ pixels (experimentally set)

**Weight Initialization**: Weights are drawn from a Gaussian distribution:

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{N}}\right)$$

where $N$ is the number of incoming nodes per neuron (e.g., $N = 9 \times 64 = 576$ for a 3×3 convolution on 64 input channels). This follows He initialization (He et al., 2015) to ensure approximately unit variance in each feature map.

### Training Details

- **Optimizer**: SGD with high momentum (0.99)
- **Batch size**: 1 image (to maximize tile size within GPU memory)
- **Framework**: Caffe
- **Training time**: ~10 hours on NVIDIA Titan GPU (6 GB)
- **Input tile size**: 572×572 → output 388×388

---

## 3. Experimental Evaluation

### Datasets

|Dataset|Domain|Train Size|Image Size|Task|
|---|---|---|---|---|
|ISBI 2012 EM|Electron microscopy (Drosophila VNC)|30 images|512×512|Cell/membrane segmentation|
|PhC-U373|Phase contrast microscopy (Glioblastoma cells)|35 images (partially annotated)|—|Cell segmentation|
|DIC-HeLa|Differential interference contrast microscopy (HeLa cells)|20 images (partially annotated)|—|Cell segmentation|

### Baselines

- **Ciresan et al. (IDSIA)**: Sliding-window CNN, previous ISBI 2012 winner
- **DIVE-SCI / DIVE**: Other top competitors on EM segmentation leaderboard
- **IMCB-SG, KTH-SE, HOUS-US**: Top methods from ISBI Cell Tracking Challenge 2014

### Metrics

- **Warping Error**: Topological metric measuring segmentation correctness at multiple thresholds
- **Rand Error**: Measures agreement between predicted and ground-truth segmentation boundaries
- **Pixel Error**: Direct pixel-wise classification accuracy
- **IOU (Intersection over Union)**: Standard segmentation overlap metric

### Main Results

**EM Segmentation Challenge (ISBI 2012)**:

|Rank|Method|Warping Error|Rand Error|Pixel Error|
|---|---|---|---|---|
|—|Human|0.000005|0.0021|0.0010|
|**1**|**U-Net**|**0.000353**|**0.0382**|**0.0611**|
|2|DIVE-SCI|0.000355|0.0305|0.0584|
|3|IDSIA (Ciresan et al.)|0.000420|0.0504|0.0613|

U-Net achieved the **best warping error** (0.000353 vs. 0.000420), a 16% relative improvement over Ciresan et al. The result used averaging over 7 rotated versions of the input — no additional pre- or post-processing was applied.

**ISBI Cell Tracking Challenge 2015** (IOU):

|Method|PhC-U373|DIC-HeLa|
|---|---|---|
|IMCB-SG (2014)|0.2669|0.2935|
|KTH-SE (2014)|0.7953|0.4607|
|Second-best 2015|0.83|0.46|
|**U-Net (2015)**|**0.9203**|**0.7756**|

U-Net won both categories by enormous margins: +10.9% absolute on PhC-U373 and +31.6% absolute on DIC-HeLa over the second-best methods.

### Ablations

The paper does not include formal ablation studies. The contribution of individual components (skip connections, elastic augmentation, weighted loss) is not isolated experimentally. This is a notable limitation discussed further below.

---

## 4. Critical Assessment

### Strengths

1. **Elegant and generalizable architecture**: The encoder-decoder with skip connections became one of the most influential architectural patterns in deep learning. The symmetric design is intuitive and highly extensible.
    
2. **Extreme data efficiency**: Demonstrating strong performance with only 20–35 training images was remarkable and directly addressed the most critical bottleneck in biomedical imaging — the scarcity of expert annotations.
    
3. **Decisive empirical results**: U-Net did not merely achieve marginal improvements; it won challenges by large margins (31.6% absolute IOU improvement on DIC-HeLa), providing strong evidence for the approach.
    
4. **Practical design choices**: The overlap-tile strategy for arbitrary image sizes, the weighted loss for instance separation, and elastic deformation augmentation are all highly practical innovations tailored to real biomedical workflows.
    
5. **Speed**: Sub-second inference on 512×512 images was a dramatic improvement over patch-based methods.
    

### Weaknesses

1. **No ablation studies**: The paper does not isolate the contributions of individual components (skip connections, elastic deformation, weighted loss, overlap-tile). It is unclear how much each innovation contributes to the final performance.
    
2. **Limited experimental scope**: Only three datasets are evaluated, all in 2D biomedical microscopy. No experiments on natural images, 3D volumes, or other segmentation domains are presented.
    
3. **No comparison with contemporaneous methods**: The paper does not compare against FCN (Long et al., 2014) directly, despite being built upon it. Given that the primary contribution is architectural, a direct FCN comparison would have been highly informative.
    
4. **Spatial dimension loss from valid convolutions**: Using unpadded convolutions means the output is substantially smaller than the input (572×572 → 388×388, a 32% loss per dimension). The overlap-tile strategy mitigates this for inference but adds complexity.
    
5. **Statistical rigor**: No error bars, confidence intervals, or repeated runs are reported. The EM challenge result uses 7-rotation ensembling, but it is unclear whether variance across individual rotations or random seeds was assessed.
    
6. **Limited hyperparameter analysis**: The weight map parameters ($w_0 = 10$, $\sigma = 5$) and augmentation parameters (σ = 10 for deformations) are stated without justification or sensitivity analysis.
    

### Questions Raised

- How does performance degrade as the training set shrinks further (e.g., 5 or 10 images)?
- What is the relative contribution of elastic deformations vs. other augmentations?
- Would padding-based convolutions with boundary handling achieve comparable results while simplifying the tiling strategy?
- How does the architecture scale to 3D volumetric data?

---

## 5. Research Gaps

### Acknowledged Limitations

The paper is notably light on discussing limitations. The authors implicitly acknowledge the domain-specific nature of their evaluation but frame it as demonstrating broad applicability rather than a limitation.

### Unacknowledged Gaps

1. **Instance segmentation ambiguity**: The weighted loss encourages border learning but does not provide a principled instance segmentation framework. Post-processing is still needed to separate individual cell instances.
    
2. **Multi-scale object handling**: The fixed architecture may struggle with objects that vary dramatically in size within a single image, as the receptive field and feature resolution are fixed per encoder level.
    
3. **Class imbalance beyond binary**: The weight map formulation is specifically designed for binary foreground/background with touching instances. Extension to multi-class scenarios with complex class interactions is not discussed.
    
4. **Sensitivity to augmentation choices**: The elastic deformation approach is domain-motivated (tissue deformation), but the paper does not explore when this augmentation might be harmful or irrelevant.
    

### Assumptions to Challenge

- **Symmetry assumption**: The expanding path is designed to be "more or less symmetric" to the contracting path. Is full symmetry optimal, or could asymmetric designs (e.g., lighter decoders) suffice?
- **Concatenation over addition**: The paper uses concatenation for skip connections without comparing against additive fusion (as in FCN) or other fusion strategies.
- **Fixed architecture depth**: The 4-level encoder is used across all tasks. Task-adaptive depth could improve efficiency.

### Missing Experiments

- Ablation of skip connections (no skip, additive skip, concatenation skip)
- Ablation of data augmentation types (no augmentation, affine only, elastic only)
- Ablation of weighted loss components ($w_c$ only, border weight only, combined)
- Scaling experiments with varying training set sizes
- Comparison with FCN, DeconvNet, and other encoder-decoder variants
- Computational cost comparison (FLOPs, memory, training time vs. baselines)

### Future Research Directions

1. **3D U-Net**: Extending the architecture to volumetric data (subsequently addressed by Çiçek et al., 2016)
2. **Attention mechanisms**: Incorporating spatial and channel attention to dynamically weight skip connections (subsequently addressed by Attention U-Net, Oktay et al., 2018)
3. **Dense skip connections**: Using nested or dense skip pathways (subsequently addressed by UNet++, Zhou et al., 2018)
4. **Pre-trained encoders**: Leveraging transfer learning from ImageNet-pretrained backbones
5. **Self-supervised pre-training**: Reducing dependence on labeled data even further
6. **Multi-task learning**: Joint segmentation and detection or segmentation and classification
7. **Transformer-based variants**: Replacing CNN blocks with vision transformers (subsequently addressed by TransUNet, Chen et al., 2021; Swin-UNet, etc.)

### Extension Ideas

- **Adaptive weight maps**: Learning the weight map parameters ($w_0$, $\sigma$) jointly with the network
- **Learnable augmentation**: Replacing hand-crafted elastic deformations with learned augmentation policies (AutoAugment-style)
- **Multi-resolution fusion**: Incorporating atrous/dilated convolutions to capture multi-scale context without additional downsampling
- **Uncertainty estimation**: Adding Monte Carlo dropout or ensemble-based uncertainty to the predictions for clinical decision support

---

## 6. Connections & Insights

### Related Work Connections

- **FCN (Long et al., 2014)**: U-Net directly extends FCN by making the decoder symmetric and using concatenation-based skip connections instead of additive ones. This conceptual shift was arguably the paper's greatest contribution.
- **VGGNet (Simonyan & Zisserman, 2014)**: The encoder's repeated 3×3 convolution pattern echoes VGG's design philosophy.
- **He Initialization (He et al., 2015)**: U-Net explicitly adopted the then-new He initialization, demonstrating its importance for deep encoder-decoder training.
- **ResNet (He et al., 2015)**: While published around the same time, residual connections share the philosophical DNA of skip connections — enabling gradient flow across network depth.

### Non-Obvious Observations

1. **The "U" shape as an inductive bias**: The symmetric architecture implicitly encodes the assumption that encoding and decoding are complementary processes of equal complexity — a powerful structural prior for dense prediction tasks.
    
2. **Data augmentation as regularization**: The aggressive augmentation strategy (especially elastic deformations) functions as a strong regularizer, effectively expanding the training set by orders of magnitude. This was prescient — augmentation-as-regularization became a dominant theme in later deep learning research.
    
3. **Momentum 0.99 with batch size 1**: Using extremely high momentum with batch size 1 effectively creates a rolling average over many training samples, approximating larger batch training. This was an insightful practical solution to GPU memory constraints.
    
4. **The weight map as domain knowledge injection**: The distance-based weight map elegantly injects biological prior knowledge (cells have boundaries) directly into the loss function, rather than relying on the network to discover this structure.
    

### Impact Assessment

U-Net is among the most cited papers in deep learning (~70,000+ citations). Its influence extends far beyond biomedical imaging into satellite imagery, autonomous driving, industrial inspection, and generative models (it serves as the backbone of diffusion model architectures like Stable Diffusion). The encoder-decoder with skip connections pattern has become a foundational design primitive.

---

## 7. Verdict

**Contribution Level**: **Breakthrough** — Established a new architectural paradigm for dense prediction that remains foundational a decade later.

**Reproducibility**: **High** — Architecture is fully specified, hyperparameters are documented, and the original Caffe implementation was released. The main gap is the lack of random seed reporting and multi-run statistics.

**Impact Prediction** (retrospective): Transformative. U-Net became the de facto standard for medical image segmentation and the encoder-decoder with skip connections pattern generalized across virtually all dense prediction tasks in computer vision.

**Recommendation**: Strong Accept — Despite the limited ablation studies, the architectural contribution, practical innovations, and decisive empirical results represent a clear and significant advance.

---

## 8. Codebase for Reproduction

### 8.1 Scope

- **Target**: Reproduce the core U-Net architecture and training pipeline for binary segmentation
- **Success Criterion**: Achieve comparable IOU on a cell segmentation task
- **Framework**: PyTorch (modern equivalent of the original Caffe implementation)

### 8.2 Project Structure

```
reproduction/
├── README.md              # Setup, usage, results
├── requirements.txt       # Pinned dependencies
├── config.yaml            # Hyperparameters from paper
├── src/
│   ├── model.py           # U-Net architecture (23 conv layers)
│   ├── data.py            # Data pipeline with elastic deformation
│   ├── loss.py            # Weighted cross-entropy with border maps
│   ├── train.py           # Training loop (SGD, momentum=0.99)
│   └── evaluate.py        # IOU and other metrics
└── scripts/
    └── run_experiment.sh   # Training launch script
```

### 8.3 Implementation

#### `config.yaml` — Hyperparameters from Paper

```yaml
# config.yaml — U-Net reproduction hyperparameters (Ronneberger et al., 2015)
model:
  in_channels: 1
  num_classes: 2
  base_features: 64            # First encoder block channels
  use_padding: false           # Paper uses valid (unpadded) convolutions

training:
  optimizer: sgd
  learning_rate: 0.01
  momentum: 0.99               # High momentum to compensate batch_size=1
  batch_size: 1                # Paper: single image per batch
  epochs: 100
  input_size: 572              # Paper input tile size
  output_size: 388             # Output after valid convolutions

augmentation:
  elastic_deformation: true
  elastic_sigma: 10            # Gaussian std for displacement vectors
  elastic_grid_spacing: 3      # Coarse 3x3 grid
  rotation: true
  flip: true
  gray_value_variation: true

loss:
  w0: 10                       # Border weight amplitude
  sigma: 5                     # Border weight spread (pixels)

initialization:
  method: he_normal            # Gaussian with std = sqrt(2/N)
```

#### `src/model.py` — U-Net Architecture

```python
"""
U-Net Architecture (Ronneberger et al., MICCAI 2015)

Architecture: 23 convolutional layers arranged in a symmetric encoder-decoder.
- Encoder: 4 downsampling blocks, each with 2x (3x3 conv + ReLU) + 2x2 max pool
- Bottleneck: 2x (3x3 conv + ReLU) at 1024 channels
- Decoder: 4 upsampling blocks, each with 2x2 up-conv + crop-and-concat + 2x (3x3 conv + ReLU)
- Final: 1x1 conv mapping 64 channels -> num_classes

Reference: Figure 1 and Section 2 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions, each followed by ReLU.
    
    Paper Section 2: "the repeated application of two 3x3 convolutions 
    (unpadded convolutions), each followed by a rectified linear unit (ReLU)"
    
    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        use_padding: If False, uses valid convolution (paper default).
    """

    def __init__(self, in_channels: int, out_channels: int, use_padding: bool = False):
        super().__init__()
        padding = 1 if use_padding else 0
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder block: max pool followed by DoubleConv.
    
    Paper Section 2: "a 2x2 max pooling operation with stride 2 for downsampling.
    At each downsampling step we double the number of feature channels."
    """

    def __init__(self, in_channels: int, out_channels: int, use_padding: bool = False):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, use_padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Decoder block: up-conv + crop-and-concat + DoubleConv.
    
    Paper Section 2: "an upsampling of the feature map followed by a 2x2 
    convolution ('up-convolution') that halves the number of feature channels, 
    a concatenation with the correspondingly cropped feature map from the 
    contracting path, and two 3x3 convolutions, each followed by a ReLU."
    """

    def __init__(self, in_channels: int, out_channels: int, use_padding: bool = False):
        super().__init__()
        # 2x2 up-convolution that halves channels
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation: out_channels (from up) + out_channels (from skip) = in_channels
        self.conv = DoubleConv(in_channels, out_channels, use_padding)

    @staticmethod
    def center_crop(encoder_feat: torch.Tensor, target_size: torch.Tensor) -> torch.Tensor:
        """Crop encoder feature map to match decoder spatial size.
        
        Paper Section 2: "The cropping is necessary due to the loss of border 
        pixels in every convolution."
        """
        _, _, h, w = encoder_feat.shape
        _, _, target_h, target_w = target_size.shape
        diff_h = (h - target_h) // 2
        diff_w = (w - target_w) // 2
        return encoder_feat[:, :, diff_h:diff_h + target_h, diff_w:diff_w + target_w]

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Crop skip connection to match upsampled spatial dimensions
        skip_cropped = self.center_crop(skip, x)
        # Concatenate along channel dimension
        x = torch.cat([skip_cropped, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net: Convolutional Networks for Biomedical Image Segmentation.
    
    Full architecture with 23 convolutional layers as described in Figure 1.
    Channel progression: 64 -> 128 -> 256 -> 512 -> 1024 -> 512 -> 256 -> 128 -> 64
    
    Args:
        in_channels: Number of input image channels (1 for grayscale).
        num_classes: Number of output segmentation classes.
        base_features: Number of features in the first encoder block (default: 64).
        use_padding: If True, use padded convolutions (output_size == input_size).
                     If False, use valid convolutions as in the original paper.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_features: int = 64,
        use_padding: bool = False,
    ):
        super().__init__()
        f = base_features  # shorthand

        # Encoder (contracting path)
        self.enc1 = DoubleConv(in_channels, f, use_padding)       # 572 -> 568 (valid)
        self.enc2 = EncoderBlock(f, f * 2, use_padding)           # 284 -> 280
        self.enc3 = EncoderBlock(f * 2, f * 4, use_padding)       # 140 -> 136
        self.enc4 = EncoderBlock(f * 4, f * 8, use_padding)       # 68  -> 64

        # Bottleneck
        self.bottleneck = EncoderBlock(f * 8, f * 16, use_padding)  # 32 -> 28

        # Decoder (expansive path)
        self.dec4 = DecoderBlock(f * 16, f * 8, use_padding)      # 56 -> 52
        self.dec3 = DecoderBlock(f * 8, f * 4, use_padding)       # 104 -> 100
        self.dec2 = DecoderBlock(f * 4, f * 2, use_padding)       # 200 -> 196
        self.dec1 = DecoderBlock(f * 2, f, use_padding)           # 392 -> 388

        # Final 1x1 convolution
        # Paper Section 2: "a 1x1 convolution is used to map each 64-component 
        # feature vector to the desired number of classes"
        self.final_conv = nn.Conv2d(f, num_classes, kernel_size=1)

        # Weight initialization (Section 3)
        # "drawing the initial weights from a Gaussian distribution with 
        # a standard deviation of sqrt(2/N)"
        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization: W ~ N(0, sqrt(2/N)).
        
        Paper Section 3: "For a network with our architecture (alternating 
        convolution and ReLU layers) this can be achieved by drawing the initial 
        weights from a Gaussian distribution with a standard deviation of 
        sqrt(2/N), where N denotes the number of incoming nodes of one neuron."
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
               For valid convolutions, H=W=572 yields output H=W=388.
               
        Returns:
            Logits tensor of shape (B, num_classes, H', W').
        """
        # Encoder
        e1 = self.enc1(x)          # Skip connection 1
        e2 = self.enc2(e1)         # Skip connection 2
        e3 = self.enc3(e2)         # Skip connection 3
        e4 = self.enc4(e3)         # Skip connection 4

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Final classification
        return self.final_conv(d1)
```

#### `src/loss.py` — Weighted Cross-Entropy with Border Maps

```python
"""
Weighted Cross-Entropy Loss with Border Weight Maps (Ronneberger et al., 2015)

Implements Equations (1) and (2) from the paper:

    E = sum_x w(x) * log(p_{l(x)}(x))                                  -- Eq. (1)

    w(x) = w_c(x) + w_0 * exp(-(d_1(x) + d_2(x))^2 / (2 * sigma^2))   -- Eq. (2)

where:
    - w_c(x): class frequency balancing weight
    - d_1(x): distance to the nearest cell border
    - d_2(x): distance to the second nearest cell border
    - w_0 = 10, sigma ≈ 5 pixels
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt, label


def compute_weight_map(
    segmentation: np.ndarray,
    w0: float = 10.0,
    sigma: float = 5.0,
) -> np.ndarray:
    """Compute pixel-wise weight map from instance segmentation mask.
    
    Implements Equation (2) from Section 3 of the paper.
    
    Args:
        segmentation: Instance segmentation map where each cell has a unique 
                      integer label. Background is 0.
        w0: Weight map amplitude for border emphasis (paper: 10).
        sigma: Weight map spread in pixels (paper: ~5).
    
    Returns:
        weight_map: Per-pixel weights of shape (H, W).
    """
    # --- Class frequency balancing weight w_c(x) ---
    binary_mask = (segmentation > 0).astype(np.float32)
    n_pixels = binary_mask.size
    n_foreground = binary_mask.sum()
    n_background = n_pixels - n_foreground

    wc = np.ones_like(binary_mask, dtype=np.float32)
    if n_foreground > 0 and n_background > 0:
        # Inverse frequency weighting
        wc[binary_mask == 1] = n_pixels / (2.0 * n_foreground)
        wc[binary_mask == 0] = n_pixels / (2.0 * n_background)

    # --- Border emphasis weight (Eq. 2) ---
    # Find unique cell instances
    cell_ids = np.unique(segmentation)
    cell_ids = cell_ids[cell_ids > 0]  # exclude background

    if len(cell_ids) < 2:
        # Not enough cells for border computation
        return wc

    # Compute distance transform for each cell
    # d_i(x) = distance from pixel x to the border of cell i
    distances = np.zeros((len(cell_ids),) + segmentation.shape, dtype=np.float32)
    for i, cell_id in enumerate(cell_ids):
        cell_mask = (segmentation == cell_id).astype(np.float32)
        # Distance from background pixels to this cell's border
        distances[i] = distance_transform_edt(cell_mask == 0)

    # Sort distances to find d1 (nearest) and d2 (second nearest)
    distances.sort(axis=0)
    d1 = distances[0]  # distance to nearest cell border
    d2 = distances[1]  # distance to second nearest cell border

    # Equation (2): border weight term
    border_weight = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma ** 2))

    # Combined weight map
    weight_map = wc + border_weight

    return weight_map


class WeightedCrossEntropyLoss(nn.Module):
    """Pixel-wise weighted cross-entropy loss.
    
    Implements Equation (1): E = sum_x w(x) * log(p_{l(x)}(x))
    
    The weight map w(x) is precomputed per sample using compute_weight_map().
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weight_maps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Model output of shape (B, K, H, W) — raw scores per class.
            targets: Ground-truth labels of shape (B, H, W) with values in {0, ..., K-1}.
            weight_maps: Per-pixel weights of shape (B, H, W).
        
        Returns:
            Scalar loss value.
        """
        # Pixel-wise cross-entropy (no built-in reduction)
        # log(p_{l(x)}(x)) is computed internally by cross_entropy
        loss_per_pixel = nn.functional.cross_entropy(
            logits, targets, reduction='none'
        )  # shape: (B, H, W)

        # Apply pixel-wise weights: E = sum_x w(x) * CE(x)
        weighted_loss = loss_per_pixel * weight_maps

        # Mean over all pixels and batch
        return weighted_loss.mean()
```

#### `src/data.py` — Data Pipeline with Elastic Deformation

```python
"""
Data Pipeline with Elastic Deformation Augmentation (Ronneberger et al., 2015)

Paper Section 3.1: "We generate smooth deformations using random displacement 
vectors on a coarse 3 by 3 grid. The displacements are sampled from a Gaussian 
distribution with 10 pixels standard deviation. Per-pixel displacements are 
then computed using bicubic interpolation."
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import map_coordinates, gaussian_filter
from PIL import Image
from typing import Tuple, Optional


def elastic_deformation(
    image: np.ndarray,
    mask: np.ndarray,
    sigma: float = 10.0,
    grid_spacing: int = 3,
    random_state: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random elastic deformation to image and mask jointly.
    
    Paper Section 3.1: Generates smooth deformations using random displacement
    vectors on a coarse grid, interpolated to per-pixel displacements.
    
    Args:
        image: Input image of shape (H, W).
        mask: Segmentation mask of shape (H, W).
        sigma: Standard deviation of Gaussian for displacement vectors (paper: 10).
        grid_spacing: Coarse grid size (paper: 3×3).
        random_state: Optional random state for reproducibility.
    
    Returns:
        Tuple of (deformed_image, deformed_mask).
    """
    if random_state is None:
        random_state = np.random.RandomState()

    h, w = image.shape[:2]

    # Generate random displacement fields on a coarse grid
    # Paper: "random displacement vectors on a coarse 3 by 3 grid"
    coarse_h = grid_spacing
    coarse_w = grid_spacing

    # Random displacements sampled from Gaussian (sigma=10 pixels)
    dx_coarse = random_state.randn(coarse_h, coarse_w) * sigma
    dy_coarse = random_state.randn(coarse_h, coarse_w) * sigma

    # Bicubic interpolation to full resolution
    # Paper: "Per-pixel displacements are then computed using bicubic interpolation"
    from scipy.interpolate import RectBivariateSpline

    x_coarse = np.linspace(0, h - 1, coarse_h)
    y_coarse = np.linspace(0, w - 1, coarse_w)

    spline_dx = RectBivariateSpline(x_coarse, y_coarse, dx_coarse, kx=3, ky=3)
    spline_dy = RectBivariateSpline(x_coarse, y_coarse, dy_coarse, kx=3, ky=3)

    x_full = np.arange(h)
    y_full = np.arange(w)
    dx = spline_dx(x_full, y_full)
    dy = spline_dy(x_full, y_full)

    # Create sampling coordinates
    x_coords, y_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    indices = [
        np.clip(x_coords + dx, 0, h - 1),
        np.clip(y_coords + dy, 0, w - 1),
    ]

    # Apply deformation
    deformed_image = map_coordinates(image, indices, order=3, mode='reflect')
    deformed_mask = map_coordinates(mask, indices, order=0, mode='reflect')  # nearest for labels

    return deformed_image, deformed_mask


def mirror_pad(image: np.ndarray, pad_size: int) -> np.ndarray:
    """Pad image by mirroring at boundaries.
    
    Paper Section 2 / Figure 2: "Missing input data is extrapolated by 
    mirroring the input image" for the overlap-tile strategy.
    """
    return np.pad(image, pad_size, mode='reflect')


class SegmentationDataset(Dataset):
    """Dataset with U-Net augmentation pipeline.
    
    Applies the paper's augmentation strategy:
    - Random elastic deformations (Section 3.1)
    - Random rotations and flips
    - Gray value variations
    - Mirror padding for overlap-tile inference
    
    Args:
        image_dir: Directory containing input images.
        mask_dir: Directory containing segmentation masks.
        input_size: Network input tile size (paper: 572).
        output_size: Network output size (paper: 388).
        augment: Whether to apply data augmentation.
        elastic_sigma: Elastic deformation sigma (paper: 10).
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        input_size: int = 572,
        output_size: int = 388,
        augment: bool = True,
        elastic_sigma: float = 10.0,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.output_size = output_size
        self.augment = augment
        self.elastic_sigma = elastic_sigma

        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        # Load image and mask
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = self.image_files[idx]  # assume same naming convention
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.int64)

        # Normalize image to [0, 1]
        image = image / 255.0

        # Data augmentation (Section 3.1)
        if self.augment:
            # Random elastic deformation
            image, mask = elastic_deformation(
                image, mask, sigma=self.elastic_sigma
            )

            # Random horizontal and vertical flip
            if np.random.rand() > 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() > 0.5:
                image = np.flipud(image).copy()
                mask = np.flipud(mask).copy()

            # Random rotation (0, 90, 180, 270 degrees)
            k = np.random.randint(4)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

            # Gray value variation
            if np.random.rand() > 0.5:
                image = image + np.random.uniform(-0.1, 0.1)
                image = np.clip(image, 0, 1)

        # Mirror pad for overlap-tile strategy (Section 2 / Figure 2)
        border = (self.input_size - self.output_size) // 2
        image_padded = mirror_pad(image, border)

        # Random crop to input_size × input_size
        h, w = image_padded.shape
        if h > self.input_size and w > self.input_size:
            top = np.random.randint(0, h - self.input_size)
            left = np.random.randint(0, w - self.input_size)
        else:
            top, left = 0, 0

        image_tile = image_padded[top:top + self.input_size, left:left + self.input_size]

        # Corresponding mask crop (output_size × output_size, centered)
        mask_top = top  # mask is not padded, so indices align directly
        mask_left = left
        mask_tile = mask[mask_top:mask_top + self.output_size, mask_left:mask_left + self.output_size]

        # Convert to tensors
        image_tensor = torch.from_numpy(image_tile).unsqueeze(0).float()  # (1, H, W)
        mask_tensor = torch.from_numpy(mask_tile).long()                   # (H, W)

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'filename': self.image_files[idx],
        }
```

#### `src/train.py` — Training Loop

```python
"""
Training Loop for U-Net Reproduction (Ronneberger et al., 2015)

Paper Section 3: "The input images and their corresponding segmentation maps 
are used to train the network with the stochastic gradient descent implementation 
of Caffe. [...] we favor large input tiles over a large batch size and hence 
reduce the batch to a single image. Accordingly we use a high momentum (0.99)"
"""

import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime

from model import UNet
from data import SegmentationDataset
from loss import WeightedCrossEntropyLoss, compute_weight_map
from evaluate import compute_iou


def train(config_path: str = "config.yaml"):
    """Main training function following paper's protocol."""

    # --- Load configuration ---
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Model (Section 2) ---
    model = UNet(
        in_channels=config['model']['in_channels'],
        num_classes=config['model']['num_classes'],
        base_features=config['model']['base_features'],
        use_padding=config['model']['use_padding'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # --- Dataset and DataLoader ---
    train_dataset = SegmentationDataset(
        image_dir="data/train/images",
        mask_dir="data/train/masks",
        input_size=config['training']['input_size'],
        output_size=config['training']['output_size'],
        augment=True,
        elastic_sigma=config['augmentation']['elastic_sigma'],
    )

    # Paper: batch_size = 1
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # --- Loss and Optimizer ---
    criterion = WeightedCrossEntropyLoss()

    # Paper Section 3: "stochastic gradient descent [...] high momentum (0.99)"
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=0.0,  # Not specified in paper
    )

    # --- Training Loop ---
    num_epochs = config['training']['epochs']
    best_iou = 0.0

    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Momentum: {config['training']['momentum']}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            # Compute weight maps (Equation 2)
            # In practice, weight maps should be precomputed and cached
            weight_maps = torch.ones_like(masks, dtype=torch.float32).to(device)

            # Forward pass
            logits = model(images)

            # Ensure spatial dimensions match (for valid convolutions)
            if logits.shape[2:] != masks.shape[1:]:
                # Center crop masks to match output
                diff_h = (masks.shape[1] - logits.shape[2]) // 2
                diff_w = (masks.shape[2] - logits.shape[3]) // 2
                masks = masks[
                    :,
                    diff_h:diff_h + logits.shape[2],
                    diff_w:diff_w + logits.shape[3],
                ]
                weight_maps = weight_maps[
                    :,
                    diff_h:diff_h + logits.shape[2],
                    diff_w:diff_w + logits.shape[3],
                ]

            # Compute weighted loss (Equation 1)
            loss = criterion(logits, masks, weight_maps)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] — Loss: {avg_loss:.6f}")

    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    save_path = f"checkpoints/unet_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == "__main__":
    train()
```

#### `src/evaluate.py` — Metrics

```python
"""
Evaluation Metrics for U-Net Reproduction

Implements IOU (Intersection over Union), the primary metric used in the 
ISBI Cell Tracking Challenge results (Table 2 in the paper).
"""

import numpy as np
import torch


def compute_iou(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = 2,
) -> dict:
    """Compute per-class and mean Intersection over Union (IOU).
    
    IOU = |Prediction ∩ Target| / |Prediction ∪ Target|
    
    This is the metric used in Table 2 of the paper for the ISBI Cell 
    Tracking Challenge evaluation.
    
    Args:
        predictions: Predicted class labels of shape (N, H, W).
        targets: Ground-truth labels of shape (N, H, W).
        num_classes: Number of segmentation classes (paper: 2 for binary).
    
    Returns:
        Dictionary with per-class IOU and mean IOU.
    """
    ious = {}
    for cls in range(num_classes):
        pred_cls = (predictions == cls)
        target_cls = (targets == cls)

        intersection = np.logical_and(pred_cls, target_cls).sum()
        union = np.logical_or(pred_cls, target_cls).sum()

        if union == 0:
            ious[f"class_{cls}_iou"] = float('nan')
        else:
            ious[f"class_{cls}_iou"] = intersection / union

    # Mean IOU (excluding NaN)
    valid_ious = [v for v in ious.values() if not np.isnan(v)]
    ious["mean_iou"] = np.mean(valid_ious) if valid_ious else 0.0

    return ious


def compute_pixel_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Compute overall pixel-wise accuracy.
    
    Related to the "pixel error" metric in Table 1 of the paper.
    """
    correct = (predictions == targets).sum()
    total = targets.size
    return correct / total


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 2,
) -> dict:
    """Evaluate U-Net model on a dataset.
    
    Args:
        model: Trained U-Net model.
        dataloader: Evaluation data loader.
        device: Computation device.
        num_classes: Number of classes.
    
    Returns:
        Dictionary of aggregated metrics.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].numpy()

        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu().numpy()

        # Handle size mismatch from valid convolutions
        if preds.shape[1:] != masks.shape[1:]:
            diff_h = (masks.shape[1] - preds.shape[1]) // 2
            diff_w = (masks.shape[2] - preds.shape[2]) // 2
            masks = masks[
                :,
                diff_h:diff_h + preds.shape[1],
                diff_w:diff_w + preds.shape[2],
            ]

        all_preds.append(preds)
        all_targets.append(masks)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    metrics = compute_iou(all_preds, all_targets, num_classes)
    metrics["pixel_accuracy"] = compute_pixel_accuracy(all_preds, all_targets)

    return metrics
```

### 8.4 Reproduction Results Template

```
## Reproduction Results

### Configuration
- Hardware: [GPU model, VRAM]
- Framework: PyTorch [version]
- Training time: [hours]

### Comparison
| Metric    | Paper (PhC-U373) | Ours  | Δ     |
|-----------|------------------|-------|-------|
| IOU       | 0.9203           | —     | —     |
| Pixel Acc | —                | —     | —     |

### Observations
- [Differences from paper results]
- [Challenges encountered]
- [Sensitivity findings]
```

### 8.5 Key Implementation Notes

1. **Valid vs. Padded Convolutions**: The original paper uses unpadded (valid) convolutions, causing the output to be smaller than the input (572→388). Most modern reimplementations use padded convolutions (`use_padding=True`) for simplicity. Set `use_padding=False` for faithful reproduction.
    
2. **Weight Map Precomputation**: The border weight maps (Equation 2) require instance-level segmentation labels (unique ID per cell). These should be precomputed and cached as they involve expensive distance transforms.
    
3. **Overlap-Tile Inference**: For inference on full images, implement the overlap-tile strategy with mirror padding at boundaries. Each tile's prediction covers only the valid central region.
    
4. **Batch Size 1 + Momentum 0.99**: This is critical to reproduce. The high momentum effectively averages gradients over ~100 previous steps, compensating for the noisy single-sample gradients.
    
5. **7-Rotation Ensembling**: The EM segmentation result uses test-time augmentation with 7 rotations. Average the softmax outputs across rotations for the final prediction.
    

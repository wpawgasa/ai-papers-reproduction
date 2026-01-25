# Deep Residual Learning for Image Recognition

A reproduction of the CIFAR-10 degradation experiment from the paper ["Deep Residual Learning for Image Recognition"](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (CVPR 2016).

## Overview

This repository implements a controlled experiment demonstrating the key finding from the ResNet paper:

1. **Degradation Problem**: Deeper plain networks have worse training error than shallow ones
2. **Residual Learning Solution**: Skip connections enable optimization of very deep networks
3. **Depth Benefits**: With residual connections, deeper networks achieve better accuracy

The experiment compares Plain networks vs ResNets at depths 20 and 56 on CIFAR-10.

## Key Concepts

### The Degradation Problem

Before ResNet, increasing network depth beyond a certain point led to higher training error (not just test error). This wasn't classic overfitting - the training error itself got worse. The paper demonstrates this clearly:
- **PlainNet-56** has higher training error than **PlainNet-20**

### Residual Learning

Instead of learning a direct mapping H(x), ResNet learns a residual function:
```
F(x) = H(x) - x
```

This makes it easier to learn identity mappings (setting F(x) → 0) and provides better gradient flow during backpropagation.

## Repository Structure

```
Deep Residual Learning for Image Recognition/
├── src/
│   └── resnet_cifar10/
│       ├── __init__.py
│       ├── config.py           # Configuration dataclasses
│       ├── data/
│       │   ├── __init__.py
│       │   └── cifar10.py      # CIFAR-10 data loading
│       ├── models/
│       │   ├── __init__.py
│       │   ├── resnet.py       # ResNet with skip connections
│       │   └── plainnet.py     # Plain network without skips
│       ├── training/
│       │   ├── __init__.py
│       │   └── trainer.py      # Training loops
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── metrics.py      # Evaluation metrics
│       └── utils/
│           ├── __init__.py
│           └── seed.py         # Reproducibility utilities
├── notebooks/
│   └── experiment.ipynb        # Step-by-step Jupyter notebook
├── pyproject.toml              # Project configuration
├── summary.md                  # Paper summary and insights
└── README.md                   # This file
```

## Installation

### Option 1: Using pip

```bash
cd "Deep Residual Learning for Image Recognition"
pip install -e .
```

### Option 2: Using uv (faster)

```bash
cd "Deep Residual Learning for Image Recognition"
uv pip install -e .
```

### With Development Tools

```bash
pip install -e ".[dev]"
```

## Quick Start

### Run the Jupyter Notebook (Recommended)

The notebook provides a **step-by-step walkthrough** with detailed explanations:

```bash
cd "Deep Residual Learning for Image Recognition"
jupyter notebook notebooks/experiment.ipynb
```

The notebook includes:
- ✅ Detailed explanations of the degradation problem
- ✅ Model architecture comparison
- ✅ Training progress visualization
- ✅ Comprehensive results analysis
- ✅ Publication-quality comparison plots

### Run as Python Script

Alternatively, you can create a script to run experiments:

```python
import torch
from resnet_cifar10 import (
    ExperimentConfig,
    get_dataloaders,
    ResNetCIFAR,
    PlainNetCIFAR,
    train_model,
    set_seed,
)

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
config = ExperimentConfig.default(depth=56)
config.train.device = device
set_seed(config.train.seed)

# Load data
loaders = get_dataloaders(
    data_dir=config.data.data_dir,
    batch_size=config.data.batch_size,
)

# Train ResNet-56
resnet_56 = ResNetCIFAR(depth=56)
history = train_model(
    resnet_56,
    loaders['train'],
    loaders['test'],
    config.train,
    device,
)

print(f"Best test accuracy: {max(history['test_acc'])*100:.2f}%")
```

## Expected Results

When you run the full experiment, you should observe:

### 1. Degradation in Plain Networks
- **PlainNet-56** has worse training error than **PlainNet-20**
- Demonstrates that depth alone doesn't help without skip connections

### 2. ResNets Scale Better
- **ResNet-56** achieves better accuracy than **ResNet-20**
- Deeper ResNets consistently improve with proper residual connections

### 3. Performance
- ResNet-56 should achieve approximately **93-94% test accuracy** on CIFAR-10

## Configuration

Adjust experiment parameters using configuration objects:

```python
from resnet_cifar10 import ExperimentConfig

# Full training (default)
config = ExperimentConfig.default(depth=56)
config.train.epochs = 200
config.train.lr = 0.1
config.data.batch_size = 128

# Quick testing
config = ExperimentConfig.quick_test(depth=20)
config.train.epochs = 50  # Fewer epochs for testing
```

## Model Architectures

### CIFAR ResNet Design (6n+2 layers)

The networks follow the paper's CIFAR-specific architecture:
- **Depth formula**: 6n + 2 (e.g., depth=20 → n=3, depth=56 → n=9)
- **3 stages** with channels: 16 → 32 → 64
- **Each stage** has n blocks
- **Downsampling** via stride-2 convolutions at stage transitions
- **Global average pooling** + fully connected layer

### BasicBlock (ResNet)
```
x → [Conv3x3 → BN → ReLU → Conv3x3 → BN] → (+) → ReLU
└────────────────────────────────────────────┘
      (skip connection / identity shortcut)
```

### PlainBlock (PlainNet)
```
x → Conv3x3 → BN → ReLU → Conv3x3 → BN → ReLU
```

## Key Insights from the Paper

> **"The degradation problem suggests that the solvers might have difficulties in approximating identity mappings by multiple nonlinear layers."**

The paper's solution:
1. **Residual formulation** makes identity mapping easier to learn
2. **Skip connections** provide direct gradient paths during backpropagation
3. **Implicit ensembling** effect from multiple paths through the network

## Computational Requirements

- **Training time**: ~2-4 hours per model on modern GPU
- **Memory**: ~4-6 GB GPU memory
- **Dataset**: CIFAR-10 (~170 MB download)

For quick testing:
```python
config = ExperimentConfig.quick_test(depth=20)
config.train.epochs = 50  # Reduced from 200
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```python
config.data.batch_size = 64  # Default is 128
```

### Slow Training

Use fewer epochs or smaller model:
```python
config.train.epochs = 50
# or
config = ExperimentConfig.quick_test(depth=20)
```

### Different Results

Results may vary due to:
- Hardware differences (GPU vs CPU)
- PyTorch version
- Random initialization

The qualitative trend (degradation in PlainNets, ResNets scale better) should remain consistent.

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={770--778},
  year={2016}
}
```

## Paper Link

[Deep Residual Learning for Image Recognition (CVPR 2016)](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

## Additional Resources

- [Paper Summary](summary.md) - Detailed summary with key insights
- [PyTorch ResNet Documentation](https://pytorch.org/vision/stable/models/resnet.html)
- [Original Paper Supplemental](https://openaccess.thecvf.com/content_cvpr_2016/supplemental/He_Deep_Residual_Learning_2016_CVPR_supplemental.pdf)

## License

This is a research reproduction for educational purposes. Please refer to the original paper for official implementations and results.

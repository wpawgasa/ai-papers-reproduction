# Papers Reproduction

A consolidated collection of implementation codebases used to replicate experimental results from AI research papers.

## Projects

| Paper | Directory | Status |
|-------|-----------|--------|
| [RL's Razor: Why Online RL Forgets Less](https://arxiv.org/abs/2509.04259) | [why_RL_forget_less/](why_RL_forget_less/) | ✅ Complete |
| [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) | [Deep Residual Learning for Image Recognition/](Deep%20Residual%20Learning%20for%20Image%20Recognition/) | ✅ Complete |
| [FNet: Mixing Tokens with Fourier Transforms](https://arxiv.org/abs/2105.03824) | [FNet/](FNet/) | ✅ Complete |

### Project Highlights

**RL's Razor (2024)**: Demonstrates why online RL forgets less than supervised fine-tuning through controlled ParityMNIST experiments, showing that forgetting is predicted by KL divergence.

**ResNet (He et al., CVPR 2016)**: Reproduces the degradation problem experiment on CIFAR-10, showing that deeper plain networks train worse while ResNets with skip connections scale effectively.

**FNet (Lee-Thorp et al., NAACL 2022)**: Replaces Transformer self-attention with 2D Fourier Transform for token mixing. Achieves 92-97% of BERT accuracy on GLUE with 1.8× faster training and zero learnable parameters in the mixing layer.

## Structure

Each subdirectory contains a self-contained reproduction of a specific paper:

```
papers-reproduction/
├── why_RL_forget_less/                         # RL's Razor paper
│   ├── src/rl_razor_paritymnist/              # Source code
│   ├── notebooks/                              # Experiment notebooks
│   └── README.md                               # Paper-specific docs
├── Deep Residual Learning for Image Recognition/  # ResNet paper
│   ├── src/resnet_cifar10/                    # Source code
│   ├── notebooks/                              # Experiment notebooks
│   └── README.md                               # Paper-specific docs
├── FNet/                                       # FNet paper
│   ├── src/                                    # Source code
│   ├── notebooks/                              # Experiment notebooks
│   ├── summary.md                              # Comprehensive paper analysis
│   └── README.md                               # Paper-specific docs
└── ...
```

## Getting Started

Each project has its own dependencies and setup instructions. Navigate to the specific project directory and follow its README.

### Example: RL's Razor

```bash
cd why_RL_forget_less
pip install -e .
jupyter notebook notebooks/experiment.ipynb
```

### Example: ResNet

```bash
cd "Deep Residual Learning for Image Recognition"
pip install -e .
jupyter notebook notebooks/experiment.ipynb
```

### Example: FNet

```bash
cd FNet
pip install -e .
jupyter notebook notebooks/experiment.ipynb
```

## Contributing

When adding a new paper reproduction:

1. Create a new directory with a descriptive name
2. Include a `README.md` with paper link and reproduction details
3. Use `pyproject.toml` for dependency management
4. Add a Jupyter notebook demonstrating the key experiments
5. Update this README's project table

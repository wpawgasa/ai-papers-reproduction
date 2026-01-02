# RL's Razor: Why Online RL Forgets Less

A reproduction of the ParityMNIST experiment from the paper ["RL's Razor: Why Online Reinforcement Learning Forgets Less"](https://arxiv.org/html/2509.04259v1) (arXiv:2509.04259v1).

## Overview

This repository implements a controlled experiment demonstrating that:

1. **RL forgets less than SFT** when fine-tuning on new tasks
2. **Forgetting is predicted by KL shift** on the new task distribution
3. **On-policy RL implicitly minimizes KL** to the base model

The experiment uses ParityMNIST (a modified MNIST task) and FashionMNIST to study catastrophic forgetting in a clean, reproducible setting.

## Key Concepts

### ParityMNIST

A task where correctness is defined by parity rather than exact digit:
- **Even digits** (0,2,4,6,8): Any even prediction is "correct"
- **Odd digits** (1,3,5,7,9): Any odd prediction is "correct"

This creates **multiple equally-correct solutions**, mimicking the underdetermination in generative tasks like language modeling.

### The Experiment

1. **Pretrain** an MLP jointly on ParityMNIST + FashionMNIST
2. **Fine-tune** only on ParityMNIST using different methods:
   - SFT with arbitrary fixed labels
   - SFT with random labels
   - On-policy REINFORCE
   - SFT with oracle (min-KL) labels
3. **Measure**:
   - New-task performance (ParityMNIST success)
   - Old-task retention (FashionMNIST accuracy)
   - KL(base || fine-tuned) on ParityMNIST

## Repository Structure

```
why_RL_forget_less/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── parity_mnist.py      # ParityMNIST dataset wrapper
│   ├── models/
│   │   ├── __init__.py
│   │   └── mlp.py               # MLP architecture
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretrain.py          # Joint pretraining
│   │   └── finetune.py          # SFT and REINFORCE fine-tuning
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py           # Accuracy, success, KL metrics
│   ├── utils/
│   │   ├── __init__.py
│   │   └── seed.py              # Reproducibility utilities
│   └── config.py                # Configuration dataclasses
├── notebooks/
│   └── experiment.ipynb         # Step-by-step Jupyter notebook
├── pyproject.toml               # Dependencies
├── summary.md                   # Paper summary and original script
└── README.md                    # This file
```

## Installation

### Option 1: Using pip

```bash
cd why_RL_forget_less
pip install -e .
```

### Option 2: Using uv (recommended for faster installation)

```bash
cd why_RL_forget_less
uv pip install -e .
```

### With Optional Dependencies

```bash
# For visualization enhancements
pip install -e ".[viz]"

# For development tools
pip install -e ".[dev]"

# All extras
pip install -e ".[viz,dev]"
```

## Quick Start

### Run the Jupyter Notebook (Recommended)

The notebook provides a **step-by-step walkthrough** with explanations:

```bash
cd why_RL_forget_less
jupyter notebook notebooks/experiment.ipynb
```

The notebook includes:
- ✅ Detailed explanations of each step
- ✅ Progress visualization
- ✅ Interactive result exploration
- ✅ Publication-quality plots
- ✅ Summary statistics

### Run as Python Script

Alternatively, create a script to run the experiment:

```python
# run_experiment.py
import sys
sys.path.insert(0, './src')

import torch
from config import ExperimentConfig
from data import get_dataloaders
from models import MLP
from training import pretrain_joint, finetune_sft_fixed, finetune_reinforce
from evaluation import fashion_accuracy, parity_success, kl_base_to_ft
from utils import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup
config = ExperimentConfig.default()
config.train.device = DEVICE
set_seed(config.train.seed)

# Load data
loaders = get_dataloaders(config)

# Pretrain base model
base_model = MLP().to(DEVICE)
pretrain_joint(
    base_model,
    loaders['parity_train'],
    loaders['fashion_train'],
    config.train,
    device=DEVICE
)

# Evaluate base
base_fashion = fashion_accuracy(base_model, loaders['fashion_test'], DEVICE)
base_parity = parity_success(base_model, loaders['parity_test'], DEVICE)
print(f"Base: Fashion={base_fashion:.3f}, Parity={base_parity:.3f}")

# Fine-tune with different methods
# ... (see notebook for full code)
```

## Expected Results

When you run the experiment, you should observe:

### 1. RL Forgets Less Than SFT

- **REINFORCE** achieves similar ParityMNIST performance with **better FashionMNIST retention**
- **SFT methods** (especially with arbitrary labels) show larger drops in old-task performance

### 2. Forgetting vs KL Relationship

The plot shows a clear correlation: methods with **larger KL shifts** exhibit **more forgetting**, regardless of whether they use RL or SFT.

### 3. Oracle SFT Validates the Theory

**SFT with oracle labels** (sampled from the min-KL distribution) achieves retention comparable to or better than REINFORCE, proving that **KL is the key factor**, not "RL magic."

## Key Findings from the Paper

> **"Among all ways to solve the new task, on-policy RL tends to find solutions that stay closest (in KL) to the base model."**

This is "RL's Razor" - an implicit Occam's Razor in policy space that leads to better retention.

## Configuration

Adjust experiment parameters in the notebook or via the config:

```python
from config import ExperimentConfig, TrainConfig

config = ExperimentConfig.default()

# Modify training settings
config.train.pretrain_steps = 1000      # More pretraining
config.train.finetune_steps = 500       # More fine-tuning
config.train.lr = 5e-4                  # Different learning rate
config.train.batch_size = 256           # Larger batches
config.train.seed = 123                 # Change seed

# Modify model architecture
config.model.hidden1 = 512              # Larger hidden layer
config.model.hidden2 = 512

# Modify data settings
config.data.pretrain_samples = 10000    # More training data
```

## Reproducing Paper Figures

The notebook generates two main plots:

### 1. Forgetting vs KL Shift

Shows that forgetting is predicted by KL divergence on the new task, supporting the paper's "forgetting law."

![Forgetting vs KL](forgetting_vs_kl.png)

### 2. Pareto Frontier

Shows that RL achieves better retention at similar new-task performance compared to SFT.

![Pareto Frontier](pareto_frontier.png)

## Outputs

Running the experiment produces:

- `forgetting_vs_kl.png` - Main result plot
- `pareto_frontier.png` - Performance comparison
- `results.csv` - Summary table with metrics
- (Optional) `checkpoints/*.pt` - Saved model weights

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{balsam2024rlsrazor,
  title={RL's Razor: Why Online Reinforcement Learning Forgets Less},
  author={Balsam, Peter and others},
  journal={arXiv preprint arXiv:2509.04259},
  year={2024}
}
```

## Paper Link

[RL's Razor: Why Online Reinforcement Learning Forgets Less](https://arxiv.org/html/2509.04259v1)

## Practical Implications

The paper's findings suggest:

1. **Track KL-to-base during fine-tuning** as a proxy for forgetting risk
2. **Prefer on-policy RL** when possible to naturally minimize KL
3. If using SFT, **choose labels that minimize KL** to the base model (oracle approach)

## License

This is a research reproduction for educational purposes. Please refer to the original paper for the official implementation and results.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```python
config.train.batch_size = 64
config.data.eval_batch_size = 256
```

### Slow Training

Use smaller model or fewer steps:
```python
config.model.hidden1 = 128
config.model.hidden2 = 128
config.train.pretrain_steps = 400
config.train.finetune_steps = 200
```

### Unexpected Results

- Check random seed is set consistently
- Ensure you're using the same data preprocessing
- Verify device (CPU vs GPU) doesn't affect results significantly

## Contributing

This is a reproduction repository. For issues or improvements:
1. Check the notebook for explanations
2. Verify against the paper's descriptions
3. Test with different random seeds to ensure results are stable

## Questions?

For questions about:
- **This reproduction**: Open an issue in this repository
- **The original paper**: See the paper's project page or contact the authors

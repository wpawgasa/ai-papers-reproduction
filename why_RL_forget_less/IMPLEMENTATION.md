# Implementation Guide

This document explains the structure and implementation of the RL's Razor reproduction.

## Module Overview

### 1. Configuration (`src/config.py`)

Defines three main configuration classes:
- **TrainConfig**: Learning rate, batch size, training steps, seed
- **ModelConfig**: Hidden layer sizes, output dimensions
- **DataConfig**: Data paths, sample sizes, data loader settings
- **ExperimentConfig**: Combines all configs with a `default()` factory method

### 2. Data Loading (`src/data/`)

**parity_mnist.py**:
- `ParityMNIST`: Dataset wrapper that returns (image, parity, digit)
- `correct_labels_for_parity(parity)`: Returns valid labels for a given parity
- `get_dataloaders(config)`: Creates all required data loaders

Key insight: ParityMNIST creates **multiple correct answers** (any even digit for even images, any odd digit for odd images), mimicking underdetermination in generative tasks.

### 3. Model (`src/models/`)

**mlp.py**:
- Simple 2-layer MLP
- Input: 784 (28×28 image) + 1 (parity bit) = 785 features
- Hidden layers: ReLU activations
- Output: 10 logits (digit classes)

The parity bit is concatenated to the input to allow the model to distinguish between tasks.

### 4. Training (`src/training/`)

**pretrain.py**:
- `pretrain_joint()`: Trains on both FashionMNIST and ParityMNIST
  - FashionMNIST: Standard CE on true labels
  - ParityMNIST: Random correct labels (uniform over valid set)

**finetune.py**:
Three fine-tuning methods:

1. **`finetune_sft_fixed()`**: Supervised fine-tuning with fixed label mapping
   - `mapping="01"`: Even→0, Odd→1 (may cause large KL shift)
   - `mapping="random"`: Random but consistent mapping

2. **`finetune_reinforce()`**: On-policy RL with binary reward
   - Sample actions from current policy
   - Reward = 1 if correct parity, 0 otherwise
   - Use advantage = reward - baseline
   - Update: -E[advantage × log_prob]

3. **`finetune_sft_oracle()`**: SFT with minimum-KL labels
   - `oracle_targets_from_base()`: Sample from p_base restricted to correct labels
   - This is the I-projection (min-KL) onto the correct set

### 5. Evaluation (`src/evaluation/`)

**metrics.py**:

1. **`fashion_accuracy()`**: Standard classification accuracy on FashionMNIST
   - Measures retention of old task

2. **`parity_success()`**: Fraction of predictions with correct parity
   - Measures new task performance

3. **`kl_base_to_ft()`**: Average KL(base || fine-tuned) on new task
   - Key predictor of forgetting according to the paper

## Experiment Flow

### Step 1: Pretrain Base Model
```
Base Model = MLP
  ↓
Joint training on:
  - FashionMNIST (old task)
  - ParityMNIST (new task, with random correct labels)
  ↓
Base model with capabilities on both tasks
```

### Step 2: Fine-tune on New Task Only
```
For each method:
  1. Clone base model
  2. Fine-tune ONLY on ParityMNIST
  3. Measure:
     - New task: parity success
     - Old task: fashion accuracy
     - KL shift: KL(base || ft)
```

### Step 3: Compare Methods
```
Plot 1: Forgetting vs KL
  - X-axis: KL(base || ft) on new task
  - Y-axis: Forgetting (base_acc - ft_acc)
  - Expected: Linear relationship

Plot 2: Pareto Frontier
  - X-axis: New task performance
  - Y-axis: Old task retention
  - Expected: RL dominates SFT
```

## Key Implementation Details

### 1. Why Oracle SFT?

The oracle is the **minimum-KL distribution** that solves the task:

```python
# For each input x with parity p:
q*(y|x) ∝ p_base(y|x) × 𝟙[y has correct parity]

# In code:
correct_labels = [0,2,4,6,8] if even else [1,3,5,7,9]
probs = softmax(base_logits)[correct_labels]
sample from renormalized probs
```

This proves that **KL is the key factor**, not something special about RL.

### 2. Why REINFORCE?

On-policy sampling means:
- Only update on outputs the model currently produces
- Rewards **reweight** existing distribution
- Doesn't force toward external targets that may be far away

Implementation:
```python
logits = model(x, parity)
action = sample(logits)           # On-policy!
reward = (action % 2 == parity)   # Binary reward
loss = -advantage * log_prob(action)
```

### 3. Why Fixed SFT Can Fail?

Example: Base model prefers [2,4] for even images
- SFT with "even→0" forces shift to [0]
- Large KL divergence
- Representation damage → forgetting

## Running the Experiment

### Quick Test (Faster)
```python
config.train.pretrain_steps = 400
config.train.finetune_steps = 200
config.data.pretrain_samples = 2000
```

### Full Reproduction (Better Results)
```python
config.train.pretrain_steps = 1000
config.train.finetune_steps = 500
config.data.pretrain_samples = 10000
```

## Expected Numerical Results

Typical outcomes (may vary with seed):

| Method | Parity Success | Fashion Acc | KL Shift | Forgetting |
|--------|---------------|-------------|----------|------------|
| Base Model | 0.85 | 0.75 | 0.0000 | 0.000 |
| SFT (0/1) | 0.99 | 0.65 | 0.25 | 0.100 |
| SFT (random) | 0.98 | 0.68 | 0.18 | 0.070 |
| REINFORCE | 0.97 | 0.72 | 0.08 | 0.030 |
| SFT (oracle) | 0.98 | 0.73 | 0.06 | 0.020 |

Key observations:
1. All methods improve on new task (parity ~0.97-0.99)
2. REINFORCE and oracle have lower KL → less forgetting
3. Fixed SFT can have large KL → more forgetting
4. Clear correlation between KL and forgetting

## Customization Ideas

### 1. Different Architectures
```python
# Try CNN instead of MLP
from torchvision.models import resnet18
```

### 2. Different Tasks
```python
# Try CIFAR-10 + CIFAR-100
# Or MNIST + SVHN
```

### 3. Additional Methods
```python
# Try PPO, DPO, or other alignment methods
# Try regularization: EWC, L2, etc.
```

### 4. Sweep Hyperparameters
```python
for lr in [1e-4, 5e-4, 1e-3, 5e-3]:
    for steps in [200, 400, 800]:
        # Run experiment
        # Track Pareto frontier
```

## Debugging Tips

### Model not learning parity?
- Check reward calculation: `(pred % 2) == parity`
- Increase learning rate or steps
- Verify data loader returns correct parity labels

### High variance in REINFORCE?
- Increase batch size
- Tune baseline smoothing factor (beta)
- Try advantage normalization

### Oracle SFT worse than expected?
- Verify base model is frozen during sampling
- Check probability normalization
- Ensure correct labels are properly masked

### Different results than paper?
- This is a toy reproduction, not the full experiment
- Check random seed consistency
- Try multiple seeds and average results
- Verify data preprocessing matches

## Further Reading

- **Original Paper**: [arXiv:2509.04259v1](https://arxiv.org/html/2509.04259v1)
- **Related Work**:
  - Catastrophic forgetting: Kirkpatrick et al. (EWC)
  - KL regularization: Schulman et al. (PPO, TRPO)
  - Continual learning: Zenke et al. (SI)

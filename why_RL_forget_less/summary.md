# Why RL forget less

## What the paper is claiming (in one sentence)

When you fine-tune a foundation model on a new task, **how much it forgets old skills is largely predicted by a single number computed only on the new task**:  
$$ 
\mathrm{KL}\big(\pi_{\text{base}}(\cdot|x);|;\pi_{\text{ft}}(\cdot|x)\big)\ \text{ averaged over new-task inputs }x.  
$$  
And **on-policy RL tends to land on solutions with smaller KL shift than SFT**, so it “forgets less.” ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
## Core ideas and contributions

### 1) Empirical finding: RL forgets less than SFT at the same new-task performance

They run many hyperparameter sweeps and plot **Pareto frontiers**: new-task performance (x-axis) vs retention on prior tasks (y-axis). Across multiple settings, **RL achieves comparable new-task gains while keeping prior-task scores much higher**, whereas SFT often buys new-task accuracy by sacrificing older capabilities. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))

They demonstrate this on:
- **LLMs** (Qwen 2.5 3B-Instruct) fine-tuned on **math**, **science Q&A**, **tool use**, and evaluated on broad benchmarks like MMLU, TruthfulQA, HellaSwag, HumanEval, etc. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
- **Robotics** (OpenVLA 7B) in **SimplerEnv**, fine-tune on one manipulation task and evaluate retention on other tasks. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    
### 2) The “forgetting law”: forgetting is predicted by KL shift _on the new task_

A key surprise: you don’t need past-task data to predict forgetting. They find a strong empirical relationship:
- **Plot forgetting vs KL(base‖fine-tuned) measured on the new-task distribution**, and results from different algorithms/hyperparams collapse to a single curve. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    
This reframes catastrophic forgetting as mainly a **distribution-shift problem**: if your fine-tuned policy moves far from the base policy on the new task, it tends to damage old capabilities too.

### 3) RL’s Razor: on-policy RL is implicitly biased toward KL-minimal solutions

Many tasks admit **multiple solutions** that all score well on the new objective. The paper’s principle:

> **Among all ways to solve the new task, on-policy RL tends to find solutions that stay closest (in KL) to the base model.** ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))

This is illustrated conceptually in **Figure 1**: RL picks a solution near the base distribution, SFT can converge to an arbitrarily distant one depending on labels. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))

---

## Why RL (specifically _on-policy_) tends to move less in KL

The paper separates two differences between RL and SFT:

1. **On-policy sampling** (data comes from the model itself, continually updated)
2. **Negative examples** (policy-gradient methods can push down bad outputs)
    
They test four objectives in a quadrant design (**Figure 4**):
- On-policy + negative examples (GRPO)
- On-policy without negative examples (their “1–0 Reinforce”)
- Offline without negatives (SFT)
- Offline with negatives (SimPO)
![[Screenshot 2568-12-24 at 13.10.39.png]]
Result: **on-policy methods are the ones that forget less and induce smaller KL shifts**, regardless of whether they use negative examples. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))

Intuition (their explanation):

- In on-policy RL, you only update on outputs the model already produces with non-negligible probability.
- Rewards **reweight** that existing distribution rather than forcing the model toward an external target distribution (labels) that may be far away. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    

---

## Theoretical justification (what they prove in a simplified setting)

They give a “probability space projection” view: policy-gradient updates behave like conservative projections that prefer staying close to the current policy while improving reward.

In the **binary reward** case, they formalize:

- A lemma connecting **rejection sampling / reweighting** to a **minimum-KL projection** onto the set of optimal policies. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    
- A theorem (Theorem 5.2) stating that under conditions (finite action set, convex policy family), **policy gradient converges to the KL-closest optimal policy relative to initialization**. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    

This is summarized visually in **Figure 5** as an alternating projection path that ends at a KL-minimal optimum. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))

![[Screenshot 2568-12-30 at 17.10.49.png]]

---

## The controlled “toy world”: ParityMNIST (why it matters)

To study forgetting cleanly, they build **ParityMNIST**:

- You’re “correct” if you output **any even digit** for an even image (0,2,4,6,8), and **any odd digit** for an odd image.
    
- That creates **many equally correct output distributions**, mimicking the underdetermination of generative tasks. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    

Setup:

- Pretrain a small MLP jointly on ParityMNIST + FashionMNIST.
    
- Fine-tune on ParityMNIST only; measure “forgetting” as loss of FashionMNIST performance. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    

Key observations:

- RL reaches high parity accuracy while retaining FashionMNIST better than standard SFT.
    
- Plotting forgetting vs KL on ParityMNIST collapses RL and SFT onto one curve. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    

### “Oracle SFT” (important sanity check)

They construct an **oracle supervision distribution**: among all perfectly-correct label distributions, choose the one **minimizing KL to the base**.  
In ParityMNIST, that oracle is essentially:  
$$  
q^*(y|x)\propto \pi_{\text{base}}(y|x)\ \text{restricted to correct labels}  
$$  
and they show that **SFT trained on this oracle can forget even less than RL**, supporting the claim that _KL is the driver_, not “RL magic.” ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))

---

## Additional evidence: representation drift

They also compare internal representation geometry using **CKA/CKNNA**:

- RL-fine-tuned models stay much closer to the base model’s representation space (high similarity),
    
- SFT drifts more. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    

This supports the story that RL changes the model more “surgically,” consistent with smaller policy KL shifts.

---

## Figures explained (what each is showing)

- **Figure 1**: Conceptual summary. Left: RL selects KL-near solutions among many that solve the new task. Right: this yields better retention at matched new-task performance. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    ![[Screenshot 2569-01-01 at 14.52.05.png]]
- **Figure 2**: Pareto frontiers (new-task vs prior-task performance). RL dominates SFT in retention for similar gains. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    ![[Screenshot 2569-01-01 at 14.54.10.png]]

---

## Practical implications (how you’d use this)

1. **Track KL-to-base on your new-task distribution during fine-tuning.** It’s measurable without old-task data and (per the paper) predicts forgetting. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    
2. If you must do SFT, consider **labeling / targets that are KL-minimizing** relative to the base model (their “oracle SFT” idea). ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    
3. If you can afford it, **on-policy RL-style updates** may naturally keep KL small and preserve broad capabilities. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))
    

---

## Is there an official codebase?

The paper links to a project webpage, but I did **not** see an official public repo link there. ([arXiv](https://arxiv.org/html/2509.04259v1 "RL’s Razor: Why Online Reinforcement Learning Forgets Less"))  
So below is a **self-contained reproduction** of the paper’s ParityMNIST-style experiment (SFT vs on-policy REINFORCE + oracle SFT), designed to generate the same kinds of curves (accuracy/forgetting vs KL).

---

## Minimal “ParityMNIST” reproduction (PyTorch)

### What this code does

- Pretrains an MLP on:
    
    - **FashionMNIST** (10-way classification) = “old task”
        
    - **ParityMNIST** (10-way outputs but many labels are “correct”) = “new task pretraining exposure”
        
- Fine-tunes only on ParityMNIST using three methods:
    
    1. **SFT-fixed**: choose one arbitrary correct label per parity (a “bad” labeling can induce big KL)
        
    2. **REINFORCE (on-policy)**: sample labels from model distribution; reward 1 if parity correct
        
    3. **SFT-oracle**: sample labels from the **minimum-KL correct distribution** (base prob renormalized over correct labels)
        
- Measures:
    
    - New-task “parity success”
        
    - Old-task FashionMNIST accuracy (“retention”)
        
    - KL(base‖ft) on ParityMNIST inputs
        

### Single-file script

```python
# rl_razor_paritymnist.py
# A small reproduction inspired by "RL’s Razor: Why Online RL Forgets Less" (arXiv:2509.04259v1).
#
# pip install torch torchvision tqdm matplotlib

import math
import random
from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------
# Data: ParityMNIST wrapper
# -----------------------
class ParityMNIST(torch.utils.data.Dataset):
    """
    Returns (x, parity, y_digit) where y_digit is the original MNIST digit.
    Parity correctness: even digits are "correct" if predicted label in {0,2,4,6,8}
                        odd digits are "correct" if predicted label in {1,3,5,7,9}
    """
    def __init__(self, root: str, train: bool, download: bool = True):
        self.ds = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y = self.ds[idx]
        parity = int(y % 2)  # 0 even, 1 odd
        return x, parity, y


def correct_labels_for_parity(parity: int) -> torch.Tensor:
    if parity == 0:
        return torch.tensor([0, 2, 4, 6, 8], dtype=torch.long)
    else:
        return torch.tensor([1, 3, 5, 7, 9], dtype=torch.long)


# -----------------------
# Model
# -----------------------
class MLP(nn.Module):
    def __init__(self, hidden1=256, hidden2=256, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28 + 1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_dim)

    def forward(self, x, parity_bit):
        # x: [B,1,28,28], parity_bit: [B] in {0,1}
        b = x.shape[0]
        x = x.view(b, -1)
        p = parity_bit.float().view(b, 1)
        h = torch.cat([x, p], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.fc3(h)  # logits


# -----------------------
# Helpers: metrics
# -----------------------
@torch.no_grad()
def fashion_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        parity_bit = torch.zeros(len(x), device=DEVICE)  # not used for FashionMNIST; keep 0
        logits = model(x, parity_bit)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


@torch.no_grad()
def parity_success(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    succ = 0
    total = 0
    for x, parity, _digit in loader:
        x = x.to(DEVICE)
        parity = parity.to(DEVICE)
        logits = model(x, parity)
        pred = logits.argmax(dim=-1)
        # success if predicted digit has correct parity
        succ += ((pred % 2) == parity).sum().item()
        total += parity.numel()
    return succ / max(1, total)


@torch.no_grad()
def kl_base_to_ft(base: nn.Module, ft: nn.Module, loader: DataLoader, max_batches=50) -> float:
    """
    Average KL( base || ft ) over new-task inputs.
    """
    base.eval()
    ft.eval()
    kls = []
    for i, (x, parity, _d) in enumerate(loader):
        if i >= max_batches:
            break
        x, parity = x.to(DEVICE), parity.to(DEVICE)
        p0 = F.softmax(base(x, parity), dim=-1)
        p1 = F.softmax(ft(x, parity), dim=-1)
        kl = (p0 * (p0.clamp_min(1e-12).log() - p1.clamp_min(1e-12).log())).sum(dim=-1)
        kls.append(kl.mean().item())
    return float(sum(kls) / max(1, len(kls)))


# -----------------------
# Training: pretrain + finetune variants
# -----------------------
@dataclass
class TrainCfg:
    lr: float = 1e-3
    batch_size: int = 128
    pretrain_steps: int = 800
    finetune_steps: int = 400
    seed: int = 0


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pretrain_joint(model: nn.Module, parity_loader: DataLoader, fashion_loader: DataLoader, cfg: TrainCfg):
    """
    Joint pretraining on:
      - FashionMNIST: standard CE on true labels.
      - ParityMNIST: *random correct digit label* for each sample (uniform over correct parity set),
        matching the paper's idea that pretraining can put mass on multiple correct labels.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    parity_iter = iter(parity_loader)
    fashion_iter = iter(fashion_loader)

    for _ in tqdm(range(cfg.pretrain_steps), desc="Pretrain"):
        try:
            x_p, parity, _d = next(parity_iter)
        except StopIteration:
            parity_iter = iter(parity_loader)
            x_p, parity, _d = next(parity_iter)

        try:
            x_f, y_f = next(fashion_iter)
        except StopIteration:
            fashion_iter = iter(fashion_loader)
            x_f, y_f = next(fashion_iter)

        x_p, parity = x_p.to(DEVICE), parity.to(DEVICE)
        x_f, y_f = x_f.to(DEVICE), y_f.to(DEVICE)

        # Parity labels sampled uniformly among correct digits
        targets_p = torch.empty(len(x_p), dtype=torch.long, device=DEVICE)
        for i in range(len(x_p)):
            cl = correct_labels_for_parity(int(parity[i].item())).to(DEVICE)
            targets_p[i] = cl[torch.randint(0, len(cl), (1,), device=DEVICE)]

        logits_p = model(x_p, parity)
        loss_p = F.cross_entropy(logits_p, targets_p)

        # Fashion task (parity bit fixed 0)
        logits_f = model(x_f, torch.zeros(len(x_f), device=DEVICE))
        loss_f = F.cross_entropy(logits_f, y_f)

        loss = loss_p + loss_f
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def finetune_sft_fixed(model: nn.Module, loader: DataLoader, cfg: TrainCfg,
                       mapping: Literal["01", "random"] = "01"):
    """
    Offline SFT on ParityMNIST using an arbitrary single-label mapping.
    - mapping="01": even->0, odd->1 (intentionally collapses correct set)
    - mapping="random": choose one fixed random even label and one fixed random odd label
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    if mapping == "01":
        even_label, odd_label = 0, 1
    else:
        even_label = random.choice([0, 2, 4, 6, 8])
        odd_label = random.choice([1, 3, 5, 7, 9])

    it = iter(loader)
    for _ in tqdm(range(cfg.finetune_steps), desc=f"Finetune SFT({mapping})"):
        try:
            x, parity, _d = next(it)
        except StopIteration:
            it = iter(loader)
            x, parity, _d = next(it)

        x, parity = x.to(DEVICE), parity.to(DEVICE)
        y = torch.where(parity == 0, torch.tensor(even_label, device=DEVICE), torch.tensor(odd_label, device=DEVICE))
        logits = model(x, parity)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def oracle_targets_from_base(base: nn.Module, x: torch.Tensor, parity: torch.Tensor) -> torch.Tensor:
    """
    Sample from q*(y|x) ∝ p_base(y|x) restricted to correct labels.
    This is the minimum-KL correct distribution (I-projection) in this discrete case.
    """
    base.eval()
    probs = F.softmax(base(x, parity), dim=-1)  # [B,10]
    out = torch.empty(len(x), dtype=torch.long, device=x.device)
    for i in range(len(x)):
        cl = correct_labels_for_parity(int(parity[i].item())).to(x.device)
        p = probs[i, cl]
        p = p / p.sum()
        out[i] = cl[torch.multinomial(p, 1)]
    return out


def finetune_sft_oracle(model: nn.Module, base_model: nn.Module, loader: DataLoader, cfg: TrainCfg):
    """
    Offline SFT using oracle supervision distribution q* (min KL to base subject to correctness).
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    it = iter(loader)

    for _ in tqdm(range(cfg.finetune_steps), desc="Finetune SFT(oracle)"):
        try:
            x, parity, _d = next(it)
        except StopIteration:
            it = iter(loader)
            x, parity, _d = next(it)

        x, parity = x.to(DEVICE), parity.to(DEVICE)
        y = oracle_targets_from_base(base_model, x, parity)
        logits = model(x, parity)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def finetune_reinforce(model: nn.Module, loader: DataLoader, cfg: TrainCfg):
    """
    On-policy REINFORCE with binary reward: 1 if sampled digit has correct parity else 0.
    This approximates the paper’s "1–0 Reinforce" idea: on-policy is the key ingredient.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    baseline = 0.0
    beta = 0.95  # baseline smoothing

    it = iter(loader)
    for _ in tqdm(range(cfg.finetune_steps), desc="Finetune REINFORCE(on-policy)"):
        try:
            x, parity, _d = next(it)
        except StopIteration:
            it = iter(loader)
            x, parity, _d = next(it)

        x, parity = x.to(DEVICE), parity.to(DEVICE)

        logits = model(x, parity)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()  # sampled digit label
        logp = dist.log_prob(a)

        reward = ((a % 2) == parity).float()  # [B]
        r_mean = reward.mean().item()
        baseline = beta * baseline + (1 - beta) * r_mean
        adv = reward - baseline

        loss = -(adv.detach() * logp).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


# -----------------------
# Main: run a small sweep and plot forgetting vs KL
# -----------------------
def main():
    cfg = TrainCfg(lr=1e-3, batch_size=128, pretrain_steps=800, finetune_steps=400, seed=0)
    set_seed(cfg.seed)

    # Small subsets for speed (paper uses small subsets in toy setting)
    root = "./data"

    fashion_train = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    )
    fashion_test = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
    )
    parity_train = ParityMNIST(root=root, train=True, download=True)
    parity_test = ParityMNIST(root=root, train=False, download=True)

    # Subsample to keep it lightweight
    fashion_train = Subset(fashion_train, list(range(0, 5000)))
    parity_train = Subset(parity_train, list(range(0, 5000)))

    fashion_train_loader = DataLoader(fashion_train, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    parity_train_loader = DataLoader(parity_train, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    fashion_test_loader = DataLoader(fashion_test, batch_size=512, shuffle=False, num_workers=2)
    parity_test_loader = DataLoader(parity_test, batch_size=512, shuffle=False, num_workers=2)

    # Base model
    base = MLP().to(DEVICE)
    pretrain_joint(base, parity_train_loader, fashion_train_loader, cfg)

    base_fashion = fashion_accuracy(base, fashion_test_loader)
    base_parity = parity_success(base, parity_test_loader)
    print(f"\nBase Fashion acc: {base_fashion:.3f}, Base Parity succ: {base_parity:.3f}\n")

    def eval_variant(name: str, ft_model: nn.Module):
        p = parity_success(ft_model, parity_test_loader)
        f = fashion_accuracy(ft_model, fashion_test_loader)
        kl = kl_base_to_ft(base, ft_model, parity_test_loader)
        print(f"{name:>18s} | parity={p:.3f} fashion={f:.3f} KL={kl:.4f}")
        return p, f, kl

    results = []

    # Variant 1: SFT with fixed mapping
    ft1 = MLP().to(DEVICE)
    ft1.load_state_dict(base.state_dict())
    finetune_sft_fixed(ft1, parity_train_loader, cfg, mapping="01")
    results.append(("SFT_fixed01",) + eval_variant("SFT_fixed01", ft1))

    # Variant 2: SFT random mapping
    set_seed(cfg.seed + 1)
    ft2 = MLP().to(DEVICE)
    ft2.load_state_dict(base.state_dict())
    finetune_sft_fixed(ft2, parity_train_loader, cfg, mapping="random")
    results.append(("SFT_random",) + eval_variant("SFT_random", ft2))

    # Variant 3: On-policy REINFORCE
    set_seed(cfg.seed + 2)
    ft3 = MLP().to(DEVICE)
    ft3.load_state_dict(base.state_dict())
    finetune_reinforce(ft3, parity_train_loader, cfg)
    results.append(("REINFORCE",) + eval_variant("REINFORCE", ft3))

    # Variant 4: Oracle SFT
    set_seed(cfg.seed + 3)
    ft4 = MLP().to(DEVICE)
    ft4.load_state_dict(base.state_dict())
    finetune_sft_oracle(ft4, base, parity_train_loader, cfg)
    results.append(("SFT_oracle",) + eval_variant("SFT_oracle", ft4))

    # Plot: forgetting vs KL
    # Forgetting = base_fashion - ft_fashion (higher means worse forgetting)
    labels = [r[0] for r in results]
    parity_succs = [r[1] for r in results]
    fashion_accs = [r[2] for r in results]
    kls = [r[3] for r in results]
    forgetting = [base_fashion - fa for fa in fashion_accs]

    plt.figure(figsize=(7, 5))
    for lab, kl, fg, ps in zip(labels, kls, forgetting, parity_succs):
        plt.scatter(kl, fg, s=120)
        plt.text(kl + 0.002, fg + 0.002, f"{lab}\n(parity={ps:.2f})", fontsize=9)

    plt.xlabel("KL(base || fine-tuned) on ParityMNIST")
    plt.ylabel("Forgetting on FashionMNIST (base_acc - ft_acc)")
    plt.title("Forgetting vs KL shift (toy reproduction of RL’s Razor)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

### How to run

```bash
python rl_razor_paritymnist.py
```

### What you should expect qualitatively

- **SFT_fixed / SFT_random** often achieve parity success but can incur larger KL and more FashionMNIST drop.
    
- **REINFORCE (on-policy)** tends to reach parity success with **smaller KL** and better retention.
    
- **SFT_oracle** often gives the best retention at a given parity success, matching the paper’s oracle argument.
    

(Exact numbers vary with seeds/steps; the paper’s main point is the _relationship_ and the role of on-policy sampling.)

---

If you want, I can also adapt this into:

- a mini “repo layout” (configs, sweep scripts, logging to CSV, plotting notebook), or
    
- a HuggingFace/TRL-style skeleton showing **SFT vs on-policy RL** and explicit **KL-to-base tracking** on a text task (lighter-weight than their full Qwen/OpenVLA experiments).
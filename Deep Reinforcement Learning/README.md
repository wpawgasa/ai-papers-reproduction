# Playing Atari with Deep Reinforcement Learning (DQN)

A reproduction of **Mnih et al., 2013 — "Playing Atari with Deep Reinforcement Learning"** (NIPS Deep Learning Workshop).

**Paper:** [arXiv:1312.5602](https://arxiv.org/abs/1312.5602) · **Analysis:** [`summary.md`](summary.md)

This project trains a single convolutional **Deep Q-Network** end-to-end from raw pixels to play **Pong**, demonstrating the paper's two central claims: experience replay stabilises Q-learning with a non-linear function approximator, and the learned value estimates improve smoothly without diverging.

## Key Ideas

- **End-to-end from pixels:** the network sees only stacked `84×84` grayscale frames and a scalar reward — no hand-crafted features.
- **Experience replay:** transitions are stored in a FIFO buffer and sampled uniformly at random, breaking temporal correlation and giving SGD i.i.d.-like batches.
- **State-in, K-Q-values-out:** one forward pass yields a Q-value per action, so action selection is O(1) per state.
- **Stability without theory:** Q-learning + a deep CNN is empirically non-divergent here, monitored via the smooth average max-Q curve (paper's Figure 2).

## Architecture (summary §2.4)

```
Input  84×84×4 stacked frames
  → Conv 16 @ 8×8 stride 4, ReLU   → 20×20×16
  → Conv 32 @ 4×4 stride 2, ReLU   →  9×9×32   (flatten = 2592)
  → FC 256, ReLU
  → FC K (linear)                  → Q(s, a₁) … Q(s, a_K)
≈ 1.7M trainable parameters
```

## Project Structure

```
src/dqn_atari/
├── config.py            # Dataclass configs (Model/Env/Replay/Train/Eval/Experiment)
├── preprocessing.py     # φ pipeline + Gymnasium Atari env factory (make_env)
├── replay_buffer.py     # FIFO uniform-sample replay buffer
├── models/dqn.py        # The 2013 DQN CNN (§4.1)
├── agent.py             # ε-greedy policy + Q-learning TD loss/update
├── training/trainer.py  # Algorithm 1 main loop
├── evaluation/evaluate.py  # ε=0.05 rollouts, avg-maxQ, video recording
└── utils/seed.py        # Reproducibility
test_dqn.py              # Verification tests
notebooks/experiment.ipynb
```

## Setup

Requires Python ≥3.9 and (for full training) an NVIDIA GPU. Atari ROMs ship with modern `ale-py` / `gymnasium[atari]` — no separate download needed.

### Using uv (recommended, isolated venv)

This repo's devcontainer ships a CUDA-matched build of PyTorch system-wide. Create the project venv with `--system-site-packages` so it **inherits that working `torch`** rather than pulling a PyPI build that may be too new for the host NVIDIA driver:

```bash
cd "Deep Reinforcement Learning"
uv venv --system-site-packages --python /usr/bin/python
uv pip install --python .venv/bin/python -e ".[dev]"

# uv may install its own torch into the venv; if `torch.cuda.is_available()`
# becomes False, drop it so the system CUDA build shows through:
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# if False:  uv pip uninstall --python .venv/bin/python torch triton
```

Run everything via `.venv/bin/python` (or `source .venv/bin/activate`).

### Plain pip / your own environment

```bash
cd "Deep Reinforcement Learning"
pip install -e ".[dev]"
```

If you manage CUDA yourself, install a `torch` build matching your driver from <https://pytorch.org> before/after this step.

## Quick Start

### Verify the install

```bash
pytest test_dqn.py -v
```

### Smoke run (minutes, any hardware)

```python
from dqn_atari import ExperimentConfig, DQN, DQNAgent, ReplayBuffer, make_env, env_spec, train, set_seed
import torch

cfg = ExperimentConfig.quick_test()          # ~20k frames — pipeline check only
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.train.device = str(device)

env, eval_env = make_env(cfg.env, seed=cfg.train.seed), make_env(cfg.env, seed=cfg.train.seed + 1)
obs_shape, n_actions = env_spec(env)
cfg.model.num_actions = n_actions

set_seed(cfg.train.seed, env)
agent = DQNAgent(DQN(cfg.model), n_actions, cfg.train, device)
buffer = ReplayBuffer(cfg.replay.capacity, obs_shape, device)
history = train(env, agent, buffer, cfg, eval_env)
```

### Notebook (recommended)

```bash
jupyter notebook notebooks/experiment.ipynb
```

The notebook runs training, plots the **episode-reward** (noisy) and **avg-maxQ** (smooth) curves side by side — the paper's Figure 2 contrast — and records a video of the trained agent.

## Configuration

```python
ExperimentConfig.default()       # Full Pong run (~2M frames, CUDA)
ExperimentConfig.quick_test()    # Tiny smoke test (~20k frames)

# Flat overrides are routed to the matching sub-config:
ExperimentConfig.default(use_target_network=True, total_frames=1_500_000, lr=1e-4)
```

**DQN variant.** The default is faithful to the 2013 workshop paper — **no target network** (the target uses a detached forward of the online net). Set `use_target_network=True` for the more stable 2015-style periodically-cloned target network.

## Expected Results (Pong)

| Config | Frames | Mean eval reward (ε=0.05) | Notes |
|---|---|---|---|
| `quick_test()` | ~20k | ≈ −20 (random-ish) | Smoke test only — verifies the pipeline, not performance |
| `default()` | ~1–2M | **≥ 15** (target 20) | Agent wins matches; smooth rising avg-maxQ curve |

Paper reports an average score of **20** on Pong (Table 1). Reaching ≥15 with a smooth, non-divergent max-Q curve reproduces the paper's core finding. Enabling the target network typically converges at least as fast and stably.

## Computational Requirements

- **GPU:** ~1–2M frames on Pong is a few hours on a single modern NVIDIA GPU.
- **Memory:** the replay buffer dominates RAM. Default capacity `250k` stacked uint8 `84×84×4` frames ≈ 7 GB. The paper uses `1e6` (≈28 GB); raise `replay.capacity` if you have the RAM, or lower it on constrained machines.
- **CPU-only:** use `quick_test()`; full training is impractical on CPU.

## Troubleshooting

- **`gymnasium.error.NamespaceNotFound: ALE`** — install Atari extras: `pip install "gymnasium[atari]" ale-py`. ROMs are bundled with recent `ale-py`.
- **Out of memory (host RAM)** — lower `replay.capacity` (e.g. `ExperimentConfig.default(capacity=100_000)`).
- **CUDA OOM** — the model is tiny; OOM usually means another process holds the GPU. Lower `batch_size` if needed.
- **Slow / unstable learning** — enable the target network: `ExperimentConfig.default(use_target_network=True)`.

## Faithfulness Notes

- Reward clipping to `{-1, 0, +1}`, frame-skip 4, frame-stack 4, ε annealed 1.0→0.1 over 1M frames, RMSProp, minibatch 32 (summary §2.2.7).
- The RMSProp learning rate (`2.5e-4`) and epsilon (`1e-2`) are unspecified in the workshop paper; we adopt the values from the 2015 Nature follow-up (summary §2.5).
- φ resizes to 84×84 via Gymnasium's `AtariPreprocessing`; `preprocess_frame` keeps the exact downsample-then-crop reference used by the tests.

## Citation

```bibtex
@article{mnih2013playing,
  title={Playing Atari with Deep Reinforcement Learning},
  author={Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex
          and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin},
  journal={arXiv preprint arXiv:1312.5602},
  year={2013}
}
```

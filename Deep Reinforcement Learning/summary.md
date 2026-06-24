# Deep Reinforcement Learning for Atari Games

**Paper**: Playing Atari with Deep Reinforcement Learning **Authors**: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, Martin Riedmiller **Affiliation**: DeepMind Technologies **Venue**: NIPS Deep Learning Workshop, 2013 **arXiv**: [1312.5602](https://arxiv.org/abs/1312.5602)

---

## 1. Summary

**Problem**: Reinforcement learning had historically depended on hand-crafted feature representations because combining model-free RL with non-linear function approximators (especially neural networks) had been shown to diverge. The challenge was learning control policies _end-to-end_ from raw high-dimensional sensory input (pixels) without any task-specific feature engineering.

**Approach**: A convolutional neural network ("Deep Q-Network", DQN) approximates the optimal action-value function $Q^*(s,a)$. It is trained with a variant of Q-learning that adds two key stabilising mechanisms: (i) **experience replay**, which stores transitions in a buffer and samples them uniformly at random for SGD updates, breaking temporal correlations; and (ii) a fixed action-set CNN that outputs one Q-value per action in a single forward pass. The network is trained on raw $84{\times}84{\times}4$ stacked grayscale frames.

**Key Contributions**:

1. The first deep neural network trained end-to-end from raw pixels via reinforcement learning to play complex video games at human level.
2. Demonstration that experience replay stabilises Q-learning with a non-linear function approximator, sidestepping the well-known Baird-style divergence pathology.
3. A _single_ network architecture and _single_ hyperparameter configuration that learns competently on 7 Atari games, providing strong evidence for generality.
4. The architectural pattern of "state-in, vector-of-Q-values-out", enabling $\mathcal{O}(1)$-action evaluation per state.

**Results**: DQN beat the previous best learning methods (Sarsa with hand-crafted features, Contingency, HyperNEAT) on 6 of 7 games and surpassed an expert human player on Breakout, Enduro, and Pong.

**Significance**: This paper launched the modern era of deep reinforcement learning. Almost every subsequent advance in the field — Double DQN, Dueling DQN, Rainbow, A3C, PPO applied to vision, AlphaGo's value network, world models — descends technically and culturally from this work.

![[dqn_architecture_2013.svg|697]]
---

## 2. Deep Analysis

### 2.1 Problem & Motivation

**Research Question**: Can a single neural network, trained end-to-end from raw pixels, learn to play diverse high-dimensional control tasks with a single set of hyperparameters?

**Context**: By 2013, deep CNNs had achieved breakthrough results in supervised vision (AlexNet, 2012). RL, by contrast, was still operating with linear function approximators and hand-engineered features (e.g., BASS / contingency-aware features in the Arcade Learning Environment). Three obstacles had blocked the marriage of deep learning and RL:

1. **Sparse, delayed, scalar rewards** — supervised learning has dense per-example labels; RL credit assignment may span thousands of timesteps.
2. **Non-i.i.d. data** — consecutive RL samples are highly correlated, violating SGD assumptions.
3. **Non-stationary distribution** — the data distribution shifts as the policy improves.

**Prior Work Limitations**:

- TD-Gammon (Tesauro, 1995) succeeded with an MLP + on-policy TD on backgammon, but extensions to chess/Go/checkers failed, suggesting backgammon's stochastic dice was special.
- Tsitsiklis & Van Roy (1997) and Baird (1995) proved that off-policy + non-linear function approximator + bootstrapping (the "deadly triad") can diverge.
- Neural Fitted Q-Iteration (NFQ; Riedmiller, 2005) used batch RPROP updates that scale poorly with data size and had not been demonstrated on raw high-dimensional vision.
- Bellemare et al.'s ALE baselines used hand-engineered BASS features and per-color binary maps — they injected substantial visual prior knowledge.

**Gap Addressed**: An end-to-end pixel-to-action RL agent that scales via SGD and avoids the deadly triad's divergence in practice.

---

### 2.2 Technical Approach

#### 2.2.1 RL Formalism

The agent interacts with an environment $\mathcal{E}$ over discrete timesteps. At each step it observes a raw screen $x_t \in \mathbb{R}^d$, takes an action $a_t \in \mathcal{A} = {1, \dots, K}$, and receives a scalar reward $r_t$.

Because individual frames are partially observable (e.g., one Pong frame doesn't tell you the ball's velocity), the _state_ is defined as the full history sequence:

$$ s_t = x_1, a_1, x_2, a_2, \dots, a_{t-1}, x_t $$

This makes the problem a finite (but enormous) Markov Decision Process. The future discounted return is

$$ R_t = \sum_{t'=t}^{T} \gamma^{t' - t}, r_{t'} $$

with discount factor $\gamma \in [0, 1)$. The optimal action-value function is

$$ Q^*(s,a) = \max_{\pi} \mathbb{E}\big[R_t \big| s_t = s,a_t = a, \pi \big]. $$

It satisfies the **Bellman optimality equation**:

$$ Q^*(s, a) = \mathbb{E}_{s' \sim \mathcal{E}}\left[ r + \gamma \max_{a'} Q^*(s', a') \big| s, a \right] \quad\text{(Eq. 1 in paper)} $$

#### 2.2.2 Neural Q-Learning

A neural Q-network $Q(s,a;\theta) \approx Q^*(s,a)$ with parameters $\theta$ is trained by minimising a sequence of losses, where each iteration $i$ has its own target:

$$ L_i(\theta_i) = \mathbb{E}_{s, a \sim \rho(\cdot)}\left[\big( y_i - Q(s,a;\theta_i) \big)^2 \right] \quad\text{(Eq. 2)} $$

with the **target**

$$ y_i = \mathbb{E}_{s' \sim \mathcal{E}}\left[r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) \big| s, a \right] $$

and $\rho(s,a)$ the **behaviour distribution** (the empirical state-action distribution generated by the $\epsilon$-greedy policy). The crucial detail is that $\theta_{i-1}$ — _the previous iteration's parameters_ — is held fixed when computing the target. This was a precursor to the explicit "target network" introduced in the 2015 Nature paper.

The gradient of the loss is

$$ \nabla_{\theta_i} L_i(\theta_i) = \mathbb{E}_{s, a \sim \rho s' \sim \mathcal{E}}\left[\Big( r + \gamma \max_{a'} Q(s', a';\theta_{i-1}) - Q(s,a;\theta_i) \Big) \nabla_{\theta_i} Q(s,a;\theta_i) \right] \quad\text{(Eq. 3)} $$

Replacing the expectation with single-sample stochastic estimates (one transition from the replay buffer) recovers the familiar Q-learning update — but applied to a deep CNN.

#### 2.2.3 Experience Replay — The Key Stability Trick

At every environment step, the transition $e_t = (s_t, a_t, r_t, s_{t+1})$ is appended to a FIFO buffer $\mathcal{D}$ of capacity $N = 10^6$. Each gradient step samples a minibatch of 32 transitions _uniformly at random_ from $\mathcal{D}$ rather than using the most recent transition.

Why this works:

- **Data efficiency**: each transition contributes to many updates.
- **Decorrelation**: random sampling breaks the strong serial correlation between consecutive frames, giving SGD i.i.d.-like batches.
- **Distributional smoothing**: averaging across many past behaviours prevents pathological feedback loops where a temporary policy bias dominates the training distribution and amplifies itself.
- **Off-policy compatibility**: because the buffer is a mix of behaviours from older parameter snapshots, the learner _must_ be off-policy — which is exactly what Q-learning is.

The paper notes that uniform sampling is suboptimal (it "gives equal importance to all transitions"), foreshadowing **prioritised experience replay** (Schaul et al., 2016).

#### 2.2.4 Algorithm 1 — Deep Q-Learning with Experience Replay

```
Initialise replay memory D with capacity N
Initialise action-value function Q with random weights θ
for episode = 1 to M:
    Initialise sequence s₁ = {x₁}, preprocessed φ₁ = φ(s₁)
    for t = 1 to T:
        With probability ε, select random action aₜ
        Otherwise select aₜ = argmax_a Q(φ(sₜ), a; θ)
        Execute aₜ in emulator; observe rₜ, x_{t+1}
        Set s_{t+1} = sₜ, aₜ, x_{t+1}; preprocess φ_{t+1} = φ(s_{t+1})
        Store transition (φₜ, aₜ, rₜ, φ_{t+1}) in D
        Sample random minibatch of (φⱼ, aⱼ, rⱼ, φ_{j+1}) from D
        Set yⱼ = rⱼ                              if φ_{j+1} terminal
                = rⱼ + γ · max_a' Q(φ_{j+1}, a'; θ)  otherwise
        Perform SGD step on (yⱼ − Q(φⱼ, aⱼ; θ))² w.r.t. θ
```

#### 2.2.5 Preprocessing Function $\phi$

Raw Atari frames are $210 \times 160$ pixels with a 128-color palette — too large and noisy to feed directly. The preprocessing pipeline $\phi$:

1. Convert RGB → grayscale.
2. Downsample to $110 \times 84$.
3. Crop to an $84 \times 84$ playing-area square (square required by the cuda-convnet conv kernel).
4. Stack the **last 4 frames** to form an $84 \times 84 \times 4$ tensor — this restores enough of the Markov property to recover velocity / direction information.

#### 2.2.6 Architecture (Section 4.1)

The exact CNN is specified in prose in the paper rather than as a figure. The architecture is:

|Layer|Type|Specification|Output shape|
|---|---|---|---|
|Input|—|Stack of 4 grayscale frames|$84 \times 84 \times 4$|
|Conv1|Conv 2D + ReLU|16 filters, $8 \times 8$, stride 4|$20 \times 20 \times 16$|
|Conv2|Conv 2D + ReLU|32 filters, $4 \times 4$, stride 2|$9 \times 9 \times 32$|
|FC1|Dense + ReLU|256 units|$256$|
|Output|Dense (linear)|One unit per action ($K = 4{-}18$)|$K$|

The two design choices worth highlighting:

- **One Q-value per action, not (state, action) → Q**: prior work fed both state and action as input, so getting Q-values for all $K$ actions required $K$ forward passes. DQN does it in one pass — essential when action selection happens every frame at 60 Hz.
- **Convolutional, not fully-connected**: spatial weight-sharing makes learning from $\sim$28k input pixels tractable on a single 2013-era GPU.

(The architecture is small by modern standards — no pooling, no batch-norm, only 2 conv layers — but matches what was feasible on then-current hardware. The 2015 Nature follow-up scaled this up to 3 conv layers and 512 FC units.)

#### 2.2.7 Training Tricks

- **Reward clipping**: all positive rewards clipped to $+1$, negative to $-1$. This standardises gradients across games but loses magnitude information.
- **Frame-skip $k=4$** ($k=3$ on Space Invaders to keep lasers visible): the agent picks an action every 4 frames; the same action repeats on skipped frames. This effectively quadruples wall-clock throughput.
- **$\epsilon$-greedy exploration**: $\epsilon$ annealed linearly from 1.0 to 0.1 over the first 1M frames, then fixed at 0.1.
- **Optimiser**: RMSProp, minibatch 32.
- **Total training**: 10M frames per game; replay capacity 1M frames.

---

### 2.3 Experimental Evaluation

**Benchmark**: 7 Atari 2600 games via the Arcade Learning Environment — Beam Rider, Breakout, Enduro, Pong, Q*bert, Seaquest, Space Invaders.

**Baselines**:

|Method|Type|Notes|
|---|---|---|
|Random|Trivial floor|Uniform random action|
|Sarsa (Bellemare 2013)|Linear, hand-engineered features|BASS feature set|
|Contingency (Bellemare 2012)|Linear + learned controllable-region features|Stronger than Sarsa|
|HNeat Best (Hausknecht 2013)|Evolutionary, hand-coded object detector|Deterministic eval (cherry-picked)|
|HNeat Pixel (Hausknecht 2013)|Evolutionary on 8-channel object map|Deterministic eval|
|Human expert|—|Median over $\sim$2 hours of play|

**Main Results (Table 1, average score, $\epsilon$-greedy with $\epsilon=0.05$):**

|Method|B. Rider|Breakout|Enduro|Pong|Q*bert|Seaquest|S. Invaders|
|---|---|---|---|---|---|---|---|
|Random|354|1.2|0|−20.4|157|110|179|
|Sarsa|996|5.2|129|−19|614|665|271|
|Contingency|1743|6|159|−17|960|723|268|
|**DQN**|**4092**|**168**|**470**|**20**|**1952**|**1705**|**581**|
|Human|7456|31|368|−3|18900|28010|3690|
|HNeat Best|3616|52|106|19|1800|920|1720|
|HNeat Pixel|1332|4|91|−16|1325|800|1145|
|DQN Best|5184|225|661|21|4500|1740|1075|

**Headline findings**:

- DQN beats every other learning method on all 7 games on average score.
- DQN exceeds _human_ performance on Breakout (168 vs 31), Enduro (470 vs 368), and Pong (20 vs −3).
- DQN remains far below human on Q*bert and Seaquest — games that demand long-horizon planning over many seconds of game-time.

**No formal ablation table**, but the paper offers two pieces of indirect evidence:

- The single-architecture / single-hyperparameter result across 7 games is implicitly a generality ablation.
- The $Q$-value learning curves (Figure 2, right two plots) show monotone smooth improvement — evidence that bootstrapping with a deep CNN does not diverge in this regime, despite no theoretical guarantee.

**Statistical rigour**: No error bars, no multi-seed runs, no significance testing. Numbers are point estimates from a single training run per game.

---

### 2.4 Architecture & Figure Walkthrough

Since the paper relies on textual description rather than an architecture diagram, I'll walk through both my reconstructed architecture diagram and the three figures in the paper.

#### Reconstructed Architecture Diagram

```
   Raw frame                Stack last 4              CNN
   210×160 RGB              84×84×4                   |
       │                       │                      │
       ▼                       ▼                      ▼
   ┌────────┐  φ          ┌─────────┐           ┌──────────────┐
   │  RGB → │ ────▶       │ stacked │ ────▶     │ Conv 16@8×8  │  stride 4, ReLU
   │  gray  │             │ frames  │           │ → 20×20×16   │
   │ +crop  │             └─────────┘           └──────┬───────┘
   └────────┘                                          ▼
                                                 ┌──────────────┐
                                                 │ Conv 32@4×4  │  stride 2, ReLU
                                                 │ → 9×9×32     │
                                                 └──────┬───────┘
                                                        ▼
                                                 ┌──────────────┐
                                                 │ Flatten +    │
                                                 │ FC 256, ReLU │
                                                 └──────┬───────┘
                                                        ▼
                                                 ┌──────────────┐
                                                 │ FC linear    │  K outputs
                                                 │ Q(s, a₁)…Q(s, aₖ)
                                                 └──────────────┘
```

The shape arithmetic: $\lfloor (84 - 8)/4 \rfloor + 1 = 20$, then $\lfloor (20 - 4)/2 \rfloor + 1 = 9$. After flattening, $9 \times 9 \times 32 = 2592$ features feed into the 256-unit FC.

Total trainable parameters $\approx$ 1.7M — modest by modern standards but already 2 orders of magnitude beyond TD-Gammon's MLP.

#### Figure 1 — Game Screenshots

Five $210 \times 160$ raw screenshots from the seven test games (left to right: Pong, Breakout, Space Invaders, Seaquest, Beam Rider). The figure does two things: (i) it conveys that the games are visually diverse — different physics, different sprite vocabularies, different scoring mechanics — and (ii) it makes the "raw pixel input" claim concrete. There is no game-specific encoding; the network sees these images, and that is all.

#### Figure 2 — Training Stability Curves

Four side-by-side plots, one row, on Breakout (cols 1, 3) and Seaquest (cols 2, 4):

|Plot|y-axis|x-axis|Behaviour|
|---|---|---|---|
|Avg reward — Breakout|Episode reward (eval, $\epsilon=0.05$, 10k steps)|Training epoch (1 epoch = 50k SGD updates ≈ 30 min)|Noisy upward trend, peak ~250|
|Avg reward — Seaquest|same|same|Very noisy, peak ~1700|
|Avg max Q — Breakout|$\max_a Q(s,a)$ over a fixed held-out state set|epoch|Smooth monotone rise to ~3.5|
|Avg max Q — Seaquest|same|epoch|Smooth monotone rise to ~9|

**The point of this figure is not the absolute numbers — it's the contrast between the left and right panels.** Episode reward is a high-variance stochastic estimate of policy quality; small policy shifts can drastically change visited-state distributions and hence reward. The average max-Q on a _fixed_ held-out state set is far smoother and shows that the _value estimates themselves_ are improving steadily. This is the paper's main empirical defence against the "Q-learning + deep nets diverges" prior. No theoretical convergence guarantee is offered, but the curves never diverge across 100 epochs on either game.

#### Figure 3 — Value Function Trajectory on Seaquest

A line plot of predicted $\max_a Q(s_t, a)$ across a 30-frame segment of Seaquest, with three frames (A, B, C) called out. The story:

- **Frame A**: an enemy submarine appears on screen → predicted value spikes (the agent recognises an opportunity to score).
- **Frame B**: the agent has fired a torpedo and the torpedo is about to connect → predicted value reaches its peak (return is imminent and nearly certain).
- **Frame C**: enemy is destroyed and disappears → predicted value relaxes back to baseline.

This is qualitative but compelling: it shows the network has learned an internally consistent value function that rises and falls in synchrony with the game's actual reward dynamics, rather than acting as an opaque pattern-matching policy. It functions as the paper's interpretability anecdote.

---

### 2.5 Critical Assessment

#### Strengths

1. **Genuine end-to-end learning from pixels.** No object detector, no per-color channel split, no hand-coded controllable-region features (which both stronger baselines used). The agent receives only $84 \times 84 \times 4$ grayscale and the scalar reward.
2. **Generality across games.** A single architecture and single hyperparameter setting works across 7 visually and mechanically distinct games (the only deviation: $k=3$ vs $k=4$ frame-skip on Space Invaders, openly disclosed).
3. **The experience replay insight is correct and durable.** Every modern off-policy deep-RL algorithm (DDPG, TD3, SAC, Rainbow, MuZero) uses some form of replay. The paper's framing of _why_ it works (decorrelation + averaging behaviour distributions) has held up.
4. **Architectural choice "state-in, K-Q-values-out".** Single forward pass for action selection — the right choice and now universal in DQN-family methods.
5. **Honest failure modes.** Q*bert and Seaquest results are presented unspun; the paper explicitly identifies long-horizon strategy as the limitation.

#### Weaknesses

1. **No statistical rigour.** Single seed per game, no error bars, no significance tests. With RL's notorious seed sensitivity (Henderson et al. 2018), the absolute numbers should be read as suggestive.
2. **No ablation studies.** The paper claims experience replay matters but doesn't quantify it. How much does replay vs. no-replay degrade performance? What's the effect of buffer size, frame stack depth, frame-skip? Inferring causality from this paper alone is hard.
3. **Reward clipping conflates magnitude.** Clipping $r \in {-1, 0, +1}$ destroys the agent's ability to distinguish a 1-point reward from a 100-point reward. This is acknowledged briefly but explains some of the failure on Q*bert / Seaquest where reward magnitudes encode strategic significance.
4. **No target network (yet).** The paper holds $\theta_{i-1}$ fixed _within an iteration's loss formula_ but in practice all weights are updated continuously. The 2015 Nature follow-up adds a periodically-cloned target network — explicitly to fix instability that this paper papers over.
5. **Frame stacking is a hack for partial observability.** Four frames is enough to recover velocity in most Atari games but fails on games with longer-horizon hidden state. A recurrent architecture would be the principled fix (DRQN, Hausknecht & Stone 2015).
6. **Game selection bias.** 7 games is a small sample, and the choice was not blind. The 2015 Nature paper's expansion to 49 games revealed many games where the 2013-era DQN is mediocre.
7. **Reproducibility.** No code released with the workshop paper. Hyperparameters are mostly specified but the optimiser learning rate and momentum / RMSProp ε aren't clearly stated in the workshop version (the Nature version specifies them: lr $2.5 \times 10^{-4}$, momentum 0.95, $\epsilon = 10^{-2}$).

#### Questions Raised

- Would performance scale with bigger CNNs, longer training, larger replay buffers? (Yes — answered by Nature 2015 and Rainbow 2018.)
- Is divergence avoided or merely deferred? (Partially deferred — target networks were needed for true robustness.)
- How dependent are results on RMSProp specifically? (Adam works similarly in practice.)
- How essential is reward clipping? (Quite — but later work like Pop-Art normalisation, Mnih et al. 2016, gave a principled alternative.)

---

### 2.6 Connections & Insights

**Direct intellectual descendants**:

- **DQN-Nature (2015)**: same backbone + target network + larger capacity → 49 games, human-level on majority.
- **Double DQN (2015)**: decouples action selection from value estimation in the target → fixes Q-overestimation.
- **Dueling DQN (2016)**: separate streams for $V(s)$ and $A(s,a)$.
- **Prioritised replay (2016)**: addresses the "uniform sampling is suboptimal" remark made in this paper.
- **Rainbow (2018)**: combines six DQN variants; baseline for any new value-based method.
- **R2D2, NGU, Agent57**: recurrent + intrinsic motivation extensions.
- **AlphaGo / AlphaZero / MuZero**: value network ancestry traces here.

**Non-obvious observations**:

- The "two stable plots, two noisy plots" of Figure 2 is the paper's most underappreciated contribution. It legitimised value-function-based progress monitoring in deep RL — a practice now standard.
- The frame-skip choice ($k=4$) is doing more than throughput: it changes the effective time horizon and discount, shifting which strategies are learnable. Too low, and the network can't see consequences; too high, and it can't react.
- The $\epsilon$-greedy schedule (1.0 → 0.1 over 1M frames) is conservative — modern work (NoisyNets, parameter-space noise) shows you can do better, but this simple schedule is robust enough to be the workshop default a decade later.

**Potential applications beyond games (subsequent decade)**:

- Robotics from pixels (Levine et al. visuomotor policies)
- Recommender systems (off-policy value learning)
- Computer-use agents (vision → discrete action mapping)
- Network packet routing, datacentre cooling (DeepMind's later applied work)

---

## 3. Research Gaps and Future Work

### 3.1 Acknowledged Limitations

- Uniform replay sampling is suboptimal (foreshadowing prioritised replay).
- Reward clipping prevents differentiation of reward magnitudes.
- Long-horizon strategy games (Q*bert, Seaquest) remain difficult.

### 3.2 Unacknowledged Gaps

- **No theoretical convergence guarantee** — the deadly triad is dodged empirically, not solved.
- **Q-value overestimation bias** from the $\max$ operator (later quantified and fixed in Double DQN).
- **Sample inefficiency**: 10M frames per game ≈ 50+ hours of game-time per game; no human plays for that long to reach competence.
- **No generalisation across games** — each game requires a fresh model. Multi-task RL was an open problem this paper did not address.
- **Catastrophic forgetting in the replay buffer**: with FIFO eviction, transitions from early-policy regions are lost permanently.

### 3.3 Assumptions to Challenge

- **MDP assumption via 4-frame stacking.** Many games have hidden state spanning seconds; 4 frames at $k=4$ skip = 16-frame window. Insufficient for memory-heavy games (Montezuma's Revenge).
- **Stationary environment dynamics during the lifetime of the buffer.** True for Atari, false for many real-world settings.
- **Discrete, low-cardinality action space.** $K \le 18$ allows the K-output architecture; doesn't extend to continuous control without a different formulation (DDPG).

### 3.4 Missing Experiments

- Replay buffer size sweep.
- Frame-stack depth ablation (1, 2, 4, 8 frames).
- Network capacity sweep (depth, width).
- Learning-rate / RMSProp sensitivity.
- Multi-seed runs for confidence intervals.
- Comparison against on-policy methods (REINFORCE, A2C avant-la-lettre).
- Transfer learning across games (freeze conv stack, retrain head).

### 3.5 Future Research Directions (subsequently realised)

1. **Stabilisation**: target networks (Nature 2015), Double DQN, Dueling DQN, distributional RL (C51, QR-DQN).
2. **Efficient exploration**: count-based bonuses, NoisyNets, RND, pseudo-counts (these papers came directly out of DQN's exploration weakness).
3. **Memory**: recurrent Q-networks (DRQN), transformer policies.
4. **Sample efficiency**: model-based RL (World Models, MuZero, DreamerV3).
5. **Generalisation**: Procgen benchmark, domain randomisation, multi-task value functions.
6. **Continuous control**: DDPG, TD3, SAC.
7. **Offline RL**: a natural extension of "everything from a replay buffer" — what if the buffer is fixed?

### 3.6 Extension Ideas Worth Re-Visiting Today

- DQN with modern vision backbones (ViT, ConvNeXt) on the full 57-game suite — does the 2013 backbone hold modern Atari back, or does sample inefficiency dominate?
- DQN with foundation-model representations (CLIP / DINO frozen features → small Q-head) — would self-supervised features make the 10M-frame requirement collapse?
- Replay distillation: train a small model on a large agent's replay buffer as an offline RL benchmark for game-playing agents.

---

## 4. Reproduction Codebase

A working reproduction targeting **Pong** (the simplest of the 7 games and the fastest to converge — typically learnable in ~1–2M frames on a single H100) is provided in `reproduction/`.

### Scope and Success Criterion

- **Target**: Pong, average reward $\geq 15$ (paper reports 20).
- **Compute budget**: 1× H100 SXM 80GB, ~3–6 hours wall-clock.
- **Success criterion**: agent consistently wins matches 21:5 or better after $\sim$1.5M training frames.

### Structure

```
reproduction/
├── README.md              # Setup, usage, expected results
├── requirements.txt       # Pinned dependencies
├── configs/
│   └── pong.yaml          # All paper hyperparameters
├── src/
│   ├── preprocessing.py   # φ: RGB→gray→resize→crop→stack
│   ├── replay_buffer.py   # FIFO uniform-sample buffer
│   ├── model.py           # The exact CNN from §4.1
│   ├── agent.py           # ε-greedy + Q-learning update
│   ├── train.py           # Algorithm 1 main loop
│   └── evaluate.py        # ε=0.05 evaluation rollouts
└── scripts/
    └── run_pong.sh        # Single-command launcher
```

See the `reproduction/` directory for the complete implementation. The code follows the paper's specifications faithfully; comments cite section / equation numbers throughout.

---

## 5. Verdict

|Dimension|Assessment|
|---|---|
|**Contribution Level**|**Breakthrough** — defined a research field.|
|**Reproducibility (workshop version)**|**Medium** — architecture clear, exact RMSProp params and seeds missing. The 2015 Nature follow-up fills these gaps.|
|**Impact**|Among the top 5 most influential ML papers of the 2010s. The "deep RL" subfield essentially begins here.|
|**Recommendation**|Foundational reading for anyone building agentic / control / decision-making systems. Pair with the 2015 Nature paper for production-quality details.|

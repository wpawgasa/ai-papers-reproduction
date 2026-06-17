"""Deep Q-Learning with Experience Replay -- Algorithm 1 (summary Section 2.2.4).

The main loop interleaves environment interaction (epsilon-greedy), storing
transitions in the replay buffer, and SGD updates on uniformly-sampled
minibatches.  Evaluation (episode reward + held-out avg-maxQ) is run
periodically and recorded in a history dict for plotting.
"""

from collections import deque

import numpy as np
from tqdm import tqdm

from ..agent import DQNAgent
from ..evaluation.evaluate import avg_max_q, collect_holdout_states, evaluate_agent
from ..preprocessing import obs_to_array
from ..replay_buffer import ReplayBuffer


def train(env, agent: DQNAgent, buffer: ReplayBuffer, config, eval_env=None, verbose: bool = True) -> dict:
    """Run the full DQN training loop.

    Args:
        env: Training environment (preprocessed, frame-stacked).
        agent: A :class:`DQNAgent` wrapping the online (and optional target) net.
        buffer: A :class:`ReplayBuffer`.
        config: An :class:`ExperimentConfig`.
        eval_env: Optional separate environment for evaluation. Falls back to
            ``env`` if not provided.
        verbose: Whether to show a progress bar and periodic logs.

    Returns:
        ``history`` dict with per-event lists keyed by frame count.
    """
    tcfg, rcfg, vcfg = config.train, config.replay, config.eval
    eval_env = eval_env if eval_env is not None else env

    optimizer = agent.build_optimizer()

    # Fixed held-out state set for the smooth avg-maxQ curve (Figure 2 right).
    holdout_states = collect_holdout_states(eval_env, vcfg.n_holdout_states)

    history = {
        "frames": [],  # frame index at each training-loss record
        "loss": [],
        "episode_frames": [],  # frame index at each episode end
        "episode_rewards": [],  # raw (clipped) episode return during training
        "eval_frames": [],  # frame index at each evaluation
        "eval_reward": [],  # mean eval episode reward (epsilon=0.05)
        "eval_maxq": [],  # avg max-Q on the held-out states
        "epsilon": [],  # epsilon at each evaluation
    }

    recent_rewards = deque(maxlen=20)
    obs, _ = env.reset(seed=tcfg.seed)
    obs = obs_to_array(obs)
    episode_reward = 0.0
    next_eval = vcfg.eval_freq_frames
    progress = tqdm(range(1, tcfg.total_frames + 1), disable=not verbose, desc="train", unit="frame")

    for frame in progress:
        epsilon = agent.current_epsilon(frame)

        # --- act ---
        action = agent.select_action(obs, epsilon)
        next_obs_raw, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_obs = obs_to_array(next_obs_raw)

        # --- store transition ---
        buffer.push(obs, action, reward, next_obs, terminated)  # store true terminal, not truncation
        obs = next_obs
        episode_reward += float(reward)

        if done:
            recent_rewards.append(episode_reward)
            history["episode_frames"].append(frame)
            history["episode_rewards"].append(episode_reward)
            episode_reward = 0.0
            obs, _ = env.reset()
            obs = obs_to_array(obs)

        # --- learn ---
        if len(buffer) >= rcfg.min_size_to_train and frame % tcfg.train_freq == 0:
            batch = buffer.sample(rcfg.batch_size)
            loss = agent.update(batch, optimizer)
            history["frames"].append(frame)
            history["loss"].append(loss)

        # --- sync target network (only if enabled) ---
        if tcfg.use_target_network and frame % tcfg.target_update_freq == 0:
            agent.sync_target()

        # --- periodic evaluation ---
        if frame >= next_eval:
            next_eval += vcfg.eval_freq_frames
            eval_stats = evaluate_agent(agent, eval_env, vcfg.eval_episodes, vcfg.eval_epsilon)
            maxq = avg_max_q(agent, holdout_states)
            history["eval_frames"].append(frame)
            history["eval_reward"].append(eval_stats["mean_reward"])
            history["eval_maxq"].append(maxq)
            history["epsilon"].append(epsilon)
            if verbose:
                progress.write(
                    f"[frame {frame:>9,}] eval_reward={eval_stats['mean_reward']:6.2f} "
                    f"avg_maxQ={maxq:6.3f} eps={epsilon:.3f} buffer={len(buffer):,}"
                )

        # --- lightweight console log of recent training return ---
        if verbose and frame % tcfg.log_freq_frames == 0 and recent_rewards:
            progress.set_postfix(
                avg_return=f"{np.mean(recent_rewards):.1f}",
                eps=f"{epsilon:.2f}",
            )

    progress.close()
    return history

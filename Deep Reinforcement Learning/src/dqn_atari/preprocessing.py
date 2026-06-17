"""Preprocessing function phi and Atari environment factory.

The 2013 paper's phi (summary Section 2.2.5):
    1. RGB -> grayscale
    2. downsample to 110x84
    3. crop to an 84x84 playing-area square
    4. stack the last 4 frames -> 84x84x4

``preprocess_frame`` implements this faithfully and is the reference used by the
verification tests.  For the live training environment we compose Gymnasium's
well-tested ``AtariPreprocessing`` (noop-reset, k-frame skip with max-pool over
the last two raw frames, grayscale, 84x84 resize) with frame stacking.  The only
deviation from phi is that ``AtariPreprocessing`` resizes the full frame to
84x84 rather than crop-after-downsample; this is the standard modern convention
and does not affect Pong.
"""

from typing import Tuple

import cv2
import numpy as np

cv2.setNumThreads(0)  # avoid thread oversubscription with DataLoader-free RL loop


def preprocess_frame(frame: np.ndarray, screen_size: int = 84) -> np.ndarray:
    """Apply phi to a single raw Atari RGB frame (summary Section 2.2.5).

    Args:
        frame: Raw frame, shape ``(210, 160, 3)`` uint8 (RGB).
        screen_size: Side length of the final square crop (default 84).

    Returns:
        ``(screen_size, screen_size)`` uint8 grayscale array.
    """
    # 1. RGB -> grayscale (luma weights).
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # 2. Downsample to 110x84 (height x width).
    resized = cv2.resize(gray, (screen_size, 110), interpolation=cv2.INTER_AREA)

    # 3. Crop the 84x84 playing area. Atari score/UI sits at the top, so we take
    #    the bottom-biased central band: rows [18, 102).
    top = 18
    cropped = resized[top : top + screen_size, :]
    return cropped.astype(np.uint8)


def _import_frame_stack():
    """Return a frame-stacking wrapper (class, kwarg-name) across Gymnasium versions."""
    try:  # gymnasium >= 1.0
        from gymnasium.wrappers import FrameStackObservation

        return FrameStackObservation, "stack_size"
    except ImportError:  # gymnasium 0.29.x
        from gymnasium.wrappers import FrameStack

        return FrameStack, "num_stack"


def make_env(env_config, render_mode: str = None, seed: int = None):
    """Build a preprocessed, frame-stacked Atari environment.

    Args:
        env_config: An :class:`EnvConfig`.
        render_mode: Optional Gymnasium render mode (e.g. ``"rgb_array"`` for
            video recording).
        seed: Optional reset seed.

    Returns:
        A Gymnasium environment whose observations are ``(frame_stack, 84, 84)``
        uint8 arrays and whose ``action_space.n`` gives the number of actions.
    """
    import gymnasium as gym

    # Register ALE environments (no-op if already registered).
    try:
        import ale_py

        gym.register_envs(ale_py)
    except (ImportError, AttributeError):
        pass

    from gymnasium.wrappers import AtariPreprocessing, TransformReward

    # frameskip=1 + repeat_action_probability=0 so AtariPreprocessing controls the
    # k-frame skip and there are no sticky actions (faithful to the 2013 setup).
    env = gym.make(
        env_config.env_id,
        frameskip=1,
        repeat_action_probability=0.0,
        render_mode=render_mode,
        max_num_frames_per_episode=env_config.max_episode_steps,
    )

    env = AtariPreprocessing(
        env,
        noop_max=env_config.noop_max,
        frame_skip=env_config.frame_skip,
        screen_size=env_config.screen_size,
        terminal_on_life_loss=env_config.terminal_on_life_loss,
        grayscale_obs=env_config.grayscale,
        scale_obs=False,  # keep uint8 [0, 255]; the model normalizes
    )

    FrameStack, kw = _import_frame_stack()
    env = FrameStack(env, **{kw: env_config.frame_stack})

    if env_config.clip_rewards:
        # Clip rewards to {-1, 0, +1} (summary Section 2.2.7).
        env = TransformReward(env, lambda r: float(np.sign(r)))

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    return env


def obs_to_array(obs) -> np.ndarray:
    """Coerce a (possibly LazyFrames) stacked observation to a uint8 ndarray."""
    return np.asarray(obs, dtype=np.uint8)


def env_spec(env) -> Tuple[Tuple[int, int, int], int]:
    """Return ``(obs_shape, num_actions)`` for a built environment."""
    obs_shape = tuple(env.observation_space.shape)
    num_actions = int(env.action_space.n)
    return obs_shape, num_actions

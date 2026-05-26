"""Environment inspection and seeding utilities."""
from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np


def get_env_dims(env: gym.Env) -> Tuple[int, int]:
    """Return (obs_dim, action_dim) for a discrete-action environment."""
    obs_space = env.observation_space
    act_space = env.action_space

    if not isinstance(obs_space, gym.spaces.Box):
        raise ValueError(f"Expected Box observation space, got {type(obs_space)}")
    if not isinstance(act_space, gym.spaces.Discrete):
        raise ValueError(f"Expected Discrete action space, got {type(act_space)}")

    obs_dim = int(np.prod(obs_space.shape))
    action_dim = int(act_space.n)
    return obs_dim, action_dim


def set_global_seed(seed: int) -> None:
    import random
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

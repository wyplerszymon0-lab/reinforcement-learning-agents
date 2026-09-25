"""Render a greedy episode of a trained agent to an animated GIF."""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from src.agents.base_agent import BaseAgent


def record_episode(
    agent: BaseAgent,
    env_id: str,
    path: Path,
    seed: int = 0,
    frame_skip: int = 2,
    scale: int = 2,
    fps: int = 30,
    max_steps: int = 1000,
    deterministic: bool = True,
) -> float:
    """Play one episode, save every `frame_skip`-th frame downscaled by `scale`.

    deterministic=False samples actions from the policy instead of taking the greedy one.

    Returns the episode's return so the caption can quote it.
    """
    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    frames, total, done, step = [], 0.0, False, 0

    while not done and step < max_steps:
        if step % frame_skip == 0:
            frames.append(np.asarray(env.render())[::scale, ::scale])
        obs, reward, terminated, truncated, _ = env.step(agent.act(obs, training=not deterministic))
        total += float(reward)
        done = terminated or truncated
        step += 1
    frames.append(np.asarray(env.render())[::scale, ::scale])
    env.close()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1000 * frame_skip / fps, loop=0)
    return total

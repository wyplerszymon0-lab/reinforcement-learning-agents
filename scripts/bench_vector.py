"""Training throughput of PPO with 1 vs N environment copies.

    python scripts/bench_vector.py --env cartpole --steps 100000 --envs 1 2 4 8

Same hyperparameters as config.yaml, and the same number of transitions per update
(num_steps is divided by N), so only the collection changes. Timed: trainer.train().
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train import ENVS, build_agent, load_hyperparams  # noqa: E402
from src.environments.utils import get_env_dims, set_global_seed  # noqa: E402
from src.environments.wrappers import make_env, make_vec_env  # noqa: E402
from src.training.trainer import PPOTrainer  # noqa: E402


def steps_per_second(env_name: str, num_envs: int, steps: int, seed: int = 0) -> float:
    env_id, _ = ENVS[env_name]
    hp = dict(load_hyperparams("ppo", env_name), total_timesteps=steps)
    set_global_seed(seed)
    env: Any  # one gym.Env, or a vector of them
    if num_envs > 1:
        hp = dict(hp, num_envs=num_envs, num_steps=hp["num_steps"] // num_envs)
        env = make_vec_env(env_id, num_envs, seed=seed)
        dims = get_env_dims(env.envs[0])
    else:
        env = make_env(env_id, seed=seed)
        dims = get_env_dims(env)
    agent = build_agent("ppo", *dims, hp)
    trainer = PPOTrainer(agent, env, total_timesteps=steps, checkpoint_interval=10**9, verbose=False)
    start = time.perf_counter()
    trainer.train()
    return agent.total_steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=sorted(ENVS), default="cartpole")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--envs", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)

    base = None
    for n in args.envs:
        runs = sorted(steps_per_second(args.env, n, args.steps, seed=s) for s in range(args.repeats))
        median = runs[len(runs) // 2]
        base = base or median
        print(f"{args.env:12s} N={n}: {median:7.0f} steps/s (median of {args.repeats}), x{median / base:.2f}", flush=True)


if __name__ == "__main__":
    main()

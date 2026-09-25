"""
Train one agent on one environment with the hyperparameters from config.yaml.

Writes the training log and evaluation to results/runs/<algo>_<env>_seed<seed>.json
and the final weights to results/models/<algo>_<env>_seed<seed>.pt.

Usage:
    python scripts/train.py --env cartpole --algo dqn --seed 1
    python scripts/train.py --env lunarlander --algo ppo --seed 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch
import yaml

from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.environments.utils import get_env_dims, set_global_seed
from src.environments.wrappers import make_env
from src.evaluation.evaluator import Evaluator
from src.training.callbacks import CallbackList, EarlyStoppingCallback
from src.training.trainer import DQNTrainer, PPOTrainer

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "config.yaml"

ENVS = {
    # name: (gymnasium id, "solved" threshold on the 100-episode mean return)
    "cartpole": ("CartPole-v1", 475.0),
    "lunarlander": ("LunarLander-v3", 200.0),
}
EVAL_EPISODES = 100


def load_hyperparams(algo: str, env: str) -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return dict(yaml.safe_load(f)[algo][env])


def build_agent(algo: str, obs_dim: int, action_dim: int, hp: dict):
    hp = dict(hp, hidden_dims=tuple(hp["hidden_dims"]))
    if algo == "dqn":
        hp.pop("total_timesteps")
        return DQNAgent(obs_dim=obs_dim, action_dim=action_dim, **hp)
    return PPOAgent(obs_dim=obs_dim, action_dim=action_dim, **hp)


def solved_at_step(rewards, steps, threshold, window=100):
    """First environment step at which the trailing `window`-episode mean hit the threshold."""
    for i in range(window - 1, len(rewards)):
        if np.mean(rewards[i - window + 1 : i + 1]) >= threshold:
            return int(steps[i])
    return None


def run(env_name: str, algo: str, seed: int, timesteps: int | None, out_dir: Path) -> dict:
    env_id, threshold = ENVS[env_name]
    hp = load_hyperparams(algo, env_name)
    if timesteps is not None:
        hp["total_timesteps"] = timesteps

    set_global_seed(seed)
    env = make_env(env_id, seed=seed)
    obs_dim, action_dim = get_env_dims(env)
    agent = build_agent(algo, obs_dim, action_dim, hp)

    trainer_cls = DQNTrainer if algo == "dqn" else PPOTrainer
    trainer = trainer_cls(
        agent=agent,
        env=env,
        total_timesteps=hp["total_timesteps"],
        checkpoint_dir=ROOT / "checkpoints" / env_name / f"{algo}_seed{seed}",
        callbacks=CallbackList([EarlyStoppingCallback(reward_threshold=threshold)]),
        # progress line roughly every 20k environment steps
        log_interval=20_000 if algo == "dqn" else max(1, 20_000 // hp["num_steps"]),
    )

    print(f"[{algo}/{env_name}/seed{seed}] training for up to {hp['total_timesteps']:,} steps", flush=True)
    start = time.time()
    metrics = trainer.train()
    wall_time = time.time() - start
    env.close()

    eval_env = gym.make(env_id)
    eval_env.reset(seed=10_000 + seed)
    evaluator = Evaluator(eval_env, n_episodes=EVAL_EPISODES)
    evaluation = evaluator.evaluate(agent)
    # PPO learns a stochastic policy; report sampling from it as well as its greedy action.
    sampled = evaluator.evaluate(agent, deterministic=False) if algo == "ppo" else None
    eval_env.close()

    name = f"{algo}_{env_name}_seed{seed}"
    agent.save(out_dir / "models" / f"{name}.pt")

    result = {
        "env": env_name,
        "env_id": env_id,
        "algo": algo,
        "seed": seed,
        "threshold": threshold,
        "hyperparams": hp,
        "total_steps": int(metrics.steps[-1]) if metrics.steps else 0,
        "solved_at_step": solved_at_step(metrics.episode_rewards, metrics.steps, threshold),
        "wall_time_s": round(wall_time, 1),
        "evaluation": evaluation,
        "evaluation_sampled": sampled,
        "episode_rewards": [round(r, 2) for r in metrics.episode_rewards],
        "episode_steps": [int(s) for s in metrics.steps],
    }
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (runs_dir / f"{name}.json").write_text(json.dumps(result), encoding="utf-8")

    solved = result["solved_at_step"]
    print(
        f"[{algo}/{env_name}/seed{seed}] done in {wall_time / 60:.1f} min | "
        f"solved at {solved if solved is not None else 'never'} | "
        f"greedy eval {evaluation['mean_reward']:.1f} ± {evaluation['std_reward']:.1f}"
        + (f" | sampled eval {sampled['mean_reward']:.1f} ± {sampled['std_reward']:.1f}" if sampled else ""),
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=sorted(ENVS), required=True)
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, help="override total_timesteps from config.yaml")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads (small nets: 1 is fastest)")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    run(args.env, args.algo, args.seed, args.timesteps, args.out)


if __name__ == "__main__":
    main()

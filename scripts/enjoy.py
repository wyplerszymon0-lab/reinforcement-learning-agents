"""
Watch a trained agent play, in a window.

Usage:
    python scripts/enjoy.py --env lunarlander --algo dqn --seed 1
    python scripts/enjoy.py --env cartpole --algo ppo --episodes 5 --sample

Uses the saved weights in results/models/ and the hyperparameters recorded in
results/runs/. Without --seed it picks the seed with the best evaluation.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym

from scripts.train import ENVS, load_trained_agent, model_path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def find_run(env: str, algo: str, seed: int | None, results_dir: Path = RESULTS) -> dict:
    """The recorded run to replay: the given seed, or the best-evaluated one with saved weights."""
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((results_dir / "runs").glob(f"{algo}_{env}_seed*.json"))]
    runs = [r for r in runs if model_path(r, results_dir).exists()]
    if seed is not None:
        runs = [r for r in runs if r["seed"] == seed]
    if not runs:
        available = sorted(p.stem for p in (results_dir / "models").glob("*.pt"))
        wanted = f"{algo}_{env}" + (f"_seed{seed}" if seed is not None else "")
        raise SystemExit(f"no saved model for {wanted}; available: {', '.join(available) or 'none'}")
    return max(runs, key=lambda r: r["evaluation"]["mean_reward"])


def play(agent, env_id: str, episodes: int, sample: bool, render_mode: str | None = "human", seed: int = 0) -> list[float]:
    """Play `episodes` episodes and return their returns. render_mode=None runs without a window."""
    env = gym.make(env_id, render_mode=render_mode)
    returns = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            done, total = False, 0.0
            while not done:
                obs, reward, terminated, truncated, _ = env.step(agent.act(obs, training=sample))
                total += float(reward)
                done = terminated or truncated
            returns.append(total)
            print(f"episode {ep + 1}: return {total:.1f}")
    finally:
        env.close()
    return returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=sorted(ENVS), required=True)
    parser.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--seed", type=int, help="training seed to replay (default: best evaluated)")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sample", action="store_true",
                        help="sample actions (PPO's stochastic policy; ε-greedy for DQN) instead of the greedy action")
    args = parser.parse_args()

    run = find_run(args.env, args.algo, args.seed)
    mode = "sampled" if args.sample else "greedy"
    print(f"{args.algo.upper()} on {run['env_id']}, seed {run['seed']}, {mode} - Ctrl+C to stop")
    play(load_trained_agent(run, RESULTS), run["env_id"], args.episodes, args.sample)


if __name__ == "__main__":
    main()

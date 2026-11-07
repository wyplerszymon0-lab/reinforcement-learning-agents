"""
How far a trained DQN's Q-values are from the returns its greedy policy actually gets.

Q(s, a) is meant to be the discounted return from s after taking a. Playing greedy
episodes and computing that return afterwards gives a direct check: a positive mean
of Q - G is overestimation, the bias Double DQN is meant to reduce.

Episodes cut by a time limit (CartPole stops at 500 steps) need care: near the cut
the return falls to zero although the state is no worse, so Q is right to stay high.
States whose remaining steps could still be worth more than `tail` of the return
(gamma ** steps_left > tail) are left out of the per-step figures for those episodes.

Usage:
    python scripts/q_bias.py                      # published models and every ablation
    python scripts/q_bias.py --episodes 50
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
import torch

from scripts.train import ENVS, ROOT, load_trained_agent

RESULTS = ROOT / "results"
TAIL = 0.05
EPISODE_SEED = 50_000  # far from training (seed) and evaluation (10_000 + seed) seeds


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """G_t = r_t + gamma * G_{t+1}, with G = 0 after the last step."""
    out = np.zeros(len(rewards))
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def kept_steps(length: int, truncated: bool, gamma: float, tail: float = TAIL) -> int:
    """How many leading steps of an episode to score. All of them if it ended for real;
    if a time limit cut it, only those with more than log(tail) / log(gamma) steps left."""
    if not truncated:
        return length
    needed = math.ceil(math.log(tail) / math.log(gamma))
    return max(0, length - needed)


def play(agent, env_id: str, episodes: int, seed: int = EPISODE_SEED):
    """Greedy episodes: per episode the Q of each chosen action, the rewards and whether
    a time limit ended it."""
    env = gym.make(env_id)
    out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        qs, rewards, done, truncated = [], [], False, False
        while not done:
            with torch.no_grad():
                q = agent.online_net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].cpu().numpy()
            action = int(q.argmax())
            qs.append(float(q[action]))
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            done = terminated or truncated
        out.append((qs, rewards, bool(truncated)))
    env.close()
    return out


def summarise(episodes, gamma: float, tail: float = TAIL) -> dict:
    """Bias at the first state and over all scored states, absolute and relative to |G|."""
    q0: list[float] = []
    g0: list[float] = []
    q_all: list[float] = []
    g_all: list[float] = []
    returns: list[float] = []
    for qs, rewards, truncated in episodes:
        g = discounted_returns(rewards, gamma)
        q0.append(qs[0])
        g0.append(g[0])
        n = kept_steps(len(rewards), truncated, gamma, tail)
        q_all.extend(qs[:n])
        g_all.extend(g[:n])
        returns.append(sum(rewards))
    q_all_arr, g_all_arr = np.array(q_all), np.array(g_all)
    start_bias = float(np.mean(q0) - np.mean(g0))
    step_bias = float(np.mean(q_all_arr - g_all_arr)) if len(q_all) else float("nan")
    return {
        "episodes": len(episodes),
        "mean_return": float(np.mean(returns)),
        "q_start": float(np.mean(q0)),
        "g_start": float(np.mean(g0)),
        "start_bias": start_bias,
        "start_bias_rel": start_bias / abs(float(np.mean(g0))),
        "scored_steps": len(q_all),
        "step_bias": step_bias,
        "step_bias_rel": step_bias / float(np.mean(np.abs(g_all_arr))) if len(q_all) else float("nan"),
    }


def find_runs(results: Path = RESULTS) -> list[dict]:
    """Every DQN run with a saved model: published ones and those under ablations/<tag>/."""
    paths = sorted((results / "runs").glob("dqn_*.json")) + sorted(results.glob("ablations/*/runs/dqn_*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    rows = []
    for run in find_runs():
        agent = load_trained_agent(run, RESULTS)
        stats = summarise(play(agent, ENVS[run["env"]][0], args.episodes), run["hyperparams"]["gamma"])
        rows.append({"env": run["env"], "seed": run["seed"], "variant": run.get("tag", "published"),
                     "double_dqn": run["hyperparams"]["double_dqn"], **stats})
        print(f"{run['env']:11s} seed{run['seed']} {rows[-1]['variant']:10s} "
              f"return {stats['mean_return']:7.1f}  start bias {stats['start_bias_rel']:+.1%}  "
              f"per-step bias {stats['step_bias_rel']:+.1%} ({stats['scored_steps']} steps)", flush=True)
    out = RESULTS / "ablations" / "q_bias.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"episodes": args.episodes, "tail": TAIL, "rows": rows}, indent=1), encoding="utf-8")
    print(f"written {out}")


if __name__ == "__main__":
    main()

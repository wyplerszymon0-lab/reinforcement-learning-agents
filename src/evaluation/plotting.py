"""
Plotting utilities for training diagnostics and algorithm comparisons.

All functions save figures as PNG and optionally return the Figure object.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")  # Non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np


_STYLE: Dict[str, Any] = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "lines.linewidth": 1.5,
    "font.family": "monospace",
}


def _apply_style() -> None:
    plt.style.use(_STYLE)


def smooth(values: List[float], window: int = 20) -> np.ndarray:
    """Simple moving average over `window` episodes."""
    arr = np.array(values, dtype=np.float32)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training_curve(
    rewards: List[float],
    title: str = "Training Curve",
    smoothing_window: int = 20,
    save_path: Optional[Path] = None,
    color: str = "#58a6ff",
) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.25, color=color, label="Raw")
    if len(rewards) >= smoothing_window:
        smoothed = smooth(rewards, smoothing_window)
        offset = len(rewards) - len(smoothed)
        ax.plot(
            np.arange(offset, len(rewards)),
            smoothed,
            color=color,
            label=f"MA-{smoothing_window}",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_comparison(
    results: Dict[str, List[float]],
    title: str = "Algorithm Comparison",
    smoothing_window: int = 20,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlay multiple algorithms' training curves on the same axes."""
    _apply_style()
    colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff"]
    fig, ax = plt.subplots(figsize=(12, 6))

    for (name, rewards), color in zip(results.items(), colors):
        episodes = np.arange(len(rewards))
        ax.plot(episodes, rewards, alpha=0.2, color=color)
        if len(rewards) >= smoothing_window:
            smoothed = smooth(rewards, smoothing_window)
            offset = len(rewards) - len(smoothed)
            ax.plot(
                np.arange(offset, len(rewards)),
                smoothed,
                color=color,
                label=name,
            )
        else:
            ax.plot(episodes, rewards, color=color, label=name)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reward_distribution(
    rewards_by_agent: Dict[str, List[float]],
    title: str = "Evaluation Reward Distribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _apply_style()
    colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for (name, rewards), color in zip(rewards_by_agent.items(), colors):
        ax.hist(rewards, bins=20, alpha=0.6, label=name, color=color)

    ax.set_xlabel("Episode Return")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_loss_curve(
    losses: List[float],
    label: str = "Loss",
    save_path: Optional[Path] = None,
    color: str = "#f78166",
) -> plt.Figure:
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    updates = np.arange(len(losses))
    ax.plot(updates, losses, alpha=0.3, color=color)
    if len(losses) > 50:
        smoothed = smooth(losses, 50)
        offset = len(losses) - len(smoothed)
        ax.plot(np.arange(offset, len(losses)), smoothed, color=color, label=label)
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title(f"{label} over Training")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_seeds_by_steps(
    runs: Dict[str, List[tuple]],
    title: str = "Learning curves",
    threshold: Optional[float] = None,
    window: int = 100,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Compare algorithms across seeds on an environment-step axis.

    `runs` maps an algorithm name to a list of (episode_end_steps, episode_rewards)
    pairs, one per seed. Each seed's trailing `window`-episode mean is interpolated
    onto a common step grid; the line is the mean over seeds and the band spans the
    best and worst seed. Steps (not episodes) make the x-axis comparable between
    algorithms whose episodes have very different lengths.
    """
    _apply_style()
    colors = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for (name, seeds), color in zip(runs.items(), colors):
        # Start where every seed has finished at least one episode.
        grid = np.linspace(max(steps[0] for steps, _ in seeds), max(steps[-1] for steps, _ in seeds), 400)
        per_seed = []
        for steps, rewards in seeds:
            returns = np.asarray(rewards, dtype=np.float64)
            trailing = np.array([returns[max(0, i - window + 1) : i + 1].mean() for i in range(len(returns))])
            # A seed that stopped early (solved) holds its final value to the end of the grid.
            per_seed.append(np.interp(grid, steps, trailing, right=trailing[-1]))
        curves = np.vstack(per_seed)
        ax.plot(grid, curves.mean(axis=0), color=color, label=f"{name} ({len(seeds)} seeds)")
        ax.fill_between(grid, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.2)

    if threshold is not None:
        ax.axhline(threshold, color="#8b949e", linestyle=":", label=f"solved ({threshold:g})")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k")
    )
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(f"Mean return (last {window} episodes)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_noise_robustness(
    payload: Dict[str, Any],
    algo_names: Dict[str, str],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Mean return vs observation-noise level, one panel per environment.

    `payload` is the structure written by scripts/robustness.py. The line is the
    mean over seeds and the band spans the best and worst seed.
    """
    _apply_style()
    colors = {"dqn": "#58a6ff", "ppo": "#3fb950"}
    levels = np.asarray(payload["noise_levels"])
    envs = payload["envs"]
    fig, axes = plt.subplots(1, len(envs), figsize=(6 * len(envs), 4.5), squeeze=False)

    for ax, env in zip(axes[0], envs.values()):
        for algo in algo_names:
            seeds = np.asarray([r["mean_return"] for r in env["runs"] if r["algo"] == algo])
            if not len(seeds):
                continue
            ax.plot(levels, seeds.mean(axis=0), marker="o", color=colors[algo],
                    label=f"{algo_names[algo]} ({len(seeds)} seeds)")
            ax.fill_between(levels, seeds.min(axis=0), seeds.max(axis=0), color=colors[algo], alpha=0.2)
        ax.axhline(env["threshold"], color="#8b949e", linestyle=":", label=f"solved ({env['threshold']:g})")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xticks(levels)
        ax.set_xticklabels([f"{l:g}σ" for l in levels])
        ax.set_xlabel("Observation noise (× per-dimension std)")
        ax.set_ylabel("Mean return")
        ax.set_title(env["env_id"])
        ax.legend(loc="lower left")
        ax.grid(True)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig

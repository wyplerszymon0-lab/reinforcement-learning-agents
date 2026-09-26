"""
Plotting utilities for training diagnostics and algorithm comparisons.

All functions save figures as PNG and optionally return the Figure object.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")  # Non-interactive backend — safe for headless servers
import matplotlib.pyplot as plt
import numpy as np


_STYLE = {
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
    plt.rcParams.update(cast(dict[Any, Any], _STYLE))


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
        curves = []
        for steps, rewards in seeds:
            rewards = np.asarray(rewards, dtype=np.float64)
            trailing = np.array([rewards[max(0, i - window + 1) : i + 1].mean() for i in range(len(rewards))])
            # A seed that stopped early (solved) holds its final value to the end of the grid.
            curves.append(np.interp(grid, steps, trailing, right=trailing[-1]))
        stacked = np.vstack(curves)
        ax.plot(grid, stacked.mean(axis=0), color=color, label=f"{name} ({len(seeds)} seeds)")
        ax.fill_between(grid, stacked.min(axis=0), stacked.max(axis=0), color=color, alpha=0.2)

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

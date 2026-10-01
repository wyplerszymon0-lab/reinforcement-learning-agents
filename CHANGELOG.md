# Changelog

All notable changes to this project are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-01

First tagged release: DQN and PPO implemented from scratch in PyTorch, trained and
evaluated on CartPole-v1 and LunarLander-v3 with 3 seeds each.

### Added

- `scripts/enjoy.py` to watch a trained agent play in a window; replays the
  best-evaluated seed unless `--seed` is given (#1).
- Robustness to sensor noise: a seeded `GaussianObservationNoise` wrapper and
  `scripts/robustness.py`, which evaluates every saved model at 6 noise levels and
  writes `results/ROBUSTNESS.md` and a plot (#5).
- mypy type checking of `src/` and `scripts/` in CI (#4).
- Measured results for all four algorithm/environment pairs (3 seeds, 100 evaluation
  episodes) with plots, GIFs and a generated `results/RESULTS.md`.
- One `scripts/train.py` driven by `config.yaml` and `scripts/make_report.py` for
  plots, GIFs and the results table.
- Sampled-policy evaluation for PPO next to greedy evaluation.
- Seed-aggregated learning curves on an environment-step axis.
- GitHub Actions CI running the test suite.

### Changed

- PPO uses separate actor and critic networks, clipped separately.
- DQN updates every `train_freq` environment steps (CartPole: every 4 steps instead of
  every step, which oscillated and collapsed).
- `load_trained_agent()` and `model_path()` live in `scripts/train.py` instead of being
  copied into each script (#1).

### Fixed

- Early stopping never stopped training; trainers now honour the stop flag.
- PPO's shared trunk let the value gradient (~2000× the policy gradient) swamp the
  policy update after global gradient clipping.
- PPO greedy evaluation sampled actions instead of taking the argmax.
- Observation normalisation raises a clear `ValueError` for observation spaces without a
  fixed shape (#4).

### Earlier development

- 2026-05-26: initial implementation of Double DQN with a dueling architecture and PPO
  with GAE, tests, training scripts and a comparison notebook.

[1.0.0]: https://github.com/wyplerszymon0-lab/reinforcement-learning-agents/releases/tag/v1.0.0

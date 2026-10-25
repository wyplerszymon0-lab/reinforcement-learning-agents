# Reinforcement Learning Agents from Scratch

[![CI](https://github.com/wyplerszymon0-lab/reinforcement-learning-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/wyplerszymon0-lab/reinforcement-learning-agents/actions/workflows/ci.yml)

DQN and PPO implemented in pure PyTorch, trained on CartPole-v1 and LunarLander-v3.  
No stable-baselines3. No RLlib. Just the math.

| DQN · CartPole | PPO · CartPole | DQN · LunarLander | PPO · LunarLander |
| :---: | :---: | :---: | :---: |
| ![DQN CartPole](results/gifs/dqn_cartpole.gif) | ![PPO CartPole](results/gifs/ppo_cartpole.gif) | ![DQN LunarLander](results/gifs/dqn_lunarlander.gif) | ![PPO LunarLander](results/gifs/ppo_lunarlander.gif) |

<sub>Best seed of each, final policy, one fresh episode. Rendered by `scripts/make_report.py`.</sub>

## Results

Every configuration was trained with **3 seeds** and evaluated on 100 fresh episodes
([full per-seed table](results/RESULTS.md)).

| Environment | Algorithm | Solved | Median steps to solve | Greedy eval | Sampled eval |
| :--- | :--- | :---: | ---: | ---: | ---: |
| CartPole-v1 | DQN | 1/3 | 464k | 429.7 ± 104.0 | — |
| CartPole-v1 | PPO | 3/3 | 413k | 500.0 ± 0.0 | 487.2 ± 7.6 |
| LunarLander-v3 | DQN | 3/3 | 163k | 217.5 ± 4.5 | — |
| LunarLander-v3 | PPO | 3/3 | 704k | 239.9 ± 6.5 | 243.5 ± 8.4 |

*Solved* = the trailing 100-episode training return reached the official threshold
(CartPole 475, LunarLander 200); training stops there. *Sampled eval* samples actions
from PPO's policy distribution instead of always taking the most likely one.

![DQN vs PPO on LunarLander](results/plots/lunarlander_dqn_vs_ppo.png)
![DQN vs PPO on CartPole](results/plots/cartpole_dqn_vs_ppo.png)

<sub>Line: mean over seeds of the trailing 100-episode return. Band: best to worst seed.
A seed that stopped early (solved) is held at its final value.</sub>

**Takeaways**

- **Both solve LunarLander on every seed; DQN needs far fewer steps.** DQN solves it in
  118k–199k environment steps, tuned PPO in 659k–828k, about 4× more. PPO ends with the
  better policy (greedy eval 240 vs 218) and finishes faster in wall time, because it
  steps 16 environments at once.
- **PPO dominates CartPole** (return ~250 after 50k steps, DQN ~40), solves it on all 3 seeds
  and its greedy policy balances perfectly (500/500) on every seed.
- **DQN on CartPole is the least stable combination**: one seed solves it, the others
  plateau between 300 and 480. Seed-to-seed variance is large for both algorithms,
  which is why single-seed results in RL are not worth much.

### What training revealed

The first full training run exposed three bugs that unit tests had not caught. Each
is fixed and now has a regression test:

1. **Early stopping never stopped anything.** The callback set
   `trainer.total_timesteps = 0`, but the loop had already built its `range(...)`.
   Trainers now check a `stop_training` flag.
2. **PPO barely learned on LunarLander** (−150 after 1M steps). Actor and critic
   shared a trunk, and the value loss (returns squared, ~6,000) produced a gradient
   **~2,000× larger** than the policy's. After global gradient clipping the policy got
   **0.025%** of each step. Fix: separate actor and critic networks, clipped separately.
3. **PPO's "greedy" evaluation was sampling.** `act(training=False)` ignored the flag.

DQN on CartPole also oscillated and collapsed when it took a gradient step on every
environment step (greedy eval 138–170 after 500k steps). Updating every 4 steps
(`train_freq: 4`) made learning monotonic: 310–500.

### Tuning PPO on LunarLander: 1/3 → 3/3 seeds

Until v1.0.0, PPO solved LunarLander on 1 seed of 3. Four changes, tried one at a time
on the same 3 seeds ([full table](results/RESULTS.md#tuning-history-ppo_lunarlander)):

| Change | Solved | Greedy eval per seed |
| :--- | :---: | :--- |
| v1.0.0: 1 env, 1,024 steps per update, minibatches of 256 | 1/3 | 96 / 143 / 200 |
| 16 envs × 1,024 steps per update, minibatches of 64 (RL Zoo setting) | 0/3 | 100 / 161 / 119 |
| + no value-loss clipping | 0/3 | 226 / 218 / 220 |
| + entropy coefficient 0.01 → 0.005 | 2/3 | 120 / 228 / 218 |
| + 2M step budget instead of 1M | **3/3** | 237 / 235 / 247 |

- **The bigger batch alone made things worse.** With 16,384 transitions per update,
  1M steps is only 61 updates, and the policy stalled around a return of 100.
- **Value clipping was starving the critic.** It keeps each new value prediction within
  ±0.2 of the old one, a sensible limit when returns are normalised, not when they are
  in the hundreds. Without it the greedy policy cleared 200 on every seed, though
  training (which samples actions) still averaged ~130.
- **Less entropy bonus** let the sampled policy sharpen towards the greedy one, which is
  what the "solved" criterion measures; and a 2M budget makes the learning rate decay
  half as fast. All three tuning seeds still stopped before 1M steps.
- **Checked on fresh seeds.** Each change was kept because it helped on seeds 1–3, so
  those numbers flatter the setting. Seeds 4–6, never used for tuning, also solved
  **3/3**: at 739k, 865k and 1,553k steps (greedy eval 222 / 243 / 232). The last one
  needed more than 1M steps, so the larger budget matters. These runs are not in the
  results table, which keeps seeds 1–3 like every other row.


### Robustness to sensor noise

How do the trained policies cope when their observations are noisy, as real sensors are? [`scripts/robustness.py`](scripts/robustness.py) adds Gaussian noise to every observation, scaled per dimension to the spread of that reading in clean episodes, and replays every seed greedily ([full table](results/ROBUSTNESS.md)):

> Re-run in October 2026 with the current models, all 12 of which are now kept in `results/models/` (before, only seed 1 of each DQN model was, so the study could not be repeated). The DQN runs retrained bit-for-bit; the PPO CartPole runs, last trained for v1.0.0, did not (the vectorised-environment rework changed their random draws), so their row in the results table was refreshed: 3/3 seeds solved instead of 2/3.

![Return vs observation noise](results/plots/robustness.png)

| Mean return (3 seeds) | clean | 0.2σ | 0.5σ | 1σ |
| :--- | ---: | ---: | ---: | ---: |
| LunarLander · DQN | 220 | 133 | 40 | 1 |
| LunarLander · PPO | **242** | **244** | **243** | **198** |
| CartPole · DQN | 414 | 425 | 393 | 329 |
| CartPole · PPO | **500** | **500** | **500** | **480** |

- **Tuned PPO is both the best clean policy and the most robust.** On LunarLander it scores 242 without noise and still 243 at 0.5σ, while DQN falls from 220 to 133 at 0.2σ (−40%) and 40 at 0.5σ (−82%). With the v1.0.0 PPO (clean 143) the two lines crossed between 0.1σ and 0.2σ; retuning PPO removed the trade-off.
- **PPO on CartPole shrugs off noise** up to half a standard deviation on every reading, and still averages 480 at 1σ.
- A plausible reason, not tested here: PPO is trained on a stochastic policy that already has to act well under its own randomness, while greedy DQN picks the argmax of Q-values that can flip with small input changes.

---


## Algorithms

### Deep Q-Network (DQN)

**Paper:** Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015  
**Extensions:** Double DQN (van Hasselt et al., 2016), Dueling Networks (Wang et al., 2016)

DQN approximates the optimal action-value function via the Bellman optimality equation:

$$Q^*(s, a) = \mathbb{E}\left[r + \gamma \max_{a'} Q^*(s', a') \,\middle|\, s, a\right]$$

The network is trained to minimize the temporal difference (TD) error:

$$\mathcal{L}(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}}\left[\left(y - Q_\theta(s, a)\right)^2\right]$$

where the target $y$ uses a **frozen target network** $\theta^-$:

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a')$$

**Why this works:**
- **Experience replay** breaks temporal correlations in the data stream, enabling stable mini-batch SGD
- **Target network** fixes the non-stationary bootstrap target for $C$ steps, preventing oscillation
- **Double DQN** decouples action *selection* (online net) from *evaluation* (target net), eliminating maximization bias: $y = r + \gamma Q_{\theta^-}(s', \arg\max_{a'} Q_\theta(s', a'))$
- **Dueling architecture** separates $V(s)$ and $A(s,a)$ streams: $Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')$

**Implementation details:**
- Huber loss (smooth L1) over MSE — bounded gradients prevent early training instability
- Hard target updates every 500–1000 steps
- Linear epsilon decay: $\varepsilon_t = \max(\varepsilon_\text{min},\ \varepsilon_0 - t \cdot \Delta\varepsilon)$
- Gradient clipping at 10.0

---

### Proximal Policy Optimization (PPO)

**Paper:** Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv 2017  
**Prerequisite:** GAE — Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016

PPO maximizes a surrogate objective that constrains how much the policy can change in a single update:

$$\mathcal{L}^\text{CLIP}(\theta) = \mathbb{E}_t\left[\min\!\left(r_t(\theta)\,\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\,\hat{A}_t\right)\right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ is the probability ratio.

Advantages are estimated via **GAE**:

$$\hat{A}_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l\,\delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)$$

- $\lambda = 1$: Monte Carlo returns (unbiased, high variance)
- $\lambda = 0$: 1-step TD (biased, low variance)  
- $\lambda = 0.95$: empirical sweet spot

**Why this works:**
- The clip $[1-\varepsilon, 1+\varepsilon]$ acts as a first-order trust region: it prevents the policy ratio from moving far from 1 without expensive Hessian computation (unlike TRPO)
- The `min` ensures the objective is a lower bound on performance — we never *gain* from a bad update
- Advantage normalization (batch-level) further reduces gradient variance
- Multiple epochs over the same rollout with mini-batches improve sample utilization while the clip constraint prevents overfitting to old data

**Implementation details:**
- Orthogonal weight initialization (Saxe et al., 2013) — critical for PPO
- Tanh activations (bounded) in actor-critic trunk
- Entropy bonus $-c_2 H[\pi_\theta]$ encourages exploration
- Learning rate annealing: $\alpha_t = \alpha_0 \cdot (1 - t/T)$
- Value function clipping mirrors policy clip

---

--|---|---|---|
| DQN (Double + Dueling) | CartPole-v1 | ~490 | ~15 | ~80k |
| PPO (GAE) | CartPole-v1 | ~500 | ~3 | ~150k |
| DQN (Double + Dueling) | LunarLander-v3 | ~220 | ~40 | ~350k |
| PPO (GAE) | LunarLander-v3 | ~240 | ~25 | ~600k |

*CartPole-v1 solved threshold: 475 · LunarLander-v3 solved threshold: 200*

---

## Project Structure

```
.
├── src/
│   ├── agents/
│   │   ├── base_agent.py       # Abstract agent interface
│   │   ├── networks.py         # DQNNetwork, ActorCriticNetwork, MLP
│   │   ├── dqn.py              # DQN + Double DQN + Dueling
│   │   └── ppo.py              # PPO + GAE + clipped objective
│   ├── environments/
│   │   ├── wrappers.py         # RewardScaling, NormalizeObs, EpisodeMonitor
│   │   └── utils.py            # get_env_dims, set_global_seed
│   ├── training/
│   │   ├── replay_buffer.py    # ReplayBuffer (off-policy), RolloutBuffer (on-policy)
│   │   ├── trainer.py          # DQNTrainer, PPOTrainer with metric logging
│   │   ├── callbacks.py        # EarlyStopping, Logging callbacks
│   │   └── loggers.py          # optional TensorBoard scalar logger
│   └── evaluation/
│       ├── evaluator.py        # Greedy evaluation with trajectory collection
│       └── plotting.py         # Training curves, comparisons, distributions
├── tests/                      # 75 pytest tests
├── scripts/
│   ├── train.py                # train one algo/env/seed from config.yaml
│   ├── make_report.py          # plots, GIFs and RESULTS.md from all runs
│   ├── robustness.py           # evaluation under observation noise
│   └── enjoy.py                # watch a trained agent play
├── notebooks/
│   └── dqn_vs_ppo_comparison.ipynb
├── results/
│   ├── runs/                   # per-seed training logs + evaluations (JSON)
│   ├── models/                 # weights of the best seed per algo/env
│   ├── plots/  gifs/           # generated figures
│   └── RESULTS.md              # generated results table
└── config.yaml                 # all hyperparameters (single source of truth)
```

---

## Quick Start

```bash
# Install
pip install -e ".[dev,notebooks]"

# Run tests
pytest

# Type check (runs in CI too)
mypy

# Train one configuration (hyperparameters come from config.yaml)
python scripts/train.py --env cartpole --algo ppo --seed 1
python scripts/train.py --env lunarlander --algo dqn --seed 1

# Rebuild plots, GIFs and results/RESULTS.md from every run in results/runs/
python scripts/make_report.py

# Watch a trained agent play in a window (best seed unless --seed is given)
python scripts/enjoy.py --env lunarlander --algo dqn

# Also log training curves to TensorBoard (logs/tensorboard/<algo>_<env>_seed<N>/)
python scripts/train.py --env cartpole --algo ppo --seed 1 --tensorboard
tensorboard --logdir logs/tensorboard

# PPO on 8 environment copies at once (faster collection, same batch per update)
python scripts/train.py --env lunarlander --algo ppo --seed 1 --num-envs 8
python scripts/bench_vector.py --env lunarlander --envs 1 4 8   # measure it

# Interactive comparison notebook
jupyter notebook notebooks/dqn_vs_ppo_comparison.ipynb
```

### TensorBoard

With `--tensorboard`, training writes TensorBoard scalars next to the usual JSON results. Every point uses the environment step as its x-axis, so DQN and PPO runs line up:

| Tag | DQN | PPO |
| :--- | :---: | :---: |
| `episode/return`, `episode/length` | every episode | every episode |
| `train/loss` | mean over each 1 000 steps | — |
| `train/epsilon` | every 1 000 steps | — |
| `train/policy_loss`, `train/value_loss` | — | every update |
| `train/entropy` (mean policy entropy) | — | every update |
| `train/approx_kl`, `train/clip_fraction` | — | every update |

![TensorBoard scalars for a short DQN and PPO run on CartPole](results/plots/tensorboard.png)

<sub>Two short CartPole demo runs (60 000 steps, seed 1) made to show the logging, not the runs in the results table above. PPO's KL and entropy shrink as its policy settles; DQN's epsilon decays linearly while its loss grows with the size of the bootstrapped targets.</sub>

The `tensorboard` package is optional (`pip install -e ".[tensorboard]"`, included in `dev`); without the flag nothing changes.

### Vectorised environments for PPO

`--num-envs N` runs N copies of the environment in a `gymnasium.vector.SyncVectorEnv` and PPO collects from all of them at once: one forward pass picks N actions, and GAE runs over a `(steps, envs)` array. `num_steps` from `config.yaml` is divided by N, so every update still learns from the same number of transitions; only the collection changes. Finished copies are reset within the same step (`AutoresetMode.SAME_STEP`); gymnasium's default would insert a no-op step after every episode.

Training throughput, `scripts/bench_vector.py` (median of 3 seeds, one CPU thread, AMD Ryzen 5 220):

| Environment | N = 1 | N = 2 | N = 4 | N = 8 |
| :--- | ---: | ---: | ---: | ---: |
| CartPole-v1 (60k steps) | 909 steps/s | 1 198 (×1.32) | 1 446 (×1.59) | 1 697 (×1.87) |
| LunarLander-v3 (40k steps) | 1 424 steps/s | — | 3 406 (×2.39) | 4 809 (×3.38) |

LunarLander gains more because each step costs more to decide than to simulate: batching the network's forward pass pays off. PPO on LunarLander now trains on 16 copies (`num_envs` in `config.yaml`); elsewhere the default is 1, and `--num-envs` overrides either.

A vector of one environment reproduces a single-environment run exactly (same episodes, same steps), which a test checks; others check that GAE over N copies equals GAE of each copy on its own, and that with 4 copies no step is lost or counted twice when copies end episodes at different times.

---

## Key Design Decisions

**Why Huber loss over MSE for DQN?**  
Early in training, TD errors can be large (the target network is random). MSE squares these errors, producing enormous gradients that destabilize the network. Smooth L1 behaves like MSE for small errors and like L1 for large ones — bounded gradient magnitude.

**Why Tanh activations in PPO but ReLU in DQN?**  
Policy gradient methods are sensitive to the scale of activations flowing into the policy head. Tanh bounds outputs, reducing variance in log-probability estimates. DQN's Q-values don't have a natural scale constraint, and ReLU's lack of saturation improves gradient flow through deeper value networks.

**Why orthogonal initialization?**  
Orthogonal matrices preserve gradient norms during the backward pass. In RL where networks are updated many thousands of times with non-i.i.d. data, this prevents the exploding/vanishing gradients that Glorot/He initialization doesn't guard against long-term.

**Why separate replay strategies?**  
DQN is explicitly off-policy: the Bellman backup is valid regardless of the behavior policy. PPO's surrogate objective is derived assuming data was collected under the current policy — using a replay buffer would invalidate the importance ratio $r_t(\theta)$, making the gradient estimate biased.

---

## Hyperparameter Sensitivity

The most impactful hyperparameters by algorithm:

**DQN:**
1. `epsilon_decay_steps` — too fast → insufficient exploration; too slow → slow convergence
2. `target_update_freq` — too frequent → unstable; too infrequent → slow credit assignment
3. `buffer_capacity` — must be large enough that samples are approximately i.i.d.

**PPO:**
1. `clip_coef` (ε) — controls the trust region size; 0.1–0.3 is the empirical range
2. `gae_lambda` — bias/variance tradeoff; 0.95 is nearly universally good
3. `num_steps` — rollout length; longer → more on-policy data but slower updates
4. `clip_vloss` — measured here, not just folklore: with unnormalised LunarLander returns, turning value clipping off lifted greedy eval from 100–161 to 218–226 on all 3 seeds (see "Tuning PPO on LunarLander")

---

## References

1. Mnih et al. (2015). *Human-level control through deep reinforcement learning.* Nature.
2. van Hasselt et al. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI.
3. Wang et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML.
4. Schulman et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.* ICLR.
5. Schulman et al. (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
6. Saxe et al. (2013). *Exact solutions to the nonlinear dynamics of learning in deep linear neural networks.* ICLR.

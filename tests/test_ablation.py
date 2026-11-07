import json
import math

import numpy as np
import pytest

from scripts import q_bias, train


def test_overrides_parse_yaml_values_of_the_same_type():
    hp = {"double_dqn": True, "lr": 1.0e-4, "batch_size": 64, "hidden_dims": [256, 256]}
    out = train.apply_overrides(hp, ["double_dqn=false", "lr=3e-4", "batch_size=32", "hidden_dims=[64, 64]"])
    assert out == {"double_dqn": False, "lr": 3e-4, "batch_size": 32, "hidden_dims": [64, 64]}
    assert hp["double_dqn"] is True  # the input is not changed


def test_an_int_may_replace_a_float_but_becomes_a_float():
    out = train.apply_overrides({"gamma": 0.99}, ["gamma=1"])
    assert out["gamma"] == 1.0 and isinstance(out["gamma"], float)


@pytest.mark.parametrize(
    "item, message",
    [
        ("dubble_dqn=false", "unknown hyperparameter"),
        ("double_dqn=0", "is bool"),
        ("batch_size=true", "is int"),
        ("lr=fast", "is float"),
        ("double_dqn", "expected key=value"),
    ],
)
def test_bad_overrides_are_rejected(item, message):
    with pytest.raises(ValueError, match=message):
        train.apply_overrides({"double_dqn": True, "batch_size": 64, "lr": 1e-4}, [item])


def test_overrides_need_a_tag(tmp_path):
    with pytest.raises(ValueError, match="needs --tag"):
        train.run("cartpole", "dqn", seed=0, timesteps=64, out_dir=tmp_path, overrides=["double_dqn=false"])


def test_a_tagged_run_goes_to_its_ablation_folder(tmp_path):
    result = train.run(
        "cartpole", "dqn", seed=0, timesteps=300, out_dir=tmp_path, overrides=["double_dqn=false"], tag="no-double"
    )
    folder = tmp_path / "ablations" / "no-double"
    saved = json.loads((folder / "runs" / "dqn_cartpole_seed0.json").read_text())
    assert saved["tag"] == result["tag"] == "no-double"
    assert saved["overrides"] == ["double_dqn=false"]
    assert saved["hyperparams"]["double_dqn"] is False
    assert (folder / "models" / "dqn_cartpole_seed0.pt").is_file()
    assert not (tmp_path / "runs").exists()  # published results are untouched
    assert train.model_path(saved, tmp_path) == folder / "models" / "dqn_cartpole_seed0.pt"
    agent = train.load_trained_agent(saved, tmp_path)
    assert agent.double_dqn is False


def test_discounted_returns():
    g = q_bias.discounted_returns([1.0, 1.0, 1.0], 0.5)
    assert g.tolist() == pytest.approx([1.75, 1.5, 1.0])


def test_episodes_that_really_ended_are_scored_in_full():
    assert q_bias.kept_steps(120, truncated=False, gamma=0.99) == 120


def test_a_time_limit_drops_the_steps_whose_tail_still_matters():
    needed = math.ceil(math.log(0.05) / math.log(0.99))  # 299 steps until gamma ** k < 5 %
    assert q_bias.kept_steps(500, truncated=True, gamma=0.99) == 500 - needed
    assert q_bias.kept_steps(200, truncated=True, gamma=0.99) == 0


def test_summary_of_an_exact_q_has_no_bias():
    rewards = [1.0, 2.0, 3.0]
    g = q_bias.discounted_returns(rewards, 0.9)
    s = q_bias.summarise([(list(g), rewards, False)], gamma=0.9)
    assert s["start_bias"] == pytest.approx(0.0)
    assert s["step_bias"] == pytest.approx(0.0)
    assert s["mean_return"] == 6.0
    assert s["scored_steps"] == 3


def test_summary_measures_overestimation_relative_to_the_return():
    rewards = [1.0] * 4
    g = q_bias.discounted_returns(rewards, 1.0 - 1e-9)  # about 4, 3, 2, 1
    s = q_bias.summarise([(list(g + 1.0), rewards, False)], gamma=1.0 - 1e-9)
    assert s["start_bias"] == pytest.approx(1.0)
    assert s["start_bias_rel"] == pytest.approx(0.25)
    assert s["step_bias"] == pytest.approx(1.0)
    assert s["step_bias_rel"] == pytest.approx(1.0 / 2.5)


def test_summary_skips_the_cut_tail_but_keeps_the_start():
    rewards = [1.0] * 10
    g = q_bias.discounted_returns(rewards, 0.5)
    qs = list(g)
    qs[-1] += 100.0  # a wild value in the tail of a truncated episode
    s = q_bias.summarise([(qs, rewards, True)], gamma=0.5)  # 0.5 ** 5 < 5 %: last 5 steps out
    assert s["scored_steps"] == 5
    assert s["step_bias"] == pytest.approx(0.0)


def test_play_records_one_q_per_step_on_a_real_agent(tmp_path):
    result = train.run("cartpole", "dqn", seed=0, timesteps=200, out_dir=tmp_path, tag="t", overrides=[])
    agent = train.load_trained_agent(result, tmp_path)
    episodes = q_bias.play(agent, "CartPole-v1", episodes=2)
    assert len(episodes) == 2
    for qs, rewards, truncated in episodes:
        assert len(qs) == len(rewards) > 0
        assert np.isfinite(qs).all()
        assert isinstance(truncated, bool)

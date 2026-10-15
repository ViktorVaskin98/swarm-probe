import numpy as np
import pytest

from swarm_probe.cost import generation_flops
from swarm_probe.voting import (
    expected_vote_accuracy,
    majority_vote,
    vote_accuracy_curve,
    weighted_vote,
)


def test_majority_vote_picks_most_common() -> None:
    assert majority_vote(["3", "5", "5", None]) == "5"


def test_majority_vote_breaks_ties_by_first_appearance() -> None:
    assert majority_vote(["7", "3", "3", "7"]) == "7"


def test_majority_vote_all_missing() -> None:
    assert majority_vote([None, None]) is None


def test_weighted_vote_can_overrule_majority() -> None:
    assert weighted_vote(["1", "1", "2"], [0.1, 0.1, 0.9]) == "2"


def test_expected_vote_accuracy_full_set_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    assert expected_vote_accuracy(["1", "1", "2"], "1", 3, 10, rng) == 1.0


def test_expected_vote_accuracy_single_sample_equals_hit_rate() -> None:
    rng = np.random.default_rng(0)
    value = expected_vote_accuracy(["1", "2", "2", "2"], "1", 1, 20_000, rng)
    assert value == pytest.approx(0.25, abs=0.02)


def test_expected_vote_accuracy_rejects_large_k() -> None:
    with pytest.raises(ValueError):
        expected_vote_accuracy(["1"], "1", 2, 10, np.random.default_rng(0))


def test_vote_curve_is_bracketed_by_ci() -> None:
    preds = {"a": ["1", "1", "2", "1"], "b": ["5", "4", "4", "4"], "c": ["9", "9", "9", "9"]}
    gold = {"a": "1", "b": "5", "c": "9"}
    points = vote_accuracy_curve(preds, gold, ks=[1, 3], n_draws=100, n_boot=200)
    assert [p.k for p in points] == [1, 3]
    for point in points:
        assert point.ci_low <= point.accuracy <= point.ci_high


def test_generation_flops() -> None:
    assert generation_flops(10, 3, 2) == 100.0

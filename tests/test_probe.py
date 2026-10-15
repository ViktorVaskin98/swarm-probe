import numpy as np

from swarm_probe.probe import evaluate_scores, out_of_fold_scores, vote_comparison


def _toy(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(40), 10)
    labels = rng.integers(0, 2, size=groups.size)
    features = rng.normal(size=(groups.size, 5))
    features[:, 0] += 2.0 * labels
    return features, labels, groups


def test_probe_finds_planted_signal() -> None:
    features, labels, groups = _toy()
    scores = out_of_fold_scores(features, labels, groups)
    result = evaluate_scores(labels, scores, groups, n_boot=200)
    assert result.auc > 0.85
    assert result.ci_low <= result.auc <= result.ci_high
    assert result.within_problem_auc > 0.8


def test_probe_on_noise_is_near_chance() -> None:
    features, labels, groups = _toy()
    features[:, 0] = np.random.default_rng(1).normal(size=labels.size)
    scores = out_of_fold_scores(features, labels, groups)
    assert abs(evaluate_scores(labels, scores, groups, n_boot=200).auc - 0.5) < 0.1


def test_vote_comparison_uses_weights() -> None:
    qids = np.array(["q", "q", "q"])
    result = vote_comparison(qids, ["1", "1", "2"], {"q": "2"}, np.array([0.1, 0.1, 0.9]))
    assert result == {"majority": 0.0, "probe_weighted": 1.0, "oracle_any_correct": 1.0}

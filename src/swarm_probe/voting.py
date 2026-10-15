from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurvePoint:
    k: int
    accuracy: float
    ci_low: float
    ci_high: float


def majority_vote(predictions: Sequence[str | None]) -> str | None:
    counts = Counter(p for p in predictions if p is not None)
    if not counts:
        return None
    top = max(counts.values())
    return next(p for p in predictions if p is not None and counts[p] == top)


def weighted_vote(predictions: Sequence[str | None], weights: Sequence[float]) -> str | None:
    scores: dict[str, float] = {}
    for prediction, weight in zip(predictions, weights, strict=True):
        if prediction is not None:
            scores[prediction] = scores.get(prediction, 0.0) + weight
    if not scores:
        return None
    best = max(scores.values())
    return next(p for p in predictions if p is not None and scores[p] == best)


def expected_vote_accuracy(
    predictions: Sequence[str | None],
    gold: str,
    k: int,
    n_draws: int,
    rng: np.random.Generator,
) -> float:
    n = len(predictions)
    if k > n:
        raise ValueError(f"k={k} exceeds the {n} available samples")
    if k == n:
        return float(majority_vote(predictions) == gold)
    hits = 0
    for _ in range(n_draws):
        subset = rng.choice(n, size=k, replace=False)
        hits += majority_vote([predictions[i] for i in sorted(subset)]) == gold
    return hits / n_draws


def bootstrap_mean_ci(
    values: Sequence[float], n_boot: int, rng: np.random.Generator, alpha: float = 0.05
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    means = rng.choice(array, size=(n_boot, array.size), replace=True).mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def vote_accuracy_curve(
    predictions_by_problem: Mapping[str, Sequence[str | None]],
    gold_by_problem: Mapping[str, str],
    ks: Sequence[int],
    n_draws: int = 200,
    n_boot: int = 2000,
    seed: int = 0,
) -> list[CurvePoint]:
    rng = np.random.default_rng(seed)
    points: list[CurvePoint] = []
    for k in ks:
        per_problem = [
            expected_vote_accuracy(preds, gold_by_problem[qid], k, n_draws, rng)
            for qid, preds in predictions_by_problem.items()
        ]
        low, high = bootstrap_mean_ci(per_problem, n_boot, rng)
        points.append(
            CurvePoint(k=k, accuracy=float(np.mean(per_problem)), ci_low=low, ci_high=high)
        )
    return points

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from swarm_probe.voting import majority_vote, weighted_vote


@dataclass(frozen=True)
class ProbeFeatures:
    correct: np.ndarray
    qid: np.ndarray
    sample_idx: np.ndarray
    answer_len: np.ndarray
    full_logprob: np.ndarray
    layers: np.ndarray
    prefixes: list[int]
    hidden: Mapping[int, np.ndarray]
    available: Mapping[int, np.ndarray]
    prefix_logprob: Mapping[int, np.ndarray]

    @classmethod
    def load(cls, path: Path) -> "ProbeFeatures":
        data = np.load(path, allow_pickle=False)
        prefixes = sorted(int(key.removeprefix("h_")) for key in data.files if key.startswith("h_"))
        return cls(
            correct=data["correct"].astype(int),
            qid=data["qid"],
            sample_idx=data["sample_idx"],
            answer_len=data["answer_len"],
            full_logprob=data["full_logprob"],
            layers=data["layers"],
            prefixes=prefixes,
            hidden={m: data[f"h_{m}"].astype(np.float32) for m in prefixes},
            available={m: data[f"avail_{m}"] for m in prefixes},
            prefix_logprob={m: data[f"prefix_logprob_{m}"] for m in prefixes},
        )


@dataclass(frozen=True)
class AucResult:
    auc: float
    ci_low: float
    ci_high: float
    within_problem_auc: float
    n_samples: int


def out_of_fold_scores(
    features: np.ndarray, labels: np.ndarray, groups: np.ndarray, n_splits: int = 5, c: float = 0.1
) -> np.ndarray:
    scores = np.zeros(len(labels))
    for train, test in GroupKFold(n_splits=n_splits).split(features, labels, groups):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
        model.fit(features[train], labels[train])
        scores[test] = model.predict_proba(features[test])[:, 1]
    return scores


def grouped_bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, groups: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(unique))]
    values = []
    for _ in range(n_boot):
        picked = rng.integers(len(unique), size=len(unique))
        index = np.concatenate([members[g] for g in picked])
        if labels[index].min() == labels[index].max():
            continue
        values.append(roc_auc_score(labels[index], scores[index]))
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def within_problem_auc(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> float:
    aucs = []
    for group in np.unique(groups):
        mask = groups == group
        if labels[mask].min() != labels[mask].max():
            aucs.append(roc_auc_score(labels[mask], scores[mask]))
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_scores(
    labels: np.ndarray, scores: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> AucResult:
    low, high = grouped_bootstrap_auc(labels, scores, groups, n_boot, seed)
    return AucResult(
        auc=float(roc_auc_score(labels, scores)),
        ci_low=low,
        ci_high=high,
        within_problem_auc=within_problem_auc(labels, scores, groups),
        n_samples=len(labels),
    )


def probe_scores_full(features: ProbeFeatures, prefix: int, layer_index: int) -> np.ndarray:
    mask = features.available[prefix]
    scores = np.full(len(features.correct), np.nan)
    scores[mask] = out_of_fold_scores(
        features.hidden[prefix][mask, layer_index], features.correct[mask], features.qid[mask]
    )
    return scores


def vote_comparison(
    qids: np.ndarray,
    predictions: list[str | None],
    gold_by_qid: Mapping[str, str],
    weights: np.ndarray,
) -> dict[str, float]:
    grouped: dict[str, list[tuple[str | None, float]]] = defaultdict(list)
    for qid, prediction, weight in zip(qids, predictions, weights, strict=True):
        grouped[str(qid)].append((prediction, float(weight)))
    majority_hits = weighted_hits = oracle_hits = 0
    for qid, samples in grouped.items():
        gold = gold_by_qid[qid]
        preds = [p for p, _ in samples]
        majority_hits += majority_vote(preds) == gold
        weighted_hits += weighted_vote(preds, [w for _, w in samples]) == gold
        oracle_hits += gold in preds
    n = len(grouped)
    return {
        "majority": majority_hits / n,
        "probe_weighted": weighted_hits / n,
        "oracle_any_correct": oracle_hits / n,
    }

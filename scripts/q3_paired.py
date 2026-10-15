import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from swarm_probe.probe import ProbeFeatures, out_of_fold_scores
from swarm_probe.voting import majority_vote, weighted_vote

PREREGISTERED_PREFIX = -1
PREREGISTERED_LAYER = 16


def per_problem_hits(
    qids: np.ndarray,
    predictions: list[str | None],
    gold: dict[str, str],
    weights: np.ndarray | None,
) -> dict[str, int]:
    grouped: dict[str, list[tuple[str | None, float]]] = defaultdict(list)
    for index, (qid, prediction) in enumerate(zip(qids, predictions, strict=True)):
        grouped[str(qid)].append((prediction, 1.0 if weights is None else float(weights[index])))
    hits = {}
    for qid, samples in grouped.items():
        preds = [p for p, _ in samples]
        choice = (
            majority_vote(preds)
            if weights is None
            else weighted_vote(preds, [w for _, w in samples])
        )
        hits[qid] = int(choice == gold[qid])
    return hits


def paired_bootstrap(
    a: dict[str, int], b: dict[str, int], n_boot: int, seed: int
) -> tuple[float, float, float]:
    keys = sorted(a)
    diff = np.array([b[k] - a[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(low), float(high)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Q3: preregistered probe-weighted vote vs baselines"
    )
    parser.add_argument("features", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--completions", default="qwen2.5-0.5b-instruct__sampled.jsonl")
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    features = ProbeFeatures.load(args.features)
    with (args.run_dir / args.completions).open(encoding="utf-8") as source:
        lookup = {(r["qid"], r["sample_idx"]): r["prediction"] for r in map(json.loads, source)}
    with (args.run_dir / "problems.jsonl").open(encoding="utf-8") as source:
        gold = {r["qid"]: r["gold"] for r in map(json.loads, source)}
    keys = zip(features.qid, features.sample_idx, strict=True)
    predictions = [lookup[(str(qid), int(sample))] for qid, sample in keys]

    layer_index = int(np.flatnonzero(features.layers == PREREGISTERED_LAYER)[0])
    probe = out_of_fold_scores(
        features.hidden[PREREGISTERED_PREFIX][:, layer_index], features.correct, features.qid
    )
    logprob = out_of_fold_scores(
        features.full_logprob.reshape(-1, 1), features.correct, features.qid
    )

    majority = per_problem_hits(features.qid, predictions, gold, None)
    strategies = {
        "majority vote": majority,
        "logprob-weighted vote": per_problem_hits(features.qid, predictions, gold, logprob),
        "probe-weighted vote (preregistered)": per_problem_hits(
            features.qid, predictions, gold, probe
        ),
    }
    rows = []
    for name, hits in strategies.items():
        delta, low, high = paired_bootstrap(majority, hits, n_boot=5000, seed=0)
        rows.append(
            {
                "strategy": name,
                "accuracy": np.mean(list(hits.values())),
                "delta_vs_majority": delta,
                "delta_ci_low": low,
                "delta_ci_high": high,
            }
        )
    logprob_hits = strategies["logprob-weighted vote"]
    probe_hits = strategies["probe-weighted vote (preregistered)"]
    delta, low, high = paired_bootstrap(logprob_hits, probe_hits, n_boot=5000, seed=1)
    rows.append(
        {
            "strategy": "probe minus logprob weighting",
            "accuracy": np.nan,
            "delta_vs_majority": delta,
            "delta_ci_low": low,
            "delta_ci_high": high,
        }
    )
    table = pd.DataFrame(rows)
    args.results.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results / "q3_preregistered.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

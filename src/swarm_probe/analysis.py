import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from swarm_probe.cost import generation_flops
from swarm_probe.voting import bootstrap_mean_ci, vote_accuracy_curve


@dataclass(frozen=True)
class RunData:
    slug: str
    n_params: int
    greedy: pd.DataFrame
    sampled: pd.DataFrame | None


def _read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as source:
        return pd.DataFrame([json.loads(line) for line in source])


def load_runs(run_dir: Path) -> list[RunData]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    runs: list[RunData] = []
    for slug, info in summary.items():
        sampled_path = run_dir / f"{slug}__sampled.jsonl"
        runs.append(
            RunData(
                slug=slug,
                n_params=info["n_params"],
                greedy=_read_jsonl(run_dir / f"{slug}__greedy.jsonl"),
                sampled=_read_jsonl(sampled_path) if sampled_path.exists() else None,
            )
        )
    return sorted(runs, key=lambda run: run.n_params)


def load_gold(run_dir: Path) -> dict[str, str]:
    problems = _read_jsonl(run_dir / "problems.jsonl")
    return dict(zip(problems["qid"], problems["gold"], strict=True))


def mean_sample_flops(frame: pd.DataFrame, n_params: int) -> float:
    flops = [
        generation_flops(n_params, prompt, generated)
        for prompt, generated in zip(frame["prompt_tokens"], frame["generated_tokens"], strict=True)
    ]
    return float(np.mean(flops))


def greedy_table(runs: Sequence[RunData], n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for run in runs:
        hits = run.greedy["correct"].astype(float).to_numpy()
        low, high = bootstrap_mean_ci(hits, n_boot, rng)
        rows.append(
            {
                "model": run.slug,
                "n_params": run.n_params,
                "accuracy": hits.mean(),
                "ci_low": low,
                "ci_high": high,
                "flops_per_problem": mean_sample_flops(run.greedy, run.n_params),
                "mean_generated_tokens": run.greedy["generated_tokens"].mean(),
                "truncated_share": run.greedy["hit_max_tokens"].mean(),
            }
        )
    return pd.DataFrame(rows)


def swarm_table(
    runs: Sequence[RunData],
    gold: dict[str, str],
    ks: Sequence[int],
    n_draws: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    rows = []
    for run in runs:
        if run.sampled is None:
            continue
        predictions: dict[str, list[str | None]] = defaultdict(list)
        for qid, prediction in zip(run.sampled["qid"], run.sampled["prediction"], strict=True):
            predictions[qid].append(prediction if isinstance(prediction, str) else None)
        n_available = min(len(v) for v in predictions.values())
        usable = [k for k in ks if k <= n_available]
        per_sample = mean_sample_flops(run.sampled, run.n_params)
        for point in vote_accuracy_curve(predictions, gold, usable, n_draws=n_draws, seed=seed):
            rows.append(
                {
                    "model": run.slug,
                    "n_params": run.n_params,
                    "k": point.k,
                    "accuracy": point.accuracy,
                    "ci_low": point.ci_low,
                    "ci_high": point.ci_high,
                    "flops_per_problem": point.k * per_sample,
                }
            )
    return pd.DataFrame(rows)

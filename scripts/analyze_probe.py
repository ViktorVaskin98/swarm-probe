import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from swarm_probe.plots import apply_style
from swarm_probe.probe import ProbeFeatures, evaluate_scores, probe_scores_full, vote_comparison

FULL = -1


def load_predictions(path: Path) -> dict[tuple[str, int], str | None]:
    lookup: dict[tuple[str, int], str | None] = {}
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            lookup[(row["qid"], row["sample_idx"])] = row["prediction"]
    return lookup


def load_gold(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as source:
        return {row["qid"]: row["gold"] for row in map(json.loads, source)}


def auc_grid(features: ProbeFeatures) -> tuple[pd.DataFrame, dict[tuple[int, int], np.ndarray]]:
    rows = []
    scores_by_cell: dict[tuple[int, int], np.ndarray] = {}
    for prefix in features.prefixes:
        mask = features.available[prefix]
        labels, groups = features.correct[mask], features.qid[mask]
        for layer_index, layer in enumerate(features.layers):
            scores = probe_scores_full(features, prefix, layer_index)
            scores_by_cell[(prefix, int(layer))] = scores
            result = evaluate_scores(labels, scores[mask], groups)
            rows.append({"prefix": prefix, "feature": f"probe L{int(layer)}", **vars(result)})
        if prefix != 0:
            baseline = features.prefix_logprob[prefix][mask]
            result = evaluate_scores(labels, baseline, groups)
            rows.append({"prefix": prefix, "feature": "mean logprob", **vars(result)})
    length_mask = features.answer_len > 0
    result = evaluate_scores(
        features.correct[length_mask],
        -features.answer_len[length_mask].astype(float),
        features.qid[length_mask],
    )
    rows.append({"prefix": FULL, "feature": "answer length (neg)", **vars(result)})
    return pd.DataFrame(rows), scores_by_cell


def prefix_order(prefix: int) -> tuple[bool, int]:
    return (prefix == FULL, prefix)


def prefix_label(prefix: int) -> str:
    return "full" if prefix == FULL else str(prefix)


def plot_auc(table: pd.DataFrame, out_path: Path) -> None:
    order = sorted(table["prefix"].unique(), key=prefix_order)
    position = {prefix: i for i, prefix in enumerate(order)}
    probes = table[table["feature"].str.startswith("probe")]
    best = probes.loc[probes.groupby("prefix")["auc"].idxmax()]
    best = best.assign(x=best["prefix"].map(position)).sort_values("x")
    logprob = table[table["feature"] == "mean logprob"]
    logprob = logprob.assign(x=logprob["prefix"].map(position)).sort_values("x")
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    pivot = probes.assign(layer=probes["feature"].str.removeprefix("probe L").astype(int))
    pivot = pivot.pivot(index="layer", columns="prefix", values="auc")
    pivot = pivot[sorted(pivot.columns, key=prefix_order)]
    image = axes[0].imshow(pivot.to_numpy(), cmap="viridis", aspect="auto", origin="lower")
    axes[0].set_xticks(range(len(pivot.columns)), [prefix_label(c) for c in pivot.columns])
    axes[0].set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    midpoint = (np.nanmin(pivot.to_numpy()) + np.nanmax(pivot.to_numpy())) / 2
    for (i, j), value in np.ndenumerate(pivot.to_numpy()):
        color = "black" if value > midpoint else "white"
        axes[0].text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=10, color=color)
    axes[0].set_xlabel("Answer tokens seen")
    axes[0].set_ylabel("Layer")
    axes[0].set_title("Probe ROC-AUC by layer and answer prefix")
    fig.colorbar(image, ax=axes[0], fraction=0.046)
    axes[1].errorbar(
        best["x"],
        best["auc"],
        yerr=[best["auc"] - best["ci_low"], best["ci_high"] - best["auc"]],
        marker="o",
        markersize=7,
        linewidth=2,
        capsize=4,
        label="probe, best layer (picked post hoc)",
        color="#2a6f97",
    )
    axes[1].errorbar(
        logprob["x"],
        logprob["auc"],
        yerr=[logprob["auc"] - logprob["ci_low"], logprob["ci_high"] - logprob["auc"]],
        marker="s",
        markersize=7,
        linewidth=2,
        capsize=4,
        label="mean token logprob",
        color="#e07a5f",
    )
    axes[1].axhline(0.5, color="grey", linewidth=0.8, linestyle="--")
    axes[1].set_xticks(range(len(order)), [prefix_label(p) for p in order])
    axes[1].set_xlabel("Answer tokens seen")
    axes[1].set_ylabel("ROC-AUC, failure prediction")
    axes[1].set_title("How early can we predict a wrong answer?")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2/Q3: failure probe and probe-weighted voting")
    parser.add_argument("features", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--completions", default="qwen2.5-0.5b-instruct__sampled.jsonl")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()

    if args.plot_only:
        plot_auc(pd.read_csv(args.results / "q2_probe_auc.csv"), args.figures / "q2_probe_auc.png")
        return

    features = ProbeFeatures.load(args.features)
    table, scores_by_cell = auc_grid(features)
    args.results.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.results / "q2_probe_auc.csv", index=False)
    plot_auc(table, args.figures / "q2_probe_auc.png")
    print(table.round(4).to_string(index=False))

    predictions = load_predictions(args.run_dir / args.completions)
    gold = load_gold(args.run_dir / "problems.jsonl")
    keys = zip(features.qid, features.sample_idx, strict=True)
    preds = [predictions[(str(qid), int(sample))] for qid, sample in keys]
    rows = []
    for (prefix, layer), scores in scores_by_cell.items():
        if np.isnan(scores).any():
            continue
        rows.append(
            {"prefix": prefix, "layer": layer, **vote_comparison(features.qid, preds, gold, scores)}
        )
    votes = pd.DataFrame(rows).sort_values("probe_weighted", ascending=False)
    votes.to_csv(args.results / "q3_weighted_vote.csv", index=False)
    print(votes.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

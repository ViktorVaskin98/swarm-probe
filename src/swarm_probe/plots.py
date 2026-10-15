from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f"]


def display_name(slug: str) -> str:
    return slug.removesuffix("-instruct").replace("qwen", "Qwen").replace("b", "B")


def plot_swarm_vs_single(swarm: pd.DataFrame, greedy: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.8), dpi=160)
    models = list(greedy.sort_values("n_params")["model"])
    colors = {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(models)}
    for model, curve in swarm.groupby("model", sort=False):
        curve = curve.sort_values("k")
        ax.plot(
            curve["flops_per_problem"],
            curve["accuracy"],
            marker="o",
            markersize=3.5,
            color=colors[model],
            label=f"{display_name(model)}: majority vote of k samples",
        )
        ax.fill_between(
            curve["flops_per_problem"],
            curve["ci_low"],
            curve["ci_high"],
            color=colors[model],
            alpha=0.15,
            linewidth=0,
        )
        if "coverage" in curve:
            ax.plot(
                curve["flops_per_problem"],
                curve["coverage"],
                linestyle="--",
                linewidth=1.0,
                color=colors[model],
                alpha=0.7,
                label=f"{display_name(model)}: pass@k (some sample correct)",
            )
        for _, row in curve[curve["k"].isin([1, 4, 16])].iterrows():
            ax.annotate(
                f"k={int(row['k'])}",
                (row["flops_per_problem"], row["accuracy"]),
                textcoords="offset points",
                xytext=(4, -10),
                fontsize=7,
                color=colors[model],
            )
    for _, row in greedy.iterrows():
        ax.errorbar(
            row["flops_per_problem"],
            row["accuracy"],
            yerr=[[row["accuracy"] - row["ci_low"]], [row["ci_high"] - row["accuracy"]]],
            marker="*",
            markersize=11,
            color=colors[row["model"]],
            markeredgecolor="black",
            markeredgewidth=0.6,
            linestyle="none",
            label=f"{display_name(row['model'])}: one greedy answer",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Inference compute per problem, FLOPs (log scale)")
    ax.set_ylabel("GSM8K accuracy")
    ax.set_title("Swarm of small models vs one larger model at equal compute")
    ax.grid(alpha=0.3, which="both", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

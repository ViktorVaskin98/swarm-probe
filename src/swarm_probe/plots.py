from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

PALETTE = ["#2a6f97", "#e07a5f", "#3d405b", "#81b29a", "#f2cc8f"]

STYLE = {
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def display_name(slug: str) -> str:
    return slug.removesuffix("-instruct").replace("qwen", "Qwen").replace("b", "B")


def plot_swarm_vs_single(swarm: pd.DataFrame, greedy: pd.DataFrame, out_path: Path) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    models = list(greedy.sort_values("n_params")["model"])
    colors = {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(models)}
    for model, curve in swarm.groupby("model", sort=False):
        curve = curve.sort_values("k")
        ax.plot(
            curve["flops_per_problem"],
            100 * curve["accuracy"],
            marker="o",
            markersize=5,
            linewidth=2,
            color=colors[model],
        )
        ax.fill_between(
            curve["flops_per_problem"],
            100 * curve["ci_low"],
            100 * curve["ci_high"],
            color=colors[model],
            alpha=0.15,
            linewidth=0,
        )
        if "coverage" in curve:
            ax.plot(
                curve["flops_per_problem"],
                100 * curve["coverage"],
                linestyle="--",
                linewidth=1.5,
                color=colors[model],
                alpha=0.8,
            )
        for _, row in curve[curve["k"].isin([1, 16])].iterrows():
            ax.annotate(
                f"k={int(row['k'])}",
                (row["flops_per_problem"], 100 * row["accuracy"]),
                textcoords="offset points",
                xytext=(5, -14),
                fontsize=10,
                color=colors[model],
            )
    for _, row in greedy.iterrows():
        ax.errorbar(
            row["flops_per_problem"],
            100 * row["accuracy"],
            yerr=[
                [100 * (row["accuracy"] - row["ci_low"])],
                [100 * (row["ci_high"] - row["accuracy"])],
            ],
            marker="*",
            markersize=16,
            color=colors[row["model"]],
            markeredgecolor="black",
            markeredgewidth=0.8,
            linestyle="none",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Inference compute per problem, FLOPs (log scale)")
    ax.set_ylabel("GSM8K accuracy, %")
    ax.set_title("A swarm of small models vs one larger model at equal compute")
    ax.grid(alpha=0.3, which="both", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    model_handles = [
        Line2D([], [], color=colors[m], linewidth=6, label=display_name(m)) for m in models
    ]
    style_handles = [
        Line2D([], [], color="grey", marker="o", linewidth=2, label="majority vote of k samples"),
        Line2D(
            [], [], color="grey", linestyle="--", linewidth=1.5, label="pass@k: some sample correct"
        ),
        Line2D(
            [],
            [],
            color="grey",
            marker="*",
            markersize=14,
            markeredgecolor="black",
            linestyle="none",
            label="one greedy answer",
        ),
    ]
    first = ax.legend(handles=model_handles, loc="lower right", frameon=False, title="Model")
    ax.add_artist(first)
    ax.legend(handles=style_handles, loc="upper left", frameon=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

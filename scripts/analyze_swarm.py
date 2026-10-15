import argparse
from pathlib import Path

from swarm_probe.analysis import greedy_table, load_gold, load_runs, swarm_table
from swarm_probe.plots import plot_swarm_vs_single

KS = [1, 2, 3, 4, 6, 8, 12, 16]


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1: swarm vs single model at equal compute")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()

    runs = load_runs(args.run_dir)
    gold = load_gold(args.run_dir)
    greedy = greedy_table(runs)
    swarm = swarm_table(runs, gold, KS)

    args.results.mkdir(parents=True, exist_ok=True)
    greedy.to_csv(args.results / "q1_greedy.csv", index=False)
    swarm.to_csv(args.results / "q1_swarm.csv", index=False)
    plot_swarm_vs_single(swarm, greedy, args.figures / "q1_swarm_vs_single.png")
    print(greedy.round(4).to_string(index=False))
    print(swarm.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Supplementary reviewer-2 experiments using the unmodified GT-IDEA v2.

python revision_experiments.py --runs 10 --generations 30 --population 20
Histories are best-so-far raw objectives, including generation zero.
Memoization caches exact raw objectives only; it does not change operators,
acceptance, random draws, or FE call counts. These are generation-matched
diagnostics, not an equal-FE comparison against other algorithms.
"""
from pathlib import Path
import argparse
import csv
import json
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from gt_idea_v2 import GTIDEAv2, GTIDEAConfigV2

ROOT = Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_one(graph, target, delta, seed, args):
    cfg = GTIDEAConfigV2(target=target, pop_size=args.population,
                        max_iter=args.generations, seed=seed,
                        diversity_threshold=delta)
    start = time.perf_counter()
    alg = GTIDEAv2(graph, cfg)
    preprocessing = time.perf_counter() - start
    objective, cache = alg._objective, {}
    counts = {"calls": 0, "computed": 0}

    def cached(h):
        counts["calls"] += 1
        key = h.tobytes()
        if key not in cache:
            counts["computed"] += 1
            cache[key] = objective(h)
        return cache[key]

    alg._objective = cached
    start = time.perf_counter()
    result = alg.run()
    elapsed = time.perf_counter() - start
    # Check the returned solution on the original graph objective.
    assert objective(result.best_hexagram) == result.best_value
    assert np.isfinite(result.history).all()
    assert np.all(np.diff(result.history) <= 0)
    return result, {"best": result.best_value, "objective_calls": counts["calls"],
                    "unique_objectives": counts["computed"], "search_seconds": elapsed,
                    "preprocess_seconds": preprocessing, "stored_automorphisms": len(alg.autos)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("outputs/revision_20261011"))
    parser.add_argument("--figures", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()
    OUT = args.out if args.out.is_absolute() else ROOT / args.out
    FIG = args.figures if args.figures.is_absolute() else ROOT / args.figures
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    graphs = {"ER40": nx.gnp_random_graph(40, 0.12, seed=1),
              "BA40": nx.barabasi_albert_graph(40, 3, seed=1),
              "Q4": nx.convert_node_labels_to_integers(nx.hypercube_graph(4)),
              "Karate": nx.karate_club_graph()}
    raw, histories = [], []
    for target in ("tau", "iprime"):
        for name, graph in graphs.items():
            for seed in range(args.runs):
                result, row = run_one(graph, target, 1, seed, args)
                row = {"study": "convergence", "graph": name, "target": target,
                       "delta": 1, "seed": seed, **row}
                raw.append(row)
                histories.extend({"graph": name, "target": target, "seed": seed,
                                  "generation": gen, "best": value}
                                 for gen, value in enumerate(result.history))
                write_csv(OUT / "runs.csv", raw)
                write_csv(OUT / "histories.csv", histories)
                print(f"convergence {target} {name} seed={seed} best={result.best_value:.6f} time={row['search_seconds']:.2f}s", flush=True)
    # Reuse delta=1 convergence runs; only the threshold changes.
    for delta in (0, 2, 3):
        for seed in range(args.runs):
            result, row = run_one(graphs["BA40"], "tau", delta, seed, args)
            raw.append({"study": "delta", "graph": "BA40", "target": "tau",
                        "delta": delta, "seed": seed, **row})
            write_csv(OUT / "runs.csv", raw)
            print(f"delta={delta} seed={seed} best={result.best_value:.6f} time={row['search_seconds']:.2f}s", flush=True)
    summary = []
    for delta in (0, 1, 2, 3):
        rows = [r for r in raw if r["graph"] == "BA40" and r["target"] == "tau" and r["delta"] == delta]
        values = [r["best"] for r in rows]
        summary.append({"delta": delta, "runs": len(rows), "mean": np.mean(values),
                        "std": np.std(values), "mean_calls": np.mean([r["objective_calls"] for r in rows])})
    write_csv(OUT / "delta_summary.csv", summary)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.labelcolor": "red", "text.color": "red",
                         "xtick.color": "red", "ytick.color": "red",
                         "axes.edgecolor": "red", "axes.titlecolor": "red"})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3))
    styles = {"ER40": "-", "BA40": "--", "Q4": "-.", "Karate": ":"}
    conv_summary = []
    for ax, target, title in zip(axes, ("tau", "iprime"), (r"$\tau$ objective", r"$I'$ objective")):
        for name in graphs:
            matrix = np.array([[r["best"] for r in histories if r["target"] == target and r["graph"] == name and r["seed"] == seed] for seed in range(args.runs)])
            average = matrix.mean(axis=0)
            ax.plot(np.arange(args.generations + 1), average, styles[name], color="red", label=name, linewidth=1.8)
            conv_summary.append({"graph": name, "target": target, "initial_mean": average[0], "final_mean": average[-1],
                                 "final_std": matrix[:, -1].std(), "last_mean_improvement_generation": int(np.flatnonzero(np.diff(average) < -1e-12)[-1] + 1) if np.any(np.diff(average) < -1e-12) else 0})
        ax.set(title=title, xlabel="Generation", ylabel="Mean best-so-far objective", xlim=(0, args.generations))
        ax.grid(alpha=.2)
        ax.legend(frameon=False, labelcolor="red", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "revision_convergence.pdf", bbox_inches="tight")
    fig.savefig(FIG / "revision_convergence.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    write_csv(OUT / "convergence_summary.csv", conv_summary)
    metadata = {"python": platform.python_version(), "networkx": nx.__version__,
                "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                "parameters": {**vars(args), "out": str(args.out), "figures": str(args.figures)},
                "seed_range": [0, args.runs - 1],
                "graph_generation_seed": 1, "objective_memoization": True,
                "delta_study_target": "tau", "delta_study_graph": "BA40",
                "graphs": {name: {"n": g.number_of_nodes(), "edges": g.number_of_edges()} for name, g in graphs.items()}}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()

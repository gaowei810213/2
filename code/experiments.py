"""
Experiment suite for GT-IDEA (real, reproducible experiments).
==============================================================

This script runs the five experiments described in the experimental design:

  Exp1  Accuracy / optimality vs brute-force exact values (small graphs).
  Exp2  Main comparison against baselines at an EQUAL fitness-evaluation budget.
  Exp3  Ablation of the group-theoretic components of GT-IDEA.
  Exp4  Symmetry sensitivity (gain vs |Aut(G)|).
  Exp5  Scalability and parameter sensitivity.

All numbers written by this script are produced by actually running the
algorithms on the listed graphs. Nothing is hard-coded or fabricated. Results
are written as CSV files into the output directory (default: ./results).

Run examples:
    python experiments.py --exp 1
    python experiments.py --exp all --runs 30 --seed 0 --out results

Dependencies: numpy, networkx; scipy is optional (used for statistical tests).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from statistics import mean, pstdev
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from gt_idea import (
    GTIDEA,
    GTIDEAConfig,
    brute_force,
    exact_automorphisms,
)
import baselines as bl

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def small_graphs() -> Dict[str, nx.Graph]:
    """Small graphs (n <= ~14) where brute force gives the exact optimum."""
    return {
        "path_P8": nx.path_graph(8),
        "cycle_C10": nx.cycle_graph(10),
        "petersen": nx.petersen_graph(),
        "grid_3x3": nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)),
        "complete_bipartite_K3_4": nx.complete_bipartite_graph(3, 4),
        "wheel_W10": nx.wheel_graph(10),
        "barbell_4_2": nx.barbell_graph(4, 2),
        "ladder_L6": nx.ladder_graph(6),
        "karate": nx.karate_club_graph(),       # n=34 -> too big for brute force
    }


def small_graphs_bruteforceable(max_n: int = 18) -> Dict[str, nx.Graph]:
    return {k: nx.convert_node_labels_to_integers(g)
            for k, g in small_graphs().items()
            if g.number_of_nodes() <= max_n}


def synthetic_graphs(seed: int = 0) -> Dict[str, Tuple[nx.Graph, str]]:
    """Medium synthetic graphs covering several random families."""
    rng = seed
    g: Dict[str, Tuple[nx.Graph, str]] = {}
    g["ER_n30_p0.2"] = (nx.gnp_random_graph(30, 0.2, seed=rng), "erdos_renyi")
    g["ER_n40_p0.15"] = (nx.gnp_random_graph(40, 0.15, seed=rng), "erdos_renyi")
    g["BA_n40_m2"] = (nx.barabasi_albert_graph(40, 2, seed=rng), "scale_free")
    g["BA_n60_m3"] = (nx.barabasi_albert_graph(60, 3, seed=rng), "scale_free")
    g["WS_n40_k4_p0.1"] = (nx.watts_strogatz_graph(40, 4, 0.1, seed=rng), "small_world")
    g["REG_n40_d3"] = (nx.random_regular_graph(3, 40, seed=rng), "regular")
    return g


def symmetric_graphs() -> Dict[str, nx.Graph]:
    """Graphs spanning a range of automorphism-group sizes (for Exp4)."""
    return {
        "petersen": nx.petersen_graph(),
        "cycle_C12": nx.cycle_graph(12),
        "complete_bipartite_K4_4": nx.complete_bipartite_graph(4, 4),
        "hypercube_Q4": nx.convert_node_labels_to_integers(nx.hypercube_graph(4)),
        "circulant_12_1_3": nx.circulant_graph(12, [1, 3]),
        "grid_3x4": nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 4)),
        "path_P12": nx.path_graph(12),
        "ER_n12_p0.3": nx.gnp_random_graph(12, 0.3, seed=1),
    }


def real_graphs() -> Dict[str, nx.Graph]:
    """Real-world networks bundled with networkx (plus optional edgelists).

    To add larger real networks, drop edge-list files (one ``u v`` per line)
    into a ``data/`` folder; they are auto-loaded here.
    """
    g: Dict[str, nx.Graph] = {
        "karate": nx.karate_club_graph(),
        "florentine": nx.florentine_families_graph(),
        "les_miserables": nx.les_miserables_graph(),
        "davis_women": nx.davis_southern_women_graph(),
    }
    data_dir = "data"
    if os.path.isdir(data_dir):
        for fn in sorted(os.listdir(data_dir)):
            if fn.endswith((".edgelist", ".txt", ".csv")):
                path = os.path.join(data_dir, fn)
                try:
                    gr = nx.read_edgelist(path, nodetype=int)
                    g[f"file_{fn}"] = gr
                except Exception:
                    pass
    return {k: nx.convert_node_labels_to_integers(v) for k, v in g.items()}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def run_gtidea_counted(G: nx.Graph, cfg: GTIDEAConfig):
    """Run GT-IDEA while counting true objective evaluations (fair FE budget).

    Returns (result, n_fe, elapsed_seconds). The counter wraps ``_objective``,
    which is the single path through which GT-IDEA evaluates fitness.
    """
    alg = GTIDEA(G, cfg)
    counter = {"n": 0}
    base_obj = alg._objective

    def counting(h):
        counter["n"] += 1
        return base_obj(h)

    alg._objective = counting
    t0 = time.time()
    result = alg.run()
    elapsed = time.time() - t0
    return result, counter["n"], elapsed


def aut_order(G: nx.Graph, cap: int = 100000) -> int:
    """Exact |Aut(G)| by counting isomorphisms (capped). Small graphs only."""
    return len(exact_automorphisms(G, limit=cap))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def fmt(x: float) -> str:
    return "inf" if (isinstance(x, float) and math.isinf(x)) else f"{x:.6g}"


# ---------------------------------------------------------------------------
# Experiment 1 -- accuracy vs brute-force exact optimum
# ---------------------------------------------------------------------------
def experiment1(out_dir: str, runs: int = 10, seed0: int = 0, max_n: int = 16) -> None:
    print("[Exp1] accuracy vs brute force")
    graphs = small_graphs_bruteforceable(max_n=max_n)
    rows = []
    for target in ("tau", "iprime"):
        for name, G in graphs.items():
            exact_val, exact_set = brute_force(G, target)
            best_over_runs = math.inf
            gaps, hits = [], 0
            for r in range(runs):
                cfg = GTIDEAConfig(target=target, pop_size=30, max_iter=80,
                                   seed=seed0 + r)
                res, fe, secs = run_gtidea_counted(G, cfg)
                best_over_runs = min(best_over_runs, res.best_value)
                if not math.isinf(exact_val) and not math.isinf(res.best_value):
                    gap = (res.best_value - exact_val) / exact_val if exact_val > 0 else res.best_value - exact_val
                    gaps.append(gap)
                    if abs(res.best_value - exact_val) < 1e-9:
                        hits += 1
            rows.append([
                target, name, G.number_of_nodes(), G.number_of_edges(),
                fmt(exact_val), fmt(best_over_runs),
                fmt(mean(gaps)) if gaps else "n/a",
                fmt(pstdev(gaps)) if len(gaps) > 1 else "0",
                f"{hits}/{runs}",
            ])
    write_csv(os.path.join(out_dir, "exp1_accuracy.csv"),
              ["target", "graph", "n", "m", "exact", "best_found",
               "mean_gap", "std_gap", "exact_hits"], rows)


# ---------------------------------------------------------------------------
# Experiment 2 -- main comparison at equal FE budget
# ---------------------------------------------------------------------------
def experiment2(out_dir: str, runs: int = 15, seed0: int = 0) -> None:
    print("[Exp2] main comparison at equal FE budget")
    graphs: Dict[str, nx.Graph] = {}
    for name, (G, _) in synthetic_graphs(seed=1).items():
        graphs[name] = G
    for name, G in real_graphs().items():
        graphs[name] = G

    raw_rows, summ_rows = [], []
    for target in ("tau", "iprime"):
        for name, G in graphs.items():
            # Calibrate the shared FE budget from one GT-IDEA reference run.
            ref_cfg = GTIDEAConfig(target=target, pop_size=40, max_iter=120, seed=seed0)
            _, budget_fe, _ = run_gtidea_counted(G, ref_cfg)

            algo_values: Dict[str, List[float]] = {}

            # GT-IDEA (stochastic)
            vals = []
            for r in range(runs):
                cfg = GTIDEAConfig(target=target, pop_size=40, max_iter=120, seed=seed0 + r)
                res, fe, secs = run_gtidea_counted(G, cfg)
                vals.append(res.best_value)
                raw_rows.append([target, name, "GT-IDEA", r, fmt(res.best_value), fe, f"{secs:.3f}"])
            algo_values["GT-IDEA"] = vals

            # Stochastic baselines at the same FE budget.
            for bname, fn in bl.STOCHASTIC_BASELINES.items():
                vals = []
                for r in range(runs):
                    res = fn(G, target, budget_fe=budget_fe, seed=seed0 + r)
                    vals.append(res.best_value)
                    raw_rows.append([target, name, bname, r, fmt(res.best_value), res.n_fe, ""])
                algo_values[bname] = vals

            # Deterministic heuristics (single run).
            for bname, fn in bl.DETERMINISTIC_BASELINES.items():
                res = fn(G, target)
                algo_values[bname] = [res.best_value]
                raw_rows.append([target, name, bname, 0, fmt(res.best_value), res.n_fe, ""])

            for bname, vals in algo_values.items():
                finite = [v for v in vals if not math.isinf(v)]
                summ_rows.append([
                    target, name, bname,
                    fmt(min(vals)),
                    fmt(mean(finite)) if finite else "inf",
                    fmt(pstdev(finite)) if len(finite) > 1 else "0",
                    len(vals),
                ])

    write_csv(os.path.join(out_dir, "exp2_raw.csv"),
              ["target", "graph", "algorithm", "run", "best_value", "n_fe", "time_s"], raw_rows)
    write_csv(os.path.join(out_dir, "exp2_summary.csv"),
              ["target", "graph", "algorithm", "best", "mean", "std", "runs"], summ_rows)


# ---------------------------------------------------------------------------
# Experiment 3 -- ablation of group-theoretic components
# ---------------------------------------------------------------------------
def _ablation_configs(target: str, seed: int) -> Dict[str, GTIDEAConfig]:
    base = dict(target=target, pop_size=40, max_iter=120, seed=seed)
    return {
        "full": GTIDEAConfig(**base),
        "no_orbit_compression": GTIDEAConfig(use_orbit_compression=False, **base),
        "no_diversity": GTIDEAConfig(use_diversity=False, **base),
        # crossover dominated by coordinate (alignment off effect approximated)
        "circle_only_crossover": GTIDEAConfig(circle_crossover_prob=1.0, **base),
        "coord_only_crossover": GTIDEAConfig(circle_crossover_prob=0.0, **base),
        # mutation = bit-flip only (≈ standard GA mutation), no group exploration
        "bitflip_only_mut": GTIDEAConfig(mutation_probs=(0.0, 0.0, 1.0, 0.0, 0.0), **base),
        # approximate automorphisms instead of exact Aut(G)
        "approx_aut": GTIDEAConfig(use_exact_aut=False, use_approx_aut=True, **base),
    }


def experiment3(out_dir: str, runs: int = 15, seed0: int = 0) -> None:
    print("[Exp3] ablation")
    graphs = {name: G for name, (G, _) in synthetic_graphs(seed=1).items()}
    graphs["petersen"] = nx.petersen_graph()
    graphs["karate"] = nx.karate_club_graph()
    rows = []
    for target in ("tau", "iprime"):
        for name, G in graphs.items():
            for variant in _ablation_configs(target, seed0).keys():
                vals = []
                for r in range(runs):
                    cfg = _ablation_configs(target, seed0 + r)[variant]
                    res, fe, secs = run_gtidea_counted(G, cfg)
                    vals.append(res.best_value)
                finite = [v for v in vals if not math.isinf(v)]
                rows.append([
                    target, name, variant,
                    fmt(min(vals)),
                    fmt(mean(finite)) if finite else "inf",
                    fmt(pstdev(finite)) if len(finite) > 1 else "0",
                ])
    write_csv(os.path.join(out_dir, "exp3_ablation.csv"),
              ["target", "graph", "variant", "best", "mean", "std"], rows)


# ---------------------------------------------------------------------------
# Experiment 4 -- symmetry sensitivity (gain vs |Aut(G)|)
# ---------------------------------------------------------------------------
def experiment4(out_dir: str, runs: int = 15, seed0: int = 0) -> None:
    print("[Exp4] symmetry sensitivity")
    graphs = symmetric_graphs()
    rows = []
    for target in ("tau", "iprime"):
        for name, G in graphs.items():
            G = nx.convert_node_labels_to_integers(G)
            order = aut_order(G, cap=100000)
            # GT-IDEA (uses symmetry) vs standard GA (does not) at equal FE.
            ref_cfg = GTIDEAConfig(target=target, pop_size=40, max_iter=120, seed=seed0)
            _, budget_fe, _ = run_gtidea_counted(G, ref_cfg)

            gt_vals, ga_vals = [], []
            for r in range(runs):
                cfg = GTIDEAConfig(target=target, pop_size=40, max_iter=120, seed=seed0 + r)
                res, fe, _ = run_gtidea_counted(G, cfg)
                gt_vals.append(res.best_value)
                ga = bl.standard_ga(G, target, budget_fe=budget_fe, seed=seed0 + r)
                ga_vals.append(ga.best_value)
            gt_f = [v for v in gt_vals if not math.isinf(v)]
            ga_f = [v for v in ga_vals if not math.isinf(v)]
            gt_mean = mean(gt_f) if gt_f else math.inf
            ga_mean = mean(ga_f) if ga_f else math.inf
            improvement = (ga_mean - gt_mean) / ga_mean if (ga_f and gt_f and ga_mean > 0) else float("nan")
            rows.append([
                target, name, G.number_of_nodes(), order,
                fmt(gt_mean), fmt(ga_mean), f"{improvement:.4f}",
            ])
    write_csv(os.path.join(out_dir, "exp4_symmetry.csv"),
              ["target", "graph", "n", "aut_order", "gtidea_mean", "ga_mean", "rel_improvement"], rows)


# ---------------------------------------------------------------------------
# Experiment 5 -- scalability and parameter sensitivity
# ---------------------------------------------------------------------------
def experiment5(out_dir: str, runs: int = 5, seed0: int = 0) -> None:
    print("[Exp5a] scalability")
    sizes = [20, 40, 60, 80, 120, 160]
    rows = []
    for target in ("tau", "iprime"):
        for n in sizes:
            G = nx.barabasi_albert_graph(n, 3, seed=1)
            times, vals = [], []
            for r in range(runs):
                cfg = GTIDEAConfig(target=target, pop_size=40, max_iter=100, seed=seed0 + r)
                res, fe, secs = run_gtidea_counted(G, cfg)
                times.append(secs)
                vals.append(res.best_value)
            finite = [v for v in vals if not math.isinf(v)]
            rows.append([
                target, n, G.number_of_edges(),
                f"{mean(times):.3f}", f"{pstdev(times):.3f}" if len(times) > 1 else "0",
                fmt(mean(finite)) if finite else "inf",
            ])
    write_csv(os.path.join(out_dir, "exp5_scalability.csv"),
              ["target", "n", "m", "mean_time_s", "std_time_s", "mean_value"], rows)

    print("[Exp5b] parameter sensitivity (pop_size, mutation mix, delta)")
    G = nx.barabasi_albert_graph(50, 3, seed=1)
    target = "tau"
    prows = []

    def sweep(label, cfg_factory, values):
        for v in values:
            vals = []
            for r in range(runs):
                cfg = cfg_factory(v, seed0 + r)
                res, fe, _ = run_gtidea_counted(G, cfg)
                vals.append(res.best_value)
            finite = [x for x in vals if not math.isinf(x)]
            prows.append([label, str(v),
                          fmt(mean(finite)) if finite else "inf",
                          fmt(pstdev(finite)) if len(finite) > 1 else "0"])

    sweep("pop_size", lambda v, s: GTIDEAConfig(target=target, pop_size=v, max_iter=100, seed=s),
          [20, 40, 60, 80])
    sweep("diversity_delta", lambda v, s: GTIDEAConfig(target=target, pop_size=40, max_iter=100,
                                                       diversity_threshold=v, seed=s),
          [0, 1, 2, 3])
    mixes = {
        "uniform": (0.2, 0.2, 0.2, 0.2, 0.2),
        "bitflip_heavy": (0.1, 0.1, 0.5, 0.1, 0.2),
        "circle_heavy": (0.35, 0.25, 0.2, 0.1, 0.1),
        "aut_heavy": (0.1, 0.1, 0.3, 0.1, 0.4),
    }
    for label, probs in mixes.items():
        sweep(f"mut_mix:{label}",
              lambda v, s, p=probs: GTIDEAConfig(target=target, pop_size=40, max_iter=100,
                                                 mutation_probs=p, seed=s),
              [label])

    write_csv(os.path.join(out_dir, "exp5_params.csv"),
              ["parameter", "value", "mean_value", "std_value"], prows)


# ---------------------------------------------------------------------------
# Statistical tests on Exp2 raw results
# ---------------------------------------------------------------------------
def statistical_tests(out_dir: str) -> None:
    raw_path = os.path.join(out_dir, "exp2_raw.csv")
    if not os.path.isfile(raw_path):
        print("[stats] exp2_raw.csv not found; run Exp2 first.")
        return
    if scipy_stats is None:
        print("[stats] scipy not available; skipping statistical tests.")
        return

    # Load per-(target,graph,algorithm) value lists.
    data: Dict[Tuple[str, str, str], List[float]] = {}
    with open(raw_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["target"], row["graph"], row["algorithm"])
            v = row["best_value"]
            val = math.inf if v == "inf" else float(v)
            data.setdefault(key, []).append(val)

    lines: List[str] = ["Statistical tests on Exp2 (lower is better).", ""]
    # Pairwise Wilcoxon: GT-IDEA vs each baseline, paired across (target,graph).
    algos = sorted({k[2] for k in data})
    targets_graphs = sorted({(k[0], k[1]) for k in data})

    for baseline in [a for a in algos if a != "GT-IDEA"]:
        gt_means, bl_means = [], []
        for (t, g) in targets_graphs:
            gt = data.get((t, g, "GT-IDEA"))
            bb = data.get((t, g, baseline))
            if not gt or not bb:
                continue
            gt_m = np.mean([x for x in gt if not math.isinf(x)] or [1e9])
            bl_m = np.mean([x for x in bb if not math.isinf(x)] or [1e9])
            gt_means.append(gt_m)
            bl_means.append(bl_m)
        if len(gt_means) >= 5:
            try:
                stat, p = scipy_stats.wilcoxon(gt_means, bl_means)
                wins = sum(1 for a, b in zip(gt_means, bl_means) if a < b)
                lines.append(f"GT-IDEA vs {baseline}: Wilcoxon p={p:.4g}, "
                             f"GT-IDEA better on {wins}/{len(gt_means)} instances")
            except Exception as e:
                lines.append(f"GT-IDEA vs {baseline}: test failed ({e})")

    out = os.path.join(out_dir, "exp2_stats.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {out}")
    for ln in lines:
        print("   ", ln)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GT-IDEA experiment suite")
    parser.add_argument("--exp", default="all",
                        choices=["1", "2", "3", "4", "5", "stats", "all"],
                        help="which experiment to run")
    parser.add_argument("--runs", type=int, default=15,
                        help="independent runs per (algorithm, instance)")
    parser.add_argument("--seed", type=int, default=0, help="base random seed")
    parser.add_argument("--out", default="results", help="output directory")
    parser.add_argument("--max-n", type=int, default=16,
                        help="max graph size for brute force in Exp1")
    args = parser.parse_args()

    ensure_dir(args.out)
    t0 = time.time()

    if args.exp in ("1", "all"):
        experiment1(args.out, runs=max(args.runs, 1), seed0=args.seed, max_n=args.max_n)
    if args.exp in ("2", "all"):
        experiment2(args.out, runs=args.runs, seed0=args.seed)
    if args.exp in ("3", "all"):
        experiment3(args.out, runs=args.runs, seed0=args.seed)
    if args.exp in ("4", "all"):
        experiment4(args.out, runs=args.runs, seed0=args.seed)
    if args.exp in ("5", "all"):
        experiment5(args.out, runs=max(args.runs // 3, 3), seed0=args.seed)
    if args.exp in ("stats", "all", "2"):
        statistical_tests(args.out)

    print(f"\nDone in {time.time() - t0:.1f}s. Results in '{args.out}/'.")


if __name__ == "__main__":
    main()

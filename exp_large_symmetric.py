"""
Experiment 6: GT-IDEA on LARGE strongly-symmetric graphs.
=========================================================

Motivation: Exp4 found no symmetry-driven advantage on small symmetric graphs
(both GT-IDEA and a plain GA reached the optimum). The question here is whether
the symmetry machinery of GT-IDEA -- orbit compression and graph-automorphism
mutation -- provides any advantage on LARGER strongly-symmetric graphs, where
the search space is no longer trivially solved.

For each large symmetric graph we compare, at an EQUAL fitness-evaluation
budget and over many independent runs:

  A = GT-IDEA "symmetry-aware"  : orbit compression ON, Aut(G) mutation ON
  B = GT-IDEA "symmetry-off"    : orbit compression OFF, no Aut(G) mutation
  C = standard GA               : reference baseline

If A is significantly better than B on these graphs, the symmetry machinery
helps. Otherwise it does not. Results are written to results/exp6_*.

All numbers are produced by real runs. Nothing is fabricated.
"""

from __future__ import annotations

import csv
import math
import os
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import networkx as nx

from gt_idea import GTIDEA, GTIDEAConfig
import baselines as bl
import experiments as E

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


def large_symmetric_graphs() -> Dict[str, Tuple[nx.Graph, str]]:
    """Large, strongly-symmetric graphs with known |Aut(G)| (theoretical).

    Graphs are chosen so that enumerating aut_limit automorphisms via
    GraphMatcher is feasible (hypercubes, cycles, sparse circulants).
    Dense or super-symmetric graphs (K_{m,m}, cocktail-party) are excluded
    because GraphMatcher's backtracking is too slow for them.
    """
    g: Dict[str, Tuple[nx.Graph, str]] = {}
    g["hypercube_Q5"] = (nx.convert_node_labels_to_integers(nx.hypercube_graph(5)),
                         str(2 ** 5 * math.factorial(5)))   # 3840,  n=32
    g["hypercube_Q6"] = (nx.convert_node_labels_to_integers(nx.hypercube_graph(6)),
                         str(2 ** 6 * math.factorial(6)))   # 46080, n=64
    g["cycle_C100"]   = (nx.cycle_graph(100), str(2 * 100)) # 200,   n=100
    g["circulant_60_1_2_3"] = (nx.circulant_graph(60, [1, 2, 3]), ">=120")  # n=60
    return {k: (nx.convert_node_labels_to_integers(G), lab) for k, (G, lab) in g.items()}


def _cfg_symmetry_on(target: str, seed: int) -> GTIDEAConfig:
    """A: symmetry-aware GT-IDEA (orbit compression + Aut mutation)."""
    return GTIDEAConfig(
        target=target, pop_size=40, max_iter=120, seed=seed,
        use_orbit_compression=True, use_exact_aut=True, aut_limit=200,
        mutation_probs=(0.20, 0.15, 0.30, 0.15, 0.20),  # p5 (Aut) active
    )


def _cfg_symmetry_off(target: str, seed: int) -> GTIDEAConfig:
    """B: same operators but symmetry machinery disabled.

    Orbit compression is off and the Aut(G) mutation share (p5) is moved to the
    bit-flip component, so the only difference from A is the use of symmetry.
    """
    return GTIDEAConfig(
        target=target, pop_size=40, max_iter=120, seed=seed,
        use_orbit_compression=False, use_exact_aut=True, aut_limit=200,
        mutation_probs=(0.20, 0.15, 0.50, 0.15, 0.00),  # p5 -> bit flips
    )


def run_experiment(out_dir: str = "results", runs: int = 30, seed0: int = 0) -> None:
    E.ensure_dir(out_dir)
    graphs = large_symmetric_graphs()
    raw_rows: List[list] = []
    summ_rows: List[list] = []
    # Collect per-(target,graph) means for paired statistics.
    means_A: List[float] = []
    means_B: List[float] = []
    means_C: List[float] = []

    for target in ("tau", "iprime"):
        for name, (G, aut_label) in graphs.items():
            print(f"[Exp6] {target} {name} (n={G.number_of_nodes()}, aut={aut_label})", flush=True)

            # Calibrate the shared FE budget from one symmetry-aware run.
            _, budget_fe, _ = E.run_gtidea_counted(G, _cfg_symmetry_on(target, seed0))

            vals_A, vals_B, vals_C = [], [], []
            for r in range(runs):
                resA, feA, secA = E.run_gtidea_counted(G, _cfg_symmetry_on(target, seed0 + r))
                vals_A.append(resA.best_value)
                raw_rows.append([target, name, "A_symmetry_on", r, E.fmt(resA.best_value), feA, f"{secA:.2f}"])

                resB, feB, secB = E.run_gtidea_counted(G, _cfg_symmetry_off(target, seed0 + r))
                vals_B.append(resB.best_value)
                raw_rows.append([target, name, "B_symmetry_off", r, E.fmt(resB.best_value), feB, f"{secB:.2f}"])

                resC = bl.standard_ga(G, target, budget_fe=budget_fe, seed=seed0 + r)
                vals_C.append(resC.best_value)
                raw_rows.append([target, name, "C_standard_ga", r, E.fmt(resC.best_value), resC.n_fe, ""])

            def stats(vals):
                fin = [v for v in vals if not math.isinf(v)]
                return (min(vals), mean(fin) if fin else math.inf,
                        pstdev(fin) if len(fin) > 1 else 0.0)

            for label, vals in (("A_symmetry_on", vals_A),
                                ("B_symmetry_off", vals_B),
                                ("C_standard_ga", vals_C)):
                bst, mn, sd = stats(vals)
                summ_rows.append([target, name, aut_label, label,
                                  E.fmt(bst), E.fmt(mn), E.fmt(sd), len(vals)])

            mA = mean([v for v in vals_A if not math.isinf(v)] or [math.inf])
            mB = mean([v for v in vals_B if not math.isinf(v)] or [math.inf])
            mC = mean([v for v in vals_C if not math.isinf(v)] or [math.inf])
            means_A.append(mA); means_B.append(mB); means_C.append(mC)

    E.write_csv(os.path.join(out_dir, "exp6_large_symmetric_raw.csv"),
                ["target", "graph", "variant", "run", "best_value", "n_fe", "time_s"], raw_rows)
    E.write_csv(os.path.join(out_dir, "exp6_large_symmetric.csv"),
                ["target", "graph", "aut_order", "variant", "best", "mean", "std", "runs"], summ_rows)

    # Paired statistics across (target, graph) instances.
    lines = ["Exp6: large strongly-symmetric graphs (lower is better).",
             "A = symmetry-aware GT-IDEA; B = symmetry-off GT-IDEA; C = standard GA.", ""]
    if scipy_stats is not None and len(means_A) >= 5:
        for other, mo, lbl in ((means_B, means_B, "B_symmetry_off"),
                               (means_C, means_C, "C_standard_ga")):
            winsA = sum(1 for a, b in zip(means_A, mo) if a < b)
            try:
                _, p = scipy_stats.wilcoxon(means_A, mo)
                lines.append(f"A vs {lbl}: Wilcoxon p={p:.4g}, A better on {winsA}/{len(means_A)} instances")
            except Exception as e:
                lines.append(f"A vs {lbl}: test failed ({e})")
    out = os.path.join(out_dir, "exp6_stats.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {out}")
    for ln in lines:
        print("   ", ln)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run_experiment(out_dir=a.out, runs=a.runs, seed0=a.seed)

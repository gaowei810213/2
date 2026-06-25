"""
Demonstration script for GT-IDEA.
=================================

Runs the Group-Theoretic I-Ching Evolutionary Algorithm on a collection of
small benchmark graphs and compares the approximate tau(G) / I'(G) values
against the exact values obtained by brute-force enumeration.

Usage:
    python demo_gt_idea.py

Dependencies: numpy, networkx (same as gt_idea.py).

Because the benchmark graphs are small (n <= ~12), the brute-force solver can
enumerate all 2^n vertex subsets and provide the ground-truth optimum, which
lets us report the optimality gap of GT-IDEA.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import networkx as nx

from gt_idea import GTIDEA, GTIDEAConfig, brute_force


def benchmark_graphs() -> Dict[str, nx.Graph]:
    """A small library of non-complete benchmark graphs (n kept small)."""
    graphs: Dict[str, nx.Graph] = {
        "path_P8": nx.path_graph(8),
        "cycle_C8": nx.cycle_graph(8),
        "petersen": nx.petersen_graph(),
        "grid_3x3": nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3)),
        "star_S9": nx.star_graph(8),               # 1 centre + 8 leaves
        "wheel_W8": nx.wheel_graph(8),             # hub + C8
        "bipartite_K3_3": nx.complete_bipartite_graph(3, 3),
        "barbell": nx.barbell_graph(4, 2),
    }
    return graphs


def run_one(
    G: nx.Graph, target: str, seed: int = 0
) -> Tuple[float, List[int], float, List[int], float]:
    """Run GT-IDEA and brute force on a single graph/target.

    Returns (approx_value, approx_set, exact_value, exact_set, elapsed_seconds).
    """
    cfg = GTIDEAConfig(
        target=target,
        pop_size=30,
        max_iter=80,
        use_exact_aut=True,
        aut_limit=200,
        seed=seed,
    )
    t0 = time.time()
    result = GTIDEA(G, cfg).run()
    elapsed = time.time() - t0

    exact_value, exact_set = brute_force(G, target)
    return result.best_value, result.best_subset, exact_value, exact_set, elapsed


def _fmt(value: float) -> str:
    return "inf" if value == float("inf") else f"{value:.4f}"


def main() -> None:
    graphs = benchmark_graphs()

    for target in ("tau", "iprime"):
        title = "tau(G)" if target == "tau" else "I'(G)"
        print("=" * 72)
        print(f"  Target parameter: {title}")
        print("=" * 72)
        header = f"{'graph':<16}{'n':>3}  {'GT-IDEA':>9}  {'exact':>9}  {'gap':>8}  {'time(s)':>8}"
        print(header)
        print("-" * len(header))

        hit = 0
        total = 0
        for name, G in graphs.items():
            n = G.number_of_nodes()
            approx_v, approx_s, exact_v, exact_s, secs = run_one(G, target)

            if exact_v == float("inf"):
                gap_str = "n/a"          # parameter undefined (e.g. complete-ish)
            elif approx_v == float("inf"):
                gap_str = "inf"
            else:
                gap = approx_v - exact_v
                gap_str = f"{gap:.4f}"
                total += 1
                if abs(gap) < 1e-9:
                    hit += 1

            print(
                f"{name:<16}{n:>3}  {_fmt(approx_v):>9}  {_fmt(exact_v):>9}  "
                f"{gap_str:>8}  {secs:>8.2f}"
            )

        if total:
            print("-" * len(header))
            print(f"  exact optimum reached on {hit}/{total} feasible instances")
        print()

    # Example of inspecting a returned critical set in detail.
    print("=" * 72)
    print("  Detailed example: tau(G) critical set on the Petersen graph")
    print("=" * 72)
    G = nx.petersen_graph()
    cfg = GTIDEAConfig(target="tau", pop_size=40, max_iter=120, seed=1)
    result = GTIDEA(G, cfg).run()
    S = result.best_subset
    H = G.copy()
    H.remove_nodes_from(S)
    print(f"critical set S      : {S}")
    print(f"|S|                 : {len(S)}")
    print(f"omega(G - S)        : {nx.number_connected_components(H)}")
    print(f"tau estimate        : {_fmt(result.best_value)}")
    print(f"history (first 10)  : {[round(x, 3) if x != float('inf') else 'inf' for x in result.history[:10]]}")


if __name__ == "__main__":
    main()

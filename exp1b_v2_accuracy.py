"""Exp1b: accuracy of GT-IDEA v2 vs brute-force exact optimum (small graphs)."""
from __future__ import annotations
import math, os
from statistics import mean, pstdev
from gt_idea import brute_force
from gt_idea_v2 import GTIDEAv2, GTIDEAConfigV2
import experiments as E


def run(out_dir="results", runs=10, seed0=0, max_n=16):
    E.ensure_dir(out_dir)
    graphs = E.small_graphs_bruteforceable(max_n=max_n)
    rows = []
    for target in ("tau", "iprime"):
        for name, G in graphs.items():
            exact_val, _ = brute_force(G, target)
            best = math.inf
            gaps, hits = [], 0
            for r in range(runs):
                cfg = GTIDEAConfigV2(target=target, pop_size=30, max_iter=40, seed=seed0 + r,
                                     use_diversity=False)
                res = GTIDEAv2(G, cfg).run()
                best = min(best, res.best_value)
                if not math.isinf(exact_val) and not math.isinf(res.best_value):
                    gaps.append((res.best_value - exact_val) / exact_val if exact_val > 0
                                else res.best_value - exact_val)
                    if abs(res.best_value - exact_val) < 1e-9:
                        hits += 1
            rows.append([target, name, G.number_of_nodes(), G.number_of_edges(),
                         E.fmt(exact_val), E.fmt(best),
                         E.fmt(mean(gaps)) if gaps else "n/a",
                         E.fmt(pstdev(gaps)) if len(gaps) > 1 else "0",
                         f"{hits}/{runs}"])
    E.write_csv(os.path.join(out_dir, "exp1b_v2_accuracy.csv"),
                ["target","graph","n","m","exact","best_found","mean_gap","std_gap","exact_hits"], rows)


if __name__ == "__main__":
    run()

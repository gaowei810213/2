"""Historical Experiment 9: GT-IDEA v2 vs GA/SA/BPSO.

V2 runs 40 generations with local descent; baseline FE budgets are calibrated
with the original v1 helper. Realized evaluation counts are unequal. The raw
CSV stores objective values only, not FE or timing; missing measurements must
not be reconstructed as if recorded. See docs/DATA_PROVENANCE.md.
"""
from __future__ import annotations

import math, os, time
from statistics import mean, pstdev
from typing import List

import networkx as nx
from gt_idea_v2 import GTIDEAv2, GTIDEAConfigV2
import baselines as bl
import experiments as E

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

V2_MAX_ITER = 40


def test_graphs():
    g = {}
    for name, (G, _) in E.synthetic_graphs(seed=1).items():
        g[name] = G
    for name, G in E.real_graphs().items():
        if name == "les_miserables":
            continue
        g[name] = G
    return g


def run(out_dir="results", runs=30, seed0=0):
    E.ensure_dir(out_dir)
    graphs = test_graphs()
    raw, summ = [], []
    means = {"GT-IDEA": [], "standard_ga": [], "simulated_annealing": [], "binary_pso": []}

    for target in ("tau", "iprime"):
        for name, G in graphs.items():
            print(f"[Exp9] {target} {name}", flush=True)

            # Historical baseline FE calibration through the v1 helper
            cfg_ref = GTIDEAConfigV2(target=target, pop_size=40, max_iter=V2_MAX_ITER, seed=seed0)
            _, budget_fe, _ = E.run_gtidea_counted(G, cfg_ref)

            def run_v2(seed):
                alg = GTIDEAv2(G, GTIDEAConfigV2(target=target, pop_size=40,
                                                 max_iter=V2_MAX_ITER, seed=seed))
                cnt = {"n": 0}
                base = alg._objective
                def counting(h):
                    cnt["n"] += 1
                    return base(h)
                alg._objective = counting
                return alg.run()

            vals = {k: [] for k in means}
            for r in range(runs):
                res = run_v2(seed0 + r)
                vals["GT-IDEA"].append(res.best_value)
                raw.append([target, name, "GT-IDEA", r, E.fmt(res.best_value)])

                for bname, fn in (("standard_ga", bl.standard_ga),
                                  ("simulated_annealing", bl.simulated_annealing),
                                  ("binary_pso", bl.binary_pso)):
                    rb = fn(G, target, budget_fe=budget_fe, seed=seed0 + r)
                    vals[bname].append(rb.best_value)
                    raw.append([target, name, bname, r, E.fmt(rb.best_value)])

            for lbl in means:
                v = vals[lbl]
                fin = [x for x in v if not math.isinf(x)]
                summ.append([target, name, lbl, E.fmt(min(v)),
                             E.fmt(mean(fin) if fin else math.inf),
                             E.fmt(pstdev(fin) if len(fin) > 1 else 0.0), len(v)])
                means[lbl].append(mean(fin) if fin else math.inf)

    E.write_csv(os.path.join(out_dir, "exp9_multibaseline_raw.csv"),
                ["target","graph","algorithm","run","best_value"], raw)
    E.write_csv(os.path.join(out_dir, "exp9_multibaseline.csv"),
                ["target","graph","algorithm","best","mean","std","runs"], summ)

    lines = ["Exp9: GT-IDEA vs GA/SA/BPSO (lower is better).", ""]
    if scipy_stats and len(means["GT-IDEA"]) >= 5:
        for lbl in ("standard_ga", "simulated_annealing", "binary_pso"):
            a, b = means["GT-IDEA"], means[lbl]
            wins = sum(1 for x, y in zip(a, b) if x < y)
            ties = sum(1 for x, y in zip(a, b) if abs(x - y) < 1e-9)
            try:
                _, p = scipy_stats.wilcoxon(a, b)
                lines.append(f"GT-IDEA vs {lbl}: Wilcoxon p={p:.4g}, "
                             f"better {wins}/{len(a)} (ties {ties})")
            except Exception as e:
                lines.append(f"GT-IDEA vs {lbl}: {e}")
    with open(os.path.join(out_dir, "exp9_stats.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    for ln in lines:
        print("   ", ln)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out",  default="results")
    a = p.parse_args()
    run(out_dir=a.out, runs=a.runs, seed0=a.seed)

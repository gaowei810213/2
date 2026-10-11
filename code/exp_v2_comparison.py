"""Historical Experiment 7: GT-IDEA v2 vs v1 vs standard GA.

This driver preserves the archived allocations: v2 runs 40 generations with
local descent; v1 runs 120 generations; the GA reference budget is calibrated
by experiments.run_gtidea_counted, which instantiates v1 even for a v2 config.
Realized fitness-evaluation counts are unequal. These results do not establish
equal-FE efficiency. See docs/DATA_PROVENANCE.md.
"""
from __future__ import annotations

import csv, math, os, time
from statistics import mean, pstdev
from typing import List

import networkx as nx
from gt_idea import GTIDEA, GTIDEAConfig, GTIDEAResult
from gt_idea_v2 import GTIDEAv2, GTIDEAConfigV2
import baselines as bl
import experiments as E

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None


def test_graphs():
    g = {}
    for name, (G, _) in E.synthetic_graphs(seed=1).items():
        g[name] = G
    # representative real networks (exclude the largest to keep v2 tractable)
    for name, G in E.real_graphs().items():
        if name in ("les_miserables",):
            continue
        g[name] = G
    return g


# v2 uses fewer generations because local descent converges fast.
V2_MAX_ITER = 40
REF_MAX_ITER = 120


def run(out_dir="results", runs=30, seed0=0):
    E.ensure_dir(out_dir)
    graphs = test_graphs()
    raw, summ = [], []
    means_v2, means_v1, means_ga = [], [], []

    for target in ("tau", "iprime"):
        for name, G in graphs.items():
            print(f"[Exp7] {target} {name}", flush=True)

            # Historical baseline FE calibration through the v1 helper
            cfg_ref = GTIDEAConfigV2(target=target, pop_size=40, max_iter=V2_MAX_ITER, seed=seed0)
            _, budget_fe, _ = E.run_gtidea_counted(G, cfg_ref)

            # helper to wrap v2 for run_gtidea_counted (which patches _objective)
            def run_v2(seed):
                alg = GTIDEAv2(G, GTIDEAConfigV2(target=target, pop_size=40, max_iter=V2_MAX_ITER, seed=seed))
                counter = {"n": 0}
                base = alg._objective
                def counting(h):
                    counter["n"] += 1
                    return base(h)
                alg._objective = counting
                t0 = time.time()
                res = alg.run()
                return res, counter["n"], time.time() - t0

            vv2, vv1, vga = [], [], []
            for r in range(runs):
                # v2
                res2, fe2, s2 = run_v2(seed0 + r)
                vv2.append(res2.best_value)
                raw.append([target, name, "GT-IDEA-v2", r, E.fmt(res2.best_value), fe2, f"{s2:.2f}"])

                # v1 at its archived fixed generation count
                cfg1 = GTIDEAConfig(target=target, pop_size=40, max_iter=REF_MAX_ITER, seed=seed0 + r)
                res1, fe1, s1 = E.run_gtidea_counted(G, cfg1)
                vv1.append(res1.best_value)
                raw.append([target, name, "GT-IDEA-v1", r, E.fmt(res1.best_value), fe1, f"{s1:.2f}"])

                # standard GA
                resg = bl.standard_ga(G, target, budget_fe=budget_fe, seed=seed0 + r)
                vga.append(resg.best_value)
                raw.append([target, name, "standard_ga", r, E.fmt(resg.best_value), resg.n_fe, ""])

            def stats(v):
                fin = [x for x in v if not math.isinf(x)]
                return min(v), mean(fin) if fin else math.inf, pstdev(fin) if len(fin) > 1 else 0.0

            for lbl, v in (("GT-IDEA-v2", vv2), ("GT-IDEA-v1", vv1), ("standard_ga", vga)):
                b, mn, sd = stats(v)
                summ.append([target, name, lbl, E.fmt(b), E.fmt(mn), E.fmt(sd), len(v)])

            mA = mean([x for x in vv2 if not math.isinf(x)] or [math.inf])
            mB = mean([x for x in vv1 if not math.isinf(x)] or [math.inf])
            mC = mean([x for x in vga if not math.isinf(x)] or [math.inf])
            means_v2.append(mA); means_v1.append(mB); means_ga.append(mC)

    E.write_csv(os.path.join(out_dir, "exp7_v2_raw.csv"),
                ["target","graph","algorithm","run","best_value","n_fe","time_s"], raw)
    E.write_csv(os.path.join(out_dir, "exp7_v2_summary.csv"),
                ["target","graph","algorithm","best","mean","std","runs"], summ)

    lines = ["Exp7: GT-IDEA v2 vs v1 vs standard GA (lower is better).", ""]
    if scipy_stats and len(means_v2) >= 5:
        for other, lbl in ((means_v1,"GT-IDEA-v1"), (means_ga,"standard_ga")):
            wins = sum(1 for a, b in zip(means_v2, other) if a < b)
            try:
                _, p = scipy_stats.wilcoxon(means_v2, other)
                lines.append(f"v2 vs {lbl}: Wilcoxon p={p:.4g}, v2 better on {wins}/{len(means_v2)}")
            except Exception as e:
                lines.append(f"v2 vs {lbl}: {e}")
    with open(os.path.join(out_dir, "exp7_stats.txt"), "w") as fh:
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

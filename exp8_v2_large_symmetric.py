"""
Experiment 8: GT-IDEA v2 on LARGE strongly-symmetric graphs.

Confirms that the v2 improvements (local descent + bit-flip/Aut-dominant
mutation) also hold at scale on strongly-symmetric graphs, the regime where
v1 was previously shown to be worst (Exp6).

Compares, at an equal fitness-evaluation budget over 30 runs:
  GT-IDEA-v2  (revised algorithm)
  GT-IDEA-v1  (original)
  standard_ga (reference baseline)

Graph set is identical to Exp6 (hypercubes, large cycle, sparse circulant).
All numbers come from real runs.
"""
from __future__ import annotations

import math, os, time
from statistics import mean, pstdev
from typing import List

import networkx as nx
from gt_idea import GTIDEA, GTIDEAConfig
from gt_idea_v2 import GTIDEAv2, GTIDEAConfigV2
import baselines as bl
import experiments as E
from exp_large_symmetric import large_symmetric_graphs

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

V2_MAX_ITER = 40
REF_MAX_ITER = 120


def run(out_dir="results", runs=30, seed0=0):
    E.ensure_dir(out_dir)
    graphs = large_symmetric_graphs()
    raw, summ = [], []
    m_v2, m_v1, m_ga = [], [], []

    for target in ("tau", "iprime"):
        for name, (G, aut_label) in graphs.items():
            print(f"[Exp8] {target} {name} (n={G.number_of_nodes()}, aut={aut_label})", flush=True)

            cfg_ref = GTIDEAConfigV2(target=target, pop_size=40, max_iter=V2_MAX_ITER, seed=seed0)
            _, budget_fe, _ = E.run_gtidea_counted(G, cfg_ref)

            def run_v2(seed):
                alg = GTIDEAv2(G, GTIDEAConfigV2(target=target, pop_size=40,
                                                 max_iter=V2_MAX_ITER, seed=seed))
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
                res2, fe2, s2 = run_v2(seed0 + r)
                vv2.append(res2.best_value)
                raw.append([target, name, "GT-IDEA-v2", r, E.fmt(res2.best_value), fe2, f"{s2:.2f}"])

                cfg1 = GTIDEAConfig(target=target, pop_size=40, max_iter=REF_MAX_ITER, seed=seed0 + r)
                res1, fe1, s1 = E.run_gtidea_counted(G, cfg1)
                vv1.append(res1.best_value)
                raw.append([target, name, "GT-IDEA-v1", r, E.fmt(res1.best_value), fe1, f"{s1:.2f}"])

                resg = bl.standard_ga(G, target, budget_fe=budget_fe, seed=seed0 + r)
                vga.append(resg.best_value)
                raw.append([target, name, "standard_ga", r, E.fmt(resg.best_value), resg.n_fe, ""])

            def stats(v):
                fin = [x for x in v if not math.isinf(x)]
                return min(v), mean(fin) if fin else math.inf, pstdev(fin) if len(fin) > 1 else 0.0

            for lbl, v in (("GT-IDEA-v2", vv2), ("GT-IDEA-v1", vv1), ("standard_ga", vga)):
                b, mn, sd = stats(v)
                summ.append([target, name, aut_label, lbl, E.fmt(b), E.fmt(mn), E.fmt(sd), len(v)])

            m_v2.append(mean([x for x in vv2 if not math.isinf(x)] or [math.inf]))
            m_v1.append(mean([x for x in vv1 if not math.isinf(x)] or [math.inf]))
            m_ga.append(mean([x for x in vga if not math.isinf(x)] or [math.inf]))

    E.write_csv(os.path.join(out_dir, "exp8_v2_large_sym_raw.csv"),
                ["target","graph","algorithm","run","best_value","n_fe","time_s"], raw)
    E.write_csv(os.path.join(out_dir, "exp8_v2_large_sym.csv"),
                ["target","graph","aut_order","algorithm","best","mean","std","runs"], summ)

    lines = ["Exp8: GT-IDEA v2 on large strongly-symmetric graphs (lower is better).", ""]
    if scipy_stats and len(m_v2) >= 5:
        for other, lbl in ((m_v1,"GT-IDEA-v1"), (m_ga,"standard_ga")):
            wins = sum(1 for a, b in zip(m_v2, other) if a < b)
            try:
                _, p = scipy_stats.wilcoxon(m_v2, other)
                lines.append(f"v2 vs {lbl}: Wilcoxon p={p:.4g}, v2 better on {wins}/{len(m_v2)}")
            except Exception as e:
                lines.append(f"v2 vs {lbl}: {e}")
    with open(os.path.join(out_dir, "exp8_stats.txt"), "w") as fh:
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

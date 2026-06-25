"""
GT-IDEA v2: revised algorithm matching the updated paper (Section 5).

Changes from v1 (gt_idea.py):
  1. Default mutation mixture shifts weight to bit-flips (p3=0.55) and
     Aut(G) moves (p5=0.30); circular rotations/reflections reduced to
     exploratory role (p1=p2=0.05, p4=0.05).
  2. Local descent refinement (Section 5.4) applied after mutation+repair:
     iterate L_ld rounds of single-bit neighbourhood improvement.
  3. GTIDEAConfigV2 exposes the new parameter ``local_descent_rounds``.

Everything else (operators, orbit compression, diversity, repair) is
inherited unchanged from GTIDEA in gt_idea.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import networkx as nx

from gt_idea import (
    GTIDEA,
    GTIDEAConfig,
    GTIDEAResult,
    Hexagram,
    fitness_from_objective,
    vertex_subset,
    act_bitflip,
)


@dataclass
class GTIDEAConfigV2(GTIDEAConfig):
    """Hyper-parameters for GT-IDEA v2 (adds local descent, revised defaults)."""
    # Revised default: bit-flip dominant, Aut(G) secondary, circle minimal.
    mutation_probs: Tuple[float, float, float, float, float] = (0.05, 0.05, 0.55, 0.05, 0.30)
    local_descent_rounds: int = 3   # L_ld: max rounds of single-bit neighbourhood search


class GTIDEAv2(GTIDEA):
    """GT-IDEA v2: adds local descent refinement after mutation/repair."""

    def __init__(self, G: nx.Graph, config: Optional[GTIDEAConfigV2] = None):
        super().__init__(G, config or GTIDEAConfigV2())

    @property
    def _ld_rounds(self) -> int:
        cfg = self.cfg
        return cfg.local_descent_rounds if isinstance(cfg, GTIDEAConfigV2) else 3

    def local_descent(self, h: Hexagram) -> Hexagram:
        """Greedy single-bit neighbourhood descent for at most L_ld rounds."""
        h = h.copy()
        e = np.eye(self.n, dtype=np.int8)
        for _ in range(self._ld_rounds):
            cur_val = self._objective(h)
            improved = False
            # Evaluate all n single-bit neighbours.
            best_val, best_h = cur_val, h
            for j in range(self.n):
                nb = act_bitflip(h, e[j])
                val = self._objective(nb)
                if val < best_val:
                    best_val, best_h = val, nb
            if best_val < cur_val:
                h = best_h
                improved = True
            if not improved:
                break
        return h

    def run(self) -> GTIDEAResult:
        """Execute GT-IDEA v2 with local descent inserted after mutation/repair."""
        pop = [self._new_feasible_individual() for _ in range(self.cfg.pop_size)]
        objs = [self._objective(h) for h in pop]
        fits = [fitness_from_objective(o) for o in objs]

        best_idx = int(np.argmax(fits))
        best_h = pop[best_idx].copy()
        best_obj = objs[best_idx]
        history = [best_obj]

        for _ in range(self.cfg.max_iter):
            new_pop: List[Hexagram] = [best_h.copy()]
            attempts = 0
            max_attempts = self.cfg.pop_size * 20

            while len(new_pop) < self.cfg.pop_size:
                attempts += 1
                relax = attempts > max_attempts
                parent_a = self._tournament_select(pop, fits)
                parent_b = self._tournament_select(pop, fits)

                if self.rng.random() < self.cfg.crossover_rate:
                    children = self.crossover(parent_a, parent_b)
                else:
                    children = (parent_a.copy(), parent_b.copy())

                for child in children:
                    child = self.mutate(child)
                    if self.cfg.use_repair and not self._feasible(child):
                        repaired = self.repair(child)
                        child = repaired if repaired is not None else self._new_feasible_individual()
                    # ---- v2: local descent refinement ----
                    child = self.local_descent(child)
                    child = self._orbit_compress(child)

                    cand_obj = self._objective(child)
                    cand_fit = fitness_from_objective(cand_obj)
                    cur_best_fit = fitness_from_objective(best_obj)
                    if relax or self._accept_diversity(child, new_pop, cand_fit, cur_best_fit):
                        new_pop.append(child)
                    if len(new_pop) >= self.cfg.pop_size:
                        break

            pop = new_pop[: self.cfg.pop_size]
            objs = [self._objective(h) for h in pop]
            fits = [fitness_from_objective(o) for o in objs]

            gen_best_idx = int(np.argmax(fits))
            if objs[gen_best_idx] < best_obj:
                best_obj = objs[gen_best_idx]
                best_h = pop[gen_best_idx].copy()
            history.append(best_obj)

        return GTIDEAResult(
            best_subset=vertex_subset(best_h),
            best_value=best_obj,
            best_hexagram=best_h,
            history=history,
        )


if __name__ == "__main__":
    import networkx as nx
    from gt_idea import brute_force

    G = nx.petersen_graph()
    for target in ("tau", "iprime"):
        cfg = GTIDEAConfigV2(target=target, pop_size=30, max_iter=60, seed=0)
        res = GTIDEAv2(G, cfg).run()
        exact_val, _ = brute_force(G, target)
        print(f"[{target}] v2={res.best_value:.4f}  exact={exact_val:.4f}")

"""
Baseline algorithms for comparison against GT-IDEA.
===================================================

All baselines optimise the same binary vertex-subset problem as GT-IDEA:
they minimise the raw objective f_P(h) (P in {tau, iprime}) over h in {0,1}^n,
where infeasible individuals receive f = +inf (penalty formulation).

Every baseline shares a single fitness-evaluation budget (``budget_fe``) so
that comparisons are fair in number of objective evaluations rather than in
wall-clock time. Each algorithm returns a ``BaselineResult``.

These are *standard* textbook metaheuristics (GA, SA, BPSO, random search) and
two centrality-based greedy heuristics. They are intentionally generic so that
any advantage of GT-IDEA reflects its group-theoretic operators, not tuning.

Dependencies: numpy, networkx, and gt_idea (same folder).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import networkx as nx

from gt_idea import (
    f_tau,
    f_iprime,
    code_to_hexagram,
    vertex_subset,
    omega_after_removal,
    isolated_after_removal,
)


@dataclass
class BaselineResult:
    """Common return type for all baselines."""

    best_value: float
    best_subset: List[int]
    history: List[Tuple[int, float]] = field(default_factory=list)  # (fe, best)
    n_fe: int = 0


class _Evaluator:
    """Objective wrapper that counts fitness evaluations and tracks the best."""

    def __init__(self, G: nx.Graph, target: str):
        self.G = G
        self.target = target
        self._f = f_tau if target == "tau" else f_iprime
        self.count = 0
        self.best_value = math.inf
        self.best_h: Optional[np.ndarray] = None
        self.history: List[Tuple[int, float]] = []
        self._snapshot_every = 1

    def f(self, h: np.ndarray) -> float:
        self.count += 1
        val = self._f(self.G, h)
        if val < self.best_value:
            self.best_value = val
            self.best_h = h.copy()
        if self.count % self._snapshot_every == 0:
            self.history.append((self.count, self.best_value))
        return val

    def result(self) -> BaselineResult:
        subset = vertex_subset(self.best_h) if self.best_h is not None else []
        return BaselineResult(self.best_value, subset, self.history, self.count)


def _rand_individual(n: int, rng: np.random.Generator, density: float = 0.5) -> np.ndarray:
    return (rng.random(n) < density).astype(np.int8)


# ---------------------------------------------------------------------------
# Random search
# ---------------------------------------------------------------------------
def random_search(G: nx.Graph, target: str, budget_fe: int = 4000, seed: int = 0) -> BaselineResult:
    """Uniformly sample binary vectors; keep the best. Lower bound baseline."""
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    rng = np.random.default_rng(seed)
    ev = _Evaluator(G, target)
    while ev.count < budget_fe:
        ev.f(_rand_individual(n, rng))
    return ev.result()


# ---------------------------------------------------------------------------
# Standard binary genetic algorithm (tournament + uniform crossover + bit flip)
# ---------------------------------------------------------------------------
def standard_ga(
    G: nx.Graph,
    target: str,
    budget_fe: int = 4000,
    pop_size: int = 40,
    crossover_rate: float = 0.9,
    mutation_rate: Optional[float] = None,
    tournament_k: int = 3,
    seed: int = 0,
) -> BaselineResult:
    """Generic binary GA with elitism (penalty for infeasible individuals)."""
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    rng = np.random.default_rng(seed)
    pyrng = random.Random(seed)
    mut = mutation_rate if mutation_rate is not None else 1.0 / n
    ev = _Evaluator(G, target)

    pop = [_rand_individual(n, rng) for _ in range(pop_size)]
    objs = [ev.f(h) for h in pop]
    fit = [0.0 if math.isinf(o) else 1.0 / (1.0 + o) for o in objs]

    def tournament() -> np.ndarray:
        idx = pyrng.sample(range(pop_size), min(tournament_k, pop_size))
        return pop[max(idx, key=lambda i: fit[i])].copy()

    while ev.count < budget_fe:
        elite = pop[int(np.argmax(fit))].copy()
        new_pop = [elite]
        while len(new_pop) < pop_size:
            pa, pb = tournament(), tournament()
            if pyrng.random() < crossover_rate:
                mask = rng.random(n) < 0.5
                c1 = np.where(mask, pa, pb).astype(np.int8)
                c2 = np.where(mask, pb, pa).astype(np.int8)
            else:
                c1, c2 = pa.copy(), pb.copy()
            for c in (c1, c2):
                flip = rng.random(n) < mut
                c = np.bitwise_xor(c, flip.astype(np.int8)).astype(np.int8)
                new_pop.append(c)
                if len(new_pop) >= pop_size:
                    break
        pop = new_pop
        objs = [ev.f(h) for h in pop]
        fit = [0.0 if math.isinf(o) else 1.0 / (1.0 + o) for o in objs]
    return ev.result()


# ---------------------------------------------------------------------------
# Simulated annealing (single-bit-flip neighbourhood)
# ---------------------------------------------------------------------------
def simulated_annealing(
    G: nx.Graph,
    target: str,
    budget_fe: int = 4000,
    t0: float = 1.0,
    cooling: float = 0.995,
    seed: int = 0,
) -> BaselineResult:
    """SA minimising f_P; infeasible (+inf) treated as a large finite penalty."""
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    rng = np.random.default_rng(seed)
    ev = _Evaluator(G, target)
    big = float(n + 1)  # finite penalty surrogate for +inf during acceptance

    def pen(val: float) -> float:
        return big if math.isinf(val) else val

    cur = _rand_individual(n, rng)
    cur_val = pen(ev.f(cur))
    temp = t0
    while ev.count < budget_fe:
        nxt = cur.copy()
        j = int(rng.integers(n))
        nxt[j] ^= 1
        nxt_val = pen(ev.f(nxt))
        if nxt_val <= cur_val or rng.random() < math.exp(-(nxt_val - cur_val) / max(temp, 1e-9)):
            cur, cur_val = nxt, nxt_val
        temp *= cooling
    return ev.result()


# ---------------------------------------------------------------------------
# Binary particle swarm optimisation (sigmoid transfer function)
# ---------------------------------------------------------------------------
def binary_pso(
    G: nx.Graph,
    target: str,
    budget_fe: int = 4000,
    swarm_size: int = 30,
    w: float = 0.7,
    c1: float = 1.4,
    c2: float = 1.4,
    seed: int = 0,
) -> BaselineResult:
    """Standard BPSO (Kennedy-Eberhart) with penalty for infeasibility."""
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    rng = np.random.default_rng(seed)
    ev = _Evaluator(G, target)

    def pen(val: float) -> float:
        return float(n + 1) if math.isinf(val) else val

    X = (rng.random((swarm_size, n)) < 0.5).astype(np.int8)
    V = rng.uniform(-1, 1, size=(swarm_size, n))
    pbest = X.copy()
    pbest_val = np.array([pen(ev.f(X[i])) for i in range(swarm_size)])
    g = int(np.argmin(pbest_val))
    gbest = pbest[g].copy()
    gbest_val = pbest_val[g]

    while ev.count < budget_fe:
        for i in range(swarm_size):
            r1, r2 = rng.random(n), rng.random(n)
            V[i] = w * V[i] + c1 * r1 * (pbest[i] - X[i]) + c2 * r2 * (gbest - X[i])
            prob = 1.0 / (1.0 + np.exp(-np.clip(V[i], -10, 10)))
            X[i] = (rng.random(n) < prob).astype(np.int8)
            val = pen(ev.f(X[i]))
            if val < pbest_val[i]:
                pbest[i], pbest_val[i] = X[i].copy(), val
                if val < gbest_val:
                    gbest, gbest_val = X[i].copy(), val
            if ev.count >= budget_fe:
                break
    return ev.result()


# ---------------------------------------------------------------------------
# Centrality-based greedy heuristics (deterministic proxies)
# ---------------------------------------------------------------------------
def _greedy_by_ranking(G: nx.Graph, target: str, ranking: List[int]) -> BaselineResult:
    """Add vertices to S in the given priority order; report the best feasible
    prefix S_k = {ranking[0], ..., ranking[k-1]} under objective f_P.

    This realises the centrality-as-proxy heuristics discussed in the paper:
    the deletion set is grown by descending centrality and every prefix is
    evaluated, so it explores at most n candidate cuts.
    """
    n = G.number_of_nodes()
    objective = f_tau if target == "tau" else f_iprime
    h = np.zeros(n, dtype=np.int8)
    best_val = math.inf
    best_h = h.copy()
    fe = 0
    history: List[Tuple[int, float]] = []
    for v in ranking:
        h = h.copy()
        h[v] = 1
        val = objective(G, h)
        fe += 1
        if val < best_val:
            best_val = val
            best_h = h.copy()
        history.append((fe, best_val))
    return BaselineResult(best_val, vertex_subset(best_h), history, fe)


def degree_greedy(G: nx.Graph, target: str, **_) -> BaselineResult:
    """Grow the deletion set by descending degree centrality."""
    G = nx.convert_node_labels_to_integers(G)
    ranking = [v for v, _ in sorted(G.degree(), key=lambda kv: kv[1], reverse=True)]
    return _greedy_by_ranking(G, target, ranking)


def betweenness_greedy(G: nx.Graph, target: str, **_) -> BaselineResult:
    """Grow the deletion set by descending betweenness centrality."""
    G = nx.convert_node_labels_to_integers(G)
    bc = nx.betweenness_centrality(G)
    ranking = [v for v, _ in sorted(bc.items(), key=lambda kv: kv[1], reverse=True)]
    return _greedy_by_ranking(G, target, ranking)


# Registry of stochastic baselines that honour a shared FE budget.
STOCHASTIC_BASELINES: dict = {
    "random_search": random_search,
    "standard_ga": standard_ga,
    "simulated_annealing": simulated_annealing,
    "binary_pso": binary_pso,
}

# Deterministic heuristics (no FE budget / no randomness).
DETERMINISTIC_BASELINES: dict = {
    "degree_greedy": degree_greedy,
    "betweenness_greedy": betweenness_greedy,
}

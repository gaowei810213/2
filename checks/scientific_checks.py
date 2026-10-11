"""Run existing scientific checks without manuscript or backup dependencies."""
from pathlib import Path
import itertools
import json
import sys
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from gt_idea import (_vertex_invariants, f_tau, f_iprime, exact_automorphisms,
                     act_permutation, code_to_hexagram)
from gt_idea_v2 import GTIDEAv2, GTIDEAConfigV2

def scientific_checks():
    cycle = nx.cycle_graph(6)
    phi = _vertex_invariants(cycle)
    assert np.allclose(phi, phi[0])
    alternating = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
    consecutive = np.array([1, 1, 1, 0, 0, 0], dtype=np.int8)
    for f in (f_tau, f_iprime):
        assert f(cycle, alternating) == 1.5
        assert np.isinf(f(cycle, consecutive))
    path = nx.path_graph(3)
    middle = np.array([0, 1, 0], dtype=np.int8)
    endpoint = np.array([1, 0, 0], dtype=np.int8)
    for f in (f_tau, f_iprime):
        assert f(path, middle) == 1
        assert np.isinf(f(path, endpoint))
    # Exhaustive check of exact invariance and neighborhood best values on C4.
    graph = nx.cycle_graph(4)
    permutations = exact_automorphisms(graph, 200)
    assert len(permutations) == 8
    for code, perm, objective in itertools.product(range(16), permutations, (f_tau, f_iprime)):
        h = code_to_hexagram(code, 4)
        image = act_permutation(h, perm)
        assert objective(graph, h) == objective(graph, image)
        def best_neighborhood(x):
            candidates = [x]
            for j in range(4):
                neighbor = x.copy()
                neighbor[j] ^= 1
                candidates.append(neighbor)
            return min(objective(graph, y) for y in candidates)
        assert best_neighborhood(h) == best_neighborhood(image)
    # Ensure exact memoization leaves v2's random trajectory unchanged.
    for target in ("tau", "iprime"):
        cfg = GTIDEAConfigV2(target=target, pop_size=6, max_iter=3, seed=7)
        plain = GTIDEAv2(nx.path_graph(8), cfg).run()
        cached_alg = GTIDEAv2(nx.path_graph(8), cfg)
        base, cache = cached_alg._objective, {}
        def cached(h):
            key = h.tobytes()
            if key not in cache:
                cache[key] = base(h)
            return cache[key]
        cached_alg._objective = cached
        cached_result = cached_alg.run()
        assert plain.history == cached_result.history
        assert plain.best_subset == cached_result.best_subset
    return {"path_counterexample": "passed", "C6_feature_counterexample": "passed",
            "C4_exhaustive_objective_and_neighborhood_invariance": "passed",
            "memoization_preserves_seeded_trajectory_both_targets": "passed"}

if __name__ == "__main__":
    results = scientific_checks()
    destination = ROOT / "outputs/verification/scientific_checks.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))

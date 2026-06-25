"""
GT-IDEA: A Group-Theoretic I-Ching Evolutionary Algorithm
=========================================================

Reference implementation of the algorithm described in

    W. Gao and H. Zhang,
    "A Group-Theoretic I-Ching Evolutionary Algorithm for Computing
     Toughness and Isolated Toughness Variants".

The algorithm approximates the two shifted-denominator graph vulnerability
parameters

    tau(G)  = min_{S : omega(G-S) > 1} |S| / (omega(G-S) - 1)
    I'(G)   = min_{S : i(G-S)    > 1} |S| / (i(G-S)    - 1)

and the corresponding critical vertex sets, by evolving binary hexagram
encodings h in {0,1}^n of vertex subsets S_h = { v_i : h_i = 1 }.

The implementation follows Sections 4 and 5 of the paper:

  * Section 4.1  hexagram encoding / circle representation  (code(h), Phi(h))
  * Section 4.2  cyclic group   C_{2^n}        (rotation rho)
  * Section 4.3  dihedral group D_{2^n}        (reflection sigma)
  * Section 4.4  hypercube automorphism group  B_n = Z_2^n  x|  S_n
  * Section 4.5  graph automorphism group      Aut(G)
  * Section 5.1  fitness functions             Fit_tau, Fit_{I'}
  * Section 5.2  group mutation                P_M = sum p_i P_{G_i}
  * Section 5.3  group-aligned crossover       (+ alignment sampling strategy)
  * Section 5.4  orbit compression + diversity (+ approximate Aut(G))
  * Section 5.6  full GT-IDEA loop (Algorithm 1, incl. feasibility repair)

Dependencies: numpy, networkx.

Vertices are assumed to be labelled 0, 1, ..., n-1. The most significant bit
of code(h) corresponds to vertex 0 (h_1 in the paper), matching
code(h) = sum_{j=1}^{n} h_j 2^{n-j}.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx

try:  # networkx >= 2.x
    from networkx.algorithms.isomorphism import GraphMatcher
except Exception:  # pragma: no cover
    GraphMatcher = None


Hexagram = np.ndarray  # 1-D array of int8 in {0, 1}


# ---------------------------------------------------------------------------
# Section 4.1 -- Hexagram encoding and circle representation
# ---------------------------------------------------------------------------
def hexagram_to_code(h: Hexagram) -> int:
    """Integer code of a hexagram: code(h) = sum_{j=1}^{n} h_j 2^{n-j}.

    Vertex 0 (h_1) is the most significant bit. Returns a Python int so the
    value is exact even for large n (where 2^n exceeds machine words).
    """
    code = 0
    for bit in h:
        code = (code << 1) | int(bit)
    return code


def code_to_hexagram(code: int, n: int) -> Hexagram:
    """Inverse of :func:`hexagram_to_code` for an n-bit hexagram."""
    return np.array([(code >> (n - 1 - j)) & 1 for j in range(n)], dtype=np.int8)


def circle_embedding(h: Hexagram) -> complex:
    """Phi(h) = exp(i * 2*pi*code(h) / 2^n) -- the unit-circle embedding."""
    n = len(h)
    return complex(np.exp(1j * 2.0 * np.pi * hexagram_to_code(h) / (2 ** n)))


def vertex_subset(h: Hexagram) -> List[int]:
    """S_h = { v_i : h_i = 1 } as a sorted list of vertex indices."""
    return np.nonzero(h)[0].tolist()


# ---------------------------------------------------------------------------
# Group actions (Sections 4.2 - 4.5)
# ---------------------------------------------------------------------------
def act_rotation(h: Hexagram, k: int) -> Hexagram:
    """Cyclic rotation rho^k : eta_i |-> eta_{i+k mod N}  (Section 4.2)."""
    n = len(h)
    N = 1 << n
    return code_to_hexagram((hexagram_to_code(h) + k) % N, n)


def act_reflection(h: Hexagram) -> Hexagram:
    """Dihedral reflection sigma : eta_i |-> eta_{-i mod N}  (Section 4.3)."""
    n = len(h)
    N = 1 << n
    return code_to_hexagram((-hexagram_to_code(h)) % N, n)


def act_bitflip(h: Hexagram, mask: Hexagram) -> Hexagram:
    """Bit flips from Z_2^n : (a . h)_j = h_j XOR a_j  (Section 4.4)."""
    return np.bitwise_xor(h, mask).astype(np.int8)


def act_permutation(h: Hexagram, perm: np.ndarray) -> Hexagram:
    """Coordinate permutation pi in S_n acting on a hexagram (Section 4.4/4.5).

    For a permutation ``perm`` with ``perm[i] = pi(i)`` the action is
    ``(pi . h)_j = h_{pi^{-1}(j)}``, which is realised by ``out[perm] = h``.
    This is consistent with pi(S) = { v_{pi(i)} : v_i in S }.
    """
    out = np.empty_like(h)
    out[perm] = h
    return out


def hamming(h1: Hexagram, h2: Hexagram) -> int:
    """Hamming distance Ham(h1, h2)."""
    return int(np.count_nonzero(h1 != h2))


# ---------------------------------------------------------------------------
# Graph parameters omega(G - S) and i(G - S)
# ---------------------------------------------------------------------------
def omega_after_removal(G: nx.Graph, h: Hexagram) -> int:
    """Number of connected components omega(G - S_h).

    By convention the empty graph (S_h = V) has 0 components, matching the
    degenerate case discussed in Section 5.1. Uses a subgraph *view* (no copy)
    for efficiency.
    """
    remaining = [int(v) for v in np.nonzero(h == 0)[0]]
    if not remaining:
        return 0
    return nx.number_connected_components(G.subgraph(remaining))


def isolated_after_removal(G: nx.Graph, h: Hexagram) -> int:
    """Number of isolated vertices i(G - S_h)."""
    S = set(np.nonzero(h)[0].tolist())
    count = 0
    for v in G.nodes():
        if v in S:
            continue
        has_neighbour = False
        for u in G.neighbors(v):
            if u not in S:
                has_neighbour = True
                break
        if not has_neighbour:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Section 5.1 -- Fitness functions
# ---------------------------------------------------------------------------
def f_tau(G: nx.Graph, h: Hexagram) -> float:
    """Raw objective f_tau(h); +inf on the infeasible region omega <= 1."""
    w = omega_after_removal(G, h)
    if w > 1:
        return int(h.sum()) / (w - 1)
    return math.inf


def f_iprime(G: nx.Graph, h: Hexagram) -> float:
    """Raw objective f_{I'}(h); +inf on the infeasible region i <= 1."""
    ic = isolated_after_removal(G, h)
    if ic >= 2:
        return int(h.sum()) / (ic - 1)
    return math.inf


def fitness_from_objective(fval: float) -> float:
    """Fit_P(h) = 1 / (1 + f_P(h)); zero for infeasible (+inf) individuals."""
    if math.isinf(fval):
        return 0.0
    return 1.0 / (1.0 + fval)


# ---------------------------------------------------------------------------
# Section 4.5 / 5.4 -- Symmetry toolkit: exact and approximate Aut(G)
# ---------------------------------------------------------------------------
def exact_automorphisms(G: nx.Graph, limit: int = 200) -> List[np.ndarray]:
    """Enumerate (up to ``limit``) graph automorphisms of G as permutations.

    Each automorphism pi is returned as an array ``perm`` with ``perm[i] =
    pi(i)``. The identity is always included. Requires integer node labels
    0..n-1.
    """
    n = G.number_of_nodes()
    identity = np.arange(n)
    if GraphMatcher is None:
        return [identity]
    autos: List[np.ndarray] = []
    matcher = GraphMatcher(G, G)
    for mapping in matcher.isomorphisms_iter():
        perm = np.fromiter((mapping[i] for i in range(n)), dtype=int, count=n)
        autos.append(perm)
        if len(autos) >= limit:
            break
    if not any(np.array_equal(p, identity) for p in autos):
        autos.append(identity)
    return autos


def _vertex_invariants(G: nx.Graph) -> np.ndarray:
    """Vertex invariant vectors phi(v): (degree, clustering, PageRank)."""
    n = G.number_of_nodes()
    deg = np.array([G.degree(v) for v in range(n)], dtype=float)
    clustering = nx.clustering(G)
    clu = np.array([clustering[v] for v in range(n)], dtype=float)
    try:
        pr = nx.pagerank(G)
        prank = np.array([pr[v] for v in range(n)], dtype=float)
    except Exception:
        prank = np.zeros(n, dtype=float)
    return np.column_stack([deg, clu, prank])


def approximate_automorphisms(
    G: nx.Graph,
    eps: float = 1e-6,
    n_samples: int = 50,
    rng: Optional[random.Random] = None,
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """Approximate Aut(G) via vertex-invariant clustering (Section 5.4).

    Vertices with invariant vectors within tolerance ``eps`` are placed in the
    same cluster C_1, ..., C_k. The returned group is a sample of cluster-
    preserving permutations (a subgroup of the Young subgroup prod Sym(C_i)).
    The identity is always included.

    Returns ``(permutations, clusters)``.
    """
    rng = rng or random.Random()
    n = G.number_of_nodes()
    phi = _vertex_invariants(G)

    # Greedy clustering by invariant similarity within tolerance eps.
    signatures: List[np.ndarray] = []
    cluster_of = [-1] * n
    for v in range(n):
        placed = False
        for cid, sig in enumerate(signatures):
            if np.linalg.norm(phi[v] - sig) <= eps:
                cluster_of[v] = cid
                placed = True
                break
        if not placed:
            signatures.append(phi[v])
            cluster_of[v] = len(signatures) - 1

    clusters: List[List[int]] = [[] for _ in signatures]
    for v in range(n):
        clusters[cluster_of[v]].append(v)

    identity = np.arange(n)
    perms: List[np.ndarray] = [identity]
    for _ in range(n_samples):
        perm = np.arange(n)
        for members in clusters:
            if len(members) > 1:
                shuffled = members[:]
                rng.shuffle(shuffled)
                perm[members] = shuffled
        if not np.array_equal(perm, identity):
            perms.append(perm)
    return perms, clusters


def orbit_representative(h: Hexagram, group: Sequence[np.ndarray]) -> Hexagram:
    """rep_Gamma(h) = lexicographically minimal hexagram in the orbit of h.

    ``group`` is a list of coordinate permutations (a subgroup of Aut(G) or an
    approximate automorphism group). Implements Section 5.4 orbit compression.
    """
    best = h
    best_key = tuple(int(x) for x in h)
    for perm in group:
        cand = act_permutation(h, perm)
        key = tuple(int(x) for x in cand)
        if key < best_key:
            best_key = key
            best = cand
    return best


def group_aligned_distance(
    h1: Hexagram, h2: Hexagram, group: Sequence[np.ndarray]
) -> int:
    """d_Gamma(h1, h2) = min_{g in Gamma} Ham(g . h1, h2)  (Section 5.4)."""
    best = hamming(h1, h2)
    for perm in group:
        d = hamming(act_permutation(h1, perm), h2)
        if d < best:
            best = d
    return best


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class GTIDEAConfig:
    """Hyper-parameters of GT-IDEA (Algorithm 1)."""

    target: str = "tau"                    # "tau" or "iprime"
    pop_size: int = 40                     # population size m
    max_iter: int = 200                    # T_max
    # Mutation mixture (p1..p5): cyclic, dihedral, bit-flip, permutation, Aut(G)
    mutation_probs: Tuple[float, float, float, float, float] = (0.20, 0.15, 0.35, 0.15, 0.15)
    crossover_rate: float = 0.9
    circle_crossover_prob: float = 0.5     # share of circle-midpoint vs coordinate crossover
    alignment_sample_cap: int = 200        # |hatG_0| upper bound (Section 5.3)
    tournament_k: int = 3
    diversity_threshold: int = 1           # delta (Section 5.4)
    use_orbit_compression: bool = True
    use_diversity: bool = True
    use_repair: bool = True
    # Symmetry source
    use_exact_aut: bool = True
    aut_limit: int = 200
    use_approx_aut: bool = False
    invariant_eps: float = 1e-6
    approx_aut_samples: int = 50
    init_density: float = 0.5              # bit probability for random init
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Section 5.6 -- The GT-IDEA optimiser (Algorithm 1)
# ---------------------------------------------------------------------------
@dataclass
class GTIDEAResult:
    """Output of a GT-IDEA run."""

    best_subset: List[int]
    best_value: float
    best_hexagram: Hexagram
    history: List[float] = field(default_factory=list)


class GTIDEA:
    """Group-Theoretic I-Ching Evolutionary Algorithm for tau(G) and I'(G)."""

    def __init__(self, G: nx.Graph, config: Optional[GTIDEAConfig] = None):
        # Normalise the graph to integer labels 0..n-1.
        self.G = nx.convert_node_labels_to_integers(G)
        self.n = self.G.number_of_nodes()
        self.N = 1 << self.n
        self.cfg = config or GTIDEAConfig()
        self.rng = random.Random(self.cfg.seed)
        self._np_rng = np.random.default_rng(self.cfg.seed)

        if self.cfg.target not in ("tau", "iprime"):
            raise ValueError("target must be 'tau' or 'iprime'")

        # Objective / feasibility helpers bound to the target parameter.
        if self.cfg.target == "tau":
            self._objective = lambda h: f_tau(self.G, h)
            self._feasible = lambda h: omega_after_removal(self.G, h) > 1
        else:
            self._objective = lambda h: f_iprime(self.G, h)
            self._feasible = lambda h: isolated_after_removal(self.G, h) >= 2

        # Symmetry groups Gamma (Section 4.5 / 5.4).
        self.autos = self._build_symmetry_group()
        # Precompute permutation matrices for vectorised orbit/diversity ops.
        # perm_matrix[g] = pi_g ; inv_perm_matrix[g] = pi_g^{-1}.
        self.perm_matrix = np.array(self.autos, dtype=int)
        self.inv_perm_matrix = np.empty_like(self.perm_matrix)
        ar = np.arange(self.n)
        for g, perm in enumerate(self.perm_matrix):
            self.inv_perm_matrix[g, perm] = ar
        # Column order (most significant first) for lexicographic comparison.
        self._lex_keys_order = list(range(self.n - 1, -1, -1))

    # ----- symmetry group construction ------------------------------------
    def _build_symmetry_group(self) -> List[np.ndarray]:
        if self.cfg.use_approx_aut:
            perms, _ = approximate_automorphisms(
                self.G,
                eps=self.cfg.invariant_eps,
                n_samples=self.cfg.approx_aut_samples,
                rng=self.rng,
            )
            return perms
        if self.cfg.use_exact_aut:
            return exact_automorphisms(self.G, limit=self.cfg.aut_limit)
        return [np.arange(self.n)]

    # ----- Section 5.2: group mutation -------------------------------------
    def mutate(self, h: Hexagram) -> Hexagram:
        """Apply a single group transformation sampled from the mixture P_M."""
        p1, p2, p3, p4, p5 = self.cfg.mutation_probs
        r = self.rng.random()
        if r < p1:                                   # cyclic rotation
            k = self.rng.randrange(1, self.N) if self.N > 1 else 0
            return act_rotation(h, k)
        r -= p1
        if r < p2:                                   # dihedral reflection
            reflected = act_reflection(h)
            k = self.rng.randrange(self.N) if self.N > 1 else 0
            return act_rotation(reflected, k)
        r -= p2
        if r < p3:                                   # bit flips (Z_2^n)
            mask = (self._np_rng.random(self.n) < max(1.0 / self.n, 0.05)).astype(np.int8)
            if mask.sum() == 0:                      # guarantee a real move
                mask[self.rng.randrange(self.n)] = 1
            return act_bitflip(h, mask)
        r -= p3
        if r < p4:                                   # coordinate permutation
            perm = self._np_rng.permutation(self.n)
            return act_permutation(h, perm)
        # graph automorphism (objective preserving)
        perm = self.autos[self.rng.randrange(len(self.autos))]
        return act_permutation(h, perm)

    # ----- Section 5.3: group-aligned crossover ----------------------------
    def _sample_alignment_transforms(self) -> List[Callable[[Hexagram], Hexagram]]:
        """Sampled candidate set hatG_0 for alignment (Section 5.3)."""
        cap = self.cfg.alignment_sample_cap
        k1 = min(100, int(math.isqrt(self.N)))
        k2 = min(50, len(self.autos))
        k3 = min(50, self.n)

        transforms: List[Callable[[Hexagram], Hexagram]] = [lambda h: h.copy()]
        # 1. random rotations
        for _ in range(k1):
            k = self.rng.randrange(self.N)
            transforms.append(lambda h, k=k: act_rotation(h, k))
        # 2. reflection
        transforms.append(act_reflection)
        # 3. sampled graph automorphisms
        idx = self.rng.sample(range(len(self.autos)), k2) if k2 > 0 else []
        for j in idx:
            perm = self.autos[j]
            transforms.append(lambda h, perm=perm: act_permutation(h, perm))
        # 4. random coordinate permutations
        for _ in range(k3):
            perm = self._np_rng.permutation(self.n)
            transforms.append(lambda h, perm=perm: act_permutation(h, perm))

        if len(transforms) > cap:
            transforms = [transforms[0]] + self.rng.sample(transforms[1:], cap - 1)
        return transforms

    def crossover(self, ha: Hexagram, hb: Hexagram) -> Tuple[Hexagram, Hexagram]:
        """Group-aligned crossover producing two offspring (Section 5.3)."""
        # Group alignment: g* = argmin_{g in hatG_0} Ham(g.ha, hb).
        transforms = self._sample_alignment_transforms()
        best_t = transforms[0]
        best_d = math.inf
        for t in transforms:
            d = hamming(t(ha), hb)
            if d < best_d:
                best_d = d
                best_t = t
        ha_aligned = best_t(ha)
        hb_aligned = hb

        if self.rng.random() < self.cfg.circle_crossover_prob:
            return self._circle_midpoint_crossover(ha_aligned, hb_aligned)
        return self._coordinate_crossover(ha_aligned, hb_aligned)

    def _circle_midpoint_crossover(self, ha: Hexagram, hb: Hexagram) -> Tuple[Hexagram, Hexagram]:
        """Circle midpoint crossover: o1 = midpoint, o2 = rho^{N/2} . o1."""
        ia = hexagram_to_code(ha)
        ib = hexagram_to_code(hb)
        N = self.N
        delta = (ib - ia) % N
        o1_code = (ia + delta // 2) % N
        o2_code = (o1_code + N // 2) % N            # opposite point rho^{N/2}
        return code_to_hexagram(o1_code, self.n), code_to_hexagram(o2_code, self.n)

    def _coordinate_crossover(self, ha: Hexagram, hb: Hexagram) -> Tuple[Hexagram, Hexagram]:
        """Uniform crossover on the aligned parents."""
        mask = self._np_rng.random(self.n) < 0.5
        o1 = np.where(mask, ha, hb).astype(np.int8)
        o2 = np.where(mask, hb, ha).astype(np.int8)
        return o1, o2

    # ----- Algorithm 1, lines 10-19: feasibility repair --------------------
    def repair(self, h: Hexagram) -> Optional[Hexagram]:
        """Repair an infeasible hexagram; return None if it cannot be fixed."""
        if self.cfg.target == "tau":
            return self._repair_tau(h)
        return self._repair_iprime(h)

    def _repair_tau(self, h: Hexagram) -> Optional[Hexagram]:
        limit = min(self.n, 20)
        h = h.copy()
        rmax = max(1, min(5, self.n // 10))
        for _ in range(limit):
            remaining = [int(v) for v in np.nonzero(h == 0)[0]]
            if not remaining:
                # Degenerate S = V: deselect a few vertices to expose nodes.
                S = sorted(int(v) for v in np.nonzero(h)[0])
                for v in self.rng.sample(S, min(len(S), rmax)):
                    h[v] = 0
                continue
            H = self.G.subgraph(remaining)
            if nx.number_connected_components(H) > 1:
                return h
            # omega == 1: split the (single) largest component.
            comp = list(max(nx.connected_components(H), key=len))
            r = self.rng.randint(1, rmax)
            for v in self.rng.sample(comp, min(len(comp), r)):
                h[v] = 1
        return h if self._feasible(h) else None

    def _repair_iprime(self, h: Hexagram) -> Optional[Hexagram]:
        limit = min(self.n, 15)
        h = h.copy()
        rmax = max(1, min(3, self.n // 20))
        for _ in range(limit):
            if isolated_after_removal(self.G, h) >= 2:
                return h
            S = set(np.nonzero(h)[0].tolist())
            non_isolated = [
                v for v in self.G.nodes()
                if v not in S and any(u not in S for u in self.G.neighbors(v))
            ]
            if not non_isolated:
                # Cannot create isolated vertices; perturb by deselecting.
                if S:
                    h[self.rng.choice(sorted(S))] = 0
                continue
            r = self.rng.randint(1, rmax)
            for v in self.rng.sample(non_isolated, min(len(non_isolated), r)):
                for u in list(self.G.neighbors(v)):
                    h[u] = 1                          # isolate v by removing its neighbours
        return h if self._feasible(h) else None

    # ----- population utilities --------------------------------------------
    def _random_individual(self) -> Hexagram:
        h = (self._np_rng.random(self.n) < self.cfg.init_density).astype(np.int8)
        return h

    def _new_feasible_individual(self, max_tries: int = 30) -> Hexagram:
        for _ in range(max_tries):
            h = self._random_individual()
            if self._feasible(h):
                return h
            if self.cfg.use_repair:
                fixed = self.repair(h)
                if fixed is not None and self._feasible(fixed):
                    return fixed
        # Fall back to whatever we have (its fitness will simply be zero).
        return self._random_individual()

    def _orbit_compress(self, h: Hexagram) -> Hexagram:
        if not (self.cfg.use_orbit_compression and len(self.autos) > 1):
            return h
        # Vectorised orbit: (pi . h)_j = h_{pi^{-1}(j)} => candidates = h[inv_perm].
        cands = h[self.inv_perm_matrix]              # shape (g, n)
        keys = [cands[:, j] for j in self._lex_keys_order]
        idx = np.lexsort(keys)                       # column 0 is most significant
        return cands[idx[0]].astype(np.int8)

    def _tournament_select(self, pop: List[Hexagram], fits: List[float]) -> Hexagram:
        k = min(self.cfg.tournament_k, len(pop))
        contenders = self.rng.sample(range(len(pop)), k)
        best = max(contenders, key=lambda idx: fits[idx])
        return pop[best].copy()

    def _min_group_distance(self, candidate: Hexagram, pop: List[Hexagram]) -> int:
        """min_{h in pop} d_Gamma(h, candidate), vectorised over Gamma."""
        best = self.n + 1
        for p in pop:
            cands = p[self.inv_perm_matrix]          # all g . p
            dist = int((cands != candidate).sum(axis=1).min())
            if dist < best:
                best = dist
                if best == 0:
                    break
        return best

    def _accept_diversity(self, candidate: Hexagram, pop: List[Hexagram], cand_fit: float, best_fit: float) -> bool:
        """Accept if it improves the incumbent best or is diverse enough."""
        if not self.cfg.use_diversity or not pop:
            return True
        if cand_fit > best_fit:
            return True
        return self._min_group_distance(candidate, pop) >= self.cfg.diversity_threshold

    # ----- main loop -------------------------------------------------------
    def run(self) -> GTIDEAResult:
        """Execute GT-IDEA and return the best critical set found."""
        # Line 1-5: initialise and evaluate.
        pop = [self._new_feasible_individual() for _ in range(self.cfg.pop_size)]
        objs = [self._objective(h) for h in pop]
        fits = [fitness_from_objective(o) for o in objs]

        best_idx = int(np.argmax(fits))
        best_h = pop[best_idx].copy()
        best_obj = objs[best_idx]
        history = [best_obj]

        for _ in range(self.cfg.max_iter):
            # Line 18 (elitism): carry the incumbent best into the next gen.
            new_pop: List[Hexagram] = [best_h.copy()]
            attempts = 0
            max_attempts = self.cfg.pop_size * 20

            while len(new_pop) < self.cfg.pop_size:
                attempts += 1
                relax = attempts > max_attempts      # stop enforcing diversity
                parent_a = self._tournament_select(pop, fits)
                parent_b = self._tournament_select(pop, fits)

                if self.rng.random() < self.cfg.crossover_rate:
                    children = self.crossover(parent_a, parent_b)
                else:
                    children = (parent_a.copy(), parent_b.copy())

                for child in children:
                    child = self.mutate(child)                    # line 9
                    if self.cfg.use_repair and not self._feasible(child):
                        repaired = self.repair(child)             # lines 10-19
                        if repaired is None:
                            child = self._new_feasible_individual()
                        else:
                            child = repaired
                    child = self._orbit_compress(child)           # line 20

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
            if objs[gen_best_idx] < best_obj:                     # lines 23-25
                best_obj = objs[gen_best_idx]
                best_h = pop[gen_best_idx].copy()
            history.append(best_obj)

        return GTIDEAResult(
            best_subset=vertex_subset(best_h),
            best_value=best_obj,
            best_hexagram=best_h,
            history=history,
        )


# ---------------------------------------------------------------------------
# Brute-force exact solver (validation aid for small graphs)
# ---------------------------------------------------------------------------
def brute_force(G: nx.Graph, target: str = "tau") -> Tuple[float, List[int]]:
    """Exact tau(G) or I'(G) by enumerating all 2^n subsets (small n only)."""
    G = nx.convert_node_labels_to_integers(G)
    n = G.number_of_nodes()
    objective = f_tau if target == "tau" else f_iprime
    best_val = math.inf
    best_subset: List[int] = []
    for code in range(1 << n):
        h = code_to_hexagram(code, n)
        val = objective(G, h)
        if val < best_val:
            best_val = val
            best_subset = vertex_subset(h)
    return best_val, best_subset


if __name__ == "__main__":
    # Quick smoke test on the Petersen graph.
    G = nx.petersen_graph()
    for target in ("tau", "iprime"):
        cfg = GTIDEAConfig(target=target, pop_size=30, max_iter=80, seed=0)
        result = GTIDEA(G, cfg).run()
        exact_val, exact_set = brute_force(G, target)
        print(f"[{target}] GT-IDEA: {result.best_value:.4f} on {result.best_subset}")
        print(f"[{target}] exact  : {exact_val:.4f} on {exact_set}")

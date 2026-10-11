# Data provenance and reproducibility

## Preserved observations

The 21 files in `data/archived/` are byte-for-byte copies of the existing top-level `results/exp*` CSV/text files. Supplementary CSVs, metadata, the existing verification report, and both convergence figures are also byte-for-byte copies. No experiment was rerun to populate this upload package. No absent raw records, timing measurements, FE counts, graph files or numerical values were invented.

The supplementary archive contains 110 distinct runs: four graphs x two objectives x 10 seeds = 80 convergence runs at delta=1, plus BA40/tau at delta=0,2,3 x 10 seeds = 30 additional runs. The 10 BA40/tau/delta=1 convergence runs are reused in the threshold table. Its 2480 history rows are 80 runs x 31 generations including generation zero.

`verification.json` in the supplementary archive is the existing historical report. Its `editorial` section describes manuscript/response checks in the original working folder, not checks intended to run in this package. The portable scientific-check entrypoint is under `checks/`.

## Graph inputs

All available experiment graph inputs are specified by the included generators:

| Study | Generator / inputs |
| --- | --- |
| Small exact benchmarks | `experiments.small_graphs_bruteforceable`: P8, C10, Petersen, 3x3 grid, K3,4, W10, barbell(4,2), ladder L6 |
| Main and v2/v1 comparisons | `experiments.synthetic_graphs(seed=1)`: ER(30,0.2), ER(40,0.15), BA(40,2), BA(60,3), WS(40,4,0.1), regular(40,3); built-in karate, Florentine, Davis |
| Original broad baseline experiment | Same synthetic graphs plus built-in networks from `real_graphs()`; includes Les Miserables |
| Large symmetric experiments | `exp_large_symmetric.large_symmetric_graphs()`: Q5, Q6, C100, circulant(60,[1,2,3]) |
| Original scalability | BA(n,3), graph seed=1; n=20,40,60,80,120,160 |
| Original parameter/operator sweeps | BA(50,3), graph seed=1, target=tau |
| Supplementary convergence | ER(40,0.12), BA(40,3), graph seed=1; Q4 and NetworkX karate |
| Supplementary delta | BA(40,3), graph seed=1, target=tau, delta=0,1,2,3 |

Built-in real-network edges are distributed by NetworkX; the included code uses their generator functions and relabels vertices to integers. No external edge-list dataset was present locally. `real_graphs()` can optionally read edge lists placed directly in `data/`, but adding one changes the set of instances and is not reproduction of the supplied archive.

Historical scripts typically use run seed `seed0 + r`; the documented paper rerun commands use seed0=0. Existing historical CSVs often store run indices instead of full runtime metadata, so their original environment cannot be recovered in full. The supplementary `metadata.json` explicitly records seeds, graph sizes, Python 3.12.7, NumPy 1.26.4, NetworkX 3.3 and Matplotlib 3.9.2. `requirements.txt` also includes the locally installed SciPy 1.13.1 needed for statistical tests. This is not a claim that all historical experiments used this environment. Timing is machine-dependent.

## Allocation and algorithm distinctions

`gt_idea.py` contains original `GTIDEA`; `gt_idea_v2.py` adds local descent and different mutation weights. Historical v2 drivers call `experiments.run_gtidea_counted` with a v2 configuration for baseline calibration, but that helper instantiates original `GTIDEA`. V2 is separately run for 40 generations, and original v1 in Exp7/8 is run for 120 generations. Their actual evaluation counts differ. This historical behavior is intentionally preserved because altering it would produce a different experiment.

Recorded examples: Exp7 ER(30,0.2)/tau/run0 records 186196 calls for v2 and 3240 for GA; Exp8 Q5/tau/run0 records 279976 calls for v2 and 3240 for GA. Exp9 raw CSV omits FE and time columns. There is no separately recorded equal-FE rerun in this package. The current manuscript reports quality at these allocations and disclaims equal-FE efficiency.

Scalability and operator-sensitivity summaries use original v1, not v2. The original timer starts after initialization; it does not include automorphism preprocessing. Only summaries are available for these sweeps, so exact per-run timings cannot be reconstructed as historical observations.

Supplementary runs use unchanged v2, population=20, generations=30, local-descent rounds=3 and default mutation weights. Objective memoization caches exact raw objectives without changing seeded trajectories or counting policy. `objective_calls` includes cache hits; `unique_objectives` counts cache misses. They are generation-matched diagnostics, not equal-FE comparisons.

## Standalone packaging changes

Core algorithm files `gt_idea.py`, `gt_idea_v2.py` and `baselines.py` are byte-identical to the working originals. The other historical scripts are also unchanged except:

- Exp7/8/9: leading docstrings and relevant comments now describe actual unequal allocations. Executable logic is unchanged.
- `exp1b_v2_accuracy.py`: added `--out`, `--runs`, `--seed`, `--max-n` CLI options. The `run()` function and defaults for trials/configurations are unchanged.
- `revision_experiments.py`: standalone root/output/figure paths and `--out`/`--figures` CLI options. Default trial configurations, graph definitions, operators, evaluation accounting and statistical calculations are unchanged. New runtime metadata may also include these output arguments.
- `checks/scientific_checks.py`: the existing `scientific_checks()` function is copied unchanged from `revision_verify.py`; imports/output paths are independent of manuscript backups.

The manifest records both current packaged and source hashes for each copied/derived file. A different hash for the explicitly adapted scripts is expected; data and figure hashes must match exactly.

## Excluded workspace material

Manuscript and reviewer response, AMCS templates, author photos/biographies, LaTeX build logs, article-rewriting and formatting helpers, PDF layout checks, theory-only finite-audit scripts/results, backups, caches and temporary files are excluded. They are not experimental code or observations. The obsolete original README (which claimed no preloaded results and uniform equal-FE comparisons) is replaced by the current package documentation.

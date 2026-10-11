# Current manuscript to experiment mapping

Table numbers refer to the current 18-page AMCS manuscript.

| Manuscript item | Driver in `code/` | Archived evidence |
| --- | --- | --- |
| Table 1: four deletion-set examples | Definitions `f_tau`, `f_iprime` in `gt_idea.py`; `checks/scientific_checks.py` | Analytic examples, not a stochastic experiment table |
| Table 2: exactly solvable instances | `exp1b_v2_accuracy.py` | `data/archived/exp1b_v2_accuracy.csv` |
| Table 3: GA/SA/BPSO comparison | `exp9_multibaseline.py` | `exp9_multibaseline_raw.csv`, `exp9_multibaseline.csv`, `exp9_stats.txt` in `data/archived/` |
| Main-comparison ablation paragraph | `exp_v2_comparison.py` | `exp7_v2_raw.csv`, `exp7_v2_summary.csv`, `exp7_stats.txt` |
| Table 4: large symmetric graphs | `exp8_v2_large_symmetric.py` | `exp8_v2_large_sym_raw.csv`, `exp8_v2_large_sym.csv`, `exp8_stats.txt` |
| Table 5: original-version scalability | `experiments.py --exp 5` | `exp5_scalability.csv` (summary only) |
| Table 6: original-version operator sensitivity | `experiments.py --exp 5` | Rows `mut_mix:*` in `exp5_params.csv` |
| Table 7: supplementary delta sensitivity | `revision_experiments.py` | `delta_summary.csv`, `runs.csv` in `data/supplementary/revision_20261011/` |
| Figure 1: supplementary convergence | `revision_experiments.py` | `histories.csv`, `convergence_summary.csv`; `figures/revision_convergence.pdf` and `.png` |

Except for supplementary files, result filenames in the table are under `data/archived/`.

## Additional preserved experiments

| Driver | Outputs / role |
| --- | --- |
| `experiments.py --exp 1` | `exp1_accuracy.csv`: original-version small-graph study, distinct from v2 Table 2 |
| `experiments.py --exp 2` | `exp2_raw.csv`, `exp2_summary.csv`, `exp2_stats.txt`: original-version broad baseline study |
| `experiments.py --exp 3` | `exp3_ablation.csv`: original-version component ablations |
| `experiments.py --exp 4` | `exp4_symmetry.csv`: original-version symmetry study |
| `experiments.py --exp 5` | Also includes original population-size and delta sweeps in `exp5_params.csv` |
| `exp_large_symmetric.py` | `exp6_large_symmetric_raw.csv`, `exp6_large_symmetric.csv`, `exp6_stats.txt`: original-version symmetry-on/off comparison |
| `demo_gt_idea.py` | Console demonstration with original-version algorithm and brute force; not the paper's v2 main experiment |

These extra files retain experimental context and dependencies. They are not substituted for the current manuscript's main tables.

## Independent checks

`checks/scientific_checks.py` reuses the existing P3 and C6 counterexamples, exhaustive C4 automorphism/neighborhood invariance checks, and seeded memoization equivalence checks for both targets. It excludes manuscript formatting and reviewer-response checks.

`checks/verify_bundle.py` checks file bytes only. Neither check creates new paper results.

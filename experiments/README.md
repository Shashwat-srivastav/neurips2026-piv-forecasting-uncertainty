# Experiments

One folder per line of investigation. Scripts write their console output next to themselves (`*_out.txt`); those logs are
the sources the report cites. Scripts read the real data through `RPDE_TRAIN_REAL` and the local forecast caches built by
`interval_recipes/cache_preds.py`.

| folder | what it investigates | report |
|---|---|---|
| `interval_recipes/` | interval-width and interval-centre recipes under the live-calibrated proxies; proxy calibration | §5–7 |
| `estimator_blend/` | v17's per-element blend of the FNO with the window mean and last frame, and its interval tables | §6–7, Table 5 |
| `per_window_rebuild/` | per-window rebuilds of v17 and v19 and the independence test | §13, Disclosure 1 |
| `v18_refits/` | table refits and the fine-tuned-checkpoint variant (v18, v18ft; not submitted) | Appendix B |
| `v19_ridge_partner/` | ridge linear forecaster as a third blend partner (v19), including its build | Appendix B |
| `tke_energy/` | diagnosis of the FNO's turbulent-kinetic-energy deficit and the energy correction | §6 |
| `piv_noise/` | PIV measurement noise and why persistence wins at lead 1 | §6 |
| `robustness/` | random-split refits (13 splits), swap ablation, leave-one-AoA/Re-band-out, bootstrap intervals | §7, §10, Table 6 |
| `clean_fno_build/` | building the clean sim-to-real FNO submission from the Run 1 checkpoint | §10 |
| `clean_fno_regimes/` | per-regime evaluation of the clean FNO build and √-scaled widths | §10 |
| `cno_check/` | the organisers' CNO under the same proxy | §10 |
| `ensemble_check/` | averaging the official FNO with our clean FNO | §10 |
| `early_diagnostics/` | first-month diagnostics: generalisation map, hold-out hardness, local references, linear gates, output combinations, AdaBN, WiSE-FT, UNet and DPOT pilots | Appendices B–C |

Scripts in `early_diagnostics/` were run against local build folders of that time (`.dist_*`) that no longer exist; they
are kept as the record of what was run.

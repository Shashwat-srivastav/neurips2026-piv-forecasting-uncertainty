# Submissions

Each folder holds the inference code (`submission.py`) and fitted tables (`_blend_tables.py`) of one build. The files are
byte-identical to the submitted packages; the packages themselves (`*.zip`, with the model checkpoint) are not in the
repository and are listed with their hashes in `../MANIFEST.sha256`.

| folder | build | live final score | notes |
|---|---|---|---|
| `exp1_sf/` | our clean sim-to-real FNO (Run 1) + interval tables with √-scaled widths | 79.238 | per-window statistics; independence test passed |
| `v17/` | organisers' FNO + estimator blend + interval tables + energy correction | 79.261 | rank bins pooled over the 5 windows of each call (report, Disclosure 1) |
| `v17_per_window/` | v17 rebuilt with per-window statistics | not scored live | compliant rebuild of v17 |
| `v19_per_window/` | v17 + ridge forecaster as a third blend partner, per-window | not scored live | compliant rebuild of v19 |

To package a build: add the checkpoint and the kit's vendored modules, zip the folder, and run
`python src/evaluation/preflight.py <zip>` before submitting. `test_independence.py` checks that every window's output is
identical whether it is predicted alone or in any batch.

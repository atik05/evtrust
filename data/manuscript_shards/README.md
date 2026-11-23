# Cold-start predictions of the paper (ACN)

Out-of-sample predictions of the cross-fitted cold-start decomposition reported in the paper (Section II-E):
35,695 sessions per seed of the 549 cohort drivers, five seeds, CatBoost and the FT-Transformer, for duration
(`duration_hours`) and energy (`kwh_delivered`). They are the starting point of the robustness analyses
(`robustness/`), so that these analyses use exactly the predictions behind the published numbers.

Each file `cs_<learner>_<target>_<seed>.parquet` has one row per session with

| Column | Content |
|---|---|
| `session_key` | ACN-Data session identifier |
| `user` | ACN-Data driver identifier |
| `site`, `k`, `is_T`, `session_index` | site, number of the driver's completed sessions, evaluation-set flag (last 20 % of the driver's sessions), row index |
| `pred_A` to `pred_E` | predictions of settings A (random split), B (driver-temporal), C (unseen driver, own history), D (cold start) and E (published-style aggregates) |
| `target`, `learner`, `seed` | as in the file name |

No session data are redistributed: durations, energies and requests are restored from the user's own copy of
ACN-Data (`runs/acn/data/processed/sessions.parquet`, written by `./reproduce.sh acn`) when the files are read,
and the preflight stage stops unless the restored cohort reproduces the published values (35,695 sessions per
seed, 549 drivers, value of own history 73.19 %, flexibility at 2 % undelivered energy 38.95, 35.77 and 20.00 %).
`SHA256SUMS` lists the checksums checked by `scripts/run_robustness.sh`.

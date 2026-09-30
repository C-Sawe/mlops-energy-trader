title: "Investigate: mlruns.db schema migration wiped MLflow run history (root cause unknown)"
labels: bug, investigation, mlflow
---
## What happened (CLAUDE.md §13, 2026-09-17)
The first time `resolve_mlflow_tracking_uri()` ran against the long-lived `mlruns.db`, MLflow ran a schema migration. New tables appeared (`webhooks`, `guardrails`, …) and the **`runs` table came back empty**: 92 runs' MLflow-side params and metrics were gone. Postgres `model_run`/`model_version` and the artifact files survived. One `model_version` row was hand-patched to a direct file path.

## Why it matters more now
A cloud deployment creates a fresh MLflow store and will upgrade it whenever the image's mlflow version changes. `requirements.txt` says `mlflow>=2.17` (unpinned). The new `constraints.txt` pins `mlflow==3.16.1`, but the upgrade path was never tested.

## Tasks
- [ ] Reproduce: copy a pre-migration `mlruns.db` (if a backup exists) and open it with 3.16.1; inspect `alembic_version` before and after
- [ ] Determine whether runs were deleted or just became unreadable (was the old DB written by a different MLflow major version?)
- [ ] Add a startup check: log the `alembic_version` and run count, and refuse to start if the count drops to 0 while `model_version` rows reference runs
- [ ] Document the "bump mlflow" procedure (back up `mlruns` volume → bump constraints → test)

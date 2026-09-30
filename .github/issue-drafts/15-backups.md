title: "Automated backups for the deployed Postgres and mlruns volumes"
labels: ops, deployment, reliability
---
## Problem
In the cloud deployment, all traceability state (NFR-07: decisions → versions → runs → partitions) lives in the `pgdata` volume, and the model artifacts live in `mlruns`. Nothing backs either up. A lost VM disk means losing the whole CT history the thesis relies on as evidence.

## Proposal
A daily cron on the VM (or a scheduled workflow) that runs `pg_dump` and tars the `mlruns` volume into object storage (S3/GCS/Spaces) with 14-day retention. Test a restore once and write down how long it took.

## Acceptance criteria
- [ ] Backup job running, documented in `docs/DEPLOYMENT.md`
- [ ] One restore rehearsed onto a scratch VM; the dashboard shows the same active model and decision log

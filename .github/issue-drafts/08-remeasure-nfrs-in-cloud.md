title: "Re-measure NFR-01/02/03/04 on the deployed stack (Postgres, real HTTP), not just local TestClient + SQLite"
labels: evaluation, nfr, thesis
---
## Why
Every NFR latency figure in CLAUDE.md §5 / Chapter 5 was measured on the M5 with `fastapi.testclient.TestClient` (no network) against **SQLite**. CLAUDE.md itself flags the NFR-01 p99 tail (77 ms, ~4.5× p50) as "worth watching against real Postgres".

## Tasks
- [ ] NFR-01: 100 `/predict` requests over real HTTP against the deployed backend (from inside the VM, to separate server latency from WAN latency). Report p50/p95/p99
- [ ] NFR-02 / NFR-03: repeat `scripts/nfr_reliability_check.py`'s method during a real retrain on the VM (2 vCPU will contend more than the M5)
- [ ] NFR-04: `docker compose kill backend` and time until `/health` answers again, now including container restart, not just application readiness (the existing 497.9 ms figure excludes the process-manager restart)
- [ ] Report both environments side by side in Chapter 5. Don't replace the local numbers

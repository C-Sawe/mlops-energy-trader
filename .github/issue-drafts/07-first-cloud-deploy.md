title: "First cloud deployment: provision a VM, set secrets, run Deploy with bootstrap"
labels: ops, deployment, priority:high
---
Tracks the first real rollout using `.github/workflows/deploy.yml`. Full guide: `docs/DEPLOYMENT.md`.

## Checklist
- [ ] VM provisioned (≥2 vCPU / 4 GB, Ubuntu 24.04, ports 22/80/443 only), Docker installed
- [ ] **Yahoo Finance reachable from the VM** (`curl -sI https://query1.finance.yahoo.com`). Some datacenter IPs are blocked, and CLAUDE.md §9 recorded a 403 from one cloud environment
- [ ] Deploy SSH key created; `DEPLOY_KNOWN_HOSTS` checked against the provider's console
- [ ] Secrets set: `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY`, `DEPLOY_KNOWN_HOSTS`, `POSTGRES_PASSWORD` (hex), `API_BEARER_TOKEN`, `BASIC_AUTH_USER`, `BASIC_AUTH_PASSWORD`
- [ ] Domain + `SITE_ADDRESS` variable set **before sharing the URL** (otherwise basic auth goes over plain HTTP)
- [ ] Actions → Deploy → Run workflow with **bootstrap** ticked
- [ ] Dashboard loads, "Last ingest" is recent, "Active model" shows a label, and the CT status ticks every 5 min
- [ ] Record in CLAUDE.md §9 what actually happened (timings, any failures)

## Known unknowns
- The backend and dashboard images have not been built yet (Docker wasn't running on the dev machine when the workflow was written). The first CI build is the first real check of the Dockerfiles.
- The first MLflow initialisation on the fresh `mlruns` volume. See the mlruns.db migration issue.

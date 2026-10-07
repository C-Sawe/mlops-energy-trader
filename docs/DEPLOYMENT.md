# Cloud deployment

This deploys the full system (Postgres, the FastAPI backend with its CT and
ingestion schedulers, and the dashboard) to a **single Linux VM** with Docker
Compose, driven by GitHub Actions. It works on any provider that gives you an
SSH-reachable VM.

A single VM is a deliberate choice. The backend runs the CT orchestrator and
ingestion scheduler as long-lived background tasks, and training runs
in-process (CLAUDE.md §6). Scale-to-zero platforms such as Cloud Run, Lambda
or free-tier Render would pause those schedulers, and a paused scheduler
cannot detect decay.

## What runs where

```
Internet ──▶ caddy (:80/:443, auto-TLS) ──▶ frontend (nginx: static + basic auth)
                                               │  /api/* (bearer token injected server-side)
                                               ▼
                                            backend (uvicorn :8000, not published)
                                               │
                                               ▼
                                            postgres (not published)
```

| Volume | Holds | Lose it and... |
|---|---|---|
| `pgdata` | market data, decisions, snapshots, `model_run`/`model_version` | you lose traceability (NFR-07) |
| `mlruns` | MLflow tracking DB and trained model artifacts | the active model can't be reloaded |
| `caddy_data` | TLS certificates | Caddy re-issues them (rate-limited) |

## Workflows

| File | When | What |
|---|---|---|
| `.github/workflows/ci.yml` | every push (not main) and PR | pytest (138) + Vitest (47) + frontend build |
| `.github/workflows/deploy.yml` | push to `main`, or manual | CI gate → build and push both images to GHCR → SSH to VM → `docker compose up -d` → health checks |

Pushing to `main` deploys. `sprint-4` is **not merged into `main` yet**
(CLAUDE.md §3). Until it is, deploy a branch by hand: Actions → Deploy → Run
workflow → pick `sprint-4`.

## One-time setup

### 1. Provision a VM

- **Size:** 2 vCPU and 4 GB RAM minimum. PPO retrains run in-process on
  CPU, and torch plus pandas need the memory. Roughly: AWS `t3.medium`,
  GCP `e2-medium`, DigitalOcean 4 GB droplet, Hetzner CX22.
- **OS:** Ubuntu 24.04 LTS.
- **Firewall:** allow inbound 22, 80 and 443 only.
- **Install Docker:** `curl -fsSL https://get.docker.com | sh`, then
  `sudo usermod -aG docker $USER` and log back in.
- **Check Yahoo Finance is reachable from the VM** before anything else:
  `curl -sI https://query1.finance.yahoo.com | head -1`. Some datacenter IP
  ranges are blocked or rate-limited (CLAUDE.md §9 saw a 403 from one cloud
  environment). If it fails, ingestion (FR-01) cannot work from this host.

### 2. Create a deploy SSH key

On your own machine:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/mlops_deploy -N "" -C "github-actions-deploy"
ssh-copy-id -i ~/.ssh/mlops_deploy.pub <user>@<vm-ip>
ssh-keyscan -t ed25519 <vm-ip>      # copy this output for DEPLOY_KNOWN_HOSTS
```

Before trusting the `ssh-keyscan` output, compare it with the host key your
provider's console shows.

### 3. Add GitHub secrets and variables

Go to Repo → Settings → Secrets and variables → Actions.

**Secrets** (all required):

| Secret | Value |
|---|---|
| `DEPLOY_HOST` | VM public IP or hostname |
| `DEPLOY_USER` | SSH user (in the `docker` group) |
| `DEPLOY_SSH_KEY` | contents of `~/.ssh/mlops_deploy` (private key) |
| `DEPLOY_KNOWN_HOSTS` | the `ssh-keyscan` line from step 2 |
| `POSTGRES_PASSWORD` | `openssl rand -hex 24`. Use hex: the password goes straight into a connection URL, so `@`, `/` and `:` would break it |
| `API_BEARER_TOKEN` | `openssl rand -hex 32` |
| `BASIC_AUTH_USER` | dashboard login name |
| `BASIC_AUTH_PASSWORD` | dashboard login password |

**Variables** (all optional):

| Variable | Default | Notes |
|---|---|---|
| `SITE_ADDRESS` | `:80` | Set a domain (e.g. `trader.example.com`) whose DNS A record points at the VM. Caddy then serves HTTPS automatically. **Do this before sharing the URL**: basic auth over plain HTTP sends the password in the clear. |
| `CT_EVALUATION_INTERVAL_SECONDS` | `300` | FR-13 cadence |
| `INGESTION_INTERVAL_SECONDS` | `86400` | FR-01 cadence. Lower it for a live demo. |
| `IMAGE_PLATFORMS` | `linux/amd64,linux/arm64` | CPU architectures to build. Run `uname -m` on the VM: `aarch64` → set `linux/arm64`, `x86_64` → `linux/amd64`. Building only the one you need roughly halves build time. |
| `PAPER_TRADING_ENABLED` | `false` | `true` turns on the daily forward paper-trading cycle (see below). Needs the two Alpaca secrets. |
| `PAPER_TRADE_TIME_UTC` | `22:00` | Weekday run time. 22:00 UTC is after the US close in both EDT and EST. |

**Paper-trading secrets** (only if `PAPER_TRADING_ENABLED=true`): `ALPACA_API_KEY`
and `ALPACA_SECRET_KEY` from a free Alpaca **paper** account. The backend's
broker URL is a fixed constant pointing at `paper-api.alpaca.markets`; no
setting can point it at live trading.

Optionally, open Settings → Environments → `production` and add yourself as
a required reviewer, so each rollout waits for a click.

### 4. First deploy (with bootstrap)

Actions → Deploy → Run workflow → pick the branch → tick **bootstrap**.

The first deploy starts with an empty database, and the ingestion
scheduler's 120-day trailing window cannot fill it (CLAUDE.md §13). The
bootstrap step therefore:

1. runs `scripts/run_ingestion.py --end <today>`, a full 2015→today backfill;
2. calls `POST /ct/evaluate`. With no incumbent, this trains and promotes
   the first model (the §10 bootstrap path).

Training runs in the background, so the first model appears on the
dashboard a minute or two after the workflow finishes. Don't tick bootstrap
on later deploys: the data and model persist in the volumes.

## Operating it

```bash
ssh <user>@<vm-ip>
cd ~/mlops-energy-trader
docker compose ps
docker compose logs -f backend          # CT ticks, retrains, FR-17 decisions
docker compose exec backend python scripts/run_ingestion.py --start 2026-01-01 --end 2026-09-30   # manual backfill after an outage longer than 120 days
```

**Rollback:** re-run the Deploy workflow from an earlier commit, or on the VM
set `IMAGE_TAG=<older sha>` in `.env` and run `docker compose up -d`. Model
promotions are *not* rolled back by an image rollback. They live in
Postgres, and FR-17's gate governs them.

**Backups:** `docker compose exec postgres pg_dump -U mlops mlops_trading > backup.sql`,
and archive the `mlruns` volume. Nothing in this setup does this for you
(see the issue backlog).

## Forward paper trading

With `PAPER_TRADING_ENABLED=true`, every weekday at `PAPER_TRADE_TIME_UTC`
the backend:

1. ingests the day's bar;
2. skips if that bar isn't today's (holiday, data lag) or today already traded;
3. calls `predict()` with the paper account's real current weights;
4. converts the raw target weights to long-only notional orders
   (`src/execution/rebalance.py`) and submits them to the paper account.
   They queue and fill at the next open.

Every `PAPER_SYNC_INTERVAL_SECONDS` (default 900) it records equity,
positions and fills. The dashboard's "Price & entries" and "Paper account"
cards read those records.

Trigger a cycle by hand (it obeys the same skip rules, so it can't
double-trade):

```bash
curl -X POST -u "$BASIC_AUTH_USER:$BASIC_AUTH_PASSWORD" https://<site>/api/paper/run   # nginx swaps basic auth for the bearer token
```

## What this deployment does not change

- It is **not live trading.** With paper trading enabled, orders go to
  Alpaca's paper venue: simulated money, real order API. Paper P&L is
  evidence that the loop runs forward unattended, not a profitability
  result (CLAUDE.md §1).
- NFR-01/02/04 were measured locally on the M5 against SQLite/TestClient.
  Numbers from the cloud VM will differ and must be re-measured before they
  are cited.

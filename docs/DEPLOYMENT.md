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
Internet ──▶ caddy (:80/:443, auto-TLS) ──▶ frontend (nginx: static + basic auth, or sign-in)
                                               │  /api/* (bearer token injected server-side in basic mode)
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
| `./backups` (host directory) | daily backups of `pgdata` and `mlruns` | you have no restore point (see [Backups](#backups)) |

## Workflows

| File | When | What |
|---|---|---|
| `.github/workflows/ci.yml` | every push (not main) and PR | pytest (176) + Vitest (78) + frontend build |
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
| `BASIC_AUTH_USER` | dashboard login name (basic mode only; see [Sign-in](#sign-in)) |
| `BASIC_AUTH_PASSWORD` | dashboard login password (basic mode only) |

**Variables** (all optional):

| Variable | Default | Notes |
|---|---|---|
| `SITE_ADDRESS` | `:80` | Set a domain (e.g. `trader.example.com`) whose DNS A record points at the VM. Caddy then serves HTTPS automatically. **Do this before sharing the URL**: basic auth over plain HTTP sends the password in the clear. |
| `CT_EVALUATION_INTERVAL_SECONDS` | `300` | FR-13 cadence |
| `INGESTION_INTERVAL_SECONDS` | `86400` | FR-01 cadence. Lower it for a live demo. |
| `IMAGE_PLATFORMS` | `linux/amd64,linux/arm64` | CPU architectures to build. Run `uname -m` on the VM: `aarch64` → set `linux/arm64`, `x86_64` → `linux/amd64`. Building only the one you need roughly halves build time. |
| `PAPER_TRADING_ENABLED` | `false` | `true` turns on the daily forward paper-trading cycle (see below). Needs the two Alpaca secrets. |
| `PAPER_TRADE_TIME_UTC` | `22:00` | Weekday run time. 22:00 UTC is after the US close in both EDT and EST. |
| `BACKUP_TIME_UTC` | `03:00` | Daily backup time (see [Backups](#backups)). |
| `BACKUP_RETENTION_DAYS` | `14` | Backups older than this are deleted after each new one. |
| `DASHBOARD_AUTH` | `basic` | `signin` replaces basic auth with the dashboard's own sign-in: email and password and/or Google, Microsoft or GitHub. See [Sign-in](#sign-in). |
| `PASSWORD_LOGIN` | `false` | `true` shows the email-and-password form on the login page (with `DASHBOARD_AUTH=signin`). |
| `MICROSOFT_TENANT` | `common` | `common` accepts personal and work/school Microsoft accounts. A tenant ID restricts sign-in to one organisation. |

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
docker compose exec backend python scripts/run_ingestion.py --start 2026-01-01 --end 2026-09-30   # manual backfill of an arbitrary range
```

An outage longer than the ingestion scheduler's 120-day window no longer
needs a manual backfill: the next tick anchors its fetch to the last stored
bar and fills the gap itself, logging a warning when it does
(`ingestion_scheduler._fetch_start`). An empty database still needs the
bootstrap step above, since there is no last stored bar to anchor to.

**Rollback:** re-run the Deploy workflow from an earlier commit, or on the VM
set `IMAGE_TAG=<older sha>` in `.env` and run `docker compose up -d`. Model
promotions are *not* rolled back by an image rollback. They live in
Postgres, and FR-17's gate governs them.

## Backups

The `backup` service (`deploy/backup.sh`) backs up both stateful volumes
every day at `BACKUP_TIME_UTC` (default 03:00) into `~/mlops-energy-trader/backups/`
on the VM, and deletes backups older than `BACKUP_RETENTION_DAYS` (default 14).
Each backup is one directory named by its UTC time, e.g. `20261007T030000Z/`:

| File | From | Notes |
|---|---|---|
| `postgres.dump` | `pgdata` | `pg_dump` custom format, same major version as the server |
| `mlruns.db` | `mlruns` | MLflow tracking store, copied with `sqlite3 .backup` so a write in progress can't tear it |
| `mlruns-artifacts.tar.gz` | `mlruns` | trained model files |
| `SHA256SUMS` | | checked before any restore |

A backup is written to a hidden `.partial-*` directory and renamed only when
complete, so a failure part-way never leaves something that looks finished.
Files are readable only by the deploy user.

```bash
cd ~/mlops-energy-trader
docker compose logs backup                                    # one line per backup, with size and duration
docker compose exec backup sh /usr/local/bin/backup.sh now    # back up right now (e.g. before a risky change)
ls backups/
```

### Restore

```bash
cd ~/mlops-energy-trader
docker compose stop backend                                   # restore refuses while anything else is connected
docker compose exec backup sh /usr/local/bin/backup.sh restore 20261007T030000Z
docker compose start backend
```

The restore verifies the checksums first and refuses a backup that fails
them. It then replaces the database contents and the whole `mlruns` volume.
On a new VM, run the first deploy without bootstrap, copy a backup
directory into `backups/`, and restore it as above.

**Rehearsed 2026-10-07, locally, not on the VM.** A throwaway
`postgres:16-alpine` was seeded at production scale (13,830 market rows,
540 model runs, 135 versions, 300 decisions and 180 snapshots), and the
`mlruns` volume held a copy of the real local MLflow store (313 MB, 1,443
files). The backup took **8 s** (260 MB). The database and volume were
then deleted and replaced with empty ones. The restore took **3 s**, and
afterwards every table's row count and content hash matched, the artifact
files were byte-identical, `mlruns.db` passed `PRAGMA integrity_check`, and
the active model version was intact. Both refusals were also checked: a
tampered backup was rejected on its checksum, and a restore with another
open connection was rejected. Re-time it on the VM's own disk before citing
a recovery time for the deployed system.

### Off-site copies

`backups/` is on the VM's own disk. It survives a broken container, a bad
deploy or a deleted volume, but **not the loss of the VM or its disk**.
Until there is an off-site copy, pull the backups to another machine
regularly:

```bash
rsync -a <user>@<vm-ip>:mlops-energy-trader/backups/ ~/mlops-backups/
```

Most providers can also snapshot the whole disk on a schedule, which covers
the same failure.

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

## Sign-in

`DASHBOARD_AUTH=signin` replaces the shared basic-auth password with
per-person sign-in (`backend/src/serving/auth.py`): email and password,
Google, Microsoft, GitHub, or any mix. Only addresses in
`ALLOWED_EMAILS` get in, and the list is re-checked on every request, so
removing an address takes effect at once. nginx then stops injecting the
bearer token: the browser's session cookie decides. Scripts can still call
the backend with the bearer token directly.

Sign-in needs HTTPS, so `SITE_ADDRESS` must be a domain, not `:80`. In
everything below, `<site>` is that domain.

### Email and password

Set the variable `PASSWORD_LOGIN=true`. Accounts are created from the VM,
not by signing up: self-sign-up would let anyone claim an allowlisted
address, since nothing emails them to check it's theirs.

```bash
docker compose exec backend python scripts/create_dashboard_user.py --email you@gmail.com
```

It asks for the password twice, at a hidden prompt (at least 12
characters). Run it again to change a password, or add `--delete` to remove
the account. The email must also be in `ALLOWED_EMAILS`. Passwords are
stored as salted scrypt hashes. After 10 wrong attempts for one email in 15
minutes, that email is blocked until the 15 minutes pass. Note that this
means someone guessing at your address can lock you out for that long; you
can still sign in with a provider meanwhile.

### Google, Microsoft and GitHub

A provider appears on the login page once both its client ID and secret are
set. You only need the ones you want.

| Provider | Where to register | Redirect URI to register |
|---|---|---|
| Google | [Google Cloud Console](https://console.cloud.google.com/apis/credentials) → Create credentials → OAuth client ID → *Web application*. Set up the consent screen first (External, scopes `openid`, `email`, `profile`). | `https://<site>/api/auth/callback/google` |
| Microsoft | [Microsoft Entra admin center](https://entra.microsoft.com) → App registrations → New registration. Supported account types: *any organisational directory and personal Microsoft accounts*. Platform *Web*. Then Certificates & secrets → New client secret (note its expiry date). | `https://<site>/api/auth/callback/microsoft` |
| GitHub | GitHub → Settings → Developer settings → OAuth Apps → New OAuth App. Homepage URL `https://<site>`. | `https://<site>/api/auth/callback/github` |

GitHub only allows one callback URL per OAuth app, so register a second app
for local development.

**Secrets** (signin mode):

| Secret | Value |
|---|---|
| `SESSION_SECRET` | `openssl rand -hex 32`. Signs the session cookie; changing it signs everyone out. |
| `ALLOWED_EMAILS` | Comma-separated, e.g. `you@gmail.com,supervisor@strathmore.edu` |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | from Google |
| `MICROSOFT_CLIENT_ID`, `MICROSOFT_CLIENT_SECRET` | Application (client) ID and the secret's *value* |
| `OAUTH_GITHUB_CLIENT_ID`, `OAUTH_GITHUB_CLIENT_SECRET` | from GitHub. The `OAUTH_` prefix is required: GitHub doesn't allow secret names that start with `GITHUB_`. |

Then set the variable `DASHBOARD_AUTH=signin` and deploy. The backend
refuses to start if sign-in is on without `SESSION_SECRET` or
`ALLOWED_EMAILS`.

**Which email is checked.** Google: the account email, only if Google has
verified it. GitHub: the account's primary *verified* email, not the public
profile email. Microsoft: personal accounts use their email. Work and school
accounts use the sign-in name (UPN), because their `email` field is set by
the organisation's admin and isn't verified. So list
`name@strathmore.edu` as the user signs in, not an alias.

**Locally**, without Docker: register `http://localhost:8000/auth/callback/<provider>`
with each provider (Google and Microsoft accept `http://localhost`), put the
same variables in the backend's environment, leave `PUBLIC_API_URL` and
`FRONTEND_URL` at their defaults, and open the dashboard at
`http://localhost:5173`. Use `localhost`, not `127.0.0.1`: the session cookie
belongs to `localhost`, and the browser treats the two as different sites.

With `DASHBOARD_AUTH=signin`, trigger the paper cycle (above) with the bearer
token from the VM instead, since nginx no longer swaps in credentials:

```bash
docker compose exec backend python -c "import os, urllib.request as u; print(u.urlopen(u.Request('http://localhost:8000/paper/run', method='POST', headers={'Authorization': 'Bearer ' + os.environ['API_BEARER_TOKEN']})).read().decode())"
```

## What this deployment does not change

- It is **not live trading.** With paper trading enabled, orders go to
  Alpaca's paper venue: simulated money, real order API. Paper P&L is
  evidence that the loop runs forward unattended, not a profitability
  result (CLAUDE.md §1).
- NFR-01/02/04 were measured locally on the M5 against SQLite/TestClient.
  Numbers from the cloud VM will differ and must be re-measured before they
  are cited.

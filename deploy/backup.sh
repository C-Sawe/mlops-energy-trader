#!/bin/sh
# Backups of the two volumes that hold this deployment's state (issue #40;
# procedure in docs/DEPLOYMENT.md, "Backups"):
#
#   pgdata -> postgres.dump            market data, decisions, snapshots and
#                                      model_run/model_version: NFR-07's whole
#                                      decision -> version -> run -> partition chain
#   mlruns -> mlruns.db                MLflow's tracking store; resolves runs:/ URIs
#             mlruns-artifacts.tar.gz  the trained model files themselves
#
# Runs as the `backup` compose service on postgres:16-alpine, the same image
# as the database, so pg_dump/pg_restore match the server's major version.
#
#   backup.sh [loop]          daily at BACKUP_TIME_UTC, keep BACKUP_RETENTION_DAYS
#   backup.sh now             one backup immediately
#   backup.sh restore <name>  restore one backup (stop the backend first)
set -eu

OUT=/backups
MLRUNS=/mlruns
APP_UID=10001  # the backend image's `app` user, which must own mlruns
: "${BACKUP_TIME_UTC:=03:00}"
: "${BACKUP_RETENTION_DAYS:=14}"
: "${BACKUP_UID:=0}"

log() { echo "$(date -u +%FT%TZ) backup: $*"; }

backup_now() {
  # MLflow's store is SQLite. Copying the file while the backend writes to
  # it can capture a torn page; sqlite3's .backup takes a consistent copy.
  command -v sqlite3 >/dev/null || apk add --no-cache sqlite >/dev/null
  umask 077
  rm -rf "$OUT"/.partial-*  # left by an earlier failed run
  name=$(date -u +%Y%m%dT%H%M%SZ)
  work="$OUT/.partial-$name"
  mkdir -p "$work"
  started=$(date +%s)

  pg_dump --format=custom --file="$work/postgres.dump"
  if [ -f "$MLRUNS/mlruns.db" ]; then
    sqlite3 "$MLRUNS/mlruns.db" ".backup '$work/mlruns.db'"
  fi
  tar -czf "$work/mlruns-artifacts.tar.gz" -C "$MLRUNS" --exclude=./mlruns.db .
  (cd "$work" && sha256sum -- * > SHA256SUMS)

  # Renamed into place only once complete, so a crash part-way through
  # never leaves something that looks like a finished backup.
  mv "$work" "$OUT/$name"
  chown -R "$BACKUP_UID" "$OUT/$name"
  log "wrote $name ($(du -sh "$OUT/$name" | cut -f1)) in $(( $(date +%s) - started ))s"

  for old in $(find "$OUT" -mindepth 1 -maxdepth 1 -type d -name '20*Z' -mtime +"$BACKUP_RETENTION_DAYS"); do
    rm -rf "$old"
    log "pruned $(basename "$old")"
  done
}

restore() {
  name=${1:?usage: backup.sh restore <name>, one of: $(ls "$OUT")}
  dir="$OUT/$name"
  [ -d "$dir" ] || { log "no backup named $name; have: $(ls "$OUT")"; exit 1; }
  (cd "$dir" && sha256sum -c -s SHA256SUMS) || { log "$name fails its checksums; not restoring it"; exit 1; }

  # Restoring under a running backend would interleave its writes with the
  # restore. Refuse unless nothing else is connected.
  others=$(psql -tAc "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() AND pid <> pg_backend_pid()")
  if [ "$others" != "0" ]; then
    log "$others other connection(s) to $PGDATABASE; run 'docker compose stop backend' first"
    exit 1
  fi

  started=$(date +%s)
  pg_restore --clean --if-exists --no-owner --single-transaction --dbname="$PGDATABASE" "$dir/postgres.dump"
  find "$MLRUNS" -mindepth 1 -delete
  tar -xzf "$dir/mlruns-artifacts.tar.gz" -C "$MLRUNS"
  if [ -f "$dir/mlruns.db" ]; then cp "$dir/mlruns.db" "$MLRUNS/mlruns.db"; fi
  chown -R "$APP_UID" "$MLRUNS"
  log "restored $name in $(( $(date +%s) - started ))s; now run 'docker compose start backend'"
}

case "${1:-loop}" in
  now) backup_now ;;
  restore) shift; restore "$@" ;;
  loop)
    log "daily at $BACKUP_TIME_UTC UTC into $OUT, keeping $BACKUP_RETENTION_DAYS days"
    while true; do
      now=$(date -u +%s)
      next=$(date -u -d "$(date -u +%Y-%m-%d) $BACKUP_TIME_UTC" +%s)
      [ "$next" -gt "$now" ] || next=$((next + 86400))
      sleep $((next - now))
      # A separate process, not a function call: `set -e` is ignored inside
      # anything on the left of `||`, so a failed pg_dump would otherwise
      # carry on and publish an incomplete backup.
      sh "$0" now || log "FAILED; the previous backups are untouched, retrying tomorrow"
    done
    ;;
  *) echo "usage: backup.sh [loop|now|restore <name>]" >&2; exit 2 ;;
esac

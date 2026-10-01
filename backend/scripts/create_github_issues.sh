#!/usr/bin/env bash
# Files every draft in .github/issue-drafts/ as a GitHub issue on this repo.
# Each draft: `title:` line, `labels:` line (comma-separated), `---`, body.
#
# Prereqs: brew install gh && gh auth login
# Usage:   scripts/create_github_issues.sh            # dry run: prints what it would file
set -euo pipefail
cd "$(dirname "$0")/../.."

apply=false
[[ "${1:-}" == "--apply" ]] && apply=true

for f in .github/issue-drafts/*.md; do
  title=$(sed -n 's/^title: "\{0,1\}\(.*\)"\{0,1\}$/\1/p' "$f" | head -1 | sed 's/"$//')
  labels=$(sed -n 's/^labels: //p' "$f" | head -1)
  body=$(sed '1,/^---$/d' "$f")

  if ! $apply; then
    echo "[dry-run] $title  (labels: $labels)"
    continue
  fi

  IFS=',' read -ra arr <<< "$labels"
  label_args=()
  for l in "${arr[@]}"; do
    l=$(echo "$l" | xargs)
    gh label create "$l" --force >/dev/null 2>&1 || true
    label_args+=(--label "$l")
  done
  gh issue create --title "$title" --body "$body" "${label_args[@]}"
done

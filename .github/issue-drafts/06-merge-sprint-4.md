title: "Merge sprint-4 into main and push, which is also the first production deploy"
labels: process, needs-owner-approval, priority:high
---
## Current state (verified 2026-09-30)
- `sprint-4` holds all Sprint 4 work plus every post-sprint fix and measurement. **138 backend + 47 frontend tests pass.**
- `sprint-4` is **7 commits ahead of `origin/sprint-4`** (unpushed): the algorithm comparison, the Chapter 5 writeup, the NFR-09/NFR-04 corrections, the frontend tests.
- `origin/main` is still at Sprint 2 (`8b8dc21`).
- CLAUDE.md §3: **no merge to main without the owner explicitly saying so.**

## Why it matters now
The new `.github/workflows/deploy.yml` deploys on every push to `main`. Merging is therefore also the first production deploy.

## Tasks
- [ ] Owner decision: merge (fast-forward) or open a PR `sprint-4 → main` for the record
- [ ] `git push origin sprint-4` first, so CI runs on the branch
- [ ] Merge only after CI is green
- [ ] Complete the deployment setup issue first, or the Deploy job will fail on missing secrets

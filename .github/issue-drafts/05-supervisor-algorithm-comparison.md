title: "Decision needed: get supervisor approval for the PPO vs A2C/SAC/TD3/ensemble comparison, then write it into Chapter 5"
labels: thesis, needs-supervisor, decision
---
## Context
`scripts/algorithm_comparison.py` (2026-09-24) ran 720 real training runs: PPO, A2C, SAC and TD3, plus a Yang et al. (2020) selection ensemble, over 18 walk-forward splits × 10 seeds. Results are in `results/algorithm_comparison/`, summarised in CLAUDE.md §10.

Headline: **no algorithm is distinguishable from PPO** (Wilcoxon p 0.35–0.77). The ensemble did slightly *worse*. Every untuned algorithm loses to buy-and-hold on average, and no fold survives deflation (0/18 DSR > 0.5 for all five).

## Why this needs the supervisor
CLAUDE.md §2 says the research scope can't change without consulting Mr. Allan Vikiru. The comparison supports the PPO choice with evidence, but it's still scope beyond the approved proposal. **Chapter 5 currently contains none of it.**

## Tasks
- [ ] Show the supervisor the results table and the §1 framing ("evidence the loop, not the learner, is the contribution")
- [ ] If approved: add a Chapter 5 subsection (5.5.4?) including the ensemble's negative result
- [ ] If not approved: note it as future work only, or leave it out

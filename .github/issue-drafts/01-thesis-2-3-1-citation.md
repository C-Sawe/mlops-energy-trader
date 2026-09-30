title: "Thesis §2.3.1: citation points to an MLOps survey, not the ARIMA + Attention-LSTM study it describes"
labels: thesis, marker-feedback, priority:high
---
## Problem
Section 2.3.1 describes an ARIMA + Attention-LSTM ensemble study but cites **Kreuzberger et al. (2023)**, an MLOps *survey*. The marker (Ms. Salome Chemiat, comment #2) flagged this independently, so it is known to cost marks.

## What's needed
Find the actual source. This takes a literature search; the codebase can't answer it.
- Search: `"ARIMA" "attention" "LSTM" ensemble stock forecasting`, `ARIMA-LSTM hybrid energy price forecasting`
- Mine the surrounding paragraph for clues (dataset, reported RMSE/accuracy, institution) and phrase-search them in Google Scholar.
- If the source can't be recovered: rewrite the paragraph around a legitimate, verifiable paper. Don't keep the description and swap in a random citation.

## Acceptance criteria
- [ ] In-text citation matches a paper that actually describes an ARIMA + Attention-LSTM ensemble
- [ ] Reference list entry added (APA, consistent with the rest)
- [ ] Kreuzberger et al. (2023) stays only where it's genuinely about MLOps

Refs: CLAUDE.md §2 item 1, THESIS_HANDOFF.md §1

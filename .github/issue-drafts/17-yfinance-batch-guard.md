title: "Guard against the latent yfinance MultiIndex collision if downloads are ever batched"
labels: dataops, tech-debt, low-priority
---
`_normalise_yf_frame()` flattens MultiIndex columns with `get_level_values(0)`. That's correct only because `_fetch_one` requests **one symbol at a time**; yfinance 1.7.0 returns a MultiIndex even for a single ticker (CLAUDE.md §9). A future change that batches tickers into one `yf.download` call would silently collide every ticker's `Close` column.

- [ ] Assert the ticker level has exactly one unique value before flattening, and raise explicitly otherwise (§11: explicit failure over silent degradation)
- [ ] Unit test with a synthetic two-ticker MultiIndex frame that expects the error

# DataOps Module

**Sprint 1 — Status: 🔄 In Progress**

This module contains the automated data engineering pipeline responsible for:

- Programmatic OHLCV ingestion via `yfinance` for global energy equities (XOM, CVX, SHEL, BP, NEE)
- Forward-fill imputation for missing daily ticks
- Corporate action adjustments (stock splits, dividend distributions)
- Rolling Z-score standardisation for neural network compatibility
- ACID-compliant persistence to PostgreSQL time-series database

## Planned Files

```
dataops/
├── ingestor.py          # DataIngestor class — yfinance API integration
├── preprocessor.py      # Normalisation, imputation, adjustment logic
├── db_store.py          # PostgreSQLStore class — connection pooling, batch I/O
├── schema.sql           # PostgreSQL DDL — market_data table schema
└── config.py            # Ticker list, DB credentials, date range config
```

## Implementation begins: Sprint 1 (August 2026)

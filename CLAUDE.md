# SMART_RETAIL — project notes

5-phase ML pipeline + Streamlit dashboard for the Bangladesh FMCG retail dataset
(MITE 431 course project). See README.md for usage, docs/plans/ for the build plan.

## Commands

- Test: `.venv/bin/pytest` (local, needs `brew install libomp` for XGBoost) or `make test` (Docker)
- Full pipeline: `make pipeline` · quick run: `python -m smart_retail.pipeline --nrows 50000`
- Dashboard: `make dashboard` → http://localhost:8501

## Hard rules

- **Never create synthetic/sample data** — the user requires all code and tests to run
  against the real Mendeley CSVs in `dataset/` (user downloads them manually).
- Raw data lives in `dataset/` (git-ignored), not `data/raw/`.
- Always read CSVs through `smart_retail.data.loader` — it pins `shop_code` to str
  (slice-dependent int64 inference otherwise breaks joins silently).

## Dataset gotchas

- Targets file: only ~614k of 832k shops have valued `Total`; the first ~6k shop-file
  rows all map to NaN targets, so small `--nrows` slices can have zero labelled shops.
- The two big CSVs are ordered differently — head-slicing both yields zero join overlap.
- ~2.4% duplicate shop_codes in targets (dedup keeps valued rows first).
- RQ3 "will-miss-target" is a proxy label (dataset has assigned targets, not realized
  sales): `at_risk = Total < 0.8 × median(Total | cluster, shop_type)`.

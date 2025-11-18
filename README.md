# EC4430  Portfolio Management (code)

Short README with setup and how to run the analysis scripts used in the project.

## Prerequisites
- Python 3.10+ (or your system Python)
- A project-local virtual environment (`.venv`) is recommended (create with `python -m venv .venv`).

# EC4430 — Portfolio Management (code)

This repository contains scripts used for: aggregating ESG scores, computing momentum-based candidate lists, and constructing mean-variance portfolios (with shrinkage and no-shrink comparisons). Generated CSVs are stored under `output/`. A plotting helper produces a Capital Market Line (CML) visualization under `graphs/`.

**Quick summary of main scripts**
- `calc_esg.py` — Aggregate ESG CSVs under `ESG/` and write `output/esg_avg_by_ticker.csv`.
- `momentum.py` — Compute momentum (12m/6m) and z-scores for a universe; updated to optionally use `output/top_esg_tickers.csv` and write candidate files (`output/top_esg_tickers_momentum_scores.csv`, `output/candidate_tickers.csv`, `output/candidate_short_term_comparison.csv`).
- `weighting_updated.py` — Read candidate tickers (default `output/candidate_tickers.csv`), estimate expected returns and covariance, apply optional mean shrinkage, compute tangency (Max-Sharpe) weights, and write several outputs:
	- `output/portfolio_weighting.csv` (primary weights)
	- `output/portfolio_weighting_long_only.csv` (long-only projection)
	- `output/portfolio(test).csv` and `output/portfolio(test)_long.csv` (no-shrink comparison portfolios)
- `graphs.py` — Generate a visualization of assets, the efficient frontier, and the Capital Market Line. Saves `graphs/cml.png`.
- `main.py` — Orchestrator that runs the pipeline end-to-end: `calc_esg.py` -> `momentum.py` -> `weighting_updated.py` -> `volatility.py`. Uses the project `.venv` Python if present.

Other utility scripts/files in this repo:
- `shortlisting.py` — builds ESG shortlists from supplied S&P risk ratings CSVs.
- `data.py` — helper for processing candidate lists and simple ranking tasks.

## Requirements
- Python 3.10+ recommended
- Dependencies are listed in `requirements.txt`. Typical packages used: `pandas`, `numpy`, `yfinance`, `matplotlib`.

Install into the project venv (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If you prefer to install only missing packages, use:
```powershell
.\.venv\Scripts\pip.exe install pandas numpy yfinance matplotlib
```

## Typical workflow / run order
0. Run the full pipeline (runs ESG aggregation, momentum, weighting, volatility):

```powershell
.\.venv\Scripts\python.exe .\main.py
```

1. Aggregate ESG averages (if you have raw ESG CSVs):
```powershell
.\.venv\Scripts\python.exe .\calc_esg.py
```
2. Compute momentum scores (optionally restrict to `output/top_esg_tickers.csv`):
```powershell
.\.venv\Scripts\python.exe .\momentum.py
```
3. Build candidate lists (12m / 6m) — the momentum script writes these (`output/candidate_tickers.csv`, `output/candidate_short_term_comparison.csv`).
4. Compute portfolio weights from candidates:
```powershell
.\.venv\Scripts\python.exe .\weighting_updated.py
```
5. Create CML / frontier plot:
```powershell
.\.venv\Scripts\python.exe .\graphs.py
```

## Unused / Manual Scripts
- `shortlisting.py` — Builds ESG shortlists from supplied S&P risk ratings CSVs. This script is available for manual use but is not invoked by `main.py` by default.
- `graphs.py` — Plotting helper to generate the CML/efficient frontier plot. It's useful for manual inspection but `main.py` does not call it automatically (you can run it manually as shown above).

Note: "Unused" here means "not executed by `main.py` in the default orchestration". These scripts may still be useful and can be run manually; if you want them included in the automated pipeline I can add them to `main.py`.

## Key output files (examples in `output/`)
- `esg_avg_by_ticker.csv` — averaged ESG scores per ticker (from `calc_esg.py`).
- `top_esg_tickers.csv` — ESG-selected tickers (shortlist).
- `top_esg_tickers_momentum_scores.csv` — momentum results for the ESG shortlist.
- `candidate_tickers.csv` — top-20% 12m candidates (used by `weighting_updated.py`).
- `candidate_short_term_comparison.csv` — 6m top-20% for comparison.
- `portfolio_weighting.csv` — primary computed tangency weights.
- `portfolio_weighting_long_only.csv` — long-only projection of the primary portfolio.
- `portfolio(test).csv`, `portfolio(test)_long.csv` — no-shrink comparison portfolios.

## Notes / behavior highlights
- `weighting_updated.py` includes an option to shrink expected returns (mu) toward the cross-sectional mean. The script also writes a no-shrink tangency portfolio for comparison.
- Borrow/lending policy in `weighting_updated.py`: lending (credit) default is 3%; borrowing default is 6% but the script sets borrowing equal to lending unless `portfolio_value > 1_000_000`.
- `graphs.py` now computes the analytic unconstrained efficient frontier (Markowitz) and plots the Capital Market Line (CML). If you need a long-only constrained frontier, tell me and I can add a quadratic-program solver (SciPy/cvxopt) to compute that.

## Troubleshooting
- If a package import fails (e.g. `ModuleNotFoundError`), make sure you installed the packages into the same interpreter you use to run the scripts. Use the explicit venv python path shown above to avoid mismatches.
- If `momentum.py` or `weighting_updated.py` fails because expected CSV inputs are missing, run the upstream steps (ESG aggregation → momentum → candidate generation) first.

## Next steps / improvements I can do
- Add a `Makefile` / PowerShell script to run the full pipeline end-to-end.
- Add unit tests for the weighting functions (tangency, shrinkage) and a small example dataset to speed up development.
- Add a long-only optimizer to compute the exact constrained efficient frontier.

If you'd like I can commit this README update, or tweak the wording/sections to match your preferred style. 


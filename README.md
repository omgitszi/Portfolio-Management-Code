# EC4430  Portfolio Management (code)

Short README with setup and how to run the analysis scripts used in the project.

## Prerequisites
- Python 3.10+ (or your system Python)
- A project-local virtual environment (`.venv`) is recommended (create with `python -m venv .venv`).

## Setup (Windows PowerShell)

1. Create the venv (if missing):

```powershell
python -m venv .venv
```

2. Install dependencies into the venv:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

3. (Optional) Activate the venv in PowerShell:

```powershell
. .\.venv\Scripts\Activate.ps1
# If activation is blocked by policy, either run the commands using the venv python directly
# or set the execution policy for the current user (run PowerShell as admin):
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Main scripts

- `momentum.py`  compute risk-adjusted momentum scores for the S&P 500 and write results to `output/`.
- `overlap_top200.py`  builds `output/top200_esg.csv` from the S&P risk ratings file (if present in repo root) and computes the overlap with momentum shortlists; outputs placed in `output/`.
- `data.py`  reads a shortlist (e.g. `output/sp500_top20_esg_overlap.csv`) and computes top-25 by Sharpe; writes `output/top25_by_sharpe.csv`.
- `weighting.py`  reads `output/top25_by_sharpe.csv`, pulls price histories, computes tangency weights (with optional mu shrinkage), and writes `output/top25_weights.csv`.

## Run examples (PowerShell)

```powershell
# run momentum (uses yfinance; long-running)
.\.venv\Scripts\python.exe .\momentum.py

# build top200 and compute overlap
.\.venv\Scripts\python.exe .\overlap_top200.py

# compute top25 by Sharpe
.\.venv\Scripts\python.exe .\data.py

# compute weights for top25
.\.venv\Scripts\python.exe .\weighting.py
```

## Outputs
All generated CSVs are written to the `output/` directory. Examples:

- `output/sp500_momentum_results.csv`
- `output/sp500_momentum_scores.csv`
- `output/sp500_top20_12m.csv`, `output/sp500_top20_6m.csv`
- `output/sp500_top20_all.csv`
- `output/top200_esg.csv` (created by `overlap_top200.py` if the risk file is present)
- `output/sp500_top20_esg_overlap.csv`
- `output/top25_by_sharpe.csv`
- `output/top25_weights.csv`

## Notes about the S&P ESG / risk file

- If you have the S&P risk ratings CSV (for example, `SP 500 ESG Risk Ratings.csv`) place it in the repo root. `overlap_top200.py` will automatically detect common filenames and create `output/top200_esg.csv`.
- The current code treats larger ESG percentile/score as a better rating by default (higher = better). If you want the opposite behavior, edit `overlap_top200.py` or tell me and I will change it.

## Troubleshooting
- If a package import fails (e.g. `ModuleNotFoundError: No module named 'yfinance'`), ensure you installed `requirements.txt` into the venv you're running. Use the explicit venv python above to install/run to avoid interpreter mismatches.


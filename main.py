import subprocess
import sys
from pathlib import Path
import pandas as pd


def find_python_executable():
    base = Path(__file__).parent
    venv_python = base / '.venv' / 'Scripts' / 'python.exe'
    if venv_python.exists():
        return str(venv_python)
    # fallback to current interpreter
    return sys.executable


def write_top_esg(threshold=7.0):
    base = Path(__file__).parent
    in_path = base / 'output' / 'esg_avg_by_ticker.csv'
    out_path = base / 'output' / 'top_esg_tickers.csv'
    if not in_path.exists():
        raise FileNotFoundError(f"ESG averages not found: {in_path}")

    df = pd.read_csv(in_path)
    # find ticker and avg_esg_score columns
    cols = {c.lower(): c for c in df.columns}
    tick_col = None
    for k in ('ticker', 'symbol'):
        if k in cols:
            tick_col = cols[k]
            break
    if tick_col is None:
        tick_col = df.columns[0]

    score_col = None
    for k in ('avg_esg_score', 'esg_score', 'avg_esg'):
        if k in cols:
            score_col = cols[k]
            break
    if score_col is None:
        raise ValueError('Could not find ESG score column in esg_avg_by_ticker.csv')

    sel = df[[tick_col, score_col]].copy()
    # normalize tickers
    sel['ticker_norm'] = sel[tick_col].astype(str).str.replace('.', '-', regex=False).str.strip().str.upper()
    sel[score_col] = pd.to_numeric(sel[score_col], errors='coerce')
    sel = sel.dropna(subset=[score_col])
    sel = sel[sel[score_col] >= threshold]

    out = sel[['ticker_norm', score_col]].copy()
    out = out.rename(columns={'ticker_norm': 'ticker', score_col: 'avg_esg_score'})
    out.to_csv(out_path, index=False)
    print(f'Wrote {len(out)} top-ESG tickers to {out_path}')
    return out_path


def run_script(python_exec, script, args=None):
    cmd = [python_exec, str(Path(__file__).parent / script)]
    if args:
        cmd += args
    print('Running:', ' '.join(cmd))
    res = subprocess.run(cmd, capture_output=False)
    if res.returncode != 0:
        raise RuntimeError(f"Script {script} exited with code {res.returncode}")


def main():
    python_exec = find_python_executable()
    print('Using python:', python_exec)

    # 1) Create top ESG tickers (>=7)
    # First run ESG aggregation script to ensure freshest ESG averages
    try:
        run_script(python_exec, 'calc_esg.py')
    except Exception as e:
        print('Warning: calc_esg.py failed or not present:', e)

    # calc_esg.py already writes `output/top_esg_tickers.csv` (ESG >= 7).
    # Rely on that file directly instead of re-filtering here.
    top_esg_path = Path(__file__).parent / 'output' / 'top_esg_tickers.csv'
    if not top_esg_path.exists():
        print('Expected output/top_esg_tickers.csv not found after calc_esg.py run. Aborting.')
        return

    # 2) Run momentum.py (will read output/top_esg_tickers.csv and write momentum outputs)
    try:
        run_script(python_exec, 'momentum.py')
    except Exception as e:
        print('Momentum step failed:', e)
        return

    # 3) Run weighting_updated.py to create portfolio weights
    try:
        run_script(python_exec, 'weighting_updated.py')
    except Exception as e:
        print('Weighting step failed:', e)
        return

    # 4) Run volatility.py to compute vol/price outputs
    try:
        run_script(python_exec, 'volatility.py')
    except Exception as e:
        print('Volatility step failed:', e)
        return

    print('Pipeline completed successfully.')


if __name__ == '__main__':
    main()

import sys
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

# covariance.py
# Usage: python covariance.py
# Requires: pandas, numpy, yfinance, python-dateutil
# Installs: pip install pandas numpy yfinance python-dateutil



import yfinance as yf

CSV_FILENAME = "./output/portfolio_weighting.csv"
COV_OUT = "covariance_matrix.csv"
CORR_OUT = "correlation_matrix.csv"

def read_tickers(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # common column names
    for col in ["ticker", "Ticker", "tickers", "Tickers", "symbol", "Symbol"]:
        if col in df.columns:
            return df[col].dropna().astype(str).str.strip().unique().tolist()
    # fallback: use first column
    first_col = df.columns[0]
    return df[first_col].dropna().astype(str).str.strip().unique().tolist()

def download_adj_close(tickers, start, end):
    # yfinance supports multiple tickers; returns DataFrame of Adj Close when tickers >1
    data = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, group_by='ticker', auto_adjust=False)
    # If single ticker, yfinance returns a DataFrame with columns like ['Open','High',...]
    if isinstance(tickers, str) or len(tickers) == 1:
        # ensure we still return a DataFrame with column = ticker
        if "Adj Close" in data.columns:
            adj = data["Adj Close"].rename(tickers if isinstance(tickers, str) else tickers[0]).to_frame()
        else:
            # sometimes returned with ticker level
            # try to select the single ticker from top-level
            try:
                adj = data[tickers[0]]["Adj Close"].rename(tickers[0]).to_frame()
            except Exception:
                adj = pd.DataFrame()
    else:
        # multiple tickers: data['Adj Close'] usually is DataFrame
        if "Adj Close" in data.columns.levels[0] if hasattr(data.columns, 'levels') else False:
            adj = data["Adj Close"]
        elif "Adj Close" in data.columns:
            adj = data["Adj Close"]
        else:
            # If group_by='ticker' produced wide format differently, try to construct
            # Fallback: try yfinance download without group_by
            # keep auto_adjust explicit to avoid FutureWarning and preserve behaviour
            data2 = yf.download(tickers, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
            if isinstance(data2, pd.DataFrame) and "Adj Close" in data2.columns:
                adj = data2["Adj Close"]
            else:
                adj = pd.DataFrame()
    return adj

def main():
    csv_path = CSV_FILENAME
    try:
        tickers = read_tickers(csv_path)
    except Exception as e:
        print("Error reading tickers:", e)
        sys.exit(1)

    if not tickers:
        print("No tickers found in CSV.")
        sys.exit(1)

    # compute 5 years history
    end = datetime.today()
    start = end - relativedelta(years=5)

    print(f"Downloading {len(tickers)} tickers from {start.date()} to {end.date()} ...")
    adj = download_adj_close(tickers, start, end)
    if adj.empty:
        print("Failed to download adjusted close prices (no data returned). Exiting.")
        print("Check: 1) tickers in the CSV at", csv_path, "2) internet connection, 3) yfinance limits/format.")
        sys.exit(1)

    # If adj is a Series (single ticker), convert to DataFrame
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()

    # Keep only requested tickers (columns may be a MultiIndex or have tickers)
    # Normalize column names to simple tickers
    adj.columns = [str(c) for c in adj.columns]

    # Drop columns that are entirely NaN
    adj = adj.dropna(axis=1, how='all')
    if adj.shape[1] == 0:
        print("No valid price series found for tickers.")
        sys.exit(1)

    # Compute daily returns
    returns = adj.pct_change().dropna(how='all')
    # Align: drop columns with insufficient data
    valid = returns.count() > 1
    returns = returns.loc[:, valid]

    if returns.shape[1] < 1:
        print("Not enough valid return series to compute matrices.")
        sys.exit(1)

    cov = returns.cov()
    corr = returns.corr()

    cov.to_csv(COV_OUT)
    corr.to_csv(CORR_OUT)

    print(f"Covariance matrix saved to {COV_OUT}")
    print(f"Correlation matrix saved to {CORR_OUT}")
    print("Covariance matrix:")
    print(cov.round(6))
    print("\nCorrelation matrix:")
    print(corr.round(4))

if __name__ == "__main__":
    main()
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
#
# Historically some scripts used 'sp_500_top20_esg_overlap.csv' (with underscore)
# while others write 'sp500_top20_esg_overlap.csv' (no underscore). Try the
# no-underscore variant by default but fall back at runtime if needed.
DEFAULT_TICKER_FILE = Path("output/sp500_top20_all.csv")
OUTPUT_TOP_CSV = Path("output/top25_by_sharpe.csv")


def read_tickers(path: Path) -> List[str]:
    """
    Read tickers from a CSV or Excel file.
    Looks for common ticker column names, otherwise uses the first column.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ticker file not found: {path}")
    # Try CSV then Excel
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_excel(path)
    # possible column names
    for col in ("ticker", "Ticker", "symbol", "Symbol", "tickers", "Tickers"):
        if col in df.columns:
            return df[col].dropna().astype(str).str.strip().tolist()
    # fallback: first column
    first_col = df.columns[0]
    return df[first_col].dropna().astype(str).str.strip().tolist()


def fetch_monthly_adjclose(tickers: List[str], start: str = None, end: str = None) -> pd.DataFrame:
    """
    Download adjusted close prices and convert to month-end prices.
    Returns a DataFrame with monthly prices columns = tickers.
    """
    if not tickers:
        return pd.DataFrame()
    # yfinance can accept a list; download closes and resample to month end
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # raw may have single-level columns or multiindex depending on number of tickers
    if isinstance(raw.columns, pd.MultiIndex):
        price_col = "Close" if "Close" in raw.columns.levels[1] else "Adj Close"
        prices = raw["Close"] if "Close" in raw else raw.iloc[:, raw.columns.get_level_values(1) == "Close"]
    else:
        # single ticker -> Series
        prices = raw["Close"] if "Close" in raw else raw
    # Ensure DataFrame
    prices = pd.DataFrame(prices)
    # Resample to month-end: take last available price of each month
    # Note: pandas deprecated 'M' alias for month-end in favor of 'ME'
    monthly = prices.resample("ME").last()
    # If single-column DataFrame with ticker name, rename to ticker
    if monthly.shape[1] == 1 and len(tickers) == 1:
        monthly.columns = [tickers[0]]
    return monthly


def compute_return_volatility(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """
    From monthly prices, compute monthly returns, then annualized return & volatility.
    Returns DataFrame indexed by ticker with columns: annual_return, annual_vol, sharpe, mean_monthly, std_monthly
    """
    # Compute simple monthly returns
    rets = monthly_prices.pct_change().dropna(how="all")
    # If any ticker is constant or all-NaN, handle safely
    metrics = []
    for col in rets.columns:
        series = rets[col].dropna()
        if series.empty:
            mean_m = np.nan
            std_m = np.nan
        else:
            mean_m = series.mean()
            std_m = series.std(ddof=1)
        # annualize
        annual_return = mean_m * 12 if not np.isnan(mean_m) else np.nan
        annual_vol = std_m * np.sqrt(12) if not np.isnan(std_m) else np.nan
        sharpe = annual_return / annual_vol if (annual_vol and not np.isnan(annual_vol) and annual_vol > 0) else np.nan
        metrics.append({
            "ticker": col,
            "mean_monthly": mean_m,
            "std_monthly": std_m,
            "annual_return": annual_return,
            "annual_vol": annual_vol,
            "sharpe": sharpe
        })
    df = pd.DataFrame(metrics).set_index("ticker")
    return df.sort_index()


def select_top_n(metrics_df: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """
    Select top n tickers by Sharpe ratio, tie-break with higher return and lower vol.
    Returns the subset sorted by sharpe descending.
    """
    df = metrics_df.copy()
    df = df.dropna(subset=["sharpe"])
    if df.empty:
        return df
    df = df.sort_values(by=["sharpe", "annual_return", "annual_vol"], ascending=[False, False, True])
    return df.head(n)


def main(ticker_file: Path = DEFAULT_TICKER_FILE, start: str = None, end: str = None, top_n: int = 25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline:
      - read tickers
      - fetch monthly prices
      - compute metrics
      - select top N by Sharpe
    Returns (metrics_df, top_df) and writes top_df to OUTPUT_TOP_CSV.
    """
    # If the provided ticker_file doesn't exist, try common alternative names
    alt1 = ticker_file
    alt2 = ticker_file.parent / (ticker_file.name.replace('sp500', 'sp_500'))
    alt3 = ticker_file.parent / (ticker_file.name.replace('sp_500', 'sp500'))

    candidate = None
    for p in (alt1, alt2, alt3):
        if p.exists():
            candidate = p
            break

    if candidate is None:
        raise FileNotFoundError(
            f"Ticker file not found. Checked: {alt1}, {alt2}, {alt3}\n"
            "Please generate the overlap file or pass the correct path to the script."
        )

    tickers = read_tickers(candidate)
    print(f"Loaded {len(tickers)} tickers from {candidate}")
    monthly_prices = fetch_monthly_adjclose(tickers, start=start, end=end)
    if monthly_prices.empty:
        raise RuntimeError("No price data downloaded. Check tickers and date range.")
    metrics = compute_return_volatility(monthly_prices)
    top = select_top_n(metrics, n=top_n)
    top.to_csv(OUTPUT_TOP_CSV)
    print(f"Selected top {len(top)} tickers by Sharpe. Saved to {OUTPUT_TOP_CSV}")
    return metrics, top


if __name__ == "__main__":
    # Example run: last 5 years
    end_date = datetime.date.today().isoformat()
    start_date = (datetime.date.today().replace(year=datetime.date.today().year - 5)).isoformat()
    main(start=start_date, end=end_date, top_n=25)
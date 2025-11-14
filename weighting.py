from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import sys


def read_top25(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Top-25 file not found: {path}")
    df = pd.read_csv(path)
    # Expect a 'ticker' column or index
    if 'ticker' in df.columns:
        tickers = df['ticker'].astype(str).str.strip().tolist()
    else:
        # fallback: first column
        tickers = df.iloc[:, 0].astype(str).str.strip().tolist()
    return tickers


def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    # Use adjusted close (auto_adjust=True) daily returns
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # If bulk download failed (empty), try per-ticker download to be robust to intermittent failures
    if raw is None or raw.empty:
        cols = {}
        for t in tickers:
            try:
                r = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if r is None or r.empty:
                    continue
                if isinstance(r.columns, pd.MultiIndex):
                    close = r['Close'] if 'Close' in r else r.iloc[:, r.columns.get_level_values(1) == 'Close']
                else:
                    close = r['Close'] if 'Close' in r else r
                close = pd.Series(close).rename(t)
                cols[t] = close
            except Exception:
                continue
        if not cols:
            return pd.DataFrame()
        prices = pd.concat(cols.values(), axis=1)
        prices.columns = list(cols.keys())
        return prices

    if isinstance(raw.columns, pd.MultiIndex):
        # multiindex: first level is ticker, second has columns like Close
        try:
            prices = raw['Close']
        except Exception:
            # fallback: select columns where second level == 'Close'
            mask = raw.columns.get_level_values(1) == 'Close'
            prices = raw.iloc[:, mask]
    else:
        # single ticker
        prices = raw['Close'] if 'Close' in raw else raw
    prices = pd.DataFrame(prices)
    return prices


def annualize_returns_and_cov(returns_daily: pd.DataFrame, trading_days: int = 252):
    mu_daily = returns_daily.mean()
    cov_daily = returns_daily.cov()
    mu_annual = mu_daily * trading_days
    cov_annual = cov_daily * trading_days
    return mu_annual, cov_annual


def tangency_weights(mu: np.ndarray, cov: np.ndarray, risk_free: float) -> np.ndarray:
    """Compute tangency portfolio weights (weights sum to 1).

    Allows shorting (weights may be negative). Uses pseudo-inverse for robustness.
    Formula: w = Sigma^{-1} (mu - rf) / (1' Sigma^{-1} (mu - rf))
    """
    excess = mu - risk_free
    inv = np.linalg.pinv(cov)
    raw = inv.dot(excess)
    denom = float(np.ones(len(raw)).T.dot(raw))
    if abs(denom) < 1e-12:
        raise ValueError("Denominator is numerically zero; cannot compute tangency weights")
    w = raw / denom
    return w


def main(top25_path: str = 'output/top25_by_sharpe.csv', start: str = None, end: str = None, 
         credit_rate: float = 0.03, borrow_rate: float = 0.06, portfolio_value: float = 1_000_000, 
         mu_shrink_lambda: float = 0.5, min_price: float = 5.0, max_price: float = 500000.0):
    base = Path(__file__).parent
    top25_file = base / top25_path
    tickers = read_top25(top25_file)
    print(f"Loaded {len(tickers)} tickers from {top25_file}")

    # Date range defaults: last 3 years
    if end is None:
        end = datetime.date.today().isoformat()
    if start is None:
        start = (datetime.date.today() - datetime.timedelta(days=365 * 3)).isoformat()

    print(f"Downloading prices {start} to {end} for {len(tickers)} tickers...")
    prices = download_prices(tickers, start=start, end=end)
    if prices.empty:
        print("No price data downloaded. Aborting.")
        return

    # Daily returns
    rets = prices.pct_change().dropna(how='all')
    rets = rets.dropna(axis=1, how='all')

    mu_annual, cov_annual = annualize_returns_and_cov(rets)

    # Align tickers order and apply price filter
    common = [t for t in tickers if t in mu_annual.index]
    if not common:
        print("No tickers with returns found. Aborting.")
        return

    # Get last prices and filter by price limits
    last_prices = prices[common].ffill().iloc[-1].astype(float)
    price_filter = (last_prices >= min_price) & (last_prices <= max_price)
    valid_tickers = last_prices[price_filter].index.tolist()
    
    if not valid_tickers:
        print(f"No tickers with prices between ${min_price} and ${max_price}. Aborting.")
        return
    
    print(f"Filtered to {len(valid_tickers)} tickers within price range ${min_price} - ${max_price}")

    mu = mu_annual.loc[valid_tickers].values
    print("Annualized returns:")
    print(mu_annual.loc[valid_tickers])
    
    # Apply James-Stein style shrinkage towards the cross-sectional mean to reduce estimation error
    if mu_shrink_lambda is not None and 0.0 <= mu_shrink_lambda <= 1.0:
        mu_mean = float(np.nanmean(mu))
        mu = (1 - mu_shrink_lambda) * mu + mu_shrink_lambda * mu_mean
        print(f"Applied shrinkage (lambda={mu_shrink_lambda}), shrunk towards mean return: {mu_mean:.2%}")
    
    cov = cov_annual.loc[valid_tickers, valid_tickers].values

    # Compute tangency weights using borrowing rate as risk-free for leverage
    w = tangency_weights(mu, cov, borrow_rate)

    # Build results
    df = pd.DataFrame({'ticker': valid_tickers, 'weight': w})
    df['price'] = df['ticker'].map(last_prices.to_dict())

    # Calculate number of shares per ticker implied by weights and portfolio value
    # Shares can be negative for short positions
    df['dollar_allocation'] = df['weight'] * portfolio_value
    df['shares'] = (df['dollar_allocation'] / df['price']).round(6)
    
    # compute portfolio stats
    port_return = float(df['weight'].values.dot(mu))
    port_vol = float(np.sqrt(df['weight'].values.T.dot(cov).dot(df['weight'].values)))
    port_sharpe = (port_return - borrow_rate) / port_vol if port_vol > 0 else np.nan

    out_dir = base / 'output'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'top25_weights.csv'
    df.to_csv(out_path, index=False)

    print(f"\nSaved weights to {out_path}")
    print(f"Tangency portfolio expected annual return: {port_return:.2%}")
    print(f"Tangency portfolio annual vol: {port_vol:.2%}")
    print(f"Tangency portfolio Sharpe (borrow rate {borrow_rate:.2%}): {port_sharpe:.3f}")
    
    # Calculate net interest impact
    long_exposure = df[df['weight'] > 0]['dollar_allocation'].sum()
    short_exposure = abs(df[df['weight'] < 0]['dollar_allocation'].sum())
    net_interest = (long_exposure * credit_rate) - (short_exposure * borrow_rate)
    
    print(f"\nExposure Summary:")
    print(f"Long exposure: ${long_exposure:,.2f}")
    print(f"Short exposure: ${short_exposure:,.2f}")
    print(f"Net portfolio value: ${portfolio_value:,.2f}")
    print(f"Credit interest earned ({(credit_rate*100):.1f}%): ${(long_exposure * credit_rate):,.2f}")
    print(f"Borrow interest paid ({(borrow_rate*100):.1f}%): ${(short_exposure * borrow_rate):,.2f}")
    print(f"Net interest: ${net_interest:,.2f}")


if __name__ == '__main__':
    # Allow passing rates and dates via command-line args
    args = sys.argv[1:]
    credit = 0.03  # 3% credit interest rate
    borrow = 0.06  # 6% borrow interest rate
    min_price = 5.0
    max_price = 500000.0
    start = None
    end = None
    
    if len(args) >= 1:
        try:
            borrow = float(args[0])
        except Exception:
            pass
    if len(args) >= 2:
        try:
            credit = float(args[1])
        except Exception:
            pass
    if len(args) >= 5:
        start = args[2]
        end = args[3]
        try:
            min_price = float(args[4])
        except Exception:
            pass
    if len(args) >= 6:
        try:
            max_price = float(args[5])
        except Exception:
            pass
            
    main(start=start, end=end, credit_rate=credit, borrow_rate=borrow, 
         min_price=min_price, max_price=max_price)
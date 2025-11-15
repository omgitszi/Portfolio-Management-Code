from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import sys


def read_candidates(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")
    df = pd.read_csv(path)
    # Prefer normalized ticker column
    if 'ticker_norm' in df.columns:
        tickers = df['ticker_norm'].astype(str).str.strip().tolist()
    elif 'ticker' in df.columns:
        tickers = df['ticker'].astype(str).str.strip().tolist()
    else:
        tickers = df.iloc[:, 0].astype(str).str.strip().tolist()
    return tickers


def download_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        cols = {}
        for t in tickers:
            try:
                r = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
                if r is None or r.empty:
                    continue
                # prefer Close
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
        try:
            prices = raw['Close']
        except Exception:
            mask = raw.columns.get_level_values(1) == 'Close'
            prices = raw.iloc[:, mask]
    else:
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
    excess = mu - risk_free
    inv = np.linalg.pinv(cov)
    raw = inv.dot(excess)
    denom = float(np.ones(len(raw)).T.dot(raw))
    if abs(denom) < 1e-12:
        raise ValueError("Denominator is numerically zero; cannot compute tangency weights")
    w = raw / denom
    return w


def main(candidates_path: str = 'output/sp500_top20_12m_with_esg.csv', top_k: int = 25,
         select_window_days: int = 252, # window used to compute risk-adjusted metric
         cov_window_years: int = 3,     # window for covariance estimation (years)
         credit_rate: float = 0.03, borrow_rate: float = 0.06,
         portfolio_value: float = 1_000_000, mu_shrink_lambda: float = 0.5,
         min_price: float = 5.0, max_price: float = 500000.0):

    base = Path(__file__).parent
    cand_file = base / candidates_path
    if not cand_file.exists():
        raise FileNotFoundError(f"Candidates file not found: {cand_file}")

    print(f"Loading candidates from {cand_file}")
    df = pd.read_csv(cand_file)
    # pick ticker column
    if 'ticker_norm' in df.columns:
        df['ticker_sel'] = df['ticker_norm'].astype(str).str.strip()
    elif 'ticker_x' in df.columns:
        df['ticker_sel'] = df['ticker_x'].astype(str).str.strip()
    elif 'ticker' in df.columns:
        df['ticker_sel'] = df['ticker'].astype(str).str.strip()
    else:
        df['ticker_sel'] = df.iloc[:, 0].astype(str).str.strip()

    tickers = df['ticker_sel'].dropna().unique().tolist()
    print(f"Found {len(tickers)} candidate tickers")

    # Determine date range for quick selection metric
    end = datetime.date.today()
    start = end - datetime.timedelta(days=select_window_days)
    start_s = start.isoformat(); end_s = end.isoformat()
    print(f"Downloading prices for selection window {start_s} to {end_s}")
    prices = download_prices(tickers, start=start_s, end=end_s)
    if prices.empty:
        print("No price data available for selection. Aborting.")
        return

    rets = prices.pct_change().dropna(how='all')
    rets = rets.dropna(axis=1, how='all')

    # Compute annualized mean and vol
    mu_ann = rets.mean() * 252
    vol_ann = rets.std() * np.sqrt(252)

    # Risk-adjusted metric: annualized mean / annualized vol
    # If vol == 0 or NaN, set metric to -inf so it's deprioritized
    ratio = pd.Series(index=mu_ann.index, dtype=float)
    for t in mu_ann.index:
        m = mu_ann.get(t, np.nan)
        v = vol_ann.get(t, np.nan)
        if pd.isna(m) or pd.isna(v) or v <= 0:
            ratio.loc[t] = -np.inf
        else:
            ratio.loc[t] = m / v

    sel_df = pd.DataFrame({'ticker': ratio.index, 'risk_adj': ratio.values})
    sel_df = sel_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['risk_adj'])
    sel_df = sel_df.sort_values('risk_adj', ascending=False)

    if sel_df.empty:
        print("No tickers with valid risk-adjusted metric. Aborting.")
        return

    selected = sel_df.head(top_k)['ticker'].tolist()
    print(f"Selected top {len(selected)} tickers by risk-adjusted metric")

    # Now download longer history for covariance estimation
    cov_days = cov_window_years * 252
    cov_start = (datetime.date.today() - datetime.timedelta(days=cov_days)).isoformat()
    cov_end = datetime.date.today().isoformat()
    print(f"Downloading historical prices for covariance estimation: {cov_start} to {cov_end}")
    prices_cov = download_prices(selected, start=cov_start, end=cov_end)
    if prices_cov.empty:
        print("No price data for covariance estimation. Aborting.")
        return

    rets_cov = prices_cov.pct_change().dropna(how='all').dropna(axis=1, how='all')
    mu_annual, cov_annual = annualize_returns_and_cov(rets_cov)

    # Align tickers and filter by price
    valid = [t for t in selected if t in mu_annual.index]
    if not valid:
        print("No valid tickers with returns for optimizer. Aborting.")
        return

    last_prices = prices_cov[valid].ffill().iloc[-1].astype(float)
    price_filter_mask = (last_prices >= min_price) & (last_prices <= max_price)
    valid_tickers = last_prices[price_filter_mask].index.tolist()
    if not valid_tickers:
        print(f"No tickers within price bounds ${min_price} - ${max_price}. Aborting.")
        return

    mu = mu_annual.loc[valid_tickers].values
    # Shrink mu towards cross-sectional mean
    if mu_shrink_lambda is not None and 0.0 <= mu_shrink_lambda <= 1.0:
        mu_mean = float(np.nanmean(mu))
        mu = (1 - mu_shrink_lambda) * mu + mu_shrink_lambda * mu_mean
        print(f"Applied shrinkage (lambda={mu_shrink_lambda}) towards mean {mu_mean:.2%}")

    cov = cov_annual.loc[valid_tickers, valid_tickers].values

    # Compute tangency weights using borrow_rate as risk-free
    w = tangency_weights(mu, cov, borrow_rate)

    out_df = pd.DataFrame({'ticker': valid_tickers, 'weight': w})
    out_df['price'] = out_df['ticker'].map(last_prices.to_dict())
    out_df['dollar_allocation'] = out_df['weight'] * portfolio_value
    out_df['shares'] = (out_df['dollar_allocation'] / out_df['price']).round(6)

    # Include momentum and ESG scores from the original candidates file when available
    try:
        if 'momentum_score_12m' in df.columns:
            momentum_map = df.set_index('ticker_sel')['momentum_score_12m'].to_dict()
        elif 'momentum_value_12m' in df.columns:
            momentum_map = df.set_index('ticker_sel')['momentum_value_12m'].to_dict()
        else:
            momentum_map = {}
    except Exception:
        momentum_map = {}

    try:
        if 'avg_esg_score' in df.columns:
            esg_map = df.set_index('ticker_sel')['avg_esg_score'].to_dict()
        else:
            esg_map = {}
    except Exception:
        esg_map = {}

    out_df['momentum_score_12m'] = out_df['ticker'].map(momentum_map)
    out_df['avg_esg_score'] = out_df['ticker'].map(esg_map)

    port_return = float(out_df['weight'].values.dot(mu))
    port_vol = float(np.sqrt(out_df['weight'].values.T.dot(cov).dot(out_df['weight'].values)))
    port_sharpe = (port_return - borrow_rate) / port_vol if port_vol > 0 else np.nan

    out_dir = base / 'output'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'weightings_updated.csv'
    out_df.to_csv(out_path, index=False)

    print(f"Saved weights to {out_path}")
    print(f"Portfolio expected annual return: {port_return:.2%}")
    print(f"Portfolio annual vol: {port_vol:.2%}")
    print(f"Portfolio Sharpe (borrow rate {borrow_rate:.2%}): {port_sharpe:.3f}")

    long_exposure = out_df[out_df['weight'] > 0]['dollar_allocation'].sum()
    short_exposure = abs(out_df[out_df['weight'] < 0]['dollar_allocation'].sum())
    net_interest = (long_exposure * credit_rate) - (short_exposure * borrow_rate)

    print(f"Long exposure: ${long_exposure:,.2f}")
    print(f"Short exposure: ${short_exposure:,.2f}")
    print(f"Net interest: ${net_interest:,.2f}")


if __name__ == '__main__':
    args = sys.argv[1:]
    credit = 0.03
    borrow = 0.06
    topk = 25
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
    if len(args) >= 3:
        try:
            topk = int(args[2])
        except Exception:
            pass

    main(top_k=topk, credit_rate=credit, borrow_rate=borrow)

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


def main(candidates_path: str = 'output/candidate_tickers.csv', top_k: int = 25,
         select_window_days: int = 365 * 5, # window used to compute risk-adjusted metric (5 calendar years)
         cov_window_years: int = 5,     # window for covariance estimation (years)
         credit_rate: float = 0.03, borrow_rate: float = 0.06,
         portfolio_value: float = 1_000_000, mu_shrink_lambda: float = 0.5,
         min_price: float = 5.0, max_price: float = 500000.0):

    base = Path(__file__).parent
    cand_file = base / candidates_path
    if not cand_file.exists():
        raise FileNotFoundError(f"Candidates file not found: {cand_file}")

    # Enforce lending/borrowing policy: lending (credit) rate is 3%.
    # Borrowing rate is 6% only when portfolio size exceeds $1,000,000; otherwise use lending rate.
    try:
        if portfolio_value > 1_000_000:
            borrow_rate = 0.06
        else:
            borrow_rate = credit_rate
    except Exception:
        # fallback to provided borrow_rate if something unexpected
        pass

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

    # Use a fixed end date for reproducibility (15 Nov 2025)
    fixed_end = datetime.date(2025, 11, 15)
    # Determine date range for quick selection metric
    end = fixed_end
    start = end - datetime.timedelta(days=select_window_days)
    start_s = start.isoformat(); end_s = end.isoformat()
    print(f"Downloading prices for selection window {start_s} to {end_s} (fixed end date)")
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
    # Use calendar days for cov window to represent full years (365 days/year)
    cov_days = cov_window_years * 365
    cov_start = (fixed_end - datetime.timedelta(days=cov_days)).isoformat()
    cov_end = fixed_end.isoformat()
    print(f"Downloading historical prices for covariance estimation: {cov_start} to {cov_end} (fixed end date)")
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

    # Build momentum and ESG maps early so downstream blocks can use them (fallback to output files)
    try:
        if 'momentum_score_12m' in df.columns:
            momentum_map = df.set_index('ticker_sel')['momentum_score_12m'].to_dict()
        elif 'momentum_value_12m' in df.columns:
            momentum_map = df.set_index('ticker_sel')['momentum_value_12m'].to_dict()
        else:
            mom_file = base / 'output' / 'top_esg_tickers_momentum_scores.csv'
            if mom_file.exists():
                mom_df = pd.read_csv(mom_file)
                # try common column names for ticker and momentum
                if 'ticker' in mom_df.columns:
                    tick_col = 'ticker'
                elif 'ticker_norm' in mom_df.columns:
                    tick_col = 'ticker_norm'
                elif 'ticker_sel' in mom_df.columns:
                    tick_col = 'ticker_sel'
                else:
                    tick_col = mom_df.columns[0]

                if 'momentum_score_12m' in mom_df.columns:
                    momentum_map = mom_df.set_index(tick_col)['momentum_score_12m'].to_dict()
                elif 'momentum_value_12m' in mom_df.columns:
                    momentum_map = mom_df.set_index(tick_col)['momentum_value_12m'].to_dict()
                else:
                    # try any numeric column as fallback
                    numeric_cols = mom_df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        momentum_map = mom_df.set_index(tick_col)[numeric_cols[0]].to_dict()
                    else:
                        momentum_map = {}
            else:
                momentum_map = {}
    except Exception:
        momentum_map = {}

    try:
        if 'avg_esg_score' in df.columns:
            esg_map = df.set_index('ticker_sel')['avg_esg_score'].to_dict()
        else:
            esg_file = base / 'output' / 'esg_avg_by_ticker.csv'
            if esg_file.exists():
                esg_df = pd.read_csv(esg_file)
                if 'ticker' in esg_df.columns and 'avg_esg_score' in esg_df.columns:
                    esg_df['ticker_norm'] = (
                        esg_df['ticker'].astype(str)
                        .str.replace('.', '-', regex=False)
                        .str.strip()
                        .str.upper()
                    )
                    esg_map = esg_df.set_index('ticker_norm')['avg_esg_score'].to_dict()
                else:
                    esg_map = {}
            else:
                esg_map = {}
    except Exception:
        esg_map = {}

    # Preserve original sample mu (no-shrink) for comparison and for test portfolios
    mu_orig = mu_annual.loc[valid_tickers].values
    mu = mu_orig.copy()
    # Shrink mu towards cross-sectional mean (primary behaviour)
    if mu_shrink_lambda is not None and 0.0 <= mu_shrink_lambda <= 1.0:
        mu_mean = float(np.nanmean(mu))
        mu = (1 - mu_shrink_lambda) * mu + mu_shrink_lambda * mu_mean
        print(f"Applied shrinkage (lambda={mu_shrink_lambda}) towards mean {mu_mean:.2%}")

    cov = cov_annual.loc[valid_tickers, valid_tickers].values

    # --- Compute tangency weights using ORIGINAL (no-shrink) mu for comparison ---
    try:
        w_test = tangency_weights(mu_orig, cov, borrow_rate)

        out_df_test = pd.DataFrame({'ticker': valid_tickers, 'weight': w_test})
        out_df_test['price'] = out_df_test['ticker'].map(last_prices.to_dict())
        out_df_test['dollar_allocation'] = out_df_test['weight'] * portfolio_value
        out_df_test['shares'] = (out_df_test['dollar_allocation'] / out_df_test['price']).round(6)
        out_df_test['momentum_score_12m'] = out_df_test['ticker'].map(momentum_map)
        out_df_test['avg_esg_score'] = out_df_test['ticker'].map(esg_map)

        out_dir = base / 'output'
        out_dir.mkdir(exist_ok=True)
        test_path = out_dir / 'portfolio(test).csv'
        out_df_test.to_csv(test_path, index=False)

        # Long-only projection for the no-shrink portfolio
        w_test_long = np.clip(w_test, 0.0, None)
        if np.allclose(w_test_long.sum(), 0.0):
            w_test_long = np.ones_like(w_test_long) / float(len(w_test_long))
        else:
            w_test_long = w_test_long / float(np.sum(w_test_long))

        out_df_test_long = out_df_test.copy()
        out_df_test_long['weight'] = w_test_long
        out_df_test_long['dollar_allocation'] = out_df_test_long['weight'] * portfolio_value
        out_df_test_long['shares'] = (out_df_test_long['dollar_allocation'] / out_df_test_long['price']).round(6)

        test_long_path = out_dir / 'portfolio(test)_long.csv'
        out_df_test_long.to_csv(test_long_path, index=False)

        # Print basic metrics for comparison
        try:
            port_return_test = float(out_df_test['weight'].values.dot(mu_orig))
            port_vol_test = float(np.sqrt(out_df_test['weight'].values.T.dot(cov).dot(out_df_test['weight'].values)))
            port_sharpe_test = (port_return_test - borrow_rate) / port_vol_test if port_vol_test > 0 else np.nan

            port_return_test_long = float(out_df_test_long['weight'].values.dot(mu_orig))
            port_vol_test_long = float(np.sqrt(out_df_test_long['weight'].values.T.dot(cov).dot(out_df_test_long['weight'].values)))
            port_sharpe_test_long = (port_return_test_long - borrow_rate) / port_vol_test_long if port_vol_test_long > 0 else np.nan

            print(f"Saved no-shrink (test) weights to {test_path}")
            print(f"No-shrink portfolio expected annual return: {port_return_test:.2%}")
            print(f"No-shrink portfolio annual vol: {port_vol_test:.2%}")
            print(f"No-shrink portfolio Sharpe (borrow rate {borrow_rate:.2%}): {port_sharpe_test:.3f}")

            print(f"Saved no-shrink long-only weights to {test_long_path}")
            print(f"No-shrink long-only expected annual return: {port_return_test_long:.2%}")
            print(f"No-shrink long-only annual vol: {port_vol_test_long:.2%}")
            print(f"No-shrink long-only Sharpe (borrow rate {borrow_rate:.2%}): {port_sharpe_test_long:.3f}")
        except Exception:
            print("Saved no-shrink portfolios (metrics calculation failed)")
    except Exception as e:
        print(f"Failed to compute no-shrink test portfolio: {e}")

    # Compute tangency weights using borrow_rate as risk-free (primary - potentially shrunk mu)
    w = tangency_weights(mu, cov, borrow_rate)

    out_df = pd.DataFrame({'ticker': valid_tickers, 'weight': w})
    out_df['price'] = out_df['ticker'].map(last_prices.to_dict())
    out_df['dollar_allocation'] = out_df['weight'] * portfolio_value
    out_df['shares'] = (out_df['dollar_allocation'] / out_df['price']).round(6)

    # (momentum_map and esg_map already constructed earlier)

    out_df['momentum_score_12m'] = out_df['ticker'].map(momentum_map)
    out_df['avg_esg_score'] = out_df['ticker'].map(esg_map)

    port_return = float(out_df['weight'].values.dot(mu))
    # raw (no-shrink) portfolio expected return for comparison
    try:
        port_return_raw = float(out_df['weight'].values.dot(mu_orig))
    except Exception:
        port_return_raw = None
    port_vol = float(np.sqrt(out_df['weight'].values.T.dot(cov).dot(out_df['weight'].values)))
    port_sharpe = (port_return - borrow_rate) / port_vol if port_vol > 0 else np.nan

    out_dir = base / 'output'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'portfolio_weighting.csv'
    out_df.to_csv(out_path, index=False)

    print(f"Saved weights to {out_path}")
    if port_return_raw is not None:
        print(f"Portfolio expected annual return (raw/no-shrink): {port_return_raw:.2%}")
    print(f"Portfolio expected annual return (used / shrunk): {port_return:.2%}")
    print(f"Portfolio annual vol: {port_vol:.2%}")
    print(f"Portfolio Sharpe (borrow rate {borrow_rate:.2%}): {port_sharpe:.3f}")

    long_exposure = out_df[out_df['weight'] > 0]['dollar_allocation'].sum()
    short_exposure = abs(out_df[out_df['weight'] < 0]['dollar_allocation'].sum())
    net_interest = (long_exposure * credit_rate) - (short_exposure * borrow_rate)

    print(f"Long exposure: ${long_exposure:,.2f}")
    print(f"Short exposure: ${short_exposure:,.2f}")
    print(f"Net interest: ${net_interest:,.2f}")

    # --- Create a long-only constrained portfolio by projecting negative weights to zero and renormalizing ---
    try:
        w_long = np.clip(w, 0.0, None)
        if np.allclose(w_long.sum(), 0.0):
            # fallback to equal weight if no positive weights
            w_long = np.ones_like(w_long) / float(len(w_long))
        else:
            w_long = w_long / float(np.sum(w_long))

        out_df_long = out_df.copy()
        out_df_long['weight'] = w_long
        out_df_long['dollar_allocation'] = out_df_long['weight'] * portfolio_value
        out_df_long['shares'] = (out_df_long['dollar_allocation'] / out_df_long['price']).round(6)

        out_path_long = out_dir / 'portfolio_weighting_long_only.csv'
        out_df_long.to_csv(out_path_long, index=False)

        # Compute long-only portfolio metrics
        port_return_long = float(out_df_long['weight'].values.dot(mu))
        # raw (no-shrink) long-only expected return for comparison
        try:
            port_return_long_raw = float(out_df_long['weight'].values.dot(mu_orig))
        except Exception:
            port_return_long_raw = None
        port_vol_long = float(np.sqrt(out_df_long['weight'].values.T.dot(cov).dot(out_df_long['weight'].values)))
        port_sharpe_long = (port_return_long - borrow_rate) / port_vol_long if port_vol_long > 0 else np.nan

        print(f"Saved long-only weights to {out_path_long}")
        if port_return_long_raw is not None:
            print(f"Long-only expected annual return (raw/no-shrink): {port_return_long_raw:.2%}")
        print(f"Long-only Portfolio expected annual return (used / shrunk): {port_return_long:.2%}")
        print(f"Long-only Portfolio annual vol: {port_vol_long:.2%}")
        print(f"Long-only Portfolio Sharpe (borrow rate {borrow_rate:.2%}): {port_sharpe_long:.3f}")
    except Exception as e:
        print(f"Failed to create long-only portfolio: {e}")


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

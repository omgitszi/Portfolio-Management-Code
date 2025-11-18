import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

def compute_portfolio_volatility(csv_path, end_date='2025-11-15'):
    end = pd.to_datetime(end_date)
    start = end - pd.DateOffset(years=5)

    df = pd.read_csv(csv_path)
    # detect ticker and weight columns (case-insensitive common names)
    cols = {c.lower(): c for c in df.columns}
    for k in ('ticker', 'symbol', 'tickers'):
        if k in cols:
            tick_col = cols[k]; break
    else:
        raise ValueError("No ticker column found in CSV (expected 'ticker'/'symbol').")
    for k in ('weight', 'weights'):
        if k in cols:
            weight_col = cols[k]; break
    else:
        raise ValueError("No weight column found in CSV (expected 'weight'/'weights').")

    tickers = df[tick_col].astype(str).str.strip().unique().tolist()
    weights = df.set_index(tick_col).loc[tickers, weight_col].astype(float).values
    if weights.sum() == 0:
        raise ValueError("Sum of weights is zero.")
    weights = weights / weights.sum()

    # Request adjusted prices for reproducibility
    data = yf.download(tickers, start=start.strftime('%Y-%m-%d'),
                       end=end.strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
    # get adjusted close prices
    # data layout can be MultiIndex (tickers x fields) or flat
    adj = None
    try:
        if isinstance(data.columns, pd.MultiIndex):
            # prefer 'Adj Close' if present, otherwise 'Close'
            lev1 = list(data.columns.get_level_values(1))
            if 'Adj Close' in lev1:
                adj = data.xs('Adj Close', axis=1, level=1)
            elif 'Close' in lev1:
                adj = data.xs('Close', axis=1, level=1)
            else:
                adj = data
        else:
            if 'Adj Close' in data.columns:
                adj = data['Adj Close']
            elif 'Close' in data.columns:
                adj = data['Close']
            else:
                adj = data
    except Exception:
        adj = data

    if isinstance(adj, pd.Series):
        adj = adj.to_frame(name=tickers[0])

    # ensure all tickers present; if some missing, try downloading them individually
    # Safely extract column names even if columns are a MultiIndex
    cols = adj.columns
    if isinstance(cols, pd.MultiIndex):
        available_cols = set([str(c[0]).strip() for c in cols])
    else:
        available_cols = set([str(c).strip() for c in cols])
    missing = [t for t in tickers if t not in available_cols]
    if missing:
        recovered = {}
        still_missing = []
        for t in missing:
            try:
                d = yf.download(t, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'),
                                progress=False, auto_adjust=True)
                # same extraction logic for single-ticker frame
                if isinstance(d.columns, pd.MultiIndex):
                    lev1 = list(d.columns.get_level_values(1))
                    if 'Adj Close' in lev1:
                        ser = d.xs('Adj Close', axis=1, level=1)
                    elif 'Close' in lev1:
                        ser = d.xs('Close', axis=1, level=1)
                    else:
                        ser = d
                else:
                    if 'Adj Close' in d.columns:
                        ser = d['Adj Close']
                    elif 'Close' in d.columns:
                        ser = d['Close']
                    else:
                        ser = d
                if isinstance(ser, pd.Series):
                    ser = ser.rename(t)
                else:
                    # if DataFrame, take the last column as series
                    ser = ser.iloc[:, 0].rename(t)
                if ser.dropna().empty:
                    still_missing.append(t)
                else:
                    recovered[t] = ser
            except Exception:
                still_missing.append(t)

        # append recovered columns
        if recovered:
            rec_df = pd.concat(recovered.values(), axis=1)
            adj = pd.concat([adj, rec_df], axis=1)
            cols = adj.columns
            if isinstance(cols, pd.MultiIndex):
                available_cols = set([str(c[0]).strip() for c in cols])
            else:
                available_cols = set([str(c).strip() for c in cols])
        # recompute missing after attempts
        missing = [t for t in tickers if t not in available_cols]
        if missing:
            # drop missing tickers (warn) and continue if at least one ticker remains
            raise ValueError(f"No price data for tickers: {', '.join(missing)}")

    returns = adj[tickers].pct_change().dropna(how='all').dropna(axis=0, how='any')
    if returns.empty:
        raise ValueError("No returns data after dropping NA. Check date range and ticker data.")

    trading_days = 252
    per_ticker_annual_vol = returns.std(ddof=1) * np.sqrt(trading_days)
    portfolio_daily = returns.dot(weights)
    portfolio_annual_vol = portfolio_daily.std(ddof=1) * np.sqrt(trading_days)

    # Save historical prices (adjusted) for inspected tickers
    out_dir = Path('./output')
    out_dir.mkdir(exist_ok=True, parents=True)
    # Use the adjusted price series aligned to the returns index
    prices_out = adj.reindex(returns.index)[returns.columns]
    try:
        prices_out.to_csv(out_dir / 'historical_prices.csv', index=True)
        print(f'Wrote historical prices to {out_dir / "historical_prices.csv"}')
    except Exception:
        print('Failed to write historical prices CSV')

    # Compute per-ticker summary statistics (daily and annualized)
    daily_mean = returns.mean()
    daily_var = returns.var(ddof=1)
    daily_std = returns.std(ddof=1)

    annual_mean = daily_mean * trading_days
    annual_var = daily_var * trading_days
    annual_std = daily_std * np.sqrt(trading_days)

    stats_df = pd.DataFrame({
        'ticker': returns.columns,
        'observations': returns.shape[0],
        'last_price': adj[returns.columns].ffill().iloc[-1].values,
        'mean_daily_return': daily_mean.values,
        'var_daily_return': daily_var.values,
        'std_daily_return': daily_std.values,
        'mean_annual_return': annual_mean.values,
        'var_annual_return': annual_var.values,
        'std_annual_return': annual_std.values
    })

    try:
        stats_df.to_csv(out_dir / 'ticker_statistics.csv', index=False)
        print(f'Wrote per-ticker statistics to {out_dir / "ticker_statistics.csv"}')
    except Exception:
        print('Failed to write ticker statistics CSV')

    result = {
        'per_ticker_annual_vol': per_ticker_annual_vol,
        'portfolio_annual_vol': float(portfolio_annual_vol),
        'start_date': start.strftime('%Y-%m-%d'),
        'end_date': end.strftime('%Y-%m-%d')
    }
    return result

if __name__ == '__main__':
    # example usage (adjust path if needed)
    res = compute_portfolio_volatility('./output/portfolio_weighting.csv', end_date='2025-11-15')
    print("Period:", res['start_date'], "to", res['end_date'])
    print("Per-ticker annual volatility:")
    print(res['per_ticker_annual_vol'])
    print("Portfolio annual volatility:", res['portfolio_annual_vol'])
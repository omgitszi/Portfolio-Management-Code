import pandas as pd
import numpy as np
from math import sqrt

# Try to read portfolio daily returns if available
import os
cwd = os.path.abspath(os.path.dirname(__file__))
portfolio_daily_path = os.path.join(cwd, 'portfolio_daily_returns.csv')
combined_path = os.path.join(cwd, 'portfolio_vs_sp500_cumulative_growth.csv')

results = {}

if os.path.exists(portfolio_daily_path):
    pdr = pd.read_csv(portfolio_daily_path, index_col=0, parse_dates=True)
    # if DataFrame, take the first column as the series
    if isinstance(pdr, pd.DataFrame):
        if pdr.shape[1] >= 1:
            pdr = pdr.iloc[:, 0]
        else:
            pdr = pdr.squeeze()
    daily_portfolio = pdr.dropna()
    cum = (1 + daily_portfolio).cumprod()
    n_days = daily_portfolio.shape[0]
    cum_return = cum.iloc[-1]
    ann_return = cum_return ** (252 / n_days) - 1
    ann_vol = daily_portfolio.std() * sqrt(252)
    max_dd = (cum / cum.cummax() - 1).min()
    results['Portfolio'] = {
        'Cumulative return': cum_return - 1,
        'Annualized return': ann_return,
        'Annualized volatility': ann_vol,
        'Max drawdown': max_dd
    }
else:
    # will try to build from combined later
    pass

if os.path.exists(combined_path):
    df = pd.read_csv(combined_path, index_col=0, parse_dates=True)
    # combined file contains rebased cumulative growth (1.0 start)
    # compute daily returns from cumulative
    df = df.dropna(how='all')
    # ensure columns named 'Portfolio' and 'S&P500' or similar
    possible_port = [c for c in df.columns if 'portfolio' in c.lower()]
    possible_sp = [c for c in df.columns if 's&p' in c.lower() or 'sp' in c.lower() or 'gspc' in c.lower()]
    if len(possible_port) == 0 and 'Portfolio' in df.columns:
        possible_port = ['Portfolio']
    if len(possible_sp) == 0 and 'S&P500' in df.columns:
        possible_sp = ['S&P500']

    def compute_from_cum(series):
        series = series.dropna()
        daily = series.pct_change().dropna()
        n = daily.shape[0]
        cum_return = series.iloc[-1]
        ann_return = cum_return ** (252 / n) - 1
        ann_vol = daily.std() * sqrt(252)
        max_dd = (series / series.cummax() - 1).min()
        return {
            'Cumulative return': cum_return - 1,
            'Annualized return': ann_return,
            'Annualized volatility': ann_vol,
            'Max drawdown': max_dd
        }

    if possible_port:
        results['Portfolio_from_combined'] = compute_from_cum(df[possible_port[0]])
    if possible_sp:
        results['S&P500'] = compute_from_cum(df[possible_sp[0]])

# Prefer Portfolio computed from daily returns file if available
if 'Portfolio' in results and 'Portfolio_from_combined' in results:
    # keep both but mark preference
    results['Portfolio_note'] = 'Portfolio metrics computed from portfolio_daily_returns.csv preferred over combined.'
elif 'Portfolio_from_combined' in results and 'Portfolio' not in results:
    results['Portfolio'] = results.pop('Portfolio_from_combined')

# Print results
for name, metrics in results.items():
    if name.endswith('_note'):
        print(metrics)
        continue
    print(f"\n{name}:")
    for k, v in metrics.items():
        if 'volatility' in k.lower() or 'return' in k.lower() or 'drawdown' in k.lower():
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v}")

if not results:
    print('No data files found. Expected: portfolio_daily_returns.csv and/or portfolio_vs_sp500_cumulative_growth.csv')

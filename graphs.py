from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import sys


def download_prices(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
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


def approx_efficient_frontier(returns_annual, cov_annual, n_portfolios=5000):
    """
    Keep random-sample scatter for visualization, but compute the exact unconstrained
    efficient frontier analytically using the Markowitz matrix formulas. Returns:
      - dfp: DataFrame of random portfolio (ret, vol)
      - frontier_vol, frontier_ret: arrays for the exact efficient frontier
      - tangency_w, tangency_ret, tangency_vol: analytic tangency portfolio
    """
    # random portfolios (long-only sampling) kept for background
    n = len(returns_annual)
    rets = []
    vols = []
    for _ in range(n_portfolios):
        w = np.random.rand(n)
        w = w / np.sum(w)
        ret = float(w.dot(returns_annual.values))
        vol = float(np.sqrt(w.T.dot(cov_annual.values).dot(w)))
        rets.append(ret)
        vols.append(vol)
    dfp = pd.DataFrame({'ret': rets, 'vol': vols})

    # Exact unconstrained frontier (allows shorting) via analytic formulas
    inv_cov = np.linalg.pinv(cov_annual.values)
    ones = np.ones(len(returns_annual))
    mu = returns_annual.values
    A = float(ones.T.dot(inv_cov).dot(ones))
    B = float(ones.T.dot(inv_cov).dot(mu))
    C = float(mu.T.dot(inv_cov).dot(mu))
    D = A * C - B * B

    # choose a range of target returns around the assets' returns
    r_min = float(np.min(mu))
    r_max = float(np.max(mu))
    r_range = np.linspace(r_min, r_max, 200)
    frontier_var = (A * r_range ** 2 - 2 * B * r_range + C) / D
    frontier_vol = np.sqrt(np.maximum(frontier_var, 0.0))
    frontier_ret = r_range

    # analytic tangency portfolio (max Sharpe) for a given risk-free rate will be
    # computed by caller when rf is known; return inv_cov to allow that computation.
    return dfp, frontier_vol, frontier_ret, inv_cov


def plot_cml_and_frontier(asset_mu, asset_vol, df_random, frontier_vol, frontier_ret, inv_cov,
                          cov_mat,
                          rf, port_ret, port_vol, out_path: Path,
                          mu_used=None):
    plt.figure(figsize=(10, 7))
    # random portfolios (background)
    plt.scatter(df_random['vol'], df_random['ret'], c='lightgray', s=7, label='Random (long-only)')
    # assets
    plt.scatter(asset_vol, asset_mu, c='red', s=40, label='Assets', zorder=10)
    for i, t in enumerate(asset_mu.index):
        plt.annotate(t, (asset_vol[i], asset_mu[i]), xytext=(4, 2), textcoords='offset points', fontsize=8)

    # exact unconstrained efficient frontier (frontier_ret vs frontier_vol)
    if len(frontier_vol) > 0:
        plt.plot(frontier_vol, frontier_ret, c='blue', lw=2, label='Exact Unconstrained Frontier')

    # plot portfolio point (from provided weights)
    plt.scatter([port_vol], [port_ret], c='green', s=80, label='Provided portfolio', zorder=11)

    # analytic tangency portfolio (based on mu_used). If mu_used is provided, compute
    # the analytic tangency portfolio and draw its CML (this is the classical CML).
    tangency_plotted = False
    if mu_used is not None:
        try:
            mu_arr = np.asarray(mu_used)
            inv = inv_cov
            ones = np.ones(len(mu_arr))
            excess = mu_arr - rf
            w_raw = inv.dot(excess)
            denom = float(ones.T.dot(w_raw))
            if abs(denom) > 1e-12:
                w_t = w_raw / denom
                tangency_ret = float(w_t.dot(mu_arr))
                tangency_vol = float(np.sqrt(w_t.T.dot(cov_mat.values).dot(w_t)))
                # plot tangency point
                plt.scatter([tangency_vol], [tangency_ret], c='purple', s=80, label='Analytic Tangency', zorder=12)
                # plot CML through tangency
                ks = np.linspace(0, 2.0, 200)
                cml_vol = ks * tangency_vol
                cml_ret = rf + ks * (tangency_ret - rf)
                plt.plot(cml_vol, cml_ret, c='black', ls='--', lw=2, label=f'CML (tangency, rf={rf:.2%})')
                tangency_plotted = True
        except Exception:
            tangency_plotted = False

    # If we didn't plot the analytic tangency-based CML, fallback to line through provided portfolio
    if not tangency_plotted:
        ks = np.linspace(0, 2.0, 200)
        cml_vol = ks * port_vol
        cml_ret = rf + ks * (port_ret - rf)
        plt.plot(cml_vol, cml_ret, c='black', ls='--', lw=2, label=f'CML (through provided portfolio, rf={rf:.2%})')

    plt.xlabel('Annual Volatility')
    plt.ylabel('Expected Annual Return')
    plt.title('Assets, Efficient Frontier and CML')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(portfolio_path: str = 'output/portfolio_weighting.csv', cov_window_years: int = 5, rf: float = 0.03, mu_shrink_lambda: float = 0.5):
    base = Path(__file__).parent
    p = base / portfolio_path
    if not p.exists():
        raise FileNotFoundError(f'Portfolio file not found: {p}')
    df = pd.read_csv(p)
    if 'ticker' not in df.columns or 'weight' not in df.columns:
        raise ValueError('Expected `ticker` and `weight` columns in portfolio CSV')

    tickers = df['ticker'].astype(str).tolist()
    weights = df.set_index('ticker')['weight'].reindex(tickers).fillna(0).values

    # Download historical prices for covariance estimation (use calendar days)
    cov_days = cov_window_years * 365
    cov_start = (datetime.date.today() - datetime.timedelta(days=cov_days)).isoformat()
    cov_end = datetime.date.today().isoformat()
    print(f'Downloading prices for {len(tickers)} tickers from {cov_start} to {cov_end}')
    prices = download_prices(tickers, start=cov_start, end=cov_end)
    if prices.empty:
        raise RuntimeError('No price data downloaded for tickers')

    rets = prices.pct_change().dropna(how='all').dropna(axis=1, how='all')
    mu_annual, cov_annual = annualize_returns_and_cov(rets)

    # compute the mean returns used by the optimizer (optionally shrink towards the grand mean)
    mu_raw = mu_annual
    if mu_shrink_lambda is None or mu_shrink_lambda == 0.0:
        mu_used_full = mu_raw
    else:
        mu_bar = mu_raw.mean()
        mu_used_full = (1.0 - mu_shrink_lambda) * mu_raw + mu_shrink_lambda * mu_bar

    # align
    available = [t for t in tickers if t in mu_annual.index]
    if not available:
        raise RuntimeError('No overlapping tickers with returns')

    # recompute weights vector aligned with available tickers
    w = np.array([df.set_index('ticker').loc[t, 'weight'] if t in df['ticker'].values else 0.0 for t in available])
    # use the (possibly shrunk) mu for frontier / tangency calculations
    mu_vec = mu_used_full.loc[available]
    cov_mat = cov_annual.loc[available, available]

    # compute portfolio returns under raw vs used (shrunk) mu for diagnostic printing
    port_ret_raw = float(w.dot(mu_annual.loc[available].values))
    port_ret = float(w.dot(mu_vec.values))
    port_vol = float(np.sqrt(w.T.dot(cov_mat.values).dot(w)))

    # individual asset stats
    asset_mu = mu_vec
    asset_vol = np.sqrt(np.diag(cov_mat))

    # approximate frontier (and exact analytic frontier) using available assets
    df_random, frontier_vol, frontier_ret, inv_cov = approx_efficient_frontier(mu_vec, cov_mat, n_portfolios=4000)

    out_path = base / 'graphs' / 'cml.png'
    plot_cml_and_frontier(asset_mu, asset_vol, df_random, frontier_vol, frontier_ret, inv_cov, cov_mat,
                          rf, port_ret, port_vol, out_path, mu_used=mu_vec)

    print(f'Wrote CML plot to {out_path}')
    print(f'Portfolio expected annual return (raw mu): {port_ret_raw:.2%}, (used mu): {port_ret:.2%}, vol: {port_vol:.2%}')


if __name__ == '__main__':
    args = sys.argv[1:]
    port = 'output/portfolio_weighting.csv'
    years = 5
    rf = 0.03
    mu_shrink_lambda = 0.5
    if len(args) >= 1:
        port = args[0]
    if len(args) >= 2:
        try:
            years = int(args[1])
        except Exception:
            pass
    if len(args) >= 3:
        try:
            rf = float(args[2])
        except Exception:
            pass
    if len(args) >= 4:
        try:
            mu_shrink_lambda = float(args[3])
        except Exception:
            pass
    main(portfolio_path=port, cov_window_years=years, rf=rf, mu_shrink_lambda=mu_shrink_lambda)

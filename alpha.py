import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
import time

def load_prices_for_tickers(tickers, start, end, max_retries=3):
    """Download adjusted prices with retry logic"""
    if not isinstance(tickers, list):
        tickers = [tickers]
    
    for attempt in range(max_retries):
        try:
            data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True, threads=True)
            if data.empty or data.isna().all().all():
                print(f"Warning: Empty data for {tickers} on attempt {attempt + 1}")
                time.sleep(2)
                continue
            break
        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"All download attempts failed for {tickers}")
                return pd.DataFrame()
    
    # Extract adjusted close prices
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if 'Adj Close' in data.columns.get_level_values(1):
                adj = data.xs('Adj Close', axis=1, level=1)
            elif 'Close' in data.columns.get_level_values(1):
                adj = data.xs('Close', axis=1, level=1)
            else:
                adj = data.iloc[:, :len(tickers)]
        else:
            if 'Adj Close' in data.columns:
                adj = data['Adj Close']
            elif 'Close' in data.columns:
                adj = data['Close']
            else:
                adj = data
    except Exception as e:
        print(f"Error extracting price data: {e}")
        adj = data

    # Ensure we have a DataFrame with proper column names
    if isinstance(adj, pd.Series):
        adj = adj.to_frame()
        if len(tickers) == 1:
            adj.columns = [tickers[0]]
    
    # Clean column names
    if adj is not None and not adj.empty:
        adj.columns = [str(col).strip().upper() for col in adj.columns]
        # Remove any completely empty columns
        adj = adj.dropna(axis=1, how='all')
    
    return adj

def compute_alphas(portfolio_csv, prices_csv=None, end_date='2025-11-15', rf_annual=0.0):
    out_dir = Path('./output')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read portfolio
    df = pd.read_csv(portfolio_csv)
    if 'ticker' not in df.columns:
        raise ValueError("Expected 'ticker' column in portfolio CSV")
    tickers = df['ticker'].astype(str).str.strip().str.upper().unique().tolist()

    trading_days = 252
    end = pd.to_datetime(end_date)
    start = end - pd.DateOffset(years=5)

    print(f"Analysis period: {start.date()} to {end.date()}")
    print(f"Portfolio tickers: {tickers}")

    # Market proxies in order of preference
    market_candidates = ['SPY', '^GSPC', 'IVV', 'VOO']
    
    # Initialize prices DataFrame
    prices = pd.DataFrame()
    
    # Load from CSV if provided
    if prices_csv and Path(prices_csv).exists():
        print("Loading prices from CSV file...")
        try:
            prices = pd.read_csv(prices_csv, parse_dates=['Date'], index_col='Date')
            prices.columns = [str(c).strip().upper() for c in prices.columns]
            prices = prices.sort_index()
            print(f"Loaded {len(prices.columns)} tickers from CSV: {list(prices.columns)}")
        except Exception as e:
            print(f"Error loading prices CSV: {e}")
            prices = pd.DataFrame()

    # Identify missing tickers (including market candidates)
    available_in_csv = set(prices.columns) if not prices.empty else set()
    missing_tickers = [t for t in tickers if t not in available_in_csv]
    
    # Always try to download market data first
    market_symbol = None
    market_data = None
    
    print("\nAttempting to download market data...")
    for market_candidate in market_candidates:
        print(f"Trying {market_candidate}...")
        candidate_data = load_prices_for_tickers([market_candidate], start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        
        if not candidate_data.empty and len(candidate_data) > 100:
            market_symbol = market_candidate.upper()
            market_data = candidate_data
            print(f"✓ Successfully downloaded {market_symbol} with {len(market_data)} observations")
            break
        else:
            print(f"✗ Failed to download {market_candidate}")
    
    if market_symbol is None:
        raise ValueError(f"Could not download any market index. Tried: {market_candidates}")
    
    # Add market data to prices
    if not prices.empty:
        # Merge existing prices with market data
        prices = pd.concat([prices, market_data], axis=1)
        # Remove duplicate columns (keep first occurrence)
        prices = prices.loc[:, ~prices.columns.duplicated()]
    else:
        prices = market_data
    
    # Download missing portfolio tickers
    if missing_tickers:
        print(f"\nDownloading {len(missing_tickers)} missing tickers: {missing_tickers}")
        downloaded_tickers = load_prices_for_tickers(missing_tickers, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        
        if not downloaded_tickers.empty:
            prices = pd.concat([prices, downloaded_tickers], axis=1)
            # Remove duplicate columns
            prices = prices.loc[:, ~prices.columns.duplicated()]
            print(f"Downloaded {len(downloaded_tickers.columns)} tickers")
        else:
            print("Warning: No additional tickers downloaded")
    
    # Final data preparation
    prices = prices.sort_index().dropna(how='all')
    
    # Verify market symbol exists in prices
    if market_symbol not in prices.columns:
        print(f"Warning: {market_symbol} not in final prices DataFrame")
        print(f"Available columns: {list(prices.columns)}")
        # Try to find the market symbol in columns (case sensitivity issue)
        matching_columns = [col for col in prices.columns if market_symbol.upper() in col.upper()]
        if matching_columns:
            market_symbol = matching_columns[0]
            print(f"Using matched column: {market_symbol}")
        else:
            raise ValueError(f"Market symbol {market_symbol} not found in prices data")
    
    # Check for missing portfolio tickers
    available_tickers = set(prices.columns)
    not_found = [t for t in tickers if t not in available_tickers]
    if not_found:
        print(f"Warning: Missing data for tickers: {not_found}")
        tickers = [t for t in tickers if t in available_tickers]
    
    if not tickers:
        raise ValueError("No portfolio tickers with price data available")
    
    print(f"\nFinal dataset:")
    print(f"- Market proxy: {market_symbol}")
    print(f"- Portfolio tickers: {tickers}")
    print(f"- Total columns: {len(prices.columns)}")
    print(f"- Time period: {len(prices)} days")
    
    # Use month-end prices and monthly returns
    monthly_prices = prices.resample('M').last()
    returns = monthly_prices.pct_change().dropna(how='all')

    if market_symbol not in returns.columns:
        raise ValueError(f"Market symbol {market_symbol} not in monthly returns columns. Available: {list(returns.columns)}")

    market_returns = returns[market_symbol].rename('market')

    # Calculate alphas on monthly returns
    results = []
    periods_per_year = 12
    rf_period = rf_annual / periods_per_year

    print(f"\nCalculating monthly alphas for {len(tickers)} assets...")

    for t in tickers:
        if t not in returns.columns:
            print(f"Skipping {t} - not in returns data")
            continue

        asset_returns = returns[t].rename('asset')
        df_pair = pd.concat([asset_returns, market_returns], axis=1).dropna()
        n = len(df_pair)

        if n < 6:
            print(f"Skipping {t} - only {n} monthly observations")
            results.append({
                'ticker': t,
                'beta': np.nan,
                'alpha_period': np.nan,
                'alpha_annual': np.nan,
                'se_alpha': np.nan,
                't_alpha': np.nan,
                'start_date': df_pair.index.min() if n > 0 else None,
                'end_date': df_pair.index.max() if n > 0 else None,
            })
            continue

        # CAPM on monthly excess returns
        excess_asset = df_pair['asset'] - rf_period
        excess_market = df_pair['market'] - rf_period

        mean_asset = excess_asset.mean()
        mean_market = excess_market.mean()

        cov = ((excess_asset - mean_asset) * (excess_market - mean_market)).sum() / (n - 1)
        var_market = excess_market.var(ddof=1)

        if var_market == 0 or np.isnan(var_market):
            beta = np.nan
            alpha_period = np.nan
            se_alpha = np.nan
            t_alpha = np.nan
        else:
            beta = cov / var_market
            alpha_period = mean_asset - beta * mean_market

            # standard errors
            predicted = alpha_period + beta * excess_market
            residuals = excess_asset - predicted
            dof = max(n - 2, 1)
            sigma2 = (residuals ** 2).sum() / dof

            sxx = ((excess_market - mean_market) ** 2).sum()
            if sxx > 0:
                se_alpha = np.sqrt(sigma2 * (1.0 / n + (mean_market ** 2) / sxx))
                t_alpha = alpha_period / se_alpha if se_alpha > 0 else np.nan
            else:
                se_alpha = np.nan
                t_alpha = np.nan

        results.append({
            'ticker': t,
            'beta': float(beta) if not np.isnan(beta) else np.nan,
            'alpha_period': float(alpha_period) if not np.isnan(alpha_period) else np.nan,
            'alpha_annual': float(alpha_period * periods_per_year) if not np.isnan(alpha_period) else np.nan,
            'se_alpha': float(se_alpha) if not np.isnan(se_alpha) else np.nan,
            't_alpha': float(t_alpha) if not np.isnan(t_alpha) else np.nan,
            'start_date': df_pair.index.min().strftime('%Y-%m-%d'),
            'end_date': df_pair.index.max().strftime('%Y-%m-%d'),
        })

        print(f"✓ {t}: monthly α={alpha_period:.6f}, β={beta:.3f}")

    out_df = pd.DataFrame(results)
    out_path = out_dir / 'alphas.csv'
    out_df.to_csv(out_path, index=False)
    print(f'\n✅ Success! Wrote alphas for {len(results)} tickers to {out_path}')
    print(f"Market proxy used: {market_symbol}")
    
    return out_df

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Compute CAPM alphas for portfolio tickers vs S&P 500')
    p.add_argument('--portfolio', default='./output/portfolio_weighting.csv', help='Portfolio CSV with a `ticker` column')
    p.add_argument('--prices', default='./output/historical_prices.csv', help='Optional historical prices CSV to use (Date index)')
    p.add_argument('--end-date', default='2025-11-15', help='End date for the 5-year window (YYYY-MM-DD)')
    p.add_argument('--rf', type=float, default=0.0, help='Annual risk-free rate (decimal, e.g. 0.03)')
    args = p.parse_args()

    compute_alphas(args.portfolio, prices_csv=args.prices, end_date=args.end_date, rf_annual=args.rf)
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import io
import time
import math
from pathlib import Path

def calculate_momentum_score(z_score):
    """Calculate momentum score from z-score according to S&P methodology
    
    If Z > 0, Momentum Score = 1 + Z
    If Z < 0, Momentum Score = 1 / (1 - Z)
    If Z = 0, Momentum Score = 1
    """
    if z_score > 0:
        return 1 + z_score
    elif z_score < 0:
        return 1 / (1 - z_score)
    else:
        return 1.0

def get_sp500_tickers():
    """Get current S&P 500 tickers from a reliable CSV source"""
    tickers = []
    raw_csv_url = (
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
        "main/data/constituents.csv"
    )
    headers = {"User-Agent": "Mozilla/5.0 (compatible; project-script/1.0)"}

    try:
        resp = requests.get(raw_csv_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            df = pd.read_csv(io.StringIO(resp.text))
            if 'Symbol' in df.columns:
                tickers = df['Symbol'].astype(str).tolist()
                print(f"Successfully fetched {len(tickers)} tickers from GitHub CSV")
    except Exception as e:
        print(f"Failed to fetch S&P 500 CSV from GitHub: {e}")

    tickers = [ticker.replace('.', '-').strip() for ticker in tickers if ticker]
    return tickers

def get_stock_data(ticker, start_date, end_date):
    """Get historical stock data with proper error handling"""
    stock = yf.Ticker(ticker)
    attempts = 2
    for attempt in range(attempts):
        try:
            end_inclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            hist = stock.history(start=start_date, end=end_inclusive, interval="1d")
            if hist is not None and not hist.empty:
                return hist
            time.sleep(0.5)
        except Exception as e:
            if attempt == attempts - 1:
                print(f"Error getting data for {ticker} after {attempts} attempts: {e}")
                return None
            time.sleep(0.5)
    return None

def calculate_momentum_value(price_start, price_end):
    """Calculate momentum value (price change ratio)"""
    if price_start == 0:
        return 0
    return (price_end / price_start) - 1

def calculate_momentum_for_ticker(ticker, reference_date):
    """Calculate momentum for a single ticker according to S&P methodology
    
    Calculates both:
    1. Standard 12-month momentum (M-14 to M-2)
    2. Short-term 6-month momentum (M-8 to M-2)
    
    Key corrections:
    1. Uses DAILY data, not monthly
    2. Calculates momentum as price change excluding most recent month
    3. Adjusts for volatility using daily returns
    4. Computes z-score across universe 
    """
    # Parse reference date
    ref_date_dt = pd.Timestamp(reference_date)
    
    # CRITICAL: S&P methodology uses M-2 and M-14 where M is rebalancing month
    # M-2 = 2 months before reference date (end point for both)
    # M-14 = 14 months before reference date (12-month momentum start)
    # M-8 = 8 months before reference date (6-month momentum start)
    price_end_date = ref_date_dt - pd.DateOffset(months=2)
    price_start_date_12m = ref_date_dt - pd.DateOffset(months=14)
    price_start_date_6m = ref_date_dt - pd.DateOffset(months=8)
    
    # Get data from 14 months before reference date to 2 months before
    # Add buffer for data availability
    data_start = price_start_date_12m - pd.DateOffset(days=10)
    data_end = price_end_date + pd.DateOffset(days=10)
    
    hist_data = get_stock_data(ticker, data_start.strftime('%Y-%m-%d'), 
                               data_end.strftime('%Y-%m-%d'))
    
    if hist_data is None or len(hist_data) < 100:  # Need at least ~125 trading days for 6 months
        return None
    
    # Find closest available prices
    try:
        # Convert hist_data index to timezone-naive for comparison
        if hist_data.index.tz is not None:
            hist_data.index = hist_data.index.tz_localize(None)
        
        # Get price closest to M-2 (end of momentum period - same for both)
        price_m2_idx = hist_data.index.get_indexer([price_end_date], method='nearest')[0]
        price_m2 = hist_data['Close'].iloc[price_m2_idx]
        actual_date_m2 = hist_data.index[price_m2_idx]
        
        # === STANDARD 12-MONTH MOMENTUM ===
        standard_result = None
        if len(hist_data) >= 200:  # Need ~250 trading days for 12 months
            price_m14_idx = hist_data.index.get_indexer([price_start_date_12m], method='nearest')[0]
            price_m14 = hist_data['Close'].iloc[price_m14_idx]
            actual_date_m14 = hist_data.index[price_m14_idx]
            
            # Calculate 12-month momentum value
            momentum_value_12m = calculate_momentum_value(price_m14, price_m2)
            
            # Get daily returns for volatility calculation
            period_data_12m = hist_data.loc[actual_date_m14:actual_date_m2]
            if len(period_data_12m) >= 200:
                daily_returns_12m = period_data_12m['Close'].pct_change().dropna()
                
                if len(daily_returns_12m) > 0:
                    volatility_12m = daily_returns_12m.std()
                    
                    if volatility_12m > 0 and not pd.isna(volatility_12m):
                        risk_adj_momentum_12m = momentum_value_12m / volatility_12m
                        
                        standard_result = {
                            'momentum_value_12m': momentum_value_12m,
                            'volatility_12m': volatility_12m,
                            'risk_adj_momentum_12m': risk_adj_momentum_12m,
                            'price_m14': price_m14,
                            'actual_date_m14': actual_date_m14,
                            'trading_days_12m': len(period_data_12m)
                        }
        
        # === SHORT-TERM 6-MONTH MOMENTUM ===
        short_term_result = None
        price_m8_idx = hist_data.index.get_indexer([price_start_date_6m], method='nearest')[0]
        price_m8 = hist_data['Close'].iloc[price_m8_idx]
        actual_date_m8 = hist_data.index[price_m8_idx]
        
        # Calculate 6-month momentum value
        momentum_value_6m = calculate_momentum_value(price_m8, price_m2)
        
        # Get daily returns for volatility calculation
        period_data_6m = hist_data.loc[actual_date_m8:actual_date_m2]
        if len(period_data_6m) >= 100:
            daily_returns_6m = period_data_6m['Close'].pct_change().dropna()
            
            if len(daily_returns_6m) > 0:
                volatility_6m = daily_returns_6m.std()
                
                if volatility_6m > 0 and not pd.isna(volatility_6m):
                    risk_adj_momentum_6m = momentum_value_6m / volatility_6m
                    
                    short_term_result = {
                        'momentum_value_6m': momentum_value_6m,
                        'volatility_6m': volatility_6m,
                        'risk_adj_momentum_6m': risk_adj_momentum_6m,
                        'price_m8': price_m8,
                        'actual_date_m8': actual_date_m8,
                        'trading_days_6m': len(period_data_6m)
                    }
        
        # Combine results
        if standard_result is None and short_term_result is None:
            return None
        
        result = {
            'ticker': ticker,
            'price_m2': price_m2,
            'actual_date_m2': actual_date_m2,
        }
        
        # Add standard momentum if available
        if standard_result:
            result.update(standard_result)
            # Use 12-month as primary for backwards compatibility
            result['momentum_value'] = standard_result['momentum_value_12m']
            result['volatility'] = standard_result['volatility_12m']
            result['risk_adj_momentum'] = standard_result['risk_adj_momentum_12m']
        
        # Add short-term momentum if available
        if short_term_result:
            result.update(short_term_result)
            # If no 12-month data, use 6-month as primary
            if standard_result is None:
                result['momentum_value'] = short_term_result['momentum_value_6m']
                result['volatility'] = short_term_result['volatility_6m']
                result['risk_adj_momentum'] = short_term_result['risk_adj_momentum_6m']
        
        return result
        
    except Exception as e:
        print(f"Error calculating momentum for {ticker}: {e}")
        return None

def calculate_z_scores_and_momentum_scores(results):
    """Calculate z-scores and momentum scores across the universe
    
    Calculates separate z-scores and momentum scores for:
    1. Standard 12-month momentum
    2. Short-term 6-month momentum
    """
    if not results:
        return []
    
    # === STANDARD 12-MONTH MOMENTUM SCORES ===
    results_with_12m = [r for r in results if 'risk_adj_momentum_12m' in r]
    if results_with_12m:
        risk_adj_momentums_12m = [r['risk_adj_momentum_12m'] for r in results_with_12m]
        mean_ram_12m = np.mean(risk_adj_momentums_12m)
        std_ram_12m = np.std(risk_adj_momentums_12m)
        
        if std_ram_12m > 0:
            for result in results_with_12m:
                ram = result['risk_adj_momentum_12m']
                z_score = (ram - mean_ram_12m) / std_ram_12m
                z_score_winsorized = np.clip(z_score, -3, 3)
                momentum_score = calculate_momentum_score(z_score_winsorized)
                
                result['z_score_12m'] = z_score
                result['z_score_winsorized_12m'] = z_score_winsorized
                result['momentum_score_12m'] = momentum_score
                result['mean_ram_12m'] = mean_ram_12m
                result['std_ram_12m'] = std_ram_12m
                
                # Set as primary scores for backwards compatibility
                result['z_score'] = z_score
                result['z_score_winsorized'] = z_score_winsorized
                result['momentum_score'] = momentum_score
    
    # === SHORT-TERM 6-MONTH MOMENTUM SCORES ===
    results_with_6m = [r for r in results if 'risk_adj_momentum_6m' in r]
    if results_with_6m:
        risk_adj_momentums_6m = [r['risk_adj_momentum_6m'] for r in results_with_6m]
        mean_ram_6m = np.mean(risk_adj_momentums_6m)
        std_ram_6m = np.std(risk_adj_momentums_6m)
        
        if std_ram_6m > 0:
            for result in results_with_6m:
                ram = result['risk_adj_momentum_6m']
                z_score = (ram - mean_ram_6m) / std_ram_6m
                z_score_winsorized = np.clip(z_score, -3, 3)
                momentum_score = calculate_momentum_score(z_score_winsorized)
                
                result['z_score_6m'] = z_score
                result['z_score_winsorized_6m'] = z_score_winsorized
                result['momentum_score_6m'] = momentum_score
                result['mean_ram_6m'] = mean_ram_6m
                result['std_ram_6m'] = std_ram_6m
                
                # If no 12-month data, use 6-month as primary
                if 'z_score' not in result:
                    result['z_score'] = z_score
                    result['z_score_winsorized'] = z_score_winsorized
                    result['momentum_score'] = momentum_score
    
    return results

if __name__ == "__main__":
    print("=" * 80)
    print("S&P 500 MOMENTUM CALCULATOR")
    print("Following S&P Dow Jones Indices Methodology")
    print("=" * 80)
    
    # Load tickers from top ESG list
    print("\nLoading tickers from 'output/top_esg_tickers.csv'...")
    base = Path(__file__).parent
    output_dir = base / 'output'
    te_path = output_dir / 'top_esg_tickers.csv'
    if not te_path.exists():
        print(f"Error: '{te_path}' not found. Run ESG calculation first to create top_esg_tickers.csv")
        exit(1)

    try:
        te_df = pd.read_csv(te_path)
        # Try to find a ticker-like column
        if 'ticker' in te_df.columns:
            selected_tickers = te_df['ticker'].astype(str).str.replace('.', '-', regex=False).str.strip().str.upper().tolist()
        else:
            # fallback to first column
            selected_tickers = te_df.iloc[:, 0].astype(str).str.replace('.', '-', regex=False).str.strip().str.upper().tolist()
        print(f"Loaded {len(selected_tickers)} tickers from {te_path}")
    except Exception as e:
        print(f"Failed to read {te_path}: {e}")
        exit(1)
    
    # Reference date (last business day of month for semi-annual rebalancing)
    # For November 14, 2025, the reference date would be October 31, 2025
    reference_date = "2025-10-31"
    print(f"\nReference Date: {reference_date}")
    print("This is the date used to determine which stocks qualify")
    print("(Typically the last business day before the rebalancing month)")
    
    # Calculate reference dates for momentum calculation
    ref_dt = pd.Timestamp(reference_date)
    m2_date = ref_dt - pd.DateOffset(months=2)
    m14_date = ref_dt - pd.DateOffset(months=14)
    print(f"\nMomentum Period:")
    print(f"  Start (M-14): {m14_date.strftime('%Y-%m-%d')}")
    print(f"  End (M-2):    {m2_date.strftime('%Y-%m-%d')}")
    print(f"  This excludes the most recent month per S&P methodology")
    
    print("\n" + "=" * 80)
    print("CALCULATING MOMENTUM SCORES...")
    print("This may take 10-15 minutes for the full S&P 500...")
    print("=" * 80 + "\n")
    
    # Calculate momentum for all tickers
    results = []
    successful = 0
    failed = 0
    delay = 0.3  # Reduced delay
    
    # Prepare output directory
    output_dir.mkdir(exist_ok=True)

    for i, ticker in enumerate(selected_tickers):
        if (i + 1) % 10 == 0:
            total = len(selected_tickers)
            print(f"Progress: {i+1}/{total} ({successful} successful, {failed} failed)")
        
        result = calculate_momentum_for_ticker(ticker, reference_date)
        if result:
            results.append(result)
            successful += 1
        else:
            failed += 1
        
        # Save partial results every 50 stocks
        if (i + 1) % 50 == 0 and results:
            try:
                pd.DataFrame(results).to_csv(output_dir / 'momentum_partial.csv', index=False)
            except Exception:
                pass
        
        time.sleep(delay)
    
    print(f"\nData collection complete: {successful} successful, {failed} failed")
    
    if not results:
        print("\n No results obtained. Please check:")
        exit(1)
    
    # Calculate z-scores and momentum scores
    print("\n Calculating z-scores and momentum scores...")
    results = calculate_z_scores_and_momentum_scores(results)
    
    if not results:
        print("\n Failed to calculate z-scores")
        exit(1)
    
    # Sort by momentum score
    results.sort(key=lambda x: x['momentum_score'], reverse=True)
    
    # Display results
    print("\n" + "=" * 120)
    print("TOP 20 MOMENTUM STOCKS (S&P 500) - STANDARD 12-MONTH")
    print("=" * 120)
    print(f"{'Rank':<6} {'Ticker':<8} {'Mom.Score':<12} {'Z-Score':<10} {'12M Ret':<10} {'Vol':<10} {'Risk-Adj':<10}")
    print("-" * 120)
    
    for i, r in enumerate(results[:20]):
        if 'momentum_score_12m' in r:
            print(f"{i+1:<6} {r['ticker']:<8} {r['momentum_score_12m']:>11.4f} {r['z_score_12m']:>9.2f} "
                  f"{r['momentum_value_12m']:>9.1%} {r['volatility_12m']:>9.4f} {r['risk_adj_momentum_12m']:>9.2f}")
    
    # Short-term momentum comparison
    print("\n" + "=" * 120)
    print("TOP 20 MOMENTUM STOCKS (S&P 500) - SHORT-TERM 6-MONTH")
    print("=" * 120)
    print(f"{'Rank':<6} {'Ticker':<8} {'Mom.Score':<12} {'Z-Score':<10} {'6M Ret':<10} {'Vol':<10} {'Risk-Adj':<10}")
    print("-" * 120)
    
    # Sort by 6-month momentum score for this table
    results_with_6m = [r for r in results if 'momentum_score_6m' in r]
    results_with_6m.sort(key=lambda x: x['momentum_score_6m'], reverse=True)
    
    for i, r in enumerate(results_with_6m[:20]):
        print(f"{i+1:<6} {r['ticker']:<8} {r['momentum_score_6m']:>11.4f} {r['z_score_6m']:>9.2f} "
              f"{r['momentum_value_6m']:>9.1%} {r['volatility_6m']:>9.4f} {r['risk_adj_momentum_6m']:>9.2f}")
    
    # Side-by-side comparison for stocks in both top 20s
    print("\n" + "=" * 140)
    print("COMPARISON: STOCKS IN TOP 20 FOR BOTH 12-MONTH AND 6-MONTH MOMENTUM")
    print("=" * 140)
    
    top20_12m = set([r['ticker'] for r in results[:20] if 'momentum_score_12m' in r])
    top20_6m = set([r['ticker'] for r in results_with_6m[:20]])
    common_tickers = top20_12m & top20_6m
    
    if common_tickers:
        print(f"{'Ticker':<8} {'12M Score':<12} {'12M Ret':<10} {'6M Score':<12} {'6M Ret':<10} {'Agreement':<12}")
        print("-" * 140)
        
        for ticker in sorted(common_tickers):
            r = next((x for x in results if x['ticker'] == ticker), None)
            if r and 'momentum_score_12m' in r and 'momentum_score_6m' in r:
                agreement = "✓ Strong" if abs(r['momentum_score_12m'] - r['momentum_score_6m']) < 0.2 else "○ Moderate"
                print(f"{ticker:<8} {r['momentum_score_12m']:>11.4f} {r['momentum_value_12m']:>9.1%} "
                      f"{r['momentum_score_6m']:>11.4f} {r['momentum_value_6m']:>9.1%} {agreement:<12}")
        print(f"\n{len(common_tickers)} stocks appear in both top 20 lists")
    else:
        print("No overlap between 12-month and 6-month top 20 stocks")
    
    print("\n" + "=" * 120)
    print("BOTTOM 20 MOMENTUM STOCKS (S&P 500) - STANDARD 12-MONTH")
    print("=" * 120)
    print(f"{'Rank':<6} {'Ticker':<8} {'Mom.Score':<12} {'Z-Score':<10} {'12M Ret':<10} {'Vol':<10} {'Risk-Adj':<10}")
    print("-" * 120)
    
    for i, r in enumerate(results[-20:]):
        if 'momentum_score_12m' in r:
            rank = len(results) - 19 + i
            print(f"{rank:<6} {r['ticker']:<8} {r['momentum_score_12m']:>11.4f} {r['z_score_12m']:>9.2f} "
                  f"{r['momentum_value_12m']:>9.1%} {r['volatility_12m']:>9.4f} {r['risk_adj_momentum_12m']:>9.2f}")
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    # 12-month stats
    results_12m = [r for r in results if 'momentum_score_12m' in r]
    if results_12m:
        momentum_scores_12m = [r['momentum_score_12m'] for r in results_12m]
        z_scores_12m = [r['z_score_12m'] for r in results_12m]
        returns_12m = [r['momentum_value_12m'] for r in results_12m]
        volatilities_12m = [r['volatility_12m'] for r in results_12m]
        
        print(f"\n12-MONTH MOMENTUM (n={len(results_12m)}):")
        print(f"\nMomentum Scores:")
        print(f"  Mean:   {np.mean(momentum_scores_12m):.4f}")
        print(f"  Median: {np.median(momentum_scores_12m):.4f}")
        print(f"  Min:    {np.min(momentum_scores_12m):.4f}")
        print(f"  Max:    {np.max(momentum_scores_12m):.4f}")
        
        print(f"\nZ-Scores:")
        print(f"  Mean:   {np.mean(z_scores_12m):.4f}")
        print(f"  Median: {np.median(z_scores_12m):.4f}")
        print(f"  Min:    {np.min(z_scores_12m):.4f}")
        print(f"  Max:    {np.max(z_scores_12m):.4f}")
        
        print(f"\n12-Month Returns:")
        print(f"  Mean:   {np.mean(returns_12m):.2%}")
        print(f"  Median: {np.median(returns_12m):.2%}")
        print(f"  Min:    {np.min(returns_12m):.2%}")
        print(f"  Max:    {np.max(returns_12m):.2%}")
        
        print(f"\nVolatility (Daily Std Dev):")
        print(f"  Mean:   {np.mean(volatilities_12m):.4f}")
        print(f"  Median: {np.median(volatilities_12m):.4f}")
    
    # 6-month stats
    results_6m = [r for r in results if 'momentum_score_6m' in r]
    if results_6m:
        momentum_scores_6m = [r['momentum_score_6m'] for r in results_6m]
        z_scores_6m = [r['z_score_6m'] for r in results_6m]
        returns_6m = [r['momentum_value_6m'] for r in results_6m]
        volatilities_6m = [r['volatility_6m'] for r in results_6m]
        
        print(f"\n6-MONTH MOMENTUM (n={len(results_6m)}):")
        print(f"\nMomentum Scores:")
        print(f"  Mean:   {np.mean(momentum_scores_6m):.4f}")
        print(f"  Median: {np.median(momentum_scores_6m):.4f}")
        print(f"  Min:    {np.min(momentum_scores_6m):.4f}")
        print(f"  Max:    {np.max(momentum_scores_6m):.4f}")
        
        print(f"\nZ-Scores:")
        print(f"  Mean:   {np.mean(z_scores_6m):.4f}")
        print(f"  Median: {np.median(z_scores_6m):.4f}")
        print(f"  Min:    {np.min(z_scores_6m):.4f}")
        print(f"  Max:    {np.max(z_scores_6m):.4f}")
        
        print(f"\n6-Month Returns:")
        print(f"  Mean:   {np.mean(returns_6m):.2%}")
        print(f"  Median: {np.median(returns_6m):.2%}")
        print(f"  Min:    {np.min(returns_6m):.2%}")
        print(f"  Max:    {np.max(returns_6m):.2%}")
        
        print(f"\nVolatility (Daily Std Dev):")
        print(f"  Mean:   {np.mean(volatilities_6m):.4f}")
        print(f"  Median: {np.median(volatilities_6m):.4f}")
    
    # Save final results
    try:
        df_results = pd.DataFrame(results)
        df_results = df_results.sort_values('momentum_score', ascending=False)
        results_path = output_dir / 'top_esg_tickers_momentum.csv'
        df_results.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to '{results_path}'")

        # Also save a condensed CSV containing only ticker and momentum score
        try:
            df_scores = df_results[['ticker', 'momentum_score']].copy()
            scores_path = output_dir / 'top_esg_tickers_momentum_scores.csv'
            df_scores.to_csv(scores_path, index=False)
            print(f"✓ Condensed ticker+score file saved to '{scores_path}'")
        except Exception as e:
            print(f"\n Failed to save condensed scores CSV: {e}")
    except Exception as e:
        print(f"\n Failed to save results: {e}")

    # Save top 20% lists for 12-month and 6-month momentum scores (tickers + scores)
    try:
        # 12-month top 20%
        if 'momentum_score_12m' in df_results.columns:
            df_12m = df_results.dropna(subset=['momentum_score_12m']).copy()
            if len(df_12m) > 0:
                top_n_12m = max(1, math.ceil(len(df_12m) * 0.20))
                df_top12 = df_12m.sort_values('momentum_score_12m', ascending=False).head(top_n_12m)
                top12_path = output_dir / 'sp500_top20_12m.csv'
                df_top12[['ticker', 'momentum_score_12m']].to_csv(top12_path, index=False)
                print(f"✓ Top {top_n_12m} tickers (top 20%) for 12-month momentum saved to '{top12_path}'")
                # Also save candidate list for 12-month momentum selection
                try:
                    candidate_path = output_dir / 'candidate_tickers.csv'
                    df_top12[['ticker', 'momentum_score_12m']].to_csv(candidate_path, index=False)
                    print(f"✓ Candidate tickers (12m top 20%) saved to '{candidate_path}'")
                except Exception as e:
                    print(f"\n Failed to save candidate_tickers.csv: {e}")

        # 6-month top 20%
        if 'momentum_score_6m' in df_results.columns:
            df_6m = df_results.dropna(subset=['momentum_score_6m']).copy()
            if len(df_6m) > 0:
                top_n_6m = max(1, math.ceil(len(df_6m) * 0.20))
                df_top6 = df_6m.sort_values('momentum_score_6m', ascending=False).head(top_n_6m)
                top6_path = output_dir / 'sp500_top20_6m.csv'
                df_top6[['ticker', 'momentum_score_6m']].to_csv(top6_path, index=False)
                print(f"✓ Top {top_n_6m} tickers (top 20%) for 6-month momentum saved to '{top6_path}'")
                # Also save candidate list for 6-month momentum selection
                try:
                    candidate_short_path = output_dir / 'candidate_short_term_comparison.csv'
                    df_top6[['ticker', 'momentum_score_6m']].to_csv(candidate_short_path, index=False)
                    print(f"✓ Candidate tickers (6m top 20%) saved to '{candidate_short_path}'")
                except Exception as e:
                    print(f"\n Failed to save candidate_short_term_comparison.csv: {e}")
    except Exception as e:
        print(f"\n Failed to save top-20% files: {e}")
    
    # Quintile analysis
    print("\n" + "=" * 100)
    print("QUINTILE ANALYSIS - 12-MONTH MOMENTUM")
    print("(Top 20% would be selected for S&P 500 Momentum Index)")
    print("=" * 100)
    
    results_12m = [r for r in results if 'momentum_score_12m' in r]
    if results_12m:
        quintile_size = len(results_12m) // 5
        for q in range(5):
            start_idx = q * quintile_size
            end_idx = start_idx + quintile_size if q < 4 else len(results_12m)
            quintile_stocks = results_12m[start_idx:end_idx]
            
            q_mom_scores = [r['momentum_score_12m'] for r in quintile_stocks]
            q_returns = [r['momentum_value_12m'] for r in quintile_stocks]
            
            print(f"\nQuintile {q+1} (Rank {start_idx+1}-{end_idx}):")
            print(f"  Stocks: {len(quintile_stocks)}")
            print(f"  Avg Momentum Score: {np.mean(q_mom_scores):.4f}")
            print(f"  Avg 12M Return: {np.mean(q_returns):.2%}")
            print(f"  Range: {quintile_stocks[0]['ticker']} to {quintile_stocks[-1]['ticker']}")
    
    # Comparison analysis
    print("\n" + "=" * 100)
    print("MOMENTUM SIGNAL COMPARISON ANALYSIS")
    print("=" * 100)
    
    both = [r for r in results if 'momentum_score_12m' in r and 'momentum_score_6m' in r]
    if both:
        # Calculate correlation
        scores_12m = [r['momentum_score_12m'] for r in both]
        scores_6m = [r['momentum_score_6m'] for r in both]
        correlation = np.corrcoef(scores_12m, scores_6m)[0, 1]
        
        print(f"\nStocks with both metrics: {len(both)}")
        print(f"Correlation between 12M and 6M momentum scores: {correlation:.3f}")
        
        # Agreement analysis
        both.sort(key=lambda x: x['momentum_score_12m'], reverse=True)
        top_quintile_12m = set([r['ticker'] for r in both[:len(both)//5]])
        
        both_by_6m = sorted(both, key=lambda x: x['momentum_score_6m'], reverse=True)
        top_quintile_6m = set([r['ticker'] for r in both_by_6m[:len(both)//5]])
        
        overlap = len(top_quintile_12m & top_quintile_6m)
        overlap_pct = (overlap / len(top_quintile_12m)) * 100 if top_quintile_12m else 0
        
        print(f"\nTop Quintile Overlap:")
        print(f"  {overlap}/{len(top_quintile_12m)} stocks appear in both top quintiles ({overlap_pct:.1f}%)")
        
        # Divergence examples
        print(f"\nLargest Divergences (12M strong, 6M weak):")
        both_sorted = sorted(both, key=lambda x: (x['momentum_score_12m'] - x['momentum_score_6m']), reverse=True)
        for r in both_sorted[:5]:
            print(f"  {r['ticker']}: 12M={r['momentum_score_12m']:.3f}, 6M={r['momentum_score_6m']:.3f}, "
                  f"Diff={r['momentum_score_12m']-r['momentum_score_6m']:.3f}")
        
        print(f"\nLargest Divergences (6M strong, 12M weak):")
        both_sorted_rev = sorted(both, key=lambda x: (x['momentum_score_6m'] - x['momentum_score_12m']), reverse=True)
        for r in both_sorted_rev[:5]:
            print(f"  {r['ticker']}: 6M={r['momentum_score_6m']:.3f}, 12M={r['momentum_score_12m']:.3f}, "
                  f"Diff={r['momentum_score_6m']-r['momentum_score_12m']:.3f}")
    
    print("\n" + "=" * 100)
    print("CALCULATION COMPLETE")
    print("=" * 100)
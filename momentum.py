import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import requests
import io
import time

import pandas as pd

def calculate_momentum(return_12m, mean_momentum_3y, std_momentum_3y):
    """Calculate momentum score according to S&P methodology"""
    if std_momentum_3y == 0:  # Avoid division by zero
        return 1.0
        
    z = (return_12m - mean_momentum_3y) / std_momentum_3y
    if z > 0:
        return 1 + z
    elif z < 0:
        return 1 / (1 - z)
    else:
        return 1.0

def get_sp500_tickers():
    """Get current S&P 500 tickers from a reliable CSV source (GitHub datasets).

    Tries to fetch the raw CSV from the datasets repo. If that fails, tries a
    local cached `constituents.csv` file. If all fails, falls back to a
    small hardcoded list to allow the script to run.
    """
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
        else:
            print(f"GitHub CSV request returned HTTP {resp.status_code}")
    except Exception as e:
        print(f"Failed to fetch S&P 500 CSV from GitHub: {e}")

    # If remote fetch failed, try local cache file `constituents.csv`
    if not tickers:
        try:
            local_path = 'constituents.csv'
            df_local = pd.read_csv(local_path)
            if 'Symbol' in df_local.columns:
                tickers = df_local['Symbol'].astype(str).tolist()
                print(f"Loaded {len(tickers)} tickers from local {local_path}")
        except Exception:
            # silent fall-through to hardcoded fallback below
            pass
    
    # If online sources fail, use a comprehensive fallback list
    if not tickers:
        print("Using comprehensive fallback list")
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
            "JPM", "V", "PG", "XOM", "HD", "CVX", "MA", "BAC", "ABBV", "PFE",
            "AVGO", "LLY", "KO", "WMT", "DIS", "NFLX", "ADBE", "CRM", "CSCO", "PEP",
            "TMO", "ABT", "DHR", "ACN", "CMCSA", "NKE", "VZ", "T", "MRK", "WFC",
            "PM", "RTX", "UPS", "SBUX", "LOW", "BMY", "TXN", "AMD", "INTU", "ISRG",
            "SPGI", "PLD", "NOW", "MDT", "GS", "BLK", "AMT", "SCHW", "LRCX", "ADI",
            "DE", "CAT", "BA", "MMM", "IBM", "GE", "F", "GM", "TGT", "COST"
        ]
    
    # If still empty, use a conservative hardcoded fallback to keep script runnable
    if not tickers:
        print("Using comprehensive fallback list")
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "UNH", "JNJ",
            "JPM", "V", "PG", "XOM", "HD", "CVX", "MA", "BAC", "ABBV", "PFE",
        ]

    # Clean tickers for yfinance (replace dots with hyphens for tickers like BRK.B -> BRK-B)
    tickers = [ticker.replace('.', '-').strip() for ticker in tickers if ticker]
    return tickers

def calculate_total_return(prices):
    """Calculate total return including dividends"""
    if len(prices) < 2:
        return 0
    return (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]

def get_stock_data(ticker, start_date, end_date):
    """Get historical stock data with proper error handling.

    Retries a few times for transient issues and makes the provided end_date
    inclusive so that monthly bins include the final month.
    """
    stock = yf.Ticker(ticker)
    attempts = 3
    for attempt in range(attempts):
        try:
            end_inclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            hist = stock.history(start=start_date, end=end_inclusive, interval="1mo")
            if hist is not None and not hist.empty:
                return hist
            # empty result - wait and retry
            time.sleep(0.5)
        except Exception as e:
            if attempt == attempts - 1:
                print(f"Error getting data for {ticker} after {attempts} attempts: {e}")
                return None
            time.sleep(0.5)

    # All attempts exhausted and no data
    return None

def calculate_momentum_for_ticker(ticker, end_date):
    """Calculate momentum for a single ticker"""
    # Define periods according to S&P methodology
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # Use pandas DateOffset for month/year alignment (more reliable than fixed days)
    momentum_end = (pd.Timestamp(end_date_dt) - pd.DateOffset(months=1)).to_pydatetime()
    momentum_start = (pd.Timestamp(momentum_end) - pd.DateOffset(months=12)).to_pydatetime()

    # Historical period for Z-score: 3 years back from momentum period
    historical_start = (pd.Timestamp(momentum_start) - pd.DateOffset(years=3)).to_pydatetime()
    
    # Get data for momentum calculation
    momentum_data = get_stock_data(ticker, momentum_start.strftime('%Y-%m-%d'), 
                                  momentum_end.strftime('%Y-%m-%d'))
    if momentum_data is None:
        print(f"Skipping {ticker}: insufficient momentum-period data (need ~12 months)")
        return None
    if len(momentum_data) < 12:
        print(f"Skipping {ticker}: momentum period has only {len(momentum_data)} rows (<12)")
        return None
    
    # Calculate 12-month momentum return
    current_momentum = calculate_total_return(momentum_data['Close'])
    
    # Get historical data for Z-score calculation
    historical_data = get_stock_data(ticker, historical_start.strftime('%Y-%m-%d'),
                                   momentum_end.strftime('%Y-%m-%d'))
    if historical_data is None or len(historical_data) < 36:
        print(f"Skipping {ticker}: insufficient historical data for z-score (need ~36 months)")
        return None
    if len(historical_data) < 36:
        print(f"Skipping {ticker}: historical period has only {len(historical_data)} rows (<36)")
        return None
    
    # Calculate rolling 12-month momentum for historical period
    historical_momentums = []
    prices = historical_data['Close']
    
    for i in range(len(prices) - 12):
        period_prices = prices.iloc[i:i+12]
        if len(period_prices) >= 12:
            momentum_val = calculate_total_return(period_prices)
            historical_momentums.append(momentum_val)
    
    if len(historical_momentums) < 12:  # Need sufficient history
        return None
    
    # Calculate mean and std for Z-score
    mean_momentum = sum(historical_momentums) / len(historical_momentums)
    variance = sum((x - mean_momentum) ** 2 for x in historical_momentums) / len(historical_momentums)
    std_momentum = variance ** 0.5
    
    if std_momentum == 0:
        return None
    
    # Calculate final momentum score
    momentum_score = calculate_momentum(current_momentum, mean_momentum, std_momentum)
    
    return {
        'ticker': ticker,
        'momentum_score': momentum_score,
        '12m_return': current_momentum,
        'z_score': (current_momentum - mean_momentum) / std_momentum,
        'mean_momentum': mean_momentum,
        'std_momentum': std_momentum
    }

if __name__ == "__main__":
    # Get S&P 500 tickers
    print("Fetching S&P 500 tickers...")
    sp500_tickers = get_sp500_tickers()
    print(f"Found {len(sp500_tickers)} tickers")
    
    # Process all tickers (full S&P 500 constituents)
    test_tickers = sp500_tickers  # full list from CSV
    print(f"Processing {len(test_tickers)} tickers (full S&P 500)")
    
    # Use historical date that definitely has data
    end_date = "2024-06-01"
    
    print("Calculating momentum scores...")
    print("This may take a few minutes...")
    
    # Calculate momentum for all tickers and save incrementally to avoid losing work
    results = []
    successful_tickers = 0
    delay_seconds = 0.5  # small delay between requests to reduce rate-limiting

    for i, ticker in enumerate(test_tickers):
        print(f"Processing {i+1}/{len(test_tickers)}: {ticker}")
        result = calculate_momentum_for_ticker(ticker, end_date)
        if result:
            results.append(result)
            successful_tickers += 1

        # Periodic incremental save every 50 processed tickers
        if (i + 1) % 50 == 0:
            try:
                pd.DataFrame(results).to_csv('momentum_results_partial.csv', index=False)
                print(f"Saved partial results after {i+1} tickers (successful: {successful_tickers})")
            except Exception as e:
                print(f"Warning: failed to save partial results: {e}")

        # polite delay
        time.sleep(delay_seconds)
    
    print(f"\nSuccessfully calculated momentum for {successful_tickers} stocks")
    print("=" * 80)
    
    # Final save of results
    if results:
        # Sort by momentum score
        results.sort(key=lambda x: x['momentum_score'], reverse=True)
        
        # Display top results
        print("Top Momentum Stocks:")
        print("Rank | Ticker | Momentum Score | 12M Return | Z-Score")
        print("-" * 60)
        
        for i, result in enumerate(results[:20]):  # Show top 20
            print(f"{i+1:4} | {result['ticker']:6} | {result['momentum_score']:14.4f} | "
                  f"{result['12m_return']:10.2%} | {result['z_score']:7.2f}")
        
        # Display summary statistics
        if len(results) >= 5:
            print(f"\nSummary Statistics (for {len(results)} stocks):")
            scores = [r['momentum_score'] for r in results]
            returns = [r['12m_return'] for r in results]
            print(f"Momentum Score - Avg: {sum(scores)/len(scores):.4f}, "
                  f"Min: {min(scores):.4f}, Max: {max(scores):.4f}")
            print(f"12M Returns - Avg: {sum(returns)/len(returns):.2%}, "
                  f"Min: {min(returns):.2%}, Max: {max(returns):.2%}")
        # write final CSV
        try:
            pd.DataFrame(results).to_csv('momentum_results.csv', index=False)
            print("Final results written to momentum_results.csv")
        except Exception as e:
            print(f"Failed to write final CSV: {e}")

    else:
        print("No results obtained. Possible issues:")
        print("1. Date range may be too recent - try an older end_date like '2024-03-01'")
        print("2. Yahoo Finance API may be rate limited - try again later")
        print("3. Some tickers may not have sufficient history")
        
        # Test with a known good ticker
        print("\nTesting with AAPL...")
        test_result = calculate_momentum_for_ticker("AAPL", "2024-03-01")
        if test_result:
            print("AAPL test successful!")
            print(f"AAPL Momentum Score: {test_result['momentum_score']:.4f}")
        else:
            print("Even AAPL failed - likely date range issue")
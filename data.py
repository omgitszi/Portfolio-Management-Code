import yfinance as yf
import pandas as pd
import requests
from typing import Dict, List, Optional

def get_esg_scores_fmp(tickers: List[str], api_key: str) -> Dict:
    """Get ESG scores from Financial Modeling Prep with enhanced error handling"""
    esg_scores = {}
    
    for ticker in tickers:
        try:
            print(f"Fetching ESG data for {ticker}...")
            url = f"https://financialmodelingprep.com/api/v4/esg-environmental-social-governance-data?symbol={ticker}&apikey={api_key}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            data = response.json()
            
            if data and len(data) > 0:
                esg_scores[ticker] = {
                    'environmentalScore': data[0].get('environmentalScore'),
                    'socialScore': data[0].get('socialScore'),
                    'governanceScore': data[0].get('governanceScore'),
                    'ESGScore': data[0].get('ESGScore'),
                    'dataYear': data[0].get('ratingYear')  # Added year for context
                }
                print(f"✓ Successfully retrieved ESG data for {ticker}")
            else:
                print(f"✗ No ESG data available for {ticker}")
                esg_scores[ticker] = None
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Network error for {ticker}: {e}")
            esg_scores[ticker] = None
        except ValueError as e:
            print(f"✗ JSON parsing error for {ticker}: {e}")
            esg_scores[ticker] = None
        except Exception as e:
            print(f"✗ Unexpected error for {ticker}: {e}")
            esg_scores[ticker] = None
    
    return esg_scores



def get_monthly_returns(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Download monthly returns data with improved data handling"""
    all_returns = {}
    successful_downloads = 0
    
    for ticker in tickers:
        try:
            print(f"Downloading price data for {ticker}...")
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, interval='1mo')
            
            if not data.empty and len(data) > 1:  # Need at least 2 points for returns
                prices = data['Close']
                returns = prices.pct_change().dropna()
                
                if not returns.empty:
                    all_returns[ticker] = returns
                    successful_downloads += 1
                    print(f"✓ Successfully downloaded {len(returns)} months of data for {ticker}")
                else:
                    print(f"✗ No return data calculated for {ticker}")
            else:
                print(f"✗ Insufficient data for {ticker}")
                
        except Exception as e:
            print(f"✗ Error downloading {ticker}: {e}")
    
    print(f"\nDownload summary: {successful_downloads}/{len(tickers)} tickers successful")
    
    if all_returns:
        monthly_returns = pd.DataFrame(all_returns)
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        return monthly_returns
    else:
        return pd.DataFrame()  # Return empty DataFrame if no data

def display_esg_results(esg_scores: Dict):
    """Display ESG results in a formatted way"""
    print("\n" + "="*60)
    print("ESG SCORES SUMMARY")
    print("="*60)
    
    for ticker, scores in esg_scores.items():
        if scores:
            print(f"\n{ticker}:")
            print(f"  Total ESG Score: {scores.get('ESGScore', 'N/A')}")
            print(f"  Environmental: {scores.get('environmentalScore', 'N/A')}")
            print(f"  Social: {scores.get('socialScore', 'N/A')}")
            print(f"  Governance: {scores.get('governanceScore', 'N/A')}")
            print(f"  Data Year: {scores.get('dataYear', 'N/A')}")
        else:
            print(f"\n{ticker}: No ESG data available")

if __name__ == "__main__":
    # Configuration
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    api_key = 'hpV7VxPOvfKa4RluprDtftU1JkDbVQzM'  # Consider using environment variables
    
    # Get ESG scores
    print("Fetching ESG data from Financial Modeling Prep...")
    esg_scores = get_esg_scores_fmp(tickers, api_key)
    display_esg_results(esg_scores)
    
    # Get monthly returns
    print(f"\nFetching monthly returns from {start_date} to {end_date}...")
    monthly_returns = get_monthly_returns(tickers, start_date, end_date)
    
    if not monthly_returns.empty:
        print(f"\nMonthly Returns Summary ({len(monthly_returns)} months):")
        print(monthly_returns.head())
        
        # Basic statistics
        print(f"\nReturn Statistics:")
        print(monthly_returns.describe())
    else:
        print("No return data available")
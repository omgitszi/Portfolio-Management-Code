import requests
from bs4 import BeautifulSoup
import time
import re

def get_esg_alphavantage(tickers, api_key):
    """Alpha Vantage offers ESG data in free tier"""
    esg_scores = {}
    
    for ticker in tickers:
        try:
            url = f"https://www.alphavantage.co/query?function=ESG_SCORE&symbol={ticker}&apikey={api_key}"
            print(f"Fetching ESG data for {ticker} from Alpha Vantage...")
            print(f"URL: {url}"
                  )
            response = requests.get(url)
            data = response.json()
            
            if 'ESG Score' in data:
                esg_scores[ticker] = {
                    'ESG_Score': data['ESG Score'],
                    'Environment_Score': data.get('Environment Score'),
                    'Social_Score': data.get('Social Score'),
                    'Governance_Score': data.get('Governance Score')
                }
            else:
                esg_scores[ticker] = None
                
        except Exception as e:
            print(f"Error with Alpha Vantage for {ticker}: {e}")
            esg_scores[ticker] = None
    
    return esg_scores

import requests
from bs4 import BeautifulSoup
import time

def get_esg_from_corporate_site(ticker):
    """Scrape ESG data from company sustainability pages"""
    
    # Map tickers to their ESG/sustainability pages
    esg_pages = {
        'AAPL': 'https://www.apple.com/environment/',
        'MSFT': 'https://www.microsoft.com/en-us/corporate-responsibility/sustainability',
        'GOOGL': 'https://sustainability.google/',
        'TSLA': 'https://www.tesla.com/impact',
        'JPM': 'https://www.jpmorganchase.com/impact'
    }
    
    if ticker not in esg_pages:
        return None
        
    try:
        url = esg_pages[ticker]
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text().lower()
        
        # Look for numeric scores (0-100 range typically)
        scores = re.findall(r'\b([0-9]{1,3})\b', text)
        potential_esg_scores = [int(score) for score in scores if 0 <= int(score) <= 100]
        
        # Look for ESG keywords with context
        esg_indicators = {}
        
        # Common ESG metrics to look for
        metrics_to_find = [
            'esg', 'sustainability', 'environment', 'social', 'governance',
            'carbon', 'emissions', 'diversity', 'inclusion'
        ]
        
        for metric in metrics_to_find:
            if metric in text:
                esg_indicators[metric] = True
        
        return {
            'source': 'corporate_site',
            'potential_scores_found': len(potential_esg_scores),
            'esg_indicators': esg_indicators,
            'url': url
        }
        
    except Exception as e:
        print(f"Corporate site scraping failed for {ticker}: {e}")
        return None

if __name__ == "__main__":
    # tickers = ["AAPL", "MSFT", "GOOGL"]
    # api_key = "ZHRDBBA8TZAYKV82"
    # data = get_esg_alphavantage(tickers, api_key)
    # for ticker, scores in data.items():
    #     if scores:
    #         print(f"{ticker}: {scores}")
    #     else:
    #         print(f"{ticker}: No ESG data available")

    print(get_esg_from_corporate_site("AAPL")   )
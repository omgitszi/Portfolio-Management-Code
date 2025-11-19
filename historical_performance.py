import pandas as pd
import numpy as np
import yfinance as yf
from math import sqrt

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# portfolio weights
weights = {
    "MCK": 0.191503, "EME": 0.087276, "COR": 0.142933, "WMB": 0.17186,
    "PWR": 0.022356, "CAH": 0.013037, "GS": 0.106156, "ANET": 0.054949,
    "WELL": 0.1351, "BK": 0.066388, "STX": 0.066737, "MS": -0.06333,
    "KMI": -0.08961, "AXP": -0.03913, "WDC": -0.0244, "TPR": 0.033225,
    "RJF": -0.04711, "JCI": -0.06141, "GILD": 0.164059, "AXON": 0.047495,
    "EQT": -0.01116, "WSM": 0.024327, "NDAQ": 0.028126, "NI": 0.083405,
    "CBRE": -0.10277,
}

tickers = list(weights.keys())

# date range: last 5 years
end = pd.Timestamp.today().normalize()
start = end - pd.DateOffset(years=5)

print(f"Downloading data for {len(tickers)} tickers from {start.date()} to {end.date()}...")

# Download data with auto_adjust=True for cleaner handling
data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

# Extract prices - simplified approach
if isinstance(data, pd.DataFrame) and isinstance(data.columns, pd.MultiIndex):
    prices = data['Close'] if 'Close' in data.columns else data.iloc[:, data.columns.get_level_values(1) == 'Close']
else:
    prices = data

# Handle single ticker case
if isinstance(prices, pd.Series):
    prices = prices.to_frame()

# Ensure we have a proper DataFrame and align with weights
available_tickers = [t for t in tickers if t in prices.columns]
if len(available_tickers) == 0:
    raise ValueError("No ticker data downloaded successfully")

print(f"Successfully downloaded data for {len(available_tickers)}/{len(tickers)} tickers")

# Filter prices and weights to available tickers
prices = prices[available_tickers].ffill().dropna()
filtered_weights = [weights[t] for t in available_tickers]

# Calculate daily returns
daily_returns = prices.pct_change().dropna()

# Portfolio daily returns (handles short positions correctly)
portfolio_daily_returns = daily_returns.dot(filtered_weights)

# Cumulative performance
initial_value = 1.0
cumulative_portfolio = (1 + portfolio_daily_returns).cumprod() * initial_value

# Performance metrics
total_days = len(portfolio_daily_returns)
annualized_return = (cumulative_portfolio.iloc[-1] ** (252 / total_days)) - 1
annualized_vol = portfolio_daily_returns.std() * sqrt(252)
sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

# Correct max drawdown calculation
rolling_max = cumulative_portfolio.cummax()
drawdown = (cumulative_portfolio - rolling_max) / rolling_max
max_drawdown = drawdown.min()

print("\n" + "="*50)
print("PORTFOLIO PERFORMANCE ANALYSIS")
print("="*50)
print(f"Analysis Period: {start.date()} to {end.date()}")
print(f"Trading Days: {total_days}")
print(f"Cumulative Return: {(cumulative_portfolio.iloc[-1] - 1):.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_vol:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"Maximum Drawdown: {max_drawdown:.2%}")

# S&P 500 Benchmark
print("\n" + "="*50)
print("S&P 500 BENCHMARK")
print("="*50)

sp500 = yf.download("^GSPC", start=start, end=end, auto_adjust=True, progress=False)

# Extract S&P 500 prices properly
if isinstance(sp500, pd.DataFrame) and isinstance(sp500.columns, pd.MultiIndex):
    sp500_prices = sp500['Close'] if 'Close' in sp500.columns else sp500.iloc[:, 0]
else:
    sp500_prices = sp500['Close'] if 'Close' in sp500.columns else sp500.iloc[:, 0]

# Ensure it's a Series
if isinstance(sp500_prices, pd.DataFrame):
    sp500_prices = sp500_prices.iloc[:, 0]

sp500_returns = sp500_prices.pct_change().dropna()
sp500_cumulative = (1 + sp500_returns).cumprod() * initial_value

sp500_total_days = len(sp500_returns)
sp500_annual_return = (sp500_cumulative.iloc[-1] ** (252 / sp500_total_days)) - 1
sp500_annual_vol = sp500_returns.std() * sqrt(252)

# FIXED: Convert to scalar for comparison
sp500_annual_vol_scalar = float(sp500_annual_vol)
sp500_sharpe = sp500_annual_return / sp500_annual_vol_scalar if sp500_annual_vol_scalar > 0 else 0

sp500_rolling_max = sp500_cumulative.cummax()
sp500_drawdown = (sp500_cumulative - sp500_rolling_max) / sp500_rolling_max
sp500_max_dd = sp500_drawdown.min()

print(f"Cumulative Return: {(sp500_cumulative.iloc[-1] - 1):.2%}")
print(f"Annualized Return: {sp500_annual_return:.2%}")
print(f"Annualized Volatility: {sp500_annual_vol_scalar:.2%}")
print(f"Sharpe Ratio: {sp500_sharpe:.2f}")
print(f"Maximum Drawdown: {sp500_max_dd:.2%}")

# Combined comparison - align dates
# Also fetch S&P 500 Scored & Screened Index (^SPESG) for comparison
spesg = yf.download("^SPESG", start=start, end=end, auto_adjust=True, progress=False)

# Extract SPESG prices
if isinstance(spesg, pd.DataFrame) and isinstance(spesg.columns, pd.MultiIndex):
    spesg_prices = spesg['Close'] if 'Close' in spesg.columns else spesg.iloc[:, 0]
else:
    # fallback: try to access Close column or take first column
    spesg_prices = spesg['Close'] if (isinstance(spesg, pd.DataFrame) and 'Close' in spesg.columns) else (spesg.iloc[:, 0] if isinstance(spesg, pd.DataFrame) and spesg.shape[1] > 0 else spesg)

if isinstance(spesg_prices, pd.DataFrame):
    spesg_prices = spesg_prices.iloc[:, 0]

spesg_returns = spesg_prices.pct_change().dropna()
spesg_cumulative = (1 + spesg_returns).cumprod() * initial_value

# SPESG metrics
spesg_total_days = len(spesg_returns)
if spesg_total_days > 0:
    spesg_annual_return = (spesg_cumulative.iloc[-1] ** (252 / spesg_total_days)) - 1
    spesg_annual_vol = spesg_returns.std() * sqrt(252)
    spesg_sharpe = spesg_annual_return / (float(spesg_annual_vol) if spesg_annual_vol > 0 else 1)
    spesg_rolling_max = spesg_cumulative.cummax()
    spesg_drawdown = (spesg_cumulative - spesg_rolling_max) / spesg_rolling_max
    spesg_max_dd = spesg_drawdown.min()
    print("\nS&P 500 Scored & Screened (SPESG) BENCHMARK")
    print(f"Cumulative Return: {(spesg_cumulative.iloc[-1] - 1):.2%}")
    print(f"Annualized Return: {spesg_annual_return:.2%}")
    print(f"Annualized Volatility: {float(spesg_annual_vol):.2%}")
    print(f"Sharpe Ratio: {spesg_sharpe:.2f}")
    print(f"Maximum Drawdown: {spesg_max_dd:.2%}")
else:
    spesg_cumulative = pd.Series(name='SPESG')

# Combined comparison - align dates (include SPESG)
combined = pd.DataFrame({
    'Our Portfolio': cumulative_portfolio,
    'S&P 500': sp500_cumulative,
    'SPESG': spesg_cumulative
}).dropna()

# Rebase to 1.0 at start for percent formatting
combined_rebased = combined / combined.iloc[0]

# Save results (include SPESG)
out_csv = "portfolio_vs_sp500_spesg_comparison.csv"
combined_rebased.to_csv(out_csv)
portfolio_daily_returns.to_csv("portfolio_daily_returns.csv")

# Create comparison chart
plt.figure(figsize=(12, 8))
plt.plot(combined_rebased.index, combined_rebased['Our Portfolio'], 
         label='Our Portfolio', linewidth=2, color='blue')
plt.plot(combined_rebased.index, combined_rebased['S&P 500'], 
         label='S&P 500', linewidth=2, color='gray', linestyle='--')
plt.plot(combined_rebased.index, combined_rebased['SPESG'], 
         label='S&P 500 Scored & Screened (SPESG)', linewidth=2, color='green', linestyle=':')

plt.title('Portfolio vs S&P 500 vs SPESG Performance (Last 5 Years)', fontsize=14, fontweight='bold')
plt.ylabel('Cumulative return', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
# Format y-axis as percent where 1.0 == 100%
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Add performance summary text
text_str = f'Portfolio: {annualized_return:.1%} ann. return\nS&P 500: {sp500_annual_return:.1%} ann. return\nSPESG: {spesg_annual_return:.1%} ann. return'
plt.figtext(0.02, 0.02, text_str, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.tight_layout()
plt.savefig('portfolio_vs_sp500_spesg_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nResults saved to:")
print(f"- {out_csv}")
print("- portfolio_daily_returns.csv") 
print("- portfolio_vs_sp500_spesg_comparison.png")
"""Calculate average ESG scores per ticker across files in the `ESG/` folder.

This script:
- Reads all CSV files under the `ESG/` folder.
- Normalizes ticker names by keeping only the first whitespace-separated token
  (e.g. 'AAPL UW Equity' -> 'AAPL').
- Parses the `ESG score` value to numeric and ignores non-numeric entries.
- Computes average, count and standard deviation of ESG scores per ticker.
- Writes `output/esg_avg_by_ticker.csv` with columns: `ticker`,
  `avg_esg_score`, `count`, `std`.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def compute_avg_esg(esg_dir: Path, out_dir: Path, out_name: str = 'esg_avg_by_ticker.csv') -> pd.DataFrame:
	"""Read all CSVs in `esg_dir`, compute averages and write result to `out_dir/out_name`.

	Returns the DataFrame written.
	"""
	esg_dir = Path(esg_dir)
	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	files = sorted([p for p in esg_dir.glob('*.csv') if p.is_file()])
	if not files:
		raise FileNotFoundError(f'No CSV files found in ESG folder: {esg_dir}')

	parts = []
	for f in files:
		try:
			df = pd.read_csv(f)
		except Exception as e:
			print(f'Warning: failed to read {f}: {e}')
			continue

		# Find ticker-like column
		ticker_col = None
		score_col = None
		for c in df.columns:
			low = c.lower()
			if 'ticker' in low or 'tickers' in low or 'symbol' in low:
				ticker_col = c
			if 'esg' in low and 'score' in low:
				score_col = c

		if ticker_col is None:
			# fallback to first column
			ticker_col = df.columns[0]

		if score_col is None:
			# try find any column that contains 'score'
			for c in df.columns:
				if 'score' in c.lower():
					score_col = c
					break

		if score_col is None:
			print(f'Warning: no ESG score column found in {f}; skipping')
			continue

		tmp = df[[ticker_col, score_col]].copy()
		tmp.columns = ['ticker_raw', 'esg_score_raw']

		# Normalize ticker: keep only first whitespace-separated token and uppercase
		tmp['ticker'] = (
			tmp['ticker_raw'].astype(str)
			.str.strip()
			.str.split()
			.str[0]
			.str.replace('.', '-', regex=False)
			.str.upper()
		)

		# Parse ESG score to numeric; coerce non-numeric (#N/A etc.) to NaN
		tmp['esg_score'] = pd.to_numeric(tmp['esg_score_raw'], errors='coerce')

		# Keep only rows with a valid numeric ESG score and a non-empty ticker
		tmp = tmp[tmp['ticker'].notna() & (tmp['ticker'].str.len() > 0)]
		tmp = tmp[~tmp['esg_score'].isna()].loc[:, ['ticker', 'esg_score']]

		parts.append(tmp)

	if not parts:
		raise RuntimeError('No valid ESG score rows found across ESG CSVs')

	all_df = pd.concat(parts, ignore_index=True)

	# Group by ticker and compute stats
	agg = all_df.groupby('ticker')['esg_score'].agg(['mean', 'count', 'std']).reset_index()
	agg = agg.rename(columns={'mean': 'avg_esg_score', 'count': 'count', 'std': 'std'})

	# Round numeric columns for readability
	agg['avg_esg_score'] = agg['avg_esg_score'].round(4)
	agg['std'] = agg['std'].round(4)

	out_path = out_dir / out_name
	agg.to_csv(out_path, index=False)
	print(f'Wrote {len(agg)} tickers to {out_path}')
	return agg

def process_top_esg(out_dir: Path = None, in_name: str = 'esg_avg_by_ticker.csv',
					out_name: str = 'top_esg_tickers.csv', threshold: float = 7.0) -> pd.DataFrame:
	"""Read the aggregated ESG file, keep tickers with avg_esg_score >= threshold,
	write them to out_dir/out_name and return the dataframe."""
	if out_dir is None:
		out_dir = Path(__file__).parent / 'output'
	out_dir = Path(out_dir)
	in_path = out_dir / in_name

	if not in_path.exists():
		raise FileNotFoundError(f'Input aggregate file not found: {in_path}')

	df = pd.read_csv(in_path)

	if 'avg_esg_score' not in df.columns:
		raise ValueError(f'Expected column "avg_esg_score" in {in_path}')

	# Ensure numeric comparison
	df['avg_esg_score'] = pd.to_numeric(df['avg_esg_score'], errors='coerce')

	top = df[df['avg_esg_score'].ge(threshold)].copy()
	# Sort by score descending, then ticker ascending for determinism
	if 'ticker' in top.columns:
		top = top.sort_values(by=['avg_esg_score', 'ticker'], ascending=[False, True]).reset_index(drop=True)
	else:
		top = top.sort_values(by='avg_esg_score', ascending=False).reset_index(drop=True)

	out_path = out_dir / out_name
	top.to_csv(out_path, index=False)
	print(f'Wrote {len(top)} tickers with avg_esg_score >= {threshold} to {out_path}')
	return top

if __name__ == '__main__':
	BASE = Path(__file__).parent
	ESG_DIR = BASE / 'ESG'
	OUT_DIR = BASE / 'output'
	try:
		agg = compute_avg_esg(ESG_DIR, OUT_DIR, out_name='esg_avg_by_ticker.csv')
		top = process_top_esg(OUT_DIR)
		print(f'Average ESG by ticker written to {OUT_DIR / "esg_avg_by_ticker.csv"}')
	except Exception as e:
		print(f'Error computing average ESG: {e}')

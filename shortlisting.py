from pathlib import Path
import pandas as pd
import warnings

base = Path(__file__).parent
output_dir = base / 'output'
output_dir.mkdir(exist_ok=True)

# Paths (all read/write should use output/)
file_12m = output_dir / 'sp500_top20_12m.csv'
file_6m = output_dir / 'sp500_top20_6m.csv'
file_top200 = output_dir / 'top200_esg.csv'

# Try common locations/names for the S&P risk ratings CSV in repo root
possible_risk_files = [
    base / 'SP 500 ESG Risk Ratings.csv',
    base / 'SP_500_ESG_Risk_Ratings.csv',
    base / 'sp500_esg_risk_ratings.csv',
]


def load_tickers(path: Path):
    """Load a set of tickers from a CSV-like file. Normalizes symbols (upper, replace . with -)."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    # find ticker-like column
    for col in ("ticker", "Ticker", "symbol", "Symbol"):
        if col in df.columns:
            return set(df[col].astype(str).str.strip().str.replace('.', '-', regex=False).str.upper().dropna().tolist())
    # fallback to first column
    first = df.columns[0]
    return set(df[first].astype(str).str.strip().str.replace('.', '-', regex=False).str.upper().dropna().tolist())


def build_top200_from_riskfile(risk_file: Path, out_path: Path, top_n: int = 200):
    """Read S&P risk rating CSV and select top-N ESG-rated tickers.

    Selection logic:
    - Prefer numeric 'ESG Risk Percentile' (lower is better). Values like '50th percentile' are parsed.
    - If percentile is missing, fall back to numeric 'Total ESG Risk score' (lower is better).
    - If both missing, drop the row.
    Writes a CSV with columns: ticker, Symbol, Total ESG Risk score, ESG Risk Percentile (num), rank_score
    Returns the DataFrame written.
    """
    if not risk_file.exists():
        raise FileNotFoundError(f"Risk ratings file not found: {risk_file}")

    df = pd.read_csv(risk_file)

    # Normalize column names to ease lookup
    cols = {c.lower(): c for c in df.columns}

    # Symbol column
    if 'symbol' in cols:
        col_symbol = cols['symbol']
    else:
        col_symbol = df.columns[0]

    # Find candidate columns for total score and percentile
    col_total = None
    col_percentile = None
    for name in df.columns:
        low = name.lower()
        if 'total' in low and 'esg' in low and 'score' in low:
            col_total = name
        if 'percentile' in low and 'esg' in low:
            col_percentile = name
    # Fallbacks
    if col_total is None:
        for name in df.columns:
            if 'total' in name.lower() and 'score' in name.lower():
                col_total = name
                break

    # Parse percentile to numeric (e.g. '50th percentile' -> 50)
    def parse_percentile(x):
        try:
            if pd.isna(x):
                return None
            s = str(x)
            # extract digits
            import re
            m = re.search(r"(\d+(?:\.\d+)?)", s)
            if m:
                return float(m.group(1))
        except Exception:
            return None
        return None

    df = df.copy()
    # Ensure symbol column exists and normalized
    df['__symbol'] = df[col_symbol].astype(str).str.strip().str.replace('.', '-', regex=False).str.upper()

    # Compute percentile numeric
    if col_percentile is not None:
        df['__esg_percentile_num'] = df[col_percentile].apply(parse_percentile)
    else:
        df['__esg_percentile_num'] = pd.NA

    # Total ESG risk score numeric
    if col_total is not None:
        df['__total_esg_score'] = pd.to_numeric(df[col_total], errors='coerce')
    else:
        df['__total_esg_score'] = pd.NA

    # Build a rank score where lower raw values are better. Keep the raw value
    # so that sorting ascending picks the best-rated companies first.
    def rank_score(row):
        if pd.notna(row['__esg_percentile_num']):
            return float(row['__esg_percentile_num'])
        if pd.notna(row['__total_esg_score']):
            return float(row['__total_esg_score'])
        return float('inf')

    df['__rank_score'] = df.apply(rank_score, axis=1)

    # Drop rows without any score
    df_valid = df[df['__rank_score'] != float('inf')].copy()

    if df_valid.empty:
        warnings.warn('No valid ESG scores found in risk file; top200 will be empty')

    # Sort ascending (best ESG first), take top_n
    df_valid = df_valid.sort_values('__rank_score', ascending=True).head(top_n)

    out_df = pd.DataFrame({
        'ticker': df_valid['__symbol'],
        'Symbol': df_valid[col_symbol] if col_symbol in df_valid.columns else df_valid['__symbol'],
        'Total ESG Risk score': df_valid.get('__total_esg_score'),
        'ESG Risk Percentile (num)': df_valid.get('__esg_percentile_num'),
        'rank_score': df_valid['__rank_score'],
    })

    out_df.to_csv(out_path, index=False)
    return out_df


def ensure_top200_exists():
    # If already present, load it; otherwise attempt to build from known risk files
    if file_top200.exists():
        return pd.read_csv(file_top200)

    for candidate in possible_risk_files:
        if candidate.exists():
            print(f"Found risk ratings file: {candidate}; building {file_top200} (top200)")
            return build_top200_from_riskfile(candidate, file_top200, top_n=200)

    # Not found
    warnings.warn('No risk ratings file found in repo root; top200_esg.csv not created')
    return pd.DataFrame(columns=['ticker'])


# Load sets
s12 = load_tickers(file_12m)
s6 = load_tickers(file_6m)
top200_df = ensure_top200_exists()
top200 = set()
if not top200_df.empty:
    # tolerate either 'ticker' column or first column
    if 'ticker' in top200_df.columns:
        top200 = set(top200_df['ticker'].astype(str).str.strip().str.replace('.', '-', regex=False).str.upper().dropna().tolist())
    else:
        first = top200_df.columns[0]
        top200 = set(top200_df[first].astype(str).str.strip().str.replace('.', '-', regex=False).str.upper().dropna().tolist())



# Build union of the momentum shortlists
union_top20 = sorted(s12 | s6)

# Create a DataFrame for the union and merge with top200 ESG data (if available)
df_union = pd.DataFrame({'ticker': union_top20})
df_union['ticker'] = df_union['ticker'].astype(str).str.strip().str.replace('.', '-', regex=False).str.upper()

df_overlap = pd.DataFrame(columns=['ticker'])
if not top200_df.empty and 'ticker' in top200_df.columns:
    # normalize top200 tickers as well
    top200_df = top200_df.copy()
    top200_df['ticker'] = top200_df['ticker'].astype(str).str.strip().str.replace('.', '-', regex=False).str.upper()

    # Merge to attach ESG scores and rank_score to the union tickers
    df_overlap = pd.merge(df_union, top200_df, on='ticker', how='inner')

    # If a rank_score column exists, filter and sort by it (lower is better)
    if 'rank_score' in df_overlap.columns:
        df_overlap = df_overlap[df_overlap['rank_score'].notna()].sort_values('rank_score', ascending=True)

    overlap = df_overlap['ticker'].tolist()
else:
    overlap = []

# Write outputs
out_union = output_dir / 'sp500_top20_all.csv'
out_overlap = output_dir / 'sp500_top20_esg_overlap.csv'

pd.DataFrame({'ticker': union_top20}).to_csv(out_union, index=False)

# Write a richer overlap file (with ESG scores & rank if available)
if not df_overlap.empty:
    # determine columns to write (preserve available ESG columns)
    write_cols = []
    for c in ['ticker', 'Total ESG Risk score', 'ESG Risk Percentile (num)', 'rank_score']:
        if c in df_overlap.columns:
            write_cols.append(c)
    df_overlap.to_csv(out_overlap, index=False, columns=write_cols)
else:
    pd.DataFrame({'ticker': overlap}).to_csv(out_overlap, index=False)

# Print summary
print(f"Loaded {len(s12)} tickers from {file_12m.name}")
print(f"Loaded {len(s6)} tickers from {file_6m.name}")
print(f"Union (all) size: {len(union_top20)} (written to: {out_union})")
print(f"Loaded {len(top200)} tickers from {file_top200.name}")
print(f"Overlap size: {len(overlap)} (written to: {out_overlap})")
if overlap:
    print('\nOverlapping tickers:')
    print(', '.join(overlap))
else:
    print('\nNo overlapping tickers found.')

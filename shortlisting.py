from pathlib import Path
import pandas as pd
import sys

def load_tickers(path: Path):
    if not path.exists():
        print(f"File not found: {path}")
        return set()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        return set()

    # Try common column names
    for col in ("ticker", "Ticker", "symbol", "Symbol", "SYMBOL"):
        if col in df.columns:
            s = df[col].astype(str).str.strip().str.replace('.', '-', regex=False).str.upper()
            return set(s.dropna().unique().tolist())

    # Fallback: use first column
    first = df.columns[0]
    s = df[first].astype(str).str.strip().str.replace('.', '-', regex=False).str.upper()
    return set(s.dropna().unique().tolist())


def main():
    base = Path(__file__).parent
    output_dir = base / 'output'
    output_dir.mkdir(exist_ok=True)

    file_12m = output_dir / 'sp500_top20_12m.csv'
    file_6m = output_dir / 'sp500_top20_6m.csv'

    tickers_12 = load_tickers(file_12m)
    tickers_6 = load_tickers(file_6m)

    if not tickers_12:
        print(f"No tickers loaded from {file_12m}")
    if not tickers_6:
        print(f"No tickers loaded from {file_6m}")

    # UNION: all tickers that appear in either top-20 list
    all_tickers = sorted(tickers_12 | tickers_6)

    out_csv = output_dir / 'sp500_top20_all.csv'

    # Write CSV
    try:
        pd.DataFrame({'ticker': all_tickers}).to_csv(out_csv, index=False)
        print(f"Wrote {len(all_tickers)} tickers to {out_csv}")
    except Exception as e:
        print(f"Failed to write CSV {out_csv}: {e}")

    # Print summary
    print('\nSummary:')
    print(f'  Top 20% 12m: {len(tickers_12)} tickers')
    print(f'  Top 20% 6m:  {len(tickers_6)} tickers')
    print(f'  Union (all): {len(all_tickers)} tickers')
    if all_tickers:
        print('  Tickers:', ', '.join(all_tickers))


if __name__ == '__main__':
    main()

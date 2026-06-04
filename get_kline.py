from pathlib import Path
from typing import List
import yfinance as yf
import pandas as pd

def get_kline(
    tickers: List[str],
    output_dir: str = "selected_kline",
    start_date: str = "2010-01-01",
) -> List[str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    for ticker in tickers:
        ticker = ticker.strip().upper()

        if not ticker:
            continue

        print(f"Downloading/updating {ticker}...")

        df = yf.download(
            ticker,
            start=start_date,
            progress=False,
            auto_adjust=False,
        )

        if df.empty:
            print(f"Warning: no data found for {ticker}")
            continue

        # Fix possible multi-index columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Keep only needed columns
        df = df[["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]]

        file_path = output_path / f"{ticker}.csv"
        df.to_csv(file_path, index=False)
        
        saved_files.append(str(file_path))
        print(f"Saved: {file_path}")

    return saved_files
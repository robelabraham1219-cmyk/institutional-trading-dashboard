"""collect.py - append recent Yahoo bars to data/<TICKER>_<interval>.csv (UTC ISO timestamp, OHLC).

Run by the GitHub Actions workflow; free data only. Analytical tool, not financial advice.
"""
import os
import time

import pandas as pd
import yfinance as yf

TICKERS = ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "USDCAD=X", "USDSEK=X", "USDCHF=X", "DX-Y.NYB"]
INTERVALS = {"30m": 59, "1h": 729}      # Yahoo limits in days (first run uses the full limit)
RECENT_DAYS = {"30m": 20, "1h": 30}     # later runs only fetch a recent window
MAX_ROWS = 60000
OUT_DIR = "data"
COLS = ["open", "high", "low", "close"]


def download(ticker: str, interval: str, days: int) -> pd.DataFrame:
    """Download recent bars with retries; empty frame on failure."""
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    for k in range(3):
        try:
            h = yf.Ticker(ticker).history(start=start, interval=interval, auto_adjust=False)
            if h is not None and len(h):
                h.columns = [str(c).lower() for c in h.columns]
                h = h[COLS].copy()
                idx = pd.DatetimeIndex(h.index)
                h.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
                h = h.apply(pd.to_numeric, errors="coerce").dropna()
                return h[(h > 0).all(axis=1)]
        except Exception as e:  # keep going with the other tickers
            print(f"{ticker} {interval}: {type(e).__name__}: {e}")
        time.sleep(2 * (2 ** k))
    return pd.DataFrame(columns=COLS)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for t in TICKERS:
        for iv, limit in INTERVALS.items():
            path = os.path.join(OUT_DIR, f"{t}_{iv}.csv")
            old = pd.DataFrame(columns=COLS)
            if os.path.exists(path):
                try:
                    old = pd.read_csv(path, index_col=0, parse_dates=True)
                    old.index = pd.to_datetime(old.index, utc=True)
                except Exception as e:
                    print(f"{path}: unreadable ({e}); rebuilding")
                    old = pd.DataFrame(columns=COLS)
            new = download(t, iv, RECENT_DAYS[iv] if len(old) else limit)
            if len(new) == 0 and len(old) == 0:
                print(f"{t} {iv}: no data")
                continue
            m = pd.concat([old, new])
            m = m[~m.index.duplicated(keep="last")].sort_index().tail(MAX_ROWS)
            m.index.name = "timestamp"
            m.index = m.index.strftime("%Y-%m-%dT%H:%M:%SZ")
            m.index.name = "timestamp"
            m[COLS].to_csv(path)
            print(f"{t} {iv}: {len(m)} rows")


if __name__ == "__main__":
    main()

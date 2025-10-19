#this gonna be used to get histrical daily close prices using yfinance and local storing
import yfinance as yf
import pandas as pd 
import os 

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok = True)


def fetch_multi_price_series(tickers, start="2016-01-01", end="2024-01-01", save=True):
    """
    Fetches daily close price DataFrame with columns=tickers, index=Datetime.
    Returns pd.DataFrame.
    """
    # Use yfinance.download with group_by='column' and auto-adjusted close to avoid splits/dividends issues
    df = yf.download(tickers, start=start, end=end, progress=False, group_by="ticker", auto_adjust=True)
    # If yfinance returns a multiindex columns (tickers x fields), extract Close for each.
    if isinstance(df.columns, pd.MultiIndex):
        closes = pd.DataFrame({t: df[t]["Close"] for t in tickers})
    else:
        # single ticker case (string)
        closes = df["Close"].to_frame(name=tickers[0]) if len(tickers) == 1 else df["Close"]
    closes = closes.dropna(how="all").ffill().dropna()
    if save:
        closes.to_csv(os.path.join(DATA_DIR, f"{'_'.join(tickers)}_closes.csv"))
        print(f"Saved {len(closes)} rows x {len(closes.columns)} cols to {DATA_DIR}/{'_'.join(tickers)}_closes.csv")
    return closes



if __name__ == "__main__":
        TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]
        df = fetch_multi_price_series(TICKERS, "2016-01-01", "2024-01-01")
        print(df.head())


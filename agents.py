import numpy as np 
import pandas as pd 


#creating three agents:
#Momentum: buys if price is going up, sells if price is going down
#Mean Reversion: buys if price is below rolling mean (which means price is going down), sells if price is going up
#Market Maker: buys and sells randomly in small amounts, earns money from the spread (simplisitc liquidity provider: makes sure there's always a buyer and seller in the market)

#these classes produce desired weights per asset at each decision step.
class MomentumAgent:
    def __init__(self, tickers, short_window=10, long_window= 50, eps = 1e-6):
        self.tickers = list(tickers)
        self.short = short_window
        self.long = long_window
        self.eps = eps

    #return: numpy array of nonnegative weights summing to 1 (len = n_assets)
    #these weights are used to determine the agent's actions
    def act(self, price_df_window):
        #this function takes in a window of price data (a dataframe) and returns an action for each ticker
        ma_short = price_df_window.rolling(self.short).mean().iloc[-1]
        ma_long = price_df_window.rolling(self.long).mean().iloc[-1]
         
        scores = (ma_short - ma_long).fillna(0.0).clip(lower=0.0)
        arr = scores.values.astype(float)

        
        if arr.sum() <= self.eps:
            return np.zeros_like(arr)
        return arr / (arr.sum() + self.eps)


class MeanReversionAgent:
    def __init__(self, tickers, window=20, threshold=0.02, eps = 1e-9):
        self.tickers = list(tickers)
        self.window = window
        self.threshold = threshold
        self.eps = eps

    def act(self, price_df_window):
        ma = price_df_window.rolling(self.window).mean().iloc[-1]
        price = price_df_window.iloc[-1]
        diff = (price - ma) / (ma + self.eps)

        #only buy when price < ma threshold 
        #scores caluclate by how much below the ma the price is
        #-diff-self.threshold means if price is above ma by threshold, score is negative, so clip to 0
        scores = np.maximum(-diff - self.threshold, 0.0)
        arr = np.nan_to_num(scores.values.astype(float))
        if arr.sum() == 0:
            return np.zeros_like(arr)
        return arr / (arr.sum() + self.eps)
    

#kinda like a dealer/jsut to make sure 
class MarketMakerAgent:
    def __init__(self, tickers, spread=0.002, inventory_limit=10, unit=1.0):
        self.tickers = list(tickers)
        self.spread = spread
        self.inventory = {t: 0.0 for t in tickers}
        self.inventory_limit = inventory_limit
        self.unit = unit

    def act(self, price_df_window):
        """
        Very simplified: market maker supplies small positions in assets that moved recently.
        Output is small positive weights to indicate small exposure; not full normalization expected.
        We'll create scores based on short-term volatility and returns sign.
        """
        returns = price_df_window.pct_change().iloc[-5:].std().fillna(0.0)
        recent_ret = price_df_window.pct_change().iloc[-3:].sum().fillna(0.0)
        # if asset moved up, offer to sell -> negative desired weight; but our env expects nonnegative weights.
        # So MarketMaker provides small long weights on lower-vol assets as a liquidity provider (very simplified).
        scores = (1.0 / (1.0 + returns)).values
        # normalize to very small sum
        arr = scores / (scores.sum() + 1e-9) * 0.05
        return arr





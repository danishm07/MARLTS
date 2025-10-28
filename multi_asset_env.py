#multi_asset_env.py
import numpy as np
import pandas as pd
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo import  ParallelEnv 
from gymnasium import spaces


class MultiAssetMarketEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "multi_asset_market_v0"}

    def __init__(self, price_df: pd.DataFrame, agents = None, episode_length = 252, window_size = 20,initial_cash = 100_000, commission =1e-3, impact_coeff=1e-6, liquidity_scale = 1e6, coop_weight = 0.05, comp_weight =0.1, render_mode = "none"):
        super().__init__()
        assert isinstance(price_df, pd.DataFrame), "price_df must be a pandas DataFrame w tickers as columns"
        self.price_df = price_df.reset_index(drop=True).astype(float)
        self.tickers = list(price_df.columns)
        self.n_assets = len(self.tickers)

        self.episode_length = episode_length
        self.window_size = window_size
        self.initial_cash = float(initial_cash)
        self.commission = float(commission) #proportional commission per trade
        self.impact_coeff = float(impact_coeff) #price impact coefficient
        self.liquidity_scale = float(liquidity_scale) #scale of liquidity (higher means more liquid, less price impact)
        self.render_mode = render_mode

        self.coop_weight = float(coop_weight) #weight for cooperative reward component
        self.comp_weight = float(comp_weight) #weight for competitive reward component

        #momentum, mean reversion, market making, rl trader
        #mean reversion is the opposite of momentum, and market making works by basiclly buying low and selling high around current price 
        # (liquidity to market)
        self.possible_agents = agents or ["mom_0", "mean_0", "mm_0", "rl_trader_0"]
        self.agents = list(self.possible_agents)


        #observations & actions

        #window_size -1 because we want to see previous window_size -1 days + today + cash
        #+1 for cash position (refers to cash available to trade)
        obs_len = (self.window_size - 1) * self.n_assets +  self.n_assets +self.n_assets + 1

        #infinite observation spaces 
        #observation spaces are a vector of length obs_len, which includes historical prices for all assets and current cash position
        self.observation_spaces = {a: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32) for a in self.possible_agents}

        #action spaces are a vector of length n_assets, each value between 0 and 1, representing desired portfolio weights for each asset
        #these weights are then converted to buy/sell/hold actions based on current portfolio
        self.action_spaces = {a: spaces.Box(low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32) for a in self.possible_agents}

        #internal state: 
        self._start = 0
        self._cur_step = 0
        self._agent_idx = 0
        self.cash = {}
        self.holdings = {}
        self.rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}


    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        try:
            if seed is not None:
                np.random.seed(seed)

            # Ensure enough data for an episode
            max_start = len(self.price_df) - self.episode_length - 1
            if max_start <= self.window_size:
                raise RuntimeError(
                    f"Price series too short for requested episode_length/window. "
                    f"max_start={max_start}, window_size={self.window_size}"
                )

            # Random starting point
            self._start = np.random.randint(self.window_size, max_start)
            self._cur_step = 0
            self._agent_idx = 0
            self.agents = list(self.possible_agents)

            # Initialize agent state
            self.rewards = {a: 0.0 for a in self.agents}
            self.terminations = {a: False for a in self.agents}
            self.truncations = {a: False for a in self.agents}
            self.infos = {a: {} for a in self.agents}
            self.cash = {a: float(self.initial_cash) for a in self.agents}
            self.holdings = {a: np.zeros(self.n_assets, dtype=float) for a in self.agents}

            # Generate initial observations
            observations = {agent: self._get_obs(agent) for agent in self.agents}
            initial_prices = self._get_price(0)
            infos = {
                agent: {
                    "start_step": self._start,
                    "portfolio_value": self.cash[agent] + np.sum(self.holdings[agent] * initial_prices),
                    "allocation": self.holdings[agent].copy(),
                } 
                for agent in self.agents
            }
            return observations, infos
        except Exception as e:
            print("Error in reset:", e)
            raise e
    

    

    #returns the observation for the given agent
    #this refers to the state of the environment as seen by the agent
    #for example, agents sees: historical prices for all assets, current holdings, and cash position
    #in vector form this would look like: [r1_t-window, r1_t-window+1, ..., r1_t, r2_t-window, ..., rN_t, holdings_1, ..., holdings_N, cash]
    # multi_asset_env.py -> _get_obs method

    def _get_obs(self, agent):
        """Return the observation for a given agent."""
        idx = self._cur_step
        rsi_window = 14
        # Ensure we have enough data for RSI + the observation window
        start_idx = max(0, self._start + idx - (self.window_size - 1) - rsi_window)
        end_idx = self._start + idx

        price_window_df = self.price_df.iloc[start_idx : end_idx + 1]

        # 1. Calculate Log Returns for the observation window
        log_returns_window = np.log(price_window_df / price_window_df.shift(1)).iloc[-(self.window_size - 1) :]
        returns_flat = log_returns_window.fillna(0.0).values.flatten().astype(np.float32)

        # 2. Calculate RSI (Relative Strength Index)
        #RSI here indicates momentum of price movements
        delta = price_window_df.diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=rsi_window - 1, adjust=False).mean()
        ema_down = down.ewm(com=rsi_window - 1, adjust=False).mean()
        rs = ema_up / (ema_down + 1e-9)
        rsi = 100 - (100 / (1 + rs))

        
        # Fill any NaN values with the neutral RSI value of 50.0 before normalizing.
        rsi = rsi.fillna(50.0)
        
        rsi_latest = rsi.iloc[-1].values.astype(np.float32)
        rsi_norm = (rsi_latest - 50) / 50.0

        # 3. Normalize Holdings
        holdings_vector = self.holdings.get(agent, np.zeros(self.n_assets, dtype=float))
        norm = np.linalg.norm(holdings_vector)
        holdings_norm = (holdings_vector / (norm + 1e-9)).astype(np.float32)

        # 4. Normalize Cash
        cash_value = self.cash.get(agent, self.initial_cash)
        cash_norm = np.array([cash_value / (self.initial_cash * 10.0)], dtype=np.float32)

        # 5. Concatenate final observation vector
        obs = np.concatenate([returns_flat, rsi_norm, holdings_norm, cash_norm]).astype(np.float32)

        # Sanity check: observation length
        expected_len = (self.window_size - 1) * self.n_assets + self.n_assets + self.n_assets + 1
        if len(obs) != expected_len:
            padding = np.zeros(expected_len - len(obs), dtype=np.float32)
            obs = np.concatenate([padding, obs])

        return obs


    def _get_price(self, step_idx):
        """Get current prices at the given step."""
        idx = self._start + step_idx
        if idx >= len(self.price_df):
            idx = len(self.price_df) - 1
        return self.price_df.iloc[idx].values.astype(float)

    def observe(self, agent):
        """Expose the observation for external use (like PPO)."""
        return self._get_obs(agent)

    def step(self, actions):
        """
        Takes an action for all agents and advances the environment by one timestep.
        """
        # In a ParallelEnv, the `actions` parameter is a dictionary
        # mapping agent IDs to their chosen actions.
        assert isinstance(actions, dict) and all(a in self.agents for a in actions), \
            "Expected actions to be a dictionary for all agents."
        
        # Get the previous prices for calculations. This assumes prices are for the current step
        # before any trades are executed.
        prev_prices = self._get_price(self._cur_step)
        
        # Reset internal state dictionaries to be ready for the new step.
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {}  # Initialize empty, will populate after calculating all_vals

        net_notional = np.zeros(self.n_assets, dtype=float)
        trades = {}
        
        # Calculate trades for all agents from the `actions` dictionary
        for a in self.agents:
            act = np.asarray(actions.get(a, np.zeros(self.n_assets)), dtype=float).clip(min=0.0)
            s = act.sum()
            
            # Calculate weights based on the actions, normalize if necessary
            weights = act / (s + 1e-9) if s > 0 else np.zeros_like(act)
            
            total_val = self.cash[a] + np.sum(self.holdings[a] * prev_prices)
            target_val = weights * total_val
            current_val = self.holdings[a] * prev_prices
            dollar_diff = target_val - current_val
            
            # Calculate quantity to trade for each asset
            qty = np.where(prev_prices > 0, dollar_diff / prev_prices, 0.0)
            
            trades[a] = qty
            net_notional += dollar_diff
            
        # Apply market impact to the prices
        avg_price = np.maximum(prev_prices, 1e-6)
        liquidity = avg_price * self.liquidity_scale
        impact = np.clip(1.0 + (self.impact_coeff * (net_notional / (liquidity + 1e-9))), 0.95, 1.05)
        exec_prices = prev_prices * impact
        
        # Execute trades and update agent portfolios
        for a in self.agents: 
            qty = trades[a]
            buy_qty = np.maximum(qty, 0.0)
            sell_qty = -np.minimum(qty, 0.0)
            
            # Recalculate costs based on the impacted execution prices
            buy_cost = np.sum(buy_qty * exec_prices * (1.0 + self.commission))
            sell_proceeds = np.sum(sell_qty * exec_prices * (1.0 - self.commission))
            
            self.holdings[a] += (buy_qty - sell_qty)
            self.cash[a] -= buy_cost
            self.cash[a] += sell_proceeds
            
            if self.cash[a] < -1e6:
                self.cash[a] = -1e6
                
        # Advance the environment to the next timestep
        self._cur_step += 1
        
        # Check for episode termination
        terminated_all = self._cur_step >= self.episode_length
        
        new_prices = self._get_price(self._cur_step) if not terminated_all else prev_prices
        
        # Compute rewards for all agents
        all_vals = {a: self.cash[a] + np.sum(self.holdings[a] * new_prices) for a in self.agents}
        mean_val = np.mean(list(all_vals.values()))
        std_val = np.std(list(all_vals.values())) + 1e-9
        
        # Calculate rewards AND populate infos in the same loop
        for a in self.agents:
            base = all_vals[a] - (self.cash[a] + np.sum(self.holdings[a] * prev_prices))
            coop_bonus = self.coop_weight * (1.0 / (1.0 + std_val))
            comp_bonus = self.comp_weight * ((mean_val - all_vals[a]) / (std_val + 1e-9))
            self.rewards[a] = base + coop_bonus - comp_bonus
            
            # Populate infos with portfolio value and allocation
            self.infos[a] = {
                "portfolio_value": all_vals[a],
                "allocation": self.holdings[a].copy(),
            }
        
        # Prepare the return values. ParallelEnv requires dictionaries for all agents.
        obs = {a: self._get_obs(a) for a in self.agents}
        rewards = {a: self.rewards[a] for a in self.agents}
        terminateds = {a: terminated_all for a in self.agents}
        truncateds = {a: False for a in self.agents}
        
        return obs, rewards, terminateds, truncateds, self.infos
    
    def render(self, mode="human"):
        print(f"Step {self._cur_step}/{self.episode_length}")

def make_env(price_df, render_mode="human", **kwargs):
    env = MultiAssetMarketEnv(price_df, render_mode=render_mode, **kwargs)
    return env
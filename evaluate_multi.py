"""
Evaluation Script for Multi-Agent Market Environment
---------------------------------------------------
This script evaluates a trained PPO agent against rule-based agents.
It loads a PPO checkpoint, runs a full episode, and prints performance metrics.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from multi_asset_env import make_env
from data_loader import fetch_multi_price_series
from agents import MomentumAgent, MeanReversionAgent, MarketMakerAgent
from utils import compute_metrics, plot_portfolios_multi, plot_allocations_stacked
import gymnasium as gym


CHECKPOINT_DIR = "checkpoints_multi_2"
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]
AGENTS = ["mom_0", "mean_0", "mm_0", "rl_trader_0"]
EPISODE_LENGTH = 252
WINDOW_SIZE = 20

# ----------------------------------------------------
# Helper Functions
# ----------------------------------------------------

def act_wrapper(agent, obs_array, tickers, window_size):
    """
    Reconstructs price data from log returns and
    converts it to a DataFrame to call the agent's act method.
    """
    num_tickers = len(tickers)
    
    # Slice the array to get only the historical log returns.
    # The last price is not available in this part of the observation.
    historical_returns_flat = obs_array[:(window_size - 1) * num_tickers]
    
    # Reshape into a 2D array of (history_length, num_tickers)
    obs_2d_log_returns = historical_returns_flat.reshape(window_size - 1, num_tickers)
    
    # Convert log returns back to a price series for the agent.
    # This is a critical step. We assume a starting price of 1 for simplicity.
    price_series = np.exp(np.cumsum(obs_2d_log_returns, axis=0))
    
    # The `pct_change` calculation in your agents relies on the full price history,
    # so we'll prepend a starting row of 1s.
    start_prices = np.ones((1, num_tickers))
    price_series = np.vstack([start_prices, price_series])
    
    # Create the DataFrame for the agent
    price_df_window = pd.DataFrame(price_series, columns=tickers)
    
    return agent.act(price_df_window)

def find_latest_checkpoint(checkpoint_root):
    """Finds the most recent checkpoint and returns its absolute path."""
    paths = glob.glob(os.path.join(checkpoint_root, "**/checkpoint_*"), recursive=True)
    if not paths:
        raise RuntimeError(f"No checkpoint directories found under {checkpoint_root}")
    latest = sorted(paths, key=os.path.getmtime)[-1]
    return os.path.abspath(latest)

def env_creator(env_config):
    """Creates a ParallelPettingZooEnv instance."""
    price_df = env_config.get("price_df")
    episode_length = env_config.get("episode_length", EPISODE_LENGTH)
    window_size = env_config.get("window_size", WINDOW_SIZE)
    
    base_env = make_env(
        price_df=price_df,
        agents=AGENTS,
        episode_length=episode_length,
        window_size=window_size
    )
    return ParallelPettingZooEnv(base_env)

# ----------------------------------------------------
# Main Evaluation Function
# ----------------------------------------------------

def run_eval(ckpt_path, price_df, episode_length=EPISODE_LENGTH, window_size=WINDOW_SIZE, seed=42):
    """
    Runs a single evaluation episode using the trained and rule-based agents.
    """
    ray.init(ignore_reinit_error=True, include_dashboard=False, logging_level="ERROR")

    # 1. Register the environment
    tune.register_env("eval_multi_asset_market_env", env_creator)
    
    # 2. Re-create the full config object to be consistent with training
    temp_env = env_creator({"price_df": price_df, "episode_length": episode_length, "window_size": window_size})
    obs_space = temp_env.observation_space["rl_trader_0"]
    act_space = temp_env.action_space["rl_trader_0"]
    temp_env.close()

    policies = {
        "policy_mom": (PPOTorchPolicy, obs_space, act_space, {}),
        "policy_mean": (PPOTorchPolicy, obs_space, act_space, {}),
        "policy_mm": (PPOTorchPolicy, obs_space, act_space, {}),
        "policy_rl": (PPOTorchPolicy, obs_space, act_space, {}),
    }

    config = PPOConfig()
    config = config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    config = config.environment(
        env="eval_multi_asset_market_env",
        env_config={"price_df": price_df, "episode_length": episode_length, "window_size": window_size}
    )
    config = config.framework("torch")
    config = config.env_runners(num_env_runners=0)
    config = config.training(
        lr=3e-4,
        train_batch_size=4000,
        num_sgd_iter=10,
        model={"fcnet_hiddens": [256, 256]}
    )
    config = config.multi_agent(
        policies=policies,
        policy_mapping_fn=lambda agent_id, **kwargs: (
            "policy_mom" if "mom" in agent_id else
            "policy_mean" if "mean" in agent_id else
            "policy_mm" if "mm" in agent_id else
            "policy_rl"
        ),
        policies_to_train=["policy_rl"],
    )
    config = config.resources(num_gpus=0)
    
    # 3. Instantiate the algorithm and restore the checkpoint
    algo = Algorithm(config)
    algo.restore(ckpt_path)

    # 4. Prepare agents 
    mom_agent = MomentumAgent(TICKERS)
    mean_agent = MeanReversionAgent(TICKERS)
    mm_agent = MarketMakerAgent(TICKERS)

    env = env_creator({"price_df": price_df, "episode_length": episode_length, "window_size": window_size})
    obs, infos = env.reset(seed=seed)

    
    start_step = None
    
    # Method 1: Try to get from infos (if env provides it)
    if AGENTS[0] in infos and "start_step" in infos[AGENTS[0]]:
        start_step = infos[AGENTS[0]]["start_step"]
        print(f"✓ Got start_step from infos: {start_step}")
    
    # Method 2: Access through the wrapper chain
    if start_step is None:
        try:
            # ParallelPettingZooEnv stores the base env in .env attribute
            start_step = env.env._start
            print(f"✓ Got start_step from env.env._start: {start_step}")
        except AttributeError:
            
            try:
                start_step = env.unwrapped._start
                print(f"✓ Got start_step from env.unwrapped._start: {start_step}")
            except AttributeError:
                # Last resort: use window_size as fallback
                start_step = window_size
                print(f"⚠ Warning: Could not access _start attribute, using window_size={window_size} as fallback")
    

    portfolios = {a: [] for a in AGENTS}
    allocations = {a: [] for a in AGENTS}

    for a in AGENTS:
        pv = infos.get(a, {}).get("portfolio_value", 100_000)
        portfolios[a].append(pv)
        allocations[a].append(np.zeros(len(TICKERS), dtype=np.float32))

    terminations = {a: False for a in env.agents}
    step_count = 0

    pbar = tqdm(total=episode_length, desc="Evaluation Steps", unit="step")

    while not all(terminations.values()):
        actions = {}
        for a in env.agents:
            if terminations.get(a, False):
                continue
            
            obs_a = obs[a]
            if "rl_trader" in a:
                actions[a] = algo.compute_single_action(obs_a, policy_id="policy_rl", explore=False)
            elif "mom" in a:
                actions[a] = act_wrapper(mom_agent, obs_a, TICKERS, window_size)
            elif "mean" in a:
                actions[a] = act_wrapper(mean_agent, obs_a, TICKERS, window_size)
            elif "mm" in a:
                actions[a] = act_wrapper(mm_agent, obs_a, TICKERS, window_size)

        obs, rewards, terminateds, truncateds, infos = env.step(actions)

        for a in AGENTS:
            pv = infos.get(a, {}).get("portfolio_value", portfolios[a][-1])
            alloc = infos.get(a, {}).get("allocation", np.zeros(len(TICKERS)))
            portfolios[a].append(pv)
            if isinstance(alloc, list):
                alloc = np.array(alloc)
            allocations[a].append(alloc.astype(np.float32))

        terminations = {a: terminateds.get(a, False) or truncateds.get(a, False) for a in AGENTS}
        step_count += 1
        pbar.update(1)

    pbar.close()
    algo.stop()
    ray.shutdown()
    
    # Return the captured start_step (no longer accessing env here)
    return portfolios, allocations, start_step

# ----------------------------------------------------
# Main execution
# ----------------------------------------------------

if __name__ == "__main__":
    price_df = fetch_multi_price_series(TICKERS, "2016-01-01", "2024-01-01", save=False)
    ckpt = find_latest_checkpoint(CHECKPOINT_DIR)
    print("Using checkpoint:", ckpt)

    # FIXED: Updated function signature (removed env from return)
    portfolios, allocations, start = run_eval(ckpt, price_df)

    for name, pv in portfolios.items():
        metrics = compute_metrics(pv)
        print(f"Agent {name}: {metrics}")

    
    nsteps = len(next(iter(portfolios.values())))
    price_slice = price_df.iloc[start:start + nsteps]

    plot_portfolios_multi(price_slice, portfolios, outdir="plots_2")
    plot_allocations_stacked(allocations, price_slice.index, outdir="plots_2")
    print("Evaluation completed.")
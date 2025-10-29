#train_multi.py
import os 
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from data_loader import fetch_multi_price_series
from multi_asset_env import make_env
import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import CLIReporter
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import numpy as np


from pettingzoo.utils import parallel_to_aec, ParallelEnv

CHECKPOINT_DIR = "/Users/danish/Desktop/01_Projects/MARLTS/checkpoints_multi_2"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

class CustomMetrics(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """
        Called at the end of each episode.
        Logs per-agent cumulative reward and portfolio value as float32.
        """
        for agent_id, rewards in episode.agent_rewards.items():
            # Ensure rewards is iterable
            if np.isscalar(rewards):
                rewards = np.array([rewards], dtype=np.float32)
            else:
                rewards = np.array(rewards, dtype=np.float32)
            
            # Total reward per agent
            total_reward = np.float32(rewards.sum())
            episode.custom_metrics[f"{agent_id}_total_reward"] = total_reward

            # Log final portfolio value if available in infos
            last_info = episode.last_info_for(agent_id)
            if last_info and "portfolio_value" in last_info:
                episode.custom_metrics[f"{agent_id}_portfolio_value"] = np.float32(last_info["portfolio_value"])
    

def env_creator_from_df(price_df, agents, episode_length = 252, window_size = 20, commission = 1e-3):
    def _creator(env_config):
        cfg = {
            "price_df": price_df,
            "agents": agents,
            "episode_length": env_config.get("episode_length", episode_length),
            "window_size": env_config.get("window_size", window_size),
            "initial_cash": env_config.get("initial_cash", 100_000.0),
            "commission": env_config.get("commission", commission),
        }

        #not usin cfg so that extra params can be passed if needed

        base_env = make_env(
            price_df=price_df, 
            agents=agents, 
            episode_length=env_config.get("episode_length", episode_length), 
            window_size=env_config.get("window_size", window_size), 
            commission=env_config.get("commission", commission), 
            initial_cash=env_config.get("initial_cash", 100_000.0),
            **env_config
        )
        
        # Wrap properly for RLlib
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        return ParallelPettingZooEnv(base_env)
        
    return _creator


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    #print(f"Policy mapping function called with agent_id: {agent_id}, kwargs: {kwargs}")
    #agent names mathced to policy names in PettingZoo env
    if "mom" in agent_id:
        return "policy_mom"
    elif "mean" in agent_id:
        return "policy_mean"
    elif "mm" in agent_id:
        return "policy_mm"
    else:
        return "policy_rl"
    
    

def main():
    TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA"]
    price_df = fetch_multi_price_series(TICKERS, "2016-01-01", "2024-01-01", save=False)
    agents = ["mom_0", "mean_0", "mm_0", "rl_trader_0"]

    #initializing ray and RLlib
    #Ray is a distributed computing framework that allows for parallel and distributed processing
    #Rllib is a library built on top of Ray for reinforcement learning
    ray.init(ignore_reinit_error=True, logging_level="ERROR", log_to_driver=True)
    env_creator = env_creator_from_df(price_df, agents)
    tune.register_env("multi_asset_market_env", env_creator)

    
    #temp
    #what are spaces?: observation space is the space of all possible observations that an agent can receive from the environment
    temp_env = env_creator({})
    obs_space = temp_env.observation_space["rl_trader_0"]
    act_space = temp_env.action_space["rl_trader_0"]
    temp_env.close()

    #configuring the PPO trainer, policies
    policies = {
        "policy_mom": (None, obs_space, act_space, {}),
        "policy_mean": (None, obs_space, act_space, {}),
        "policy_mm": (None, obs_space, act_space, {}),
        "policy_rl": (None, obs_space, act_space, {}),
    }
    
    config = PPOConfig()
    config = config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )


    

    
    config = config.environment(
        env="multi_asset_market_env",
        env_config = {
            "coop_weight": 0.01,
            "comp_weight": 0.5
        })
    config = config.callbacks(CustomMetrics)
    config = config.env_runners(
        num_env_runners=1, 
        rollout_fragment_length=200,
        batch_mode="truncate_episodes",
    )
    config = config.training(
        train_batch_size=4000,
        lr=3e-4,
        #sgd_minibatch_size=256,
        num_sgd_iter=10,
        #entropy for exploraton stuff
        entropy_coeff = 0.01,
        #use generalized advantage estiamtiaon for staility
        use_gae = True,
        lambda_ = 0.95,
        )
    config = config.framework(framework="torch")
    config = config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        #policies_to_train=["policy_rl"],
        policies_to_train=["policy_rl"],
    )
    config = config.resources(num_gpus=0)
    config = config.training(model={
        "fcnet_hiddens": [256, 256],
    })

    stop = {"training_iteration": 100}

    reporter = CLIReporter(
    metric_columns=["episode_reward_mean", "training_iteration"]
)

    results = tune.run(
        "PPO",
        config=config.to_dict(),
        stop=stop,
        storage_path=CHECKPOINT_DIR,
        checkpoint_at_end=True,
        progress_reporter=reporter,
    )

    print("Training Completed. Checkpoints Saved to:", CHECKPOINT_DIR)
    ray.shutdown()

if __name__ == "__main__":
    main()


#tensorboard --logdir .../MARLTS/checkpoints_multi_2
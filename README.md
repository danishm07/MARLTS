Multi-Agent Reinforcement Learning Trading System (MARLTS)
-------------------------------------------------------------

This project implements a sophisticated Multi-Agent Reinforcement Learning (MARL) environment to train an autonomous trading agent. The agent learns to develop a profitable strategy not against static historical data, but by actively competing against a dynamic set of heuristic agents (Momentum, Mean-Reversion, and Market-Maker) in a realistic market simulation.

The core of the project is a Proximal Policy Optimization (PPO) agent, trained using the Ray RLlib framework, that learns to navigate a custom financial market built with the PettingZoo library.
------------------------------------------------------------------------------------

📈 Results: Learned Performance
After training, the RL agent successfully developed a unique and superior trading strategy, consistently outperforming all three heuristic benchmarks in the evaluation period. The agent learned to balance aggressive, profit-seeking behavior with effective risk management.

Final evaluation showing the RL Trader (red) achieving the highest portfolio value.

✨ Key Features
Custom MARL Environment: A financial market simulation built from scratch using PettingZoo, featuring multiple interacting agents.

Market Realism: The environment models crucial real-world factors like transaction costs and market price impact, where the collective trading volume of all agents affects execution prices.

Advanced Agent AI: The primary agent uses Proximal Policy Optimization (PPO), a state-of-the-art reinforcement learning algorithm, implemented via Ray RLlib.

Sophisticated Reward Shaping: The agent's learning is guided by a nuanced reward function that balances three objectives: raw profit, market stability (cooperative), and outperforming peers (competitive).

Engineered Features: The agent's perception is enhanced with quantitative indicators like the Relative Strength Index (RSI) to enable more informed decision-making.

------------------------------------------------------------------------------------

🛠️ How It Works
The system is composed of three main parts:

The Environment (multi_asset_env.py): This is the simulated stock market. It manages the price data, agent portfolios, and the rules of interaction. At each step, it takes actions from all agents, calculates the market impact of their combined trades, executes orders, and returns new observations and a shaped reward to each agent.

The Agents (agents.py & PPO):

Heuristic Agents: Simple, rule-based bots (Momentum, Mean-Reversion, Market-Maker) that act as predictable competitors.

RL Agent: A PPO-based agent with an Actor-Critic neural network architecture. It learns its strategy from scratch through trial and error, optimizing its behavior based on the rewards received from the environment.

The Training & Evaluation Scripts (train_multi.py, evaluate.py):

train_multi.py uses Ray RLlib to orchestrate the learning process. It runs thousands of simulations in parallel, collecting experience and using it to update the RL agent's neural network weights via backpropagation.

evaluate.py loads a trained agent's checkpoint and runs it through a deterministic episode to measure its final performance against the heuristic benchmarks.

data_loader.py loads historic stock ticker data from yfinance 

------------------------------------------------------------------------------------

💻 Tech Stack
Core: Python 3.9+

Reinforcement Learning: Ray RLlib, PettingZoo, Gymnasium

Neural Networks: PyTorch

Data Handling: Pandas, NumPy

Visualization: Matplotlib, TensorBoard

🚀 Setup and Usage
Prerequisites

Python 3.9 or later

Pip package manager

------------------------------------------------------------------------------------

1. Installation

Clone the repository and install the required packages. It's highly recommended to use a virtual environment.

Bash
# Clone the repository
git clone <your-repo-link>
cd <your-repo-name>

# Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2. Training the Agent

Run the training script to start the learning process. This will train a new PPO agent from scratch and save the model checkpoints in the checkpoints_multi/ directory.

Bash
python train_multi.py
You can monitor the training progress in real-time using TensorBoard:

Bash
# Open a new terminal and run:
tensorboard --logdir checkpoints_multi/
3. Evaluating the Agent

Once training is complete, run the evaluation script. It will automatically load the latest trained checkpoint and generate a plot of the agent's performance against the benchmarks.

Bash
python evaluate.py
The performance chart will be saved in the plots/ directory.
------------------------------------------------------------------------------------

Future Improvements
Hyperparameter Tuning: Systematically tune the PPO algorithm's hyperparameters (e.g., learning rate, network size) using Ray Tune to optimize performance.

Expanded Feature Set: Incorporate more advanced quantitative features into the observation space, such as MACD, Bollinger Bands, or market-wide volatility indices (VIX).

Curriculum Learning: Implement a curriculum where the agent first learns in a simplified environment (e.g., no transaction costs) before moving to the full, complex simulation.
------------------------------------------------------------------------------------

License
This project is licensed under the MIT License. See the LICENSE file for details.
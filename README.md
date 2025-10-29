Multi-Agent Reinforcement Learning Trading System
--------------------------------------------------------

A competitive trading environment where a PPO-trained agent learns to outperform rule-based strategies through adversarial multi-agent dynamics and reward shaping.
The Challenge: Train an RL agent that doesn't just backtest against historical data, but learns to compete in real-time against opponents whose strategies directly affect market conditions.

🎯 The Story
--------------------------------------------------------
Version 1: The Safe Player
Initial training produced an agent that learned the "correct" local optimum: mimic the low-risk market maker and avoid volatility. It maximized the original reward function perfectly. The reward function was just wrong.
Problem: The agent optimized for stability over differentiation.

Version 2: The Aggressive Strategist
After re-engineering the reward function (5x competitive penalty, RSI features, entropy bonuses), the same PPO algorithm developed a fundamentally different strategy—aggressive momentum-based trading that decisively outperformed all benchmarks.
Insight: In multi-agent RL, the reward function doesn't just guide learning—it defines the Nash equilibrium your agent converges to.

📊 Results
--------------------------------------------------------
Initial Results (/plots/): RL agent mimicked market maker, minimal differentiation
Improved Results (/plots_2/): RL agent (red line) achieved highest portfolio value, outperforming all heuristic baselines + AAPL benchmark
Show Image

--------------------------------------------------------

🏗️ Technical Architecture
--------------------------------------------------------
Environment Design

Framework: Custom PettingZoo multi-agent environment
Market Mechanics:

Transaction costs and slippage
Price impact from collective agent volume
Non-stationary dynamics (agents affect each other's optimal strategies)



Agents
--------------------------------------------------------

RL Agent: PPO (Proximal Policy Optimization) via Ray RLlib

Actor-Critic architecture with continuous action space
Observation space: prices, portfolio state, RSI(14), opponent positions
Trained with entropy regularization for exploration


Heuristic Opponents:
--------------------------------------------------------

Momentum trader (trend-following)
Mean-reversion trader (contrarian)
Market maker (liquidity provider)



Reward Shaping (Critical Component)
--------------------------------------------------------
The agent optimizes a multi-objective reward function:
pythonreward = profit_term + coop_weight * stability_term - comp_weight * relative_performance_term
Key parameters:

comp_weight: 0.1 → 0.5 (increased competitive pressure)
coop_weight: 0.3 → 0.1 (reduced stability emphasis)
entropy_coeff: 0.0 → 0.01 (encouraged exploration)


🔬 Mathematical Foundation
--------------------------------------------------------
See /notes/ for handwritten derivations including:

PPO objective function from first principles
Generalized Advantage Estimation (GAE-λ)
Policy gradient theorem
Full forward pass example with reward shaping


💻 Tech Stack
--------------------------------------------------------
Core: Python 3.9+
RL Framework: Ray RLlib, PettingZoo, Gymnasium
Neural Networks: PyTorch
Data: yfinance, Pandas, NumPy
Visualization: Matplotlib, TensorBoard

🚀 Setup
--------------------------------------------------------
Prerequisites

Python 3.9+
Virtual environment (recommended)

Installation
--------------------------------------------------------
bashgit clone https://github.com/yourusername/marl-trading
cd marl-trading
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

Training
--------------------------------------------------------
`
python train_multi.py
`

Monitor training (separate terminal):
`
tensorboard --logdir checkpoints_multi/
`

Evaluation
--------------------------------------------------------
`
python evaluate.py
`

🎓 Key Learnings
--------------------------------------------------------
Reward engineering > Algorithm selection: The same PPO implementation produced radically different strategies based solely on reward function design.
Multi-agent complexity: Training against dynamic opponents that respond to your strategy is fundamentally harder than backtesting—and more realistic.
Emergent behavior: Small changes to incentive structures (competitive penalties, entropy bonuses) can cause phase transitions in learned strategies.


🔮 Future Work
--------------------------------------------------------
Hyperparameter optimization via Ray Tune
Extended feature set: MACD, Bollinger Bands, order book imbalance
Curriculum learning: Progressive difficulty (no costs → full simulation)
Multi-asset generalization: Test transfer learning across different market regimes


📄 License
--------------------------------------------------------
MIT License - See LICENSE file for details

🔗 Related Resources
--------------------------------------------------------

Technical Blog Post - Full deep dive into reward shaping and PPO mechanics
Handwritten Notes - Complete mathematical derivations
LinkedIn Post - Project summary and key insights


Built by Danish Mohammed | www.linkedin.com/in/danish-mohammed-2b959b1a6

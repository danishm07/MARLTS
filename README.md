Multi-Agent Trading — Minimal Repo
----------------------------------

Files:
- data_loader.py        : fetches historical price (yfinance)
- multi_agent_env.py    : PettingZoo AEC environment for synchronous multi-agent trading
- train.py              : launches Ray RLlib PPO multi-agent training
- evaluate.py           : loads checkpoint, runs deterministic eval, computes metrics and plots
- utils.py              : helper metric functions (Sharpe, drawdown, plotting)

Usage:
1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
4. python data_loader.py    # pulls price CSV locally
5. python train.py         # trains; creates checkpoint in ./checkpoints
6. python evaluate.py      # evaluate & plot

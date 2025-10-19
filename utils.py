"""
utils.py

Metrics and plotting helpers for multi-asset environment.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def compute_metrics(portfolio_values):
    pv = np.asarray(portfolio_values, dtype=float)
    if len(pv) < 2:
        return {}
    daily_ret = np.diff(pv) / pv[:-1]
    cum_return = float(pv[-1] / pv[0] - 1.0)
    ann_ret = (1 + cum_return) ** (252.0 / len(daily_ret)) - 1 if len(daily_ret) > 0 else 0.0
    ann_vol = float(np.std(daily_ret) * np.sqrt(252)) if len(daily_ret) > 0 else 0.0
    sharpe = float(ann_ret / (ann_vol + 1e-9)) if ann_vol > 0 else 0.0
    peak = np.maximum.accumulate(pv)
    drawdowns = (pv - peak) / (peak + 1e-9)
    max_dd = float(drawdowns.min())
    return {"cum_return": cum_return, "ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe, "max_drawdown": max_dd}

def plot_portfolios_multi(price_series, portfolios, outdir="plots"):
    os.makedirs(outdir, exist_ok=True)
    nsteps = len(next(iter(portfolios.values())))
    dates = price_series.index[:nsteps]
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    # plot price series (for first asset as reference)
    ax.plot(dates, price_series.iloc[:nsteps, 0].values, label=f"Price ({price_series.columns[0]})", alpha=0.6)
    ax.set_ylabel("Reference Price")
    ax.legend(loc="upper left")
    # twin axis for portfolios
    ax2 = ax.twinx()
    for name, pv in portfolios.items():
        ax2.plot(dates, pv, label=f"Portfolio: {name}")
    ax2.set_ylabel("Portfolio Value")
    ax2.legend(loc="upper right")
    plt.title("Reference Price and Agent Portfolios Over Time")
    plt.tight_layout()
    path = f"{outdir}/portfolios_multi.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved portfolios plot to {path}")

def plot_allocations_stacked(allocations, dates, outdir="plots"):
    """
    allocations: dict agent -> list of allocation arrays (length nsteps, each array length n_assets)
    dates: pandas DatetimeIndex length nsteps
    """
    os.makedirs(outdir, exist_ok=True)
    for agent, alloc_list in allocations.items():
        arr = np.vstack(alloc_list)  # shape (nsteps, n_assets)
        n_assets = arr.shape[1]
        plt.figure(figsize=(12, 4))
        labels = [f"asset_{i}" for i in range(n_assets)]
        plt.stackplot(dates[:arr.shape[0]], arr.T, labels=labels)
        plt.legend(loc="upper left")
        plt.title(f"Agent {agent} Allocation Over Time (stacked weights)")
        plt.xlabel("Date")
        plt.ylabel("Weight")
        plt.tight_layout()
        path = f"{outdir}/alloc_{agent}.png"
        plt.savefig(path)
        plt.close()
        print(f"Saved allocation plot for {agent} to {path}")
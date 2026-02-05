"""
metrics.py

PHASE 5B: Evaluation Metrics for RL Trading System

Implements:
- Buy-and-hold baseline
- Total return
- Max drawdown
- Sharpe ratio
- Volatility comparison
"""

import numpy as np
import pandas as pd


def calculate_total_return(initial_value: float, final_value: float) -> float:
    """Calculate total percentage return."""
    return ((final_value - initial_value) / initial_value) * 100


def calculate_max_drawdown(portfolio_values: list) -> float:
    """
    Calculate maximum drawdown as percentage.
    
    Max drawdown = largest peak-to-trough decline during the period.
    """
    values = np.array(portfolio_values)
    peak = np.maximum.accumulate(values)
    drawdown = (peak - values) / peak
    return np.max(drawdown) * 100


def calculate_sharpe_ratio(returns: list, risk_free_rate: float = 0.0) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Assumes daily returns, annualizes by sqrt(252).
    """
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / 252
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)


def calculate_volatility(returns: list) -> float:
    """Calculate annualized volatility."""
    return np.std(returns) * np.sqrt(252) * 100


def buy_and_hold_baseline(prices: pd.Series, initial_cash: float, 
                          transaction_cost: float = 0.001) -> dict:
    """
    Calculate buy-and-hold performance.
    
    Strategy: Buy on day 1, hold until end.
    """
    buy_price = prices.iloc[0]
    sell_price = prices.iloc[-1]
    
    # Apply transaction costs
    effective_cash = initial_cash * (1 - transaction_cost)
    shares = effective_cash / buy_price
    final_value = shares * sell_price * (1 - transaction_cost)
    
    total_return = calculate_total_return(initial_cash, final_value)
    
    # Calculate daily returns for Sharpe
    daily_returns = prices.pct_change().dropna().values
    sharpe = calculate_sharpe_ratio(daily_returns)
    volatility = calculate_volatility(daily_returns)
    
    # Calculate max drawdown
    portfolio_values = (prices / prices.iloc[0]) * initial_cash
    max_dd = calculate_max_drawdown(portfolio_values.values)
    
    return {
        "initial_value": initial_cash,
        "final_value": final_value,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "volatility_pct": volatility
    }


def evaluate_strategy(portfolio_values: list, initial_cash: float) -> dict:
    """
    Evaluate RL strategy performance.
    
    Args:
        portfolio_values: List of portfolio values at each step
        initial_cash: Starting capital
    
    Returns:
        Dictionary of performance metrics
    """
    values = np.array(portfolio_values)
    
    # Daily returns
    returns = np.diff(values) / values[:-1]
    
    total_return = calculate_total_return(initial_cash, values[-1])
    max_dd = calculate_max_drawdown(values)
    sharpe = calculate_sharpe_ratio(returns)
    volatility = calculate_volatility(returns)
    
    return {
        "initial_value": initial_cash,
        "final_value": values[-1],
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "sharpe_ratio": sharpe,
        "volatility_pct": volatility
    }


def print_comparison(rl_metrics: dict, baseline_metrics: dict):
    """Print side-by-side comparison of RL vs Buy-and-Hold."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON: RL Agent vs Buy-and-Hold")
    print("=" * 60)
    print(f"{'Metric':<25} {'RL Agent':>15} {'Buy & Hold':>15}")
    print("-" * 60)
    print(f"{'Initial Value':<25} ${rl_metrics['initial_value']:>14,.2f} ${baseline_metrics['initial_value']:>14,.2f}")
    print(f"{'Final Value':<25} ${rl_metrics['final_value']:>14,.2f} ${baseline_metrics['final_value']:>14,.2f}")
    print(f"{'Total Return':<25} {rl_metrics['total_return_pct']:>14.2f}% {baseline_metrics['total_return_pct']:>14.2f}%")
    print(f"{'Max Drawdown':<25} {rl_metrics['max_drawdown_pct']:>14.2f}% {baseline_metrics['max_drawdown_pct']:>14.2f}%")
    print(f"{'Sharpe Ratio':<25} {rl_metrics['sharpe_ratio']:>15.3f} {baseline_metrics['sharpe_ratio']:>15.3f}")
    print(f"{'Volatility (Ann.)':<25} {rl_metrics['volatility_pct']:>14.2f}% {baseline_metrics['volatility_pct']:>14.2f}%")
    print("=" * 60)
    
    # Verdict
    outperformed = rl_metrics['total_return_pct'] > baseline_metrics['total_return_pct']
    risk_adjusted = rl_metrics['sharpe_ratio'] > baseline_metrics['sharpe_ratio']
    
    print("\nVERDICT:")
    if outperformed and risk_adjusted:
        print("✅ RL Agent OUTPERFORMS Buy-and-Hold (both return and risk-adjusted)")
    elif outperformed:
        print("⚠️ RL Agent has higher returns but worse risk-adjusted performance")
    elif risk_adjusted:
        print("⚠️ RL Agent has better risk-adjusted returns but lower absolute return")
    else:
        print("❌ RL Agent UNDERPERFORMS Buy-and-Hold")

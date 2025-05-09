"""
Performance metrics for portfolio evaluation.
"""

import numpy as np
from typing import List, Dict


def calculate_returns(portfolio_values: List[float]) -> np.ndarray:
    """
    Calculate returns from portfolio values.
    
    Args:
        portfolio_values: List of portfolio values
        
    Returns:
        Array of returns
    """
    values = np.array(portfolio_values)
    returns = values[1:] / values[:-1] - 1
    return returns


def calculate_cumulative_returns(returns: np.ndarray) -> np.ndarray:
    """
    Calculate cumulative returns from returns.
    
    Args:
        returns: Array of returns
        
    Returns:
        Array of cumulative returns
    """
    return np.cumprod(1 + returns) - 1


def calculate_annualized_return(returns: np.ndarray, trading_days: int = 252) -> float:
    """
    Calculate annualized return.
    
    Args:
        returns: Array of returns
        trading_days: Number of trading days in a year
        
    Returns:
        Annualized return
    """
    if len(returns) == 0:
        return 0.0
    
    total_return = np.prod(1 + returns) - 1
    n_days = len(returns)
    
    return ((1 + total_return) ** (trading_days / n_days)) - 1


def calculate_annualized_volatility(returns: np.ndarray, trading_days: int = 252) -> float:
    """
    Calculate annualized volatility.
    
    Args:
        returns: Array of returns
        trading_days: Number of trading days in a year
        
    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0
    
    return np.std(returns) * np.sqrt(trading_days)


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0, trading_days: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        trading_days: Number of trading days in a year
        
    Returns:
        Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe * np.sqrt(trading_days)


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0, trading_days: int = 252) -> float:
    """
    Calculate Sortino ratio.
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate
        trading_days: Number of trading days in a year
        
    Returns:
        Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    sortino = np.mean(excess_returns) / np.std(downside_returns)
    return sortino * np.sqrt(trading_days)


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Array of returns
        
    Returns:
        Maximum drawdown
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate cumulative returns
    cum_returns = np.cumprod(1 + returns)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdown
    drawdown = (running_max - cum_returns) / running_max
    
    return np.max(drawdown)


def calculate_calmar_ratio(returns: np.ndarray, trading_days: int = 252) -> float:
    """
    Calculate Calmar ratio.
    
    Args:
        returns: Array of returns
        trading_days: Number of trading days in a year
        
    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0
    
    ann_return = calculate_annualized_return(returns, trading_days)
    max_dd = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return 0.0
    
    return ann_return / max_dd


def calculate_metrics(portfolio_values: List[float], trading_days: int = 252) -> Dict[str, float]:
    """
    Calculate all performance metrics.
    
    Args:
        portfolio_values: List of portfolio values
        trading_days: Number of trading days in a year
        
    Returns:
        Dictionary with performance metrics
    """
    if len(portfolio_values) < 2:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0
        }
    
    # Calculate returns
    returns = calculate_returns(portfolio_values)
    
    # Calculate total return
    total_return = portfolio_values[-1] / portfolio_values[0] - 1
    
    # Calculate metrics
    ann_return = calculate_annualized_return(returns, trading_days)
    ann_vol = calculate_annualized_volatility(returns, trading_days)
    sharpe = calculate_sharpe_ratio(returns, 0, trading_days)
    sortino = calculate_sortino_ratio(returns, 0, trading_days)
    max_dd = calculate_max_drawdown(returns)
    calmar = calculate_calmar_ratio(returns, trading_days)
    
    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar
    }
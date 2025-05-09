"""
Graph utilities for constructing graph structures for GCN.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple

try:
    from pgmpy.estimators import PC
    from pgmpy.models import BayesianNetwork
    HAS_PGMPY = True
except ImportError:
    HAS_PGMPY = False

try:
    from sklearn.covariance import GraphicalLassoCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def normalize_adj_matrix(adj_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize adjacency matrix for GCN.
    
    A_normalized = D^(-1/2) * A * D^(-1/2)
    where D is the degree matrix of A.
    
    Args:
        adj_matrix: Adjacency matrix
        
    Returns:
        Normalized adjacency matrix
    """
    # Add self-loops
    adj_matrix = adj_matrix + np.eye(adj_matrix.shape[0])
    
    # Calculate degree matrix
    rowsum = np.array(adj_matrix.sum(1))
    
    # Calculate D^(-1/2)
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Calculate normalized adjacency matrix
    return adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def create_industry_graph(tickers: List[str], industry_map: Dict[str, str]) -> np.ndarray:
    """
    Create graph based on industry classification.
    
    Args:
        tickers: List of stock tickers
        industry_map: Dictionary mapping ticker to industry
        
    Returns:
        Adjacency matrix
    """
    n = len(tickers)
    adj_matrix = np.zeros((n, n))
    
    for i, ticker_i in enumerate(tickers):
        industry_i = industry_map.get(ticker_i, "Unknown")
        
        for j, ticker_j in enumerate(tickers):
            industry_j = industry_map.get(ticker_j, "Unknown")
            
            if industry_i == industry_j:
                adj_matrix[i, j] = 1
    
    return normalize_adj_matrix(adj_matrix)


def create_correlation_graph(returns: pd.DataFrame, threshold: float = 0.3) -> np.ndarray:
    """
    Create graph based on correlation.
    
    Args:
        returns: DataFrame with stock returns
        threshold: Correlation threshold
        
    Returns:
        Adjacency matrix
    """
    # Compute correlation matrix
    corr_matrix = returns.corr().abs().values
    
    # Set threshold to create sparse matrix
    adj_matrix = (corr_matrix > threshold).astype(float)
    
    # Set diagonal to 1
    np.fill_diagonal(adj_matrix, 1)
    
    return normalize_adj_matrix(adj_matrix)


def create_partial_correlation_graph(returns: pd.DataFrame) -> np.ndarray:
    """
    Create graph based on partial correlation using Graphical Lasso.
    
    Args:
        returns: DataFrame with stock returns
        
    Returns:
        Adjacency matrix
    """
    if not HAS_SKLEARN:
        print("scikit-learn not available, falling back to correlation")
        return create_correlation_graph(returns)
    
    n = returns.shape[1]
    
    try:
        # Fit graphical lasso
        model = GraphicalLassoCV()
        model.fit(returns)
        
        # Get precision matrix (inverse of covariance)
        precision = model.precision_
        
        # Convert to adjacency matrix
        adj_matrix = (np.abs(precision) > 0).astype(float)
        
        # Set diagonal to 1
        np.fill_diagonal(adj_matrix, 1)
    
    except Exception as e:
        print(f"Error in partial correlation: {e}")
        # Fallback to correlation
        return create_correlation_graph(returns)
    
    return normalize_adj_matrix(adj_matrix)


def create_causal_graph(returns: pd.DataFrame, max_cond_vars: int = 3) -> np.ndarray:
    """
    Create graph based on causal discovery using PC algorithm.
    
    Args:
        returns: DataFrame with stock returns
        max_cond_vars: Maximum number of conditioning variables for PC algorithm
        
    Returns:
        Adjacency matrix
    """
    if not HAS_PGMPY:
        print("pgmpy not available, falling back to correlation")
        return create_correlation_graph(returns)
    
    n = returns.shape[1]
    tickers = returns.columns
    
    try:
        # Run PC algorithm for causal discovery
        est = PC(returns)
        skeleton = est.estimate(variant="stable", max_cond_vars=max_cond_vars)
        
        # Convert to adjacency matrix
        adj_matrix = np.zeros((n, n))
        
        for i, ticker_i in enumerate(tickers):
            for j, ticker_j in enumerate(tickers):
                if skeleton.has_edge(ticker_i, ticker_j):
                    adj_matrix[i, j] = 1
        
        # Set diagonal to 1
        np.fill_diagonal(adj_matrix, 1)
    
    except Exception as e:
        print(f"Error in causal discovery: {e}")
        # Fallback to correlation
        return create_correlation_graph(returns)
    
    return normalize_adj_matrix(adj_matrix)


def create_graph(
    tickers: List[str],
    data: Optional[pd.DataFrame] = None,
    industry_map: Optional[Dict[str, str]] = None,
    graph_type: str = 'industry'
) -> np.ndarray:
    """
    Create graph structure for GCN.
    
    Args:
        tickers: List of stock tickers
        data: DataFrame with stock data (required for correlation and causal graphs)
        industry_map: Dictionary mapping ticker to industry (required for industry graph)
        graph_type: Type of graph structure to create
            - 'industry': Based on industry classification
            - 'correlation': Based on price correlation
            - 'partial_correlation': Based on partial correlation
            - 'causal': Based on causal discovery
            
    Returns:
        Normalized adjacency matrix
    """
    if graph_type == 'industry':
        if industry_map is None:
            raise ValueError("industry_map is required for industry graph")
        return create_industry_graph(tickers, industry_map)
    
    elif graph_type in ['correlation', 'partial_correlation', 'causal']:
        if data is None:
            raise ValueError(f"data is required for {graph_type} graph")
        
        # Calculate returns if not already
        if 'return' not in data.columns:
            if 'close' in data.columns:
                returns = data['close'].pct_change().dropna()
            else:
                raise ValueError("data must contain 'close' or 'return' column")
        else:
            returns = data['return']
        
        if graph_type == 'correlation':
            return create_correlation_graph(returns)
        elif graph_type == 'partial_correlation':
            return create_partial_correlation_graph(returns)
        else:  # causal
            return create_causal_graph(returns)
    
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")
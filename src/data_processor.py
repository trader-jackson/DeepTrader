"""
Data processor for loading and preprocessing financial data.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf
from sklearn.preprocessing import StandardScaler
import networkx as nx
from pgmpy.estimators import PC
from pgmpy.models import BayesianNetwork

from config import DATA_DIR, DATASETS, STOCK_INDICATORS, MARKET_INDICATORS


class DataProcessor:
    """Class for loading and preprocessing financial data."""
    
    def __init__(self, dataset_name):
        """
        Initialize data processor.
        
        Args:
            dataset_name: Name of the dataset (dow30, nasdaq100, sse50)
        """
        self.dataset_name = dataset_name
        self.config = DATASETS[dataset_name]
        self.data_dir = DATA_DIR / dataset_name
        self.tickers = self._load_tickers()
        self.industry_map = self._load_industry_map()
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _load_tickers(self):
        """Load ticker symbols for the dataset."""
        ticker_file = self.data_dir / "tickers.txt"
        
        if os.path.exists(ticker_file):
            with open(ticker_file, "r") as f:
                tickers = [line.strip() for line in f.readlines()]
                return tickers[:self.config["num_stocks"]]
        else:
            # If ticker file doesn't exist, use predefined tickers
            if self.dataset_name == "dow30":
                # Dow Jones 30 tickers
                tickers = [
                    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
                     "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO",
                    "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V",
                    "VZ", "WBA", "WMT"
                ]
            elif self.dataset_name == "nasdaq100":
                # Use the top 100 NASDAQ stocks (simplified for implementation)
                tickers = yf.Tickers("^NDX").tickers[0].constituents
                tickers = list(tickers)[:self.config["num_stocks"]]
            elif self.dataset_name == "sse50":
                # For SSE50, we would typically load from a specific source
                # This is a placeholder for the top Chinese stocks
                tickers = [f"600{i:03d}.SS" for i in range(1, 51)]
            
            # Save tickers to file
            os.makedirs(os.path.dirname(ticker_file), exist_ok=True)
            with open(ticker_file, "w") as f:
                for ticker in tickers:
                    f.write(f"{ticker}\n")
            
            return tickers
    
    def _load_industry_map(self):
        """Load industry classification for tickers."""
        industry_file = self.data_dir / "industry_map.csv"
        
        if os.path.exists(industry_file):
            industry_df = pd.read_csv(industry_file)
            return dict(zip(industry_df["ticker"], industry_df["industry"]))
        else:
            # If industry file doesn't exist, attempt to fetch from Yahoo Finance
            industries = {}
            for ticker in self.tickers:
                try:
                    info = yf.Ticker(ticker).info
                    industries[ticker] = info.get("sector", "Unknown")
                except:
                    industries[ticker] = "Unknown"
            
            # Save industry map to file
            industry_df = pd.DataFrame({
                "ticker": list(industries.keys()),
                "industry": list(industries.values())
            })
            industry_df.to_csv(industry_file, index=False)
            
            return industries
    
    def download_data(self, start_date, end_date):
        """
        Download stock data for the specified period.
        
        Args:
            start_date: Start date
            end_date: End date
        """
        for ticker in self.tickers:
            stock_file = self.data_dir / f"{ticker}.csv"
            
            if not os.path.exists(stock_file):
                try:
                    # Download data
                    data = yf.download(ticker, start=start_date, end=end_date)
                    
                    # Handle MultiIndex columns if present
                    if data.columns.nlevels > 1:
                        data.columns = data.columns.droplevel(1)
                    
                    if not data.empty:
                        # Make sure column names are consistent
                        data.columns = [col.lower() for col in data.columns]
                        
                        # Save to CSV
                        data.to_csv(stock_file)
                        print(f"Downloaded {ticker} data")
                    else:
                        print(f"No data found for {ticker}")
                except Exception as e:
                    print(f"Error downloading {ticker}: {e}")
        
        # Download market index data
        market_index = "^DJI" if self.dataset_name == "dow30" else \
                    "^IXIC" if self.dataset_name == "nasdaq100" else \
                    "000001.SS"  # SSE Composite Index for sse50
        
        index_file = self.data_dir / f"{market_index}.csv"
        
        if not os.path.exists(index_file):
            try:
                # Download market index data
                index_data = yf.download(market_index, start=start_date, end=end_date)
                
                # Handle MultiIndex columns if present
                if index_data.columns.nlevels > 1:
                    index_data.columns = index_data.columns.droplevel(1)
                
                if not index_data.empty:
                    # Make sure column names are consistent
                    index_data.columns = [col.lower() for col in index_data.columns]
                    
                    index_data.to_csv(index_file)
                    print(f"Downloaded {market_index} data")
                else:
                    print(f"No data found for {market_index}")
            except Exception as e:
                print(f"Error downloading {market_index}: {e}")
    
    def load_stock_data(self):
        """Load and preprocess stock data."""
        all_data = {}
        valid_tickers = []
        
        for ticker in self.tickers:
            stock_file = self.data_dir / f"{ticker}.csv"
            if os.path.exists(stock_file):
                try:
                    data = pd.read_csv(stock_file, index_col=0, parse_dates=True)
                    # Ensure all required columns exist
                    required_cols = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in data.columns for col in required_cols):
                        all_data[ticker] = data
                        valid_tickers.append(ticker)
                except Exception as e:
                    print(f"Error loading {ticker}: {e}")
        
        self.tickers = valid_tickers
        return all_data
    
    def load_market_data(self):
        """Load and preprocess market index data."""
        market_index = "^DJI" if self.dataset_name == "dow30" else \
                       "^IXIC" if self.dataset_name == "nasdaq100" else \
                       "000001.SS"  # SSE Composite Index for sse50
        
        index_file = self.data_dir / f"{market_index}.csv"
        
        if os.path.exists(index_file):
            try:
                data = pd.read_csv(index_file, index_col=0, parse_dates=True)
                return data
            except Exception as e:
                print(f"Error loading market index: {e}")
                return None
        else:
            print(f"Market index file not found: {index_file}")
            return None
    
    def compute_technical_indicators(self, stock_data):
        """
        Compute technical indicators for stock data.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame with OHLCV data
            
        Returns:
            Dictionary mapping ticker to DataFrame with technical indicators
        """
        result = {}
        
        for ticker, data in stock_data.items():
            df = data.copy()
            
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Calculate moving averages
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            
            # Calculate RSI (Relative Strength Index)
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD (Moving Average Convergence Divergence)
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            
            # Calculate Bollinger Bands
            df['boll_mid'] = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            df['boll_upper'] = df['boll_mid'] + 2 * std
            df['boll_lower'] = df['boll_mid'] - 2 * std
            
            # Drop NaN values
            df = df.dropna()
            
            result[ticker] = df
        
        return result
    
    def compute_market_indicators(self, market_data, stock_data):
        """
        Compute market indicators.
        
        Args:
            market_data: DataFrame with market index data
            stock_data: Dictionary mapping ticker to DataFrame with stock data
            
        Returns:
            DataFrame with market indicators
        """
        market_df = market_data.copy()
        
        # Rename columns to lowercase
        market_df.columns = [col.lower() for col in market_df.columns]
        
        # Calculate market return
        market_df['index_return'] = market_df['close'].pct_change()
        
        # Calculate moving averages
        market_df['index_ma5'] = market_df['close'].rolling(window=5).mean()
        market_df['index_ma10'] = market_df['close'].rolling(window=10).mean()
        
        # Calculate advance-decline ratio and volume ratio
        dates = sorted(list(set.intersection(
            *[set(data.index) for data in stock_data.values()]
        )))
        
        advance_decline = []
        vol_ratio = []
        
        for date in dates:
            # Count advancing vs declining stocks
            prev_date_idx = dates.index(date) - 1 if dates.index(date) > 0 else 0
            prev_date = dates[prev_date_idx]
            
            advancing = 0
            declining = 0
            rising_vol = 0
            falling_vol = 0
            
            for ticker, data in stock_data.items():
                if date in data.index and prev_date in data.index:
                    if data.loc[date, 'close'] > data.loc[prev_date, 'close']:
                        advancing += 1
                    else:
                        declining += 1
                    
                    if data.loc[date, 'volume'] > data.loc[prev_date, 'volume']:
                        rising_vol += 1
                    else:
                        falling_vol += 1
            
            # Avoid division by zero
            ad_ratio = advancing / max(declining, 1)
            v_ratio = rising_vol / max(falling_vol, 1)
            
            advance_decline.append(ad_ratio)
            vol_ratio.append(v_ratio)
        
        # Create DataFrame with market indicators
        market_indicators = pd.DataFrame({
            'date': dates,
            'advance_decline': advance_decline,
            'vol_ratio': vol_ratio
        })
        market_indicators.set_index('date', inplace=True)
        
        # Join with market data
        result = pd.merge(
            market_df, market_indicators,
            left_index=True, right_index=True,
            how='inner'
        )
        
        # Drop NaN values
        result = result.dropna()
        
        return result
    
    def align_data(self, stock_data, market_data):
        """
        Align stock and market data to have the same dates.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame with stock data
            market_data: DataFrame with market data
            
        Returns:
            Tuple of (aligned stock data, aligned market data)
        """
        # Find common dates
        common_dates = sorted(list(set.intersection(
            set(market_data.index),
            *[set(data.index) for data in stock_data.values()]
        )))
        
        # Filter data to common dates
        aligned_stock_data = {
            ticker: data.loc[common_dates]
            for ticker, data in stock_data.items()
        }
        aligned_market_data = market_data.loc[common_dates]
        
        return aligned_stock_data, aligned_market_data
    
    def create_graph_structure(self, stock_data, graph_type='industry'):
        """
        Create graph structure for GCN.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame with stock data
            graph_type: Type of graph structure to create
                - 'industry': Based on industry classification
                - 'correlation': Based on price correlation
                - 'partial_correlation': Based on partial correlation
                - 'causal': Based on causal discovery
                
        Returns:
            Adjacency matrix (numpy array)
        """
        n = len(self.tickers)
        adj_matrix = np.zeros((n, n))
        
        if graph_type == 'industry':
            # Create graph based on industry classification
            for i, ticker_i in enumerate(self.tickers):
                industry_i = self.industry_map.get(ticker_i, "Unknown")
                
                for j, ticker_j in enumerate(self.tickers):
                    industry_j = self.industry_map.get(ticker_j, "Unknown")
                    
                    if industry_i == industry_j:
                        adj_matrix[i, j] = 1
        
        elif graph_type in ['correlation', 'partial_correlation']:
            # Extract close prices
            close_prices = pd.DataFrame({
                ticker: data['close']
                for ticker, data in stock_data.items()
            })
            
            # Calculate returns
            returns = close_prices.pct_change().dropna()
            
            if graph_type == 'correlation':
                # Compute correlation matrix
                corr_matrix = returns.corr().abs().values
                
                # Set threshold to create sparse matrix
                threshold = 0.5
                adj_matrix = (corr_matrix > threshold).astype(float)
                
                # Set diagonal to 1
                np.fill_diagonal(adj_matrix, 1)
            
            else:  # partial_correlation
                from sklearn.covariance import GraphicalLassoCV
                
                # Fit graphical lasso
                model = GraphicalLassoCV()
                model.fit(returns)
                
                # Get precision matrix (inverse of covariance)
                precision = model.precision_
                
                # Convert to adjacency matrix
                adj_matrix = (np.abs(precision) > 0).astype(float)
                
                # Set diagonal to 1
                np.fill_diagonal(adj_matrix, 1)
        
        elif graph_type == 'causal':
            # Extract close prices
            close_prices = pd.DataFrame({
                ticker: data['close']
                for ticker, data in stock_data.items()
            })
            
            # Calculate returns
            returns = close_prices.pct_change().dropna()
            
            # Create empty graph
            G = nx.DiGraph()
            G.add_nodes_from(self.tickers)
            
            try:
                # Run PC algorithm for causal discovery
                est = PC(returns)
                skeleton = est.estimate(variant="stable", max_cond_vars=3)
                
                # Convert to adjacency matrix
                for i, ticker_i in enumerate(self.tickers):
                    for j, ticker_j in enumerate(self.tickers):
                        if skeleton.has_edge(ticker_i, ticker_j):
                            adj_matrix[i, j] = 1
                
                # Set diagonal to 1
                np.fill_diagonal(adj_matrix, 1)
            
            except Exception as e:
                print(f"Error in causal discovery: {e}")
                # Fallback to correlation
                corr_matrix = returns.corr().abs().values
                threshold = 0.3
                adj_matrix = (corr_matrix > threshold).astype(float)
                np.fill_diagonal(adj_matrix, 1)
        
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")
        
        return adj_matrix
    
    def normalize_data(self, stock_data, market_data):
        """
        Normalize data using StandardScaler.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame with stock data
            market_data: DataFrame with market data
            
        Returns:
            Tuple of (normalized stock data, normalized market data, scalers)
        """
        # Initialize scalers
        stock_scalers = {ticker: {} for ticker in self.tickers}
        market_scaler = {}
        
        # Normalize stock data
        normalized_stock_data = {}
        
        for ticker, data in stock_data.items():
            normalized_df = data.copy()
            
            for col in STOCK_INDICATORS:
                if col in normalized_df.columns:
                    scaler = StandardScaler()
                    normalized_df[col] = scaler.fit_transform(normalized_df[[col]])
                    stock_scalers[ticker][col] = scaler
            
            normalized_stock_data[ticker] = normalized_df
        
        # Normalize market data
        normalized_market_data = market_data.copy()
        
        for col in MARKET_INDICATORS:
            if col in normalized_market_data.columns:
                scaler = StandardScaler()
                normalized_market_data[col] = scaler.fit_transform(normalized_market_data[[col]])
                market_scaler[col] = scaler
        
        return normalized_stock_data, normalized_market_data, (stock_scalers, market_scaler)
    
    def create_time_series_dataset(self, stock_data, market_data, sequence_length, train_test_split=None):
        """
        Create time series dataset for training and testing.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame with stock data
            market_data: DataFrame with market data
            sequence_length: Length of input sequence
            train_test_split: Tuple of (train_start, train_end, test_start, test_end)
                If None, split according to dataset config
                
        Returns:
            Dictionary with train and test datasets
        """
        if train_test_split is None:
            train_start, train_end = self.config["train_period"]
            test_start, test_end = self.config["test_period"]
        else:
            train_start, train_end, test_start, test_end = train_test_split
        
        # Convert string dates to datetime
        train_start = pd.to_datetime(train_start)
        train_end = pd.to_datetime(train_end)
        test_start = pd.to_datetime(test_start)
        test_end = pd.to_datetime(test_end)
        
        # Get common dates
        dates = sorted(market_data.index)
        
        # Split into train and test
        train_dates = [d for d in dates if train_start <= d <= train_end]
        test_dates = [d for d in dates if test_start <= d <= test_end]
        
        # Create sequences
        train_sequences = self._create_sequences(stock_data, market_data, train_dates, sequence_length)
        test_sequences = self._create_sequences(stock_data, market_data, test_dates, sequence_length)
        
        return {
            'train': train_sequences,
            'test': test_sequences
        }
    
    def _create_sequences(self, stock_data, market_data, dates, sequence_length):
        """
        Create sequences for time series data.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame with stock data
            market_data: DataFrame with market data
            dates: List of dates
            sequence_length: Length of input sequence
            
        Returns:
            Dictionary with sequences
        """
        sequences = []
        
        for i in range(sequence_length, len(dates)):
            seq_dates = dates[i-sequence_length:i]
            target_date = dates[i]
            
            # Extract stock features
            stock_features = []
            
            for ticker in self.tickers:
                if ticker in stock_data:
                    # Extract features for the sequence
                    features = []
                    
                    for col in STOCK_INDICATORS:
                        if col in stock_data[ticker].columns:
                            values = stock_data[ticker].loc[seq_dates, col].values
                            features.append(values)
                    
                    features = np.vstack(features).T  # Shape: [sequence_length, features]
                    stock_features.append(features)
            
            stock_features = np.stack(stock_features)  # Shape: [num_stocks, sequence_length, features]
            
            # Extract market features
            market_features = []
            
            for col in MARKET_INDICATORS:
                if col in market_data.columns:
                    values = market_data.loc[seq_dates, col].values
                    market_features.append(values)
            
            market_features = np.vstack(market_features).T  # Shape: [sequence_length, features]
            
            # Calculate target returns (next day's returns)
            target_returns = []
            
            for ticker in self.tickers:
                if ticker in stock_data and target_date in stock_data[ticker].index:
                    close_price = stock_data[ticker].loc[target_date, 'close']
                    prev_close = stock_data[ticker].loc[seq_dates[-1], 'close']
                    
                    # Calculate return
                    ret = (close_price / prev_close) - 1
                    target_returns.append(ret)
                else:
                    # If data is missing, use 0 as return
                    target_returns.append(0.0)
            
            target_returns = np.array(target_returns)
            
            sequences.append({
                'stock_features': stock_features,
                'market_features': market_features,
                'target_returns': target_returns,
                'date': target_date
            })
        
        return sequences
    
    def prepare_data(self, train_test_split=None, sequence_length=50, graph_type='industry'):
        """
        Prepare data for training and testing.
        
        Args:
            train_test_split: Tuple of (train_start, train_end, test_start, test_end)
            sequence_length: Length of input sequence
            graph_type: Type of graph structure to create
            
        Returns:
            Dictionary with prepared data
        """
        # Load raw data
        stock_data = self.load_stock_data()

        market_data = self.load_market_data()
        
        if not stock_data or market_data is None:
            print("Error loading data")
            return None
        
        # Compute technical indicators
        print("\nComputing technical indicators...")
        print(f"Stock data before computing indicators: {len(stock_data)} stocks")
        for ticker in list(stock_data.keys())[:3]:  # Print first 3 stocks as examples
            print(f"  {ticker} shape: {stock_data[ticker].shape}, columns: {stock_data[ticker].columns.tolist()}")

        stock_data = self.compute_technical_indicators(stock_data)
        print(f"Stock data after computing indicators: {len(stock_data)} stocks")
        for ticker in list(stock_data.keys())[:3]:  # Print first 3 stocks as examples
            print(f"  {ticker} shape: {stock_data[ticker].shape}, columns: {stock_data[ticker].columns.tolist()}")

        print("\nComputing market indicators...")
        print(f"Market data shape before: {market_data.shape}, columns: {market_data.columns.tolist()}")
        market_data = self.compute_market_indicators(market_data, stock_data)
        print(f"Market data shape after: {market_data.shape}, columns: {market_data.columns.tolist()}")

        # Align data
        print("\nAligning data...")
        before_align = {
            "stock_dates": [data.index.min().strftime('%Y-%m-%d') + " to " + data.index.max().strftime('%Y-%m-%d') 
                            for ticker, data in list(stock_data.items())[:3]],
            "market_dates": market_data.index.min().strftime('%Y-%m-%d') + " to " + market_data.index.max().strftime('%Y-%m-%d'),
            "stock_shapes": [data.shape for ticker, data in list(stock_data.items())[:3]],
            "market_shape": market_data.shape
        }
        print(f"Data before alignment:")
        print(f"  Stock date ranges (first 3): {before_align['stock_dates']}")
        print(f"  Market date range: {before_align['market_dates']}")
        print(f"  Stock shapes (first 3): {before_align['stock_shapes']}")
        print(f"  Market shape: {before_align['market_shape']}")

        stock_data, market_data = self.align_data(stock_data, market_data)

        after_align = {
            "stock_dates": [data.index.min().strftime('%Y-%m-%d') + " to " + data.index.max().strftime('%Y-%m-%d') 
                        for ticker, data in list(stock_data.items())[:3]],
            "market_dates": market_data.index.min().strftime('%Y-%m-%d') + " to " + market_data.index.max().strftime('%Y-%m-%d'),
            "stock_shapes": [data.shape for ticker, data in list(stock_data.items())[:3]],
            "market_shape": market_data.shape
        }
        print(f"Data after alignment:")
        print(f"  Stock date ranges (first 3): {after_align['stock_dates']}")
        print(f"  Market date range: {after_align['market_dates']}")
        print(f"  Stock shapes (first 3): {after_align['stock_shapes']}")
        print(f"  Market shape: {after_align['market_shape']}")

        # Create graph structure
        print(f"\nCreating graph structure using {graph_type} method...")
        adj_matrix = self.create_graph_structure(stock_data, graph_type)
        print(f"Adjacency matrix shape: {adj_matrix.shape}")
        print(f"Adjacency matrix statistics:")
        print(f"  Non-zero connections: {np.count_nonzero(adj_matrix)}")
        print(f"  Sparsity: {1 - (np.count_nonzero(adj_matrix) / adj_matrix.size):.4f}")
        print(f"  Average connections per stock: {np.count_nonzero(adj_matrix) / adj_matrix.shape[0]:.2f}")

        # Normalize data
        print("\nNormalizing data...")
        for ticker in list(stock_data.keys())[:3]:
            print(f"  Before normalization - {ticker} mean: {stock_data[ticker]['close'].mean():.4f}, std: {stock_data[ticker]['close'].std():.4f}")

        stock_data, market_data, scalers = self.normalize_data(stock_data, market_data)

        for ticker in list(stock_data.keys())[:3]:
            print(f"  After normalization - {ticker} mean: {stock_data[ticker]['close'].mean():.4f}, std: {stock_data[ticker]['close'].std():.4f}")

        # Create time series dataset
        print("\nCreating time series dataset...")
        if train_test_split:
            print(f"Using custom train-test split: {train_test_split}")
        else:
            print(f"Using default train-test split: {self.config['train_period']} / {self.config['test_period']}")

        print(f"Sequence length: {sequence_length}")
        dataset = self.create_time_series_dataset(
            stock_data, market_data, sequence_length, train_test_split
        )

        print(f"Dataset created:")
        print(f"  Training sequences: {len(dataset['train'])}")
        print(f"  Testing sequences: {len(dataset['test'])}")

        if len(dataset['train']) > 0:
            sample = dataset['train'][0]
            print("\nSample sequence structure:")
            print(f"  stock_features shape: {sample['stock_features'].shape}")
            print(f"  market_features shape: {sample['market_features'].shape}")
            print(f"  target_returns shape: {sample['target_returns'].shape}")
            print(f"  date: {sample['date']}")

        return {
            'dataset': dataset,
            'adj_matrix': adj_matrix,
            'tickers': self.tickers,
            'scalers': scalers
        }
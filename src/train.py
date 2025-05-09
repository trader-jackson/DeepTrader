"""
Training module for DeepTrader model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
from typing import Dict, List, Tuple, Optional

from src.models.deeptrader import DeepTrader
from src.environment import TradingEnvironment
from config import MODEL_CONFIG


class AssetScoringReward:
    """
    Reward function for Asset Scoring Unit based on price rising rate.
    """
    
    def __call__(self, asset_scores: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """
        Calculate reward for Asset Scoring Unit.
        
        Args:
            asset_scores: Asset scores from Asset Scoring Unit of shape [batch_size, num_stocks]
            returns: Actual returns of shape [batch_size, num_stocks]
            
        Returns:
            Reward of shape [batch_size]
        """
        # Reward is weighted returns, with weights determined by asset scores
        reward = torch.sum(asset_scores * returns, dim=1)
        return reward


class MarketScoringReward:
    """
    Reward function for Market Scoring Unit based on various risk measures.
    """
    
    def __init__(self, reward_type: str = 'mdd'):
        """
        Initialize Market Scoring Reward.
        
        Args:
            reward_type: Type of reward function
                - 'ror': Rate of Return
                - 'sr': Sharpe Ratio
                - 'mdd': Maximum Drawdown
                - 'cr': Calmar Ratio
        """
        self.reward_type = reward_type
    
    def __call__(self, portfolio_values: List[float]) -> float:
        """
        Calculate reward for Market Scoring Unit.
        
        Args:
            portfolio_values: List of portfolio values
            
        Returns:
            Reward value
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        # Convert to numpy array
        values = np.array(portfolio_values)
        
        # Calculate returns
        returns = values[1:] / values[:-1] - 1
        
        if self.reward_type == 'ror':
            # Rate of Return
            reward = returns[-1]
        
        elif self.reward_type == 'sr':
            # Sharpe Ratio
            if len(returns) < 2 or np.std(returns) == 0:
                reward = 0.0
            else:
                reward = np.mean(returns) / np.std(returns)
        
        elif self.reward_type == 'mdd':
            # Negative Maximum Drawdown
            running_max = np.maximum.accumulate(values)
            drawdown = (running_max - values) / running_max
            reward = -np.max(drawdown)  # Negative MDD as reward
        
        elif self.reward_type == 'cr':
            # Calmar Ratio
            running_max = np.maximum.accumulate(values)
            drawdown = (running_max - values) / running_max
            max_dd = np.max(drawdown)
            
            if max_dd == 0:
                reward = 0.0
            else:
                reward = np.mean(returns) / max_dd
        
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        
        return reward


class DeepTraderTrainer:
    """
    Trainer for DeepTrader model.
    """
    
    def __init__(self, 
                 model: DeepTrader,
                 device: torch.device,
                 learning_rate: float = 1e-4,
                 reward_type: str = 'mdd',
                 market_reward_weight: float = 0.5,
                 entropy_weight: float = 0.001,
                 gamma: float = 0.99,
                 checkpoint_dir: str = './checkpoints'):
        """
        Initialize trainer.
        
        Args:
            model: DeepTrader model
            device: Device to use for training
            learning_rate: Learning rate for optimizer
            reward_type: Type of reward function for Market Scoring Unit
            market_reward_weight: Weight for Market Scoring Unit reward
            entropy_weight: Weight for entropy regularization
            gamma: Discount factor for future rewards
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.market_reward_weight = market_reward_weight
        self.entropy_weight = entropy_weight
        self.gamma = gamma
        self.checkpoint_dir = checkpoint_dir
        
        # Create optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Create reward functions
        self.asset_reward_fn = AssetScoringReward()
        self.market_reward_fn = MarketScoringReward(reward_type)
        
        # Create directory for checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def train_epoch(self, 
                   dataloader: torch.utils.data.DataLoader,
                   env: TradingEnvironment) -> Dict[str, float]:
        """
        Train model for one epoch.
        
        Args:
            dataloader: Dataloader for training data
            env: Trading environment
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_asset_reward = 0.0
        total_market_reward = 0.0
        total_entropy = 0.0
        
        # Reset environment
        state = env.reset()
        
        # Store for market scoring reward calculation
        portfolio_values = [env._get_portfolio_value()]
        
        for batch in tqdm(dataloader, desc="Training"):
            # Extract batch data
            stock_features = batch["stock_features"].to(self.device)
            market_features = batch["market_features"].to(self.device)
            target_returns = batch["target_returns"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            output = self.model(stock_features, market_features)
            
            # Extract portfolio weights and distribution parameters
            asset_scores = output["asset_scores"]
            long_weights = output["long_weights"]
            short_weights = output["short_weights"]
            short_ratio = output["short_ratio"]
            mu = output["mu"]
            sigma = output["sigma"]
            
            # Execute actions in environment
            batch_size = stock_features.size(0)
            batch_returns = []
            
            for b in range(batch_size):
                action = {
                    "long_weights": long_weights[b].cpu().detach().numpy(),
                    "short_weights": short_weights[b].cpu().detach().numpy(),
                    "short_ratio": short_ratio[b].cpu().detach().numpy(),
                }
                
                # Execute action in environment
                prices = target_returns[b].cpu().detach().numpy() + 1.0  # Convert returns to price relatives
                next_state, reward, done, info = env.step(action, prices)
                
                # Store portfolio value for market scoring reward
                portfolio_values.append(info["portfolio_value"])
                batch_returns.append(reward)
            
            # Calculate rewards
            # Asset Scoring Reward
            asset_reward = self.asset_reward_fn(asset_scores, target_returns)
            
            # Market Scoring Reward
            market_reward = self.market_reward_fn(portfolio_values)
            
            # Calculate loss for Asset Scoring Unit (Policy Gradient)
            asset_loss = -torch.mean(asset_reward)
            
            # Calculate loss for Market Scoring Unit (Policy Gradient with Gaussian Policy)
            # Log probability of short ratio under Gaussian distribution
            log_prob = -0.5 * ((short_ratio - mu) / sigma).pow(2) - 0.5 * torch.log(2 * np.pi * sigma.pow(2))
            
            # Entropy of Gaussian distribution
            entropy = 0.5 * torch.log(2 * np.pi * np.e * sigma.pow(2))
            
            # Market loss
            market_loss = -market_reward * log_prob.mean()
            
            # Total loss
            loss = asset_loss + self.market_reward_weight * market_loss - self.entropy_weight * entropy.mean()
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_asset_reward += asset_reward.mean().item()
            total_market_reward += market_reward
            total_entropy += entropy.mean().item()
        
        # Calculate average metrics
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_asset_reward = total_asset_reward / num_batches
        avg_market_reward = total_market_reward / num_batches
        avg_entropy = total_entropy / num_batches
        
        return {
            "loss": avg_loss,
            "asset_reward": avg_asset_reward,
            "market_reward": avg_market_reward,
            "entropy": avg_entropy,
            "final_portfolio_value": portfolio_values[-1]
        }
    
    def train(self, 
              train_dataloader: torch.utils.data.DataLoader,
              val_dataloader: Optional[torch.utils.data.DataLoader] = None,
              num_epochs: int = 100,
              eval_freq: int = 5,
              save_freq: int = 10,
              ) -> Dict[str, List[float]]:
        """
        Train model for multiple epochs.
        
        Args:
            train_dataloader: Dataloader for training data
            val_dataloader: Dataloader for validation data
            num_epochs: Number of epochs to train
            eval_freq: Frequency of evaluation
            save_freq: Frequency of saving checkpoints
            
        Returns:
            Dictionary with training history
        """
        # Create environment
        env = TradingEnvironment(
            tickers=self.model.tickers if hasattr(self.model, 'tickers') else None,
            initial_capital=10000
        )
        
        # Training history
        history = {
            "train_loss": [],
            "train_asset_reward": [],
            "train_market_reward": [],
            "train_entropy": [],
            "train_portfolio_value": [],
            "val_loss": [],
            "val_asset_reward": [],
            "val_market_reward": [],
            "val_entropy": [],
            "val_portfolio_value": []
        }
        
        # Start training
        best_val_reward = float('-inf')
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train epoch
            train_metrics = self.train_epoch(train_dataloader, env)
            
            # Record training metrics
            history["train_loss"].append(train_metrics["loss"])
            history["train_asset_reward"].append(train_metrics["asset_reward"])
            history["train_market_reward"].append(train_metrics["market_reward"])
            history["train_entropy"].append(train_metrics["entropy"])
            history["train_portfolio_value"].append(train_metrics["final_portfolio_value"])
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                 f"Asset Reward: {train_metrics['asset_reward']:.4f}, "
                 f"Market Reward: {train_metrics['market_reward']:.4f}, "
                 f"Entropy: {train_metrics['entropy']:.4f}, "
                 f"Portfolio Value: {train_metrics['final_portfolio_value']:.2f}")
            
            # Evaluate on validation set
            if val_dataloader is not None and (epoch + 1) % eval_freq == 0:
                self.model.eval()
                
                # Reset environment
                env.reset()
                portfolio_values = [env._get_portfolio_value()]
                
                # Validation metrics
                val_loss = 0.0
                val_asset_reward = 0.0
                val_entropy = 0.0
                
                with torch.no_grad():
                    for batch in val_dataloader:
                        # Extract batch data
                        stock_features = batch["stock_features"].to(self.device)
                        market_features = batch["market_features"].to(self.device)
                        target_returns = batch["target_returns"].to(self.device)
                        
                        # Forward pass
                        output = self.model(stock_features, market_features)
                        
                        # Extract portfolio weights and distribution parameters
                        asset_scores = output["asset_scores"]
                        long_weights = output["long_weights"]
                        short_weights = output["short_weights"]
                        short_ratio = output["short_ratio"]
                        mu = output["mu"]
                        sigma = output["sigma"]
                        
                        # Execute actions in environment
                        batch_size = stock_features.size(0)
                        
                        for b in range(batch_size):
                            action = {
                                "long_weights": long_weights[b].cpu().numpy(),
                                "short_weights": short_weights[b].cpu().numpy(),
                                "short_ratio": short_ratio[b].cpu().numpy(),
                            }
                            
                            # Execute action in environment
                            prices = target_returns[b].cpu().numpy() + 1.0
                            next_state, reward, done, info = env.step(action, prices)
                            
                            # Store portfolio value
                            portfolio_values.append(info["portfolio_value"])
                        
                        # Calculate rewards
                        asset_reward = self.asset_reward_fn(asset_scores, target_returns)
                        
                        # Calculate entropy
                        entropy = 0.5 * torch.log(2 * np.pi * np.e * sigma.pow(2))
                        
                        # Track metrics
                        val_asset_reward += asset_reward.mean().item()
                        val_entropy += entropy.mean().item()
                
                # Calculate market reward
                val_market_reward = self.market_reward_fn(portfolio_values)
                
                # Calculate average metrics
                num_batches = len(val_dataloader)
                val_asset_reward /= num_batches
                val_entropy /= num_batches
                
                # Record validation metrics
                history["val_asset_reward"].append(val_asset_reward)
                history["val_market_reward"].append(val_market_reward)
                history["val_entropy"].append(val_entropy)
                history["val_portfolio_value"].append(portfolio_values[-1])
                
                print(f"Validation - Asset Reward: {val_asset_reward:.4f}, "
                     f"Market Reward: {val_market_reward:.4f}, "
                     f"Entropy: {val_entropy:.4f}, "
                     f"Portfolio Value: {portfolio_values[-1]:.2f}")
                
                # Save best model
                combined_reward = val_asset_reward + self.market_reward_weight * val_market_reward
                if combined_reward > best_val_reward:
                    best_val_reward = combined_reward
                    self.save_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pth"))
                    print(f"Saved best model with combined reward: {combined_reward:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}.pth"))
        
        return history
    
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def prepare_dataloader(dataset: List[Dict], batch_size: int, shuffle: bool = True) -> torch.utils.data.DataLoader:
    """
    Prepare dataloader from dataset.
    
    Args:
        dataset: List of data samples
        batch_size: Batch size
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader
    """
    class FinancialDataset(torch.utils.data.Dataset):
        def __init__(self, data: List[Dict]):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            
            return {
                "stock_features": torch.tensor(item["stock_features"], dtype=torch.float32),
                "market_features": torch.tensor(item["market_features"], dtype=torch.float32),
                "target_returns": torch.tensor(item["target_returns"], dtype=torch.float32),
                "date": item["date"]
            }
    
    dataset = FinancialDataset(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    return dataloader
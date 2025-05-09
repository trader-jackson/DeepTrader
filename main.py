"""
Main entry point for training and evaluating DeepTrader model.
"""

import os
import argparse
import torch
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from config import DATASETS, MODEL_CONFIG
from src.data_processor import DataProcessor
from src.models.deeptrader import DeepTrader
from src.environment import TradingEnvironment
from src.train import DeepTraderTrainer, prepare_dataloader
from src.evaluation import Evaluator, ComparativeEvaluator, evaluate_baseline_strategies


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    """
    Train DeepTrader model.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = os.path.join(
        args.output_dir, 
        f"{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    config = {
        "dataset": args.dataset,
        "graph_type": args.graph_type,
        "reward_type": args.reward_type,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "market_reward_weight": args.market_reward_weight,
        "winner_size": args.winner_size,
        "seed": args.seed
    }
    
    pd.DataFrame([config]).to_csv(os.path.join(output_dir, "config.csv"), index=False)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare data
    print(f"Preparing data for {args.dataset}...")
    data_processor = DataProcessor(args.dataset)
    
    # Download data if needed
    if args.download:
        train_start, train_end = DATASETS[args.dataset]["train_period"]
        test_start, test_end = DATASETS[args.dataset]["test_period"]
        data_processor.download_data(train_start, test_end)
    
    # Prepare data
    data = data_processor.prepare_data(
        graph_type=args.graph_type,
        sequence_length=MODEL_CONFIG["train"]["sequence_length"]
    )
    
    if data is None:
        print("Error preparing data")
        return
    
    dataset = data["dataset"]
    adj_matrix = torch.tensor(data["adj_matrix"], dtype=torch.float32).to(device)
    tickers = data["tickers"]
    
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Prepare dataloaders
    train_dataloader = prepare_dataloader(train_dataset, args.batch_size, shuffle=True)
    test_dataloader = prepare_dataloader(test_dataset, args.batch_size, shuffle=False)
    
    # Create model
    model = DeepTrader(
        tickers=tickers,
        stock_feature_dim=train_dataset[0]["stock_features"].shape[1],
        market_feature_dim=train_dataset[0]["market_features"].shape[1],
        hidden_dim=MODEL_CONFIG["asset_scoring"]["hidden_dim"],
        num_layers_asset=MODEL_CONFIG["asset_scoring"]["num_layers"],
        num_layers_market=MODEL_CONFIG["market_scoring"]["num_layers"],
        dropout=MODEL_CONFIG["asset_scoring"]["dropout"],
        winner_size=args.winner_size,
        adj_matrix=adj_matrix,
        graph_type=args.graph_type
    ).to(device)
    
    # Create trainer
    trainer = DeepTraderTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        reward_type=args.reward_type,
        market_reward_weight=args.market_reward_weight,
        checkpoint_dir=os.path.join(output_dir, "checkpoints")
    )
    
    # Train model
    print("Training model...")
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq
    )
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(output_dir, "history.csv"), index=False)
    
    # Plot training history
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(history["train_asset_reward"], label="Train Asset Reward")
    plt.plot(history["train_market_reward"], label="Train Market Reward")
    if history["val_asset_reward"]:
        epochs = list(range(args.eval_freq - 1, args.num_epochs, args.eval_freq))
        plt.plot(epochs, history["val_asset_reward"], label="Val Asset Reward")
        plt.plot(epochs, history["val_market_reward"], label="Val Market Reward")
    plt.title("Rewards during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "rewards.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.title("Loss during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "loss.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(history["train_entropy"], label="Train Entropy")
    if history["val_entropy"]:
        plt.plot(epochs, history["val_entropy"], label="Val Entropy")
    plt.title("Entropy during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Entropy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "entropy.png"))
    plt.close()
    
    plt.figure(figsize=(12, 6))
    plt.plot(history["train_portfolio_value"], label="Train Portfolio Value")
    if history["val_portfolio_value"]:
        plt.plot(epochs, history["val_portfolio_value"], label="Val Portfolio Value")
    plt.title("Portfolio Value during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, "portfolio_value.png"))
    plt.close()
    
    print(f"Training completed. Results saved to {output_dir}")


def evaluate(args):
    """
    Evaluate trained DeepTrader model.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(
        args.output_dir, 
        f"eval_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    print(f"Preparing data for {args.dataset}...")
    data_processor = DataProcessor(args.dataset)
    
    # Prepare data
    data = data_processor.prepare_data(
        graph_type=args.graph_type,
        sequence_length=MODEL_CONFIG["train"]["sequence_length"]
    )
    
    if data is None:
        print("Error preparing data")
        return
    
    dataset = data["dataset"]
    adj_matrix = torch.tensor(data["adj_matrix"], dtype=torch.float32).to(device)
    tickers = data["tickers"]
    
    test_dataset = dataset["test"]
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    model = DeepTrader(
        num_stocks=len(tickers),
        stock_feature_dim=test_dataset[0]["stock_features"].shape[1],
        market_feature_dim=test_dataset[0]["market_features"].shape[1],
        hidden_dim=MODEL_CONFIG["asset_scoring"]["hidden_dim"],
        num_layers_asset=MODEL_CONFIG["asset_scoring"]["num_layers"],
        num_layers_market=MODEL_CONFIG["market_scoring"]["num_layers"],
        dropout=MODEL_CONFIG["asset_scoring"]["dropout"],
        winner_size=args.winner_size,
        adj_matrix=adj_matrix,
        graph_type=args.graph_type
    ).to(device)
    
    # Load model weights
    if args.model_path is None:
        print("No model path provided. Using random weights.")
    else:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded model from {args.model_path}")
    
    # Create evaluator
    evaluator = Evaluator(model, tickers, device)
    
    # Create environment
    env = TradingEnvironment(
        tickers=tickers,
        initial_capital=args.initial_capital
    )
    
    # Evaluate model
    print("Evaluating model...")
    metrics = evaluator.evaluate(test_dataset, env, output_dir)
    
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Evaluate baseline strategies
    if args.baselines:
        print("\nEvaluating baseline strategies...")
        baseline_metrics = evaluate_baseline_strategies(
            test_dataset, tickers, args.initial_capital, 
            os.path.join(output_dir, "baselines")
        )
        
        # Compare with baselines
        print("\nComparison with baselines:")
        comparison = pd.DataFrame({
            "DeepTrader": metrics,
            **baseline_metrics
        }).T
        
        comparison.to_csv(os.path.join(output_dir, "comparison.csv"))
        print(comparison[["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "calmar_ratio"]])
        
        # Plot comparison
        compare_dir = os.path.join(output_dir, "comparison_plots")
        os.makedirs(compare_dir, exist_ok=True)
        
        metrics_to_plot = [
            "annualized_return", 
            "annualized_volatility", 
            "sharpe_ratio", 
            "sortino_ratio", 
            "max_drawdown", 
            "calmar_ratio"
        ]
        
        for metric in metrics_to_plot:
            plt.figure(figsize=(10, 6))
            data = comparison[metric]
            plt.bar(data.index, data.values)
            plt.title(f"Comparison of {metric}")
            plt.ylabel(metric)
            plt.grid(True, axis="y")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(compare_dir, f"{metric}.png"))
            plt.close()
    
    print(f"Evaluation completed. Results saved to {output_dir}")


def ablation(args):
    """
    Perform ablation study on DeepTrader components.
    
    Args:
        args: Command-line arguments
    """
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = os.path.join(
        args.output_dir, 
        f"ablation_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and prepare data
    print(f"Preparing data for {args.dataset}...")
    data_processor = DataProcessor(args.dataset)
    
    # Prepare data
    data = data_processor.prepare_data(
        graph_type=args.graph_type,
        sequence_length=MODEL_CONFIG["train"]["sequence_length"]
    )
    
    if data is None:
        print("Error preparing data")
        return
    
    dataset = data["dataset"]
    tickers = data["tickers"]
    
    test_dataset = dataset["test"]
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Define ablation configurations
    ablation_configs = {}
    
    # Graph type ablation
    for graph_type in ['industry', 'correlation', 'partial_correlation', 'causal']:
        print(f"\nPreparing data with graph_type={graph_type}...")
        graph_data = data_processor.prepare_data(
            graph_type=graph_type,
            sequence_length=MODEL_CONFIG["train"]["sequence_length"]
        )
        
        if graph_data is None:
            print(f"Error preparing data for graph_type={graph_type}")
            continue
        
        adj_matrix = torch.tensor(graph_data["adj_matrix"], dtype=torch.float32).to(device)
        
        ablation_configs[f"DT-{graph_type}"] = {
            "adj_matrix": adj_matrix,
            "graph_type": graph_type,
            "use_market_scoring": True,
            "use_asset_scoring": True
        }
    
    # Component ablation
    causal_data = data_processor.prepare_data(
        graph_type='causal',
        sequence_length=MODEL_CONFIG["train"]["sequence_length"]
    )
    
    if causal_data is not None:
        causal_adj_matrix = torch.tensor(causal_data["adj_matrix"], dtype=torch.float32).to(device)
        
        ablation_configs["DT-NS"] = {
            "adj_matrix": causal_adj_matrix,
            "graph_type": "correlation",  # Just use correlation for DT-NS (no spatial attention)
            "use_market_scoring": True,
            "use_asset_scoring": True
        }
        
        ablation_configs["DT-NM"] = {
            "adj_matrix": causal_adj_matrix,
            "graph_type": "causal",
            "use_market_scoring": False,
            "use_asset_scoring": True
        }
        
        ablation_configs["DT"] = {
            "adj_matrix": causal_adj_matrix,
            "graph_type": "causal",
            "use_market_scoring": True,
            "use_asset_scoring": True
        }
    
    # Create models
    models = {}
    for name, config in ablation_configs.items():
        print(f"Creating model {name}...")
        
        model = DeepTrader(
            num_stocks=len(tickers),
            stock_feature_dim=test_dataset[0]["stock_features"].shape[1],
            market_feature_dim=test_dataset[0]["market_features"].shape[1],
            hidden_dim=MODEL_CONFIG["asset_scoring"]["hidden_dim"],
            num_layers_asset=MODEL_CONFIG["asset_scoring"]["num_layers"],
            num_layers_market=MODEL_CONFIG["market_scoring"]["num_layers"],
            dropout=MODEL_CONFIG["asset_scoring"]["dropout"],
            winner_size=args.winner_size,
            adj_matrix=config["adj_matrix"],
            graph_type=config["graph_type"]
        ).to(device)
        
        models[name] = model
    
    # Load model weights if available
    if args.model_path is not None:
        checkpoint = torch.load(args.model_path, map_location=device)
        for name, model in models.items():
            try:
                model.load_state_dict(checkpoint["model_state_dict"])
                print(f"Loaded weights for {name} from {args.model_path}")
            except:
                print(f"Could not load weights for {name}. Using random weights.")
    
    # Create comparative evaluator
    comp_evaluator = ComparativeEvaluator(tickers, device)
    
    # Evaluate models
    print("Evaluating models...")
    results = comp_evaluator.evaluate_models(
        models=models,
        dataset=test_dataset,
        initial_capital=args.initial_capital,
        output_dir=output_dir
    )
    
    print("\nAblation study results:")
    comparison = pd.DataFrame(results).T
    comparison.to_csv(os.path.join(output_dir, "ablation_results.csv"))
    print(comparison[["annualized_return", "annualized_volatility", "sharpe_ratio", "max_drawdown", "calmar_ratio"]])
    
    print(f"Ablation study completed. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train DeepTrader model")
    train_parser.add_argument("--dataset", type=str, default="dow30", choices=["dow30", "nasdaq100", "sse50"], help="Dataset to use")
    train_parser.add_argument("--graph-type", type=str, default="causal", choices=["industry", "correlation", "partial_correlation", "causal"], help="Graph type for asset scoring unit")
    train_parser.add_argument("--reward-type", type=str, default="mdd", choices=["ror", "sr", "mdd", "cr"], help="Reward type for market scoring unit")
    train_parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--market-reward-weight", type=float, default=0.5, help="Weight for market scoring reward")
    train_parser.add_argument("--winner-size", type=int, default=5, help="Number of stocks to select as winners/losers")
    train_parser.add_argument("--eval-freq", type=int, default=5, help="Frequency of evaluation during training")
    train_parser.add_argument("--save-freq", type=int, default=10, help="Frequency of saving checkpoints")
    train_parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save results")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    train_parser.add_argument("--download", action="store_true", help="Download data before training")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate DeepTrader model")
    eval_parser.add_argument("--dataset", type=str, default="dow30", choices=["dow30", "nasdaq100", "sse50"], help="Dataset to use")
    eval_parser.add_argument("--model-path", type=str, default=None, help="Path to saved model checkpoint")
    eval_parser.add_argument("--graph-type", type=str, default="causal", choices=["industry", "correlation", "partial_correlation", "causal"], help="Graph type for asset scoring unit")
    eval_parser.add_argument("--winner-size", type=int, default=5, help="Number of stocks to select as winners/losers")
    eval_parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital for environment")
    eval_parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save results")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    eval_parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    eval_parser.add_argument("--baselines", action="store_true", help="Evaluate baseline strategies")
    
    # Ablation command
    ablation_parser = subparsers.add_parser("ablation", help="Perform ablation study on DeepTrader")
    ablation_parser.add_argument("--dataset", type=str, default="dow30", choices=["dow30", "nasdaq100", "sse50"], help="Dataset to use")
    ablation_parser.add_argument("--model-path", type=str, default=None, help="Path to saved model checkpoint")
    ablation_parser.add_argument("--graph-type", type=str, default="causal", choices=["industry", "correlation", "partial_correlation", "causal"], help="Graph type for asset scoring unit")
    ablation_parser.add_argument("--winner-size", type=int, default=5, help="Number of stocks to select as winners/losers")
    ablation_parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital for environment")
    ablation_parser.add_argument("--output-dir", type=str, default="./results", help="Directory to save results")
    ablation_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    ablation_parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    elif args.command == "ablation":
        ablation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
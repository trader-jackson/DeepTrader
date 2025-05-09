"""
Asset Scoring Unit for DeepTrader.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism as described in the paper.
    """
    
    def __init__(self, num_stocks: int, hidden_dim: int, seq_length: int):
        """
        Initialize spatial attention module.
        
        Args:
            num_stocks: Number of stocks
            hidden_dim: Hidden dimension
            seq_length: Sequence length
        """
        super(SpatialAttention, self).__init__()
        
        self.num_stocks = num_stocks
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        
        # Parameters for spatial attention
        # Change W1 to have 2 dimensions instead of 1
        self.W1 = nn.Parameter(torch.Tensor(1, seq_length))
        self.W2 = nn.Parameter(torch.Tensor(hidden_dim, seq_length))
        self.W3 = nn.Parameter(torch.Tensor(hidden_dim))
        self.Vs = nn.Parameter(torch.Tensor(num_stocks, num_stocks))
        self.bs = nn.Parameter(torch.Tensor(num_stocks, num_stocks))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        # Use different initialization for W1 since it's now 2D
        nn.init.xavier_uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        
        # For 1D tensors, use normal initialization instead of xavier
        # since xavier requires at least 2D tensors
        nn.init.normal_(self.W3, mean=0.0, std=0.01)
        
        nn.init.xavier_uniform_(self.Vs)
        nn.init.zeros_(self.bs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, num_stocks, hidden_dim, seq_length]
            
        Returns:
            Attention weights of shape [batch_size, num_stocks, num_stocks]
        """
        batch_size = x.size(0)
        
        # x_transpose: [batch_size, hidden_dim, seq_length, num_stocks]
        x_transpose = x.permute(0, 2, 3, 1)
        
        # Compute attention weights
        # Step 1: W1 * x (adjust calculation for 2D W1)
        # Use squeeze to handle the first dimension of W1
        step1 = torch.matmul(x, self.W1.squeeze(0))
        
        # Step 2: W2 * (W3 * x_transpose) -> [batch_size, seq_length, num_stocks]
        step2 = torch.matmul(self.W3, x_transpose)
        step2 = torch.matmul(self.W2, step2)
        
        # Step 3: Combine and add bias -> [batch_size, num_stocks, num_stocks]
        step3 = torch.matmul(step1, step2)
        step3 = step3 + self.bs.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Step 4: Apply sigmoid and scale by Vs
        att_weights = torch.sigmoid(step3)
        att_weights = att_weights * self.Vs.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Normalize with softmax over the last dimension
        att_weights = F.softmax(att_weights, dim=2)
        
        return att_weights


class GraphConvolution(nn.Module):
    """
    Graph Convolution Layer as described in the paper.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initialize graph convolution layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            bias: Whether to use bias
        """
        super(GraphConvolution, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input: Input tensor of shape [batch_size, num_nodes, in_features]
            adj: Adjacency matrix of shape [batch_size, num_nodes, num_nodes]
            
        Returns:
            Output tensor of shape [batch_size, num_nodes, out_features]
        """
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class DilatedCausalConv1d(nn.Module):
    """
    Dilated Causal Convolution layer as described in the paper.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 dilation: int = 1, bias: bool = True):
        """
        Initialize dilated causal convolution layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            dilation: Dilation factor
            bias: Whether to use bias
        """
        super(DilatedCausalConv1d, self).__init__()
        
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, seq_length]
            
        Returns:
            Output tensor of shape [batch_size, out_channels, seq_length]
        """
        # Apply convolution
        output = self.conv(x)
        
        # Remove padding at the end to ensure causality
        if self.padding != 0:
            output = output[:, :, :-self.padding]
        
        return output


class SpatialTCNBlock(nn.Module):
    """
    Spatial-TCN block as described in the paper.
    """
    
    def __init__(self, num_stocks: int, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, adj_matrix: torch.Tensor,
                 dropout: float = 0.1, graph_type: str = 'industry'):
        """
        Initialize Spatial-TCN block.
        
        Args:
            num_stocks: Number of stocks
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            dilation: Dilation factor
            adj_matrix: Adjacency matrix
            dropout: Dropout rate
            graph_type: Type of graph structure
        """
        super(SpatialTCNBlock, self).__init__()
        
        self.num_stocks = num_stocks
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.adj_matrix = adj_matrix
        self.graph_type = graph_type
        
        # Temporal convolution layer
        self.tcn = DilatedCausalConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        
        # Spatial attention layer
        self.spatial_attn = SpatialAttention(
            num_stocks=num_stocks,
            hidden_dim=out_channels,
            seq_length=1  # We apply attention after TCN
        )
        
        # Graph convolution layers
        if graph_type in ['industry', 'correlation', 'partial_correlation']:
            # Using static graph structure
            self.gcn1 = GraphConvolution(out_channels, out_channels)
        else:  # 'causal'
            # Using dynamic learned graph structure
            self.gcn1 = GraphConvolution(out_channels, out_channels)
            self.gcn2 = GraphConvolution(out_channels, out_channels)
            
            # Parameters for learned correlation
            self.E = nn.Parameter(torch.Tensor(num_stocks))
            nn.init.normal_(self.E)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, num_stocks, in_channels, seq_length]
            
        Returns:
            Output tensor of shape [batch_size, num_stocks, out_channels, seq_length]
        """
        batch_size = x.size(0)
        seq_length = x.size(3)
        
        # Process each stock's time series with TCN
        out = []
        for i in range(self.num_stocks):
            # Extract time series for stock i
            # Shape: [batch_size, in_channels, seq_length]
            stock_i = x[:, i, :, :]
            
            # Apply TCN
            # Shape: [batch_size, out_channels, seq_length]
            tcn_i = self.tcn(stock_i)
            
            out.append(tcn_i)
        
        # Stack outputs
        # Shape: [batch_size, num_stocks, out_channels, seq_length]
        out = torch.stack(out, dim=1)
        
        # Apply spatial attention
        # First reshape to fit the attention module
        out_attn = out.permute(0, 3, 1, 2)  # [batch_size, seq_length, num_stocks, out_channels]
        
        # Process each time step
        spatial_out = []
        for t in range(seq_length):
            # Extract features at time step t
            # Shape: [batch_size, num_stocks, out_channels]
            features_t = out_attn[:, t, :, :]
            
            # Apply spatial attention
            # Shape: [batch_size, num_stocks, num_stocks]
            attn_weights = self.spatial_attn(features_t.unsqueeze(-1))
            
            # Apply graph convolution
            if self.graph_type in ['industry', 'correlation', 'partial_correlation']:
                # Use static adjacency matrix
                adj = self.adj_matrix.unsqueeze(0).expand(batch_size, -1, -1).to(features_t.device)
                gcn_out = self.gcn1(features_t, attn_weights * adj)
            else:  # 'causal'
                # Compute dynamic adjacency matrix
                E_matrix = torch.outer(self.E, self.E)
                A_c = F.softmax(F.relu(E_matrix), dim=1)
                
                # Combine with attention weights
                adj = attn_weights * A_c.unsqueeze(0).expand(batch_size, -1, -1).to(features_t.device)
                
                # Apply GCN with both matrices
                gcn_out1 = self.gcn1(features_t, adj)
                gcn_out2 = self.gcn2(features_t, self.adj_matrix.unsqueeze(0).expand(batch_size, -1, -1).to(features_t.device))
                
                # Combine outputs
                gcn_out = gcn_out1 + gcn_out2
            
            spatial_out.append(gcn_out)
        
        # Stack outputs
        # Shape: [batch_size, seq_length, num_stocks, out_channels]
        spatial_out = torch.stack(spatial_out, dim=1)
        
        # Reshape to original format
        # Shape: [batch_size, num_stocks, out_channels, seq_length]
        spatial_out = spatial_out.permute(0, 2, 3, 1)
        
        # Apply dropout
        out = self.dropout(spatial_out)
        
        return out


class AssetScoringUnit(nn.Module):
    """
    Asset Scoring Unit as described in the paper.
    """
    
    def __init__(self, num_stocks: int, input_dim: int, hidden_dim: int, 
                 kernel_size: int = 3, num_layers: int = 4, dilation_base: int = 2, 
                 dropout: float = 0.1, adj_matrix: torch.Tensor = None,
                 graph_type: str = 'industry'):
        """
        Initialize Asset Scoring Unit.
        
        Args:
            num_stocks: Number of stocks
            input_dim: Number of input features
            hidden_dim: Hidden dimension
            kernel_size: Size of the convolutional kernel
            num_layers: Number of Spatial-TCN blocks
            dilation_base: Base for dilation in TCN
            dropout: Dropout rate
            adj_matrix: Adjacency matrix
            graph_type: Type of graph structure
        """
        super(AssetScoringUnit, self).__init__()
        
        self.num_stocks = num_stocks
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Spatial-TCN blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilation_base ** i
            block = SpatialTCNBlock(
                num_stocks=num_stocks,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                adj_matrix=adj_matrix,
                dropout=dropout,
                graph_type=graph_type
            )
            self.blocks.append(block)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, num_stocks, input_dim, seq_length]
            
        Returns:
            Asset scores of shape [batch_size, num_stocks]
        """
        batch_size = x.size(0)
        seq_length = x.size(3)
        
        # Project input features
        # Process each stock and time step
        projected = []
        for i in range(self.num_stocks):
            stock_i = []
            for t in range(seq_length):
                # Shape: [batch_size, input_dim]
                features = x[:, i, :, t]
                
                # Project to hidden dimension
                # Shape: [batch_size, hidden_dim]
                proj = self.input_proj(features)
                stock_i.append(proj)
            
            # Stack time steps
            # Shape: [batch_size, hidden_dim, seq_length]
            stock_i = torch.stack(stock_i, dim=2)
            projected.append(stock_i)
        
        # Stack stocks
        # Shape: [batch_size, num_stocks, hidden_dim, seq_length]
        h = torch.stack(projected, dim=1)
        
        # Apply Spatial-TCN blocks with residual connections
        for block in self.blocks:
            h = h + block(h)
        
        # Extract the last time step
        # Shape: [batch_size, num_stocks, hidden_dim]
        h_last = h[:, :, :, -1]
        
        # Apply output layer
        # Shape: [batch_size, num_stocks, 1]
        scores = self.output_layer(h_last)
        
        # Squeeze the last dimension
        # Shape: [batch_size, num_stocks]
        scores = scores.squeeze(-1)
        
        # Apply sigmoid to scale scores to [0, 1]
        scores = torch.sigmoid(scores)
        
        return scores
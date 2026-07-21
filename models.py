"""
All model architectures in one file.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math


# ==================== TCN Components ====================

class Chomp1d(nn.Module):
    """Remove padding from end for causal convolutions."""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Single TCN residual block with two dilated causal convolutions."""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                          stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Full TCN with stacked temporal blocks.
    
    Args:
        num_inputs: Input channels
        num_channels: List of output channels per layer
        kernel_size: Convolutional kernel size
        dropout: Dropout probability
    """
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            ))

        self.network = nn.Sequential(*layers)
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.num_levels = num_levels
        self.receptive_field = self._calculate_rf()

    def _calculate_rf(self):
        """RF = 1 + 2*(kernel_size - 1)*sum(dilations)"""
        dilations = [2**i for i in range(self.num_levels)]
        return 1 + 2 * (self.kernel_size - 1) * sum(dilations)

    def forward(self, x):
        return self.network(x)


# ==================== Attention ====================

class MultiHeadSpatialAttention(nn.Module):
    """
    Multi-head attention over spatial (electrode) dimension.
    
    Args:
        num_channels: TCN output channels
        num_electrodes: Number of EMG electrodes
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    def __init__(self, num_channels, num_electrodes, num_heads=4, dropout=0.1):
        super().__init__()
        assert num_channels % num_heads == 0
        
        self.num_channels = num_channels
        self.num_electrodes = num_electrodes
        self.num_heads = num_heads
        self.head_dim = num_channels // num_heads
        
        self.query = nn.Linear(num_channels, num_channels)
        self.key = nn.Linear(num_channels, num_channels)
        self.value = nn.Linear(num_channels, num_channels)
        self.out_proj = nn.Linear(num_channels, num_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, electrodes, time)
        Returns:
            (batch, channels, time)
        """
        batch_size, channels, num_electrodes, time_steps = x.shape
        
        # (batch, time, electrodes, channels)
        x = x.permute(0, 3, 2, 1)
        
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Reshape for multi-head: (batch, time, electrodes, heads, head_dim)
        Q = Q.view(batch_size, time_steps, num_electrodes, self.num_heads, self.head_dim)
        K = K.view(batch_size, time_steps, num_electrodes, self.num_heads, self.head_dim)
        V = V.view(batch_size, time_steps, num_electrodes, self.num_heads, self.head_dim)
        
        # (batch, time, heads, electrodes, head_dim)
        Q = Q.permute(0, 1, 3, 2, 4)
        K = K.permute(0, 1, 3, 2, 4)
        V = V.permute(0, 1, 3, 2, 4)
        
        # Attention: (batch, time, heads, electrodes, electrodes)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # (batch, time, heads, electrodes, head_dim)
        attended = torch.matmul(attn_weights, V)
        
        # Concatenate heads: (batch, time, electrodes, channels)
        attended = attended.permute(0, 1, 3, 2, 4).contiguous()
        attended = attended.view(batch_size, time_steps, num_electrodes, channels)
        
        out = self.out_proj(attended)
        
        # Global average pool over electrodes: (batch, time, channels)
        out = out.mean(dim=2)
        
        # (batch, channels, time)
        return out.permute(0, 2, 1)


# ==================== Hybrid Model ====================

class HybridTCNAttention(nn.Module):
    """
    Complete hybrid architecture: TCN + Spatial Attention + Classifier.
    
    Args:
        num_electrodes: Number of EMG electrodes (16 for NinaPro DB8)
        num_classes: Number of gesture classes (52 for NinaPro DB8)
        tcn_channels: List of TCN channel sizes
        kernel_size: TCN kernel size
        dropout: Dropout probability
        num_heads: Number of attention heads
    """
    def __init__(self, num_electrodes=16, num_classes=52, tcn_channels=[64, 64, 128, 128, 256, 256, 256, 256],
                 kernel_size=3, dropout=0.2, num_heads=4):
        super().__init__()
        
        self.num_electrodes = num_electrodes
        self.num_classes = num_classes
        
        # TCN for temporal features
        self.tcn = TemporalConvNet(
            num_inputs=1,
            num_channels=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Spatial attention
        self.attention = MultiHeadSpatialAttention(
            num_channels=tcn_channels[-1],
            num_electrodes=num_electrodes,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(tcn_channels[-1], num_classes)
        )
        
        self.receptive_field = self.tcn.receptive_field
        
    def forward(self, x):
        """
        Args:
            x: (batch, electrodes, time)
        Returns:
            (batch, num_classes)
        """
        batch_size, num_electrodes, time_steps = x.shape
        
        # (batch, electrodes, 1, time)
        x = x.unsqueeze(2)
        
        # Apply TCN per electrode: (batch*electrodes, 1, time)
        x_reshaped = x.view(batch_size * num_electrodes, 1, time_steps)
        tcn_out = self.tcn(x_reshaped)
        
        # Reshape: (batch, electrodes, tcn_channels, time)
        _, tcn_channels, tcn_time = tcn_out.shape
        tcn_out = tcn_out.view(batch_size, num_electrodes, tcn_channels, tcn_time)
        
        # (batch, tcn_channels, electrodes, time)
        tcn_out = tcn_out.permute(0, 2, 1, 3)
        
        # Attention: (batch, tcn_channels, time)
        attended = self.attention(tcn_out)
        
        # Classify: (batch, num_classes)
        return self.classifier(attended)
    
    def summary(self):
        """Print parameter summary."""
        tcn_params = sum(p.numel() for p in self.tcn.parameters())
        attn_params = sum(p.numel() for p in self.attention.parameters())
        clf_params = sum(p.numel() for p in self.classifier.parameters())
        total = sum(p.numel() for p in self.parameters())
        
        print(f"\nModel Summary:")
        print(f"  TCN:        {tcn_params:>10,} params")
        print(f"  Attention:  {attn_params:>10,} params")
        print(f"  Classifier: {clf_params:>10,} params")
        print(f"  Total:      {total:>10,} params")
        print(f"  RF:         {self.receptive_field:>10,} samples ({self.receptive_field/2000*1000:.1f} ms)\n")


def load_model_from_config(config_name, config_dict, num_electrodes=16, num_classes=52):
    """Load model from config dictionary."""
    cfg = config_dict['rf_configs'][config_name]
    model_cfg = config_dict['model']
    
    return HybridTCNAttention(
        num_electrodes=num_electrodes,
        num_classes=num_classes,
        tcn_channels=cfg['channels'],
        kernel_size=model_cfg['kernel_size'],
        dropout=model_cfg['dropout'],
        num_heads=model_cfg['num_heads']
    )
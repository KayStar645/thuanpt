"""
Module chứa các lớp feed-forward network.
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class PointWiseFeedForward(nn.Module):
    """Point-wise Feed-Forward Network.
    
    Args:
        d_model (int): Kích thước input và output
        d_ff (int): Kích thước hidden layer
        dropout (float): Tỷ lệ dropout
        activation (str): Hàm kích hoạt ('relu' hoặc 'gelu')
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Chọn hàm kích hoạt
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Không hỗ trợ hàm kích hoạt: {activation}")
        
        # Các layer
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Khởi tạo weights
        self._init_weights()
        
        logger.info(
            f"Khởi tạo PointWiseFeedForward: d_model={d_model}, "
            f"d_ff={d_ff}, dropout={dropout}, activation={activation}"
        )
    
    def _init_weights(self):
        """Khởi tạo weights với phân phối chuẩn."""
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)
        nn.init.zeros_(self.w_1.bias)
        nn.init.zeros_(self.w_2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor shape (batch_size, seq_len, d_model)
        """
        # Log shapes nếu debug
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"FFN input shape: {x.shape}")
        
        # Feed-forward
        x = self.w_1(x)  # (batch_size, seq_len, d_ff)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.w_2(x)  # (batch_size, seq_len, d_model)
        
        # Log shapes nếu debug
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"FFN output shape: {x.shape}")
        
        return x 
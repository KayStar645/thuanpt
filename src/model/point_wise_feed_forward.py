# point_wise_feed_forward.py
import torch
import torch.nn as nn
from typing import Optional, Union, Callable

class PointWiseFeedForward(nn.Module):
    """Point-wise Feed Forward Network (FFN) theo kiến trúc Transformer.
    
    Module này thực hiện chuyển đổi biểu diễn phi tuyến với kiến trúc:
    FFN(x) = LayerNorm(x + Dropout(Linear_2(GELU(Dropout(Linear_1(x))))))
    
    Args:
        d_model (int): Kích thước đầu vào và đầu ra của module
        d_ff (int): Kích thước của hidden layer
        dropout (float, optional): Tỷ lệ dropout. Mặc định là 0.1
        activation (Union[str, Callable], optional): Hàm activation. 
            Có thể là 'gelu', 'relu', hoặc một callable. Mặc định là 'gelu'
        layer_norm_eps (float, optional): Epsilon cho layer normalization. Mặc định là 1e-5
        use_residual (bool, optional): Có sử dụng residual connection hay không. Mặc định là True
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: Union[str, Callable] = 'gelu',
        layer_norm_eps: float = 1e-5,
        use_residual: bool = True
    ):
        super(PointWiseFeedForward, self).__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        
        # Linear layers
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Activation function
        if isinstance(activation, str):
            if activation.lower() == 'gelu':
                self.activation = nn.GELU()
            elif activation.lower() == 'relu':
                self.activation = nn.ReLU()
            else:
                raise ValueError(f"Activation {activation} không được hỗ trợ")
        else:
            self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass của FFN.
        
        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Tensor đầu ra có kích thước (batch_size, seq_len, d_model)
        """
        residual = x
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # FFN
        x = self.w_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        
        # Residual connection
        if self.use_residual:
            x = x + residual
            
        return x
    
    def get_config(self) -> dict:
        """Lấy cấu hình của module.
        
        Returns:
            dict: Dictionary chứa các tham số cấu hình
        """
        return {
            'd_model': self.d_model,
            'd_ff': self.d_ff,
            'dropout': self.dropout.p,
            'activation': 'gelu' if isinstance(self.activation, nn.GELU) else 'relu',
            'layer_norm_eps': self.layer_norm.eps,
            'use_residual': self.use_residual
        }
    
    @classmethod
    def from_config(cls, config: dict) -> 'PointWiseFeedForward':
        """Tạo instance từ cấu hình.
        
        Args:
            config (dict): Dictionary chứa các tham số cấu hình
            
        Returns:
            PointWiseFeedForward: Instance mới được tạo từ cấu hình
        """
        return cls(**config)

if __name__ == "__main__":
    # Test code
    batch_size = 2
    seq_len = 10
    d_model = 512
    d_ff = 2048
    
    # Tạo input tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Khởi tạo model
    ffn = PointWiseFeedForward(d_model=d_model, d_ff=d_ff)
    
    # Forward pass
    output = ffn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model config: {ffn.get_config()}")

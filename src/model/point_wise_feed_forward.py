# point_wise_feed_forward.py
import torch
import torch.nn as nn

class PointWiseFeedForward(nn.Module):
    """Lớp Point-wise Feed Forward Network (FFN) cho model.
    
    Args:
        d_model (int): Kích thước đầu vào và đầu ra
        d_ff (int): Kích thước của hidden layer
        dropout (float): Tỷ lệ dropout
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PointWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # Sử dụng GELU như trong BERT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass của FFN.
        
        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Tensor đầu ra có kích thước (batch_size, seq_len, d_model)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

if __name__ == "__main__":
    print("Hello, World!")
    # from point_wise_feed_forward import PositionwiseFeedForward

    # ffn = PositionwiseFeedForward(d_hid=1024, d_inner_hid=2048)
    # x_out = ffn(x_encoded)  # x_encoded: (batch, seq_len, 1024)

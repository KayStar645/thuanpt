# -*- coding: utf-8 -*-
# file: dynamic_rnn.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import logging
from typing import Optional, Tuple, Union, Dict, Any

logger = logging.getLogger(__name__)

class DynamicRNN(nn.Module):
    """Dynamic RNN với 2 lớp BiLSTM và các tính năng nâng cao.
    
    Module này thực hiện xử lý chuỗi có độ dài thay đổi với:
    - 2 lớp BiLSTM với dropout
    - Layer normalization
    - Residual connection
    - Packed sequence optimization
    
    Args:
        input_size (int): Kích thước đầu vào
        hidden_size (int): Kích thước hidden state cho mỗi hướng
        num_layers (int, optional): Số lớp RNN. Mặc định là 2
        batch_first (bool, optional): Nếu True, batch dimension ở vị trí đầu tiên. Mặc định là True
        bidirectional (bool, optional): Nếu True, sử dụng bidirectional RNN. Mặc định là True
        rnn_type (str, optional): Loại RNN ('lstm', 'gru', 'rnn'). Mặc định là 'lstm'
        dropout (float, optional): Tỷ lệ dropout. Mặc định là 0.1
        layer_norm_eps (float, optional): Epsilon cho layer normalization. Mặc định là 1e-5
        use_residual (bool, optional): Có sử dụng residual connection hay không. Mặc định là True
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,  # Mặc định 2 lớp
        batch_first: bool = True,
        bidirectional: bool = True,
        rnn_type: str = 'lstm',
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_residual: bool = True
    ):
        super(DynamicRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        self.use_residual = use_residual
        
        # Output size sẽ là hidden_size * 2 nếu bidirectional
        self.output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(input_size, eps=layer_norm_eps)
        
        # Khởi tạo RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
            
        logger.info(f"Khởi tạo DynamicRNN: input_size={input_size}, "
                   f"hidden_size={hidden_size}, "
                   f"num_layers={num_layers}, "
                   f"bidirectional={bidirectional}, "
                   f"rnn_type={rnn_type}, "
                   f"output_size={self.output_size}")

    def forward(
        self, 
        x: torch.Tensor, 
        mask: torch.Tensor,
        return_hidden: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Forward pass của DynamicRNN.
        
        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, seq_len, input_size)
            mask (torch.Tensor): Attention mask có kích thước (batch_size, seq_len)
                True cho các token thực tế, False cho padding
            return_hidden (bool, optional): Có trả về hidden states hay không. Mặc định là True
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: 
                - Nếu return_hidden=True: (output, hidden_states)
                - Nếu return_hidden=False: output
                output có kích thước (batch_size, seq_len, output_size)
        """
        # Layer normalization
        x = self.layer_norm(x)
        residual = x
        
        # Tính độ dài thực tế của mỗi chuỗi từ mask
        lengths = mask.sum(dim=1).long()
        
        # Kiểm tra và xử lý lengths
        if lengths.max() == 0:
            logger.warning("All sequences have length 0!")
            return (x, None) if return_hidden else x
            
        # Sắp xếp lại chuỗi theo độ dài giảm dần
        lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)
        _, original_indices = torch.sort(sorted_indices)
        
        # Đảm bảo sorted_indices là vector
        sorted_indices = sorted_indices.view(-1)
        original_indices = original_indices.view(-1)
        
        # Sắp xếp lại input và mask
        if self.batch_first:
            x = x[sorted_indices]
            mask = mask[sorted_indices]
        else:
            x = x[:, sorted_indices]
            mask = mask[:, sorted_indices]
        
        # Pack padded sequence
        packed_x = rnn_utils.pack_padded_sequence(
            x, lengths_sorted.cpu(), batch_first=self.batch_first
        )
        
        # Chạy RNN
        if self.rnn_type == 'lstm':
            packed_output, (h_n, c_n) = self.rnn(packed_x)
            # Unpack output
            output, _ = rnn_utils.pad_packed_sequence(
                packed_output, 
                batch_first=self.batch_first,
                total_length=x.size(1)
            )
            
            # Khôi phục lại thứ tự ban đầu
            if self.batch_first:
                output = output[original_indices]
                h_n = h_n[:, original_indices]
                c_n = c_n[:, original_indices]
            else:
                output = output[:, original_indices]
                h_n = h_n[:, original_indices]
                c_n = c_n[:, original_indices]
            hidden = (h_n, c_n)
        else:
            packed_output, h_n = self.rnn(packed_x)
            # Unpack output
            output, _ = rnn_utils.pad_packed_sequence(
                packed_output, 
                batch_first=self.batch_first,
                total_length=x.size(1)
            )
            
            # Khôi phục lại thứ tự ban đầu
            if self.batch_first:
                output = output[original_indices]
                h_n = h_n[:, original_indices]
            else:
                output = output[:, original_indices]
                h_n = h_n[:, original_indices]
            hidden = h_n
        
        # Residual connection
        if self.use_residual and output.size(-1) == residual.size(-1):
            output = output + residual
        
        # Kiểm tra shape cuối cùng
        if output.size(-1) != self.output_size:
            raise ValueError(
                f"Output shape không khớp. Expected last dim: {self.output_size}, "
                f"Got: {output.size(-1)}"
            )
        
        return (output, hidden) if return_hidden else output
    
    def get_config(self) -> Dict[str, Any]:
        """Lấy cấu hình của module.
        
        Returns:
            Dict[str, Any]: Dictionary chứa các tham số cấu hình
        """
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'batch_first': self.batch_first,
            'bidirectional': self.bidirectional,
            'rnn_type': self.rnn_type,
            'dropout': self.rnn.dropout if self.num_layers > 1 else 0,
            'layer_norm_eps': self.layer_norm.eps,
            'use_residual': self.use_residual
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'DynamicRNN':
        """Tạo instance từ cấu hình.
        
        Args:
            config (Dict[str, Any]): Dictionary chứa các tham số cấu hình
            
        Returns:
            DynamicRNN: Instance mới được tạo từ cấu hình
        """
        return cls(**config)

if __name__ == "__main__":
    # Test code
    batch_size = 2
    seq_len = 10
    input_size = 512
    hidden_size = 256
    
    # Tạo input tensor và mask
    x = torch.randn(batch_size, seq_len, input_size)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 5:] = False  # Một số token padding
    
    # Khởi tạo model
    rnn = DynamicRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        bidirectional=True
    )
    
    # Forward pass
    output, hidden = rnn(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    if isinstance(hidden, tuple):
        print(f"Hidden states shape: {hidden[0].shape}")
        print(f"Cell states shape: {hidden[1].shape}")
    else:
        print(f"Hidden states shape: {hidden.shape}")
    print(f"Model config: {rnn.get_config()}")

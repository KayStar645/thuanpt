# -*- coding: utf-8 -*-
# file: dynamic_rnn.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import logging

logger = logging.getLogger(__name__)

class DynamicRNN(nn.Module):
    """Lớp RNN động để xử lý chuỗi có độ dài khác nhau.
    
    Args:
        input_size (int): Kích thước đầu vào
        hidden_size (int): Kích thước hidden state
        num_layers (int): Số lớp RNN
        batch_first (bool): Nếu True, batch dimension sẽ ở vị trí đầu tiên
        bidirectional (bool): Nếu True, sử dụng bidirectional RNN
        rnn_type (str): Loại RNN ('lstm', 'gru', 'rnn')
        dropout (float): Tỷ lệ dropout
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bidirectional: bool = True,
        rnn_type: str = 'lstm',
        dropout: float = 0.1
    ):
        super(DynamicRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        
        # Tính toán kích thước thực tế cho hidden state
        self.hidden_size_per_direction = hidden_size // 2 if bidirectional else hidden_size
        
        logger.debug(f"DynamicRNN config - input_size: {input_size}, "
                    f"hidden_size: {hidden_size}, "
                    f"hidden_size_per_direction: {self.hidden_size_per_direction}")
        
        # Khởi tạo RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=self.hidden_size_per_direction,  # Chia 2 cho bidirectional
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=self.hidden_size_per_direction,  # Chia 2 cho bidirectional
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            self.rnn = nn.RNN(
                input_size=input_size,
                hidden_size=self.hidden_size_per_direction,  # Chia 2 cho bidirectional
                num_layers=num_layers,
                batch_first=batch_first,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
            
        logger.debug(f"Khởi tạo DynamicRNN: input_size={input_size}, "
                    f"hidden_size={hidden_size}, "
                    f"num_layers={num_layers}, "
                    f"bidirectional={bidirectional}, "
                    f"rnn_type={rnn_type}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple:
        """Forward pass của DynamicRNN.
        
        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, seq_len, input_size)
            mask (torch.Tensor): Attention mask có kích thước (batch_size, seq_len)
                True cho các token thực tế, False cho padding
            
        Returns:
            tuple: (output, (h_n, c_n)) cho LSTM hoặc (output, h_n) cho GRU/RNN
                output có kích thước (batch_size, seq_len, hidden_size * num_directions)
        """
        # Log kích thước đầu vào
        logger.debug(f"Input shapes - x: {x.shape}, mask: {mask.shape}")
        
        # Tính độ dài thực tế của mỗi chuỗi từ mask
        lengths = mask.sum(dim=1).long()
        logger.debug(f"Sequence lengths: {lengths}")
        
        # Kiểm tra và xử lý lengths
        if lengths.max() == 0:
            logger.warning("All sequences have length 0!")
            return x, None
            
        # Sắp xếp lại chuỗi theo độ dài giảm dần
        lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)
        _, original_indices = torch.sort(sorted_indices)
        
        # Đảm bảo sorted_indices là vector
        sorted_indices = sorted_indices.view(-1)
        original_indices = original_indices.view(-1)
        
        logger.debug(f"Sorted indices shape: {sorted_indices.shape}")
        
        # Sắp xếp lại input và mask
        if self.batch_first:
            x = x[sorted_indices]
            mask = mask[sorted_indices]
        else:
            x = x[:, sorted_indices]
            mask = mask[:, sorted_indices]
            
        logger.debug(f"After sorting - x: {x.shape}, mask: {mask.shape}")
        
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
                total_length=x.size(1)  # Giữ nguyên độ dài ban đầu
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
                total_length=x.size(1)  # Giữ nguyên độ dài ban đầu
            )
            # Khôi phục lại thứ tự ban đầu
            if self.batch_first:
                output = output[original_indices]
                h_n = h_n[:, original_indices]
            else:
                output = output[:, original_indices]
                h_n = h_n[:, original_indices]
            hidden = h_n
            
        logger.debug(f"Output shape: {output.shape}")
        return output, hidden

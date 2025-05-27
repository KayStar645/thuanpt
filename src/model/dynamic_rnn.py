# -*- coding: utf-8 -*-
# file: dynamic_rnn.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class DynamicLSTM(nn.Module):
    """Lớp LSTM động có thể xử lý chuỗi có độ dài thay đổi.
    
    Args:
        input_size (int): Kích thước vector đầu vào
        hidden_size (int): Kích thước vector ẩn
        num_layers (int): Số lớp LSTM
        batch_first (bool): Nếu True, batch dimension sẽ ở vị trí đầu tiên
        bidirectional (bool): Nếu True, sử dụng BiLSTM
        dropout (float): Tỷ lệ dropout
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bidirectional: bool = True,
        dropout: float = 0.1
    ):
        super(DynamicLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first

    def forward(
        self,
        x: torch.Tensor,
        x_len: torch.Tensor,
        h0: torch.Tensor = None,
        c0: torch.Tensor = None
    ) -> tuple:
        """Forward pass của LSTM động.
        
        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, seq_len, input_size)
            x_len (torch.Tensor): Độ dài thực tế của mỗi chuỗi trong batch
            h0 (torch.Tensor, optional): Hidden state ban đầu
            c0 (torch.Tensor, optional): Cell state ban đầu
            
        Returns:
            tuple: (output, (h_n, c_n))
                - output: Tensor đầu ra có kích thước (batch_size, seq_len, hidden_size * num_directions)
                - h_n: Hidden state cuối cùng
                - c_n: Cell state cuối cùng
        """
        # Sắp xếp lại chuỗi theo độ dài giảm dần
        x_len_sorted, idx_sort = torch.sort(x_len, descending=True)
        _, idx_unsort = torch.sort(idx_sort)
        
        if self.batch_first:
            x = x[idx_sort]
        else:
            x = x[:, idx_sort]
        
        # Pack padded sequence
        x_packed = rnn_utils.pack_padded_sequence(
            x, x_len_sorted.cpu(), batch_first=self.batch_first
        )
        
        # Forward pass qua LSTM
        if h0 is None or c0 is None:
            output, (h_n, c_n) = self.lstm(x_packed)
        else:
            output, (h_n, c_n) = self.lstm(x_packed, (h0, c0))
        
        # Unpack padded sequence
        output, _ = rnn_utils.pad_packed_sequence(output, batch_first=self.batch_first)
        
        # Khôi phục lại thứ tự ban đầu
        if self.batch_first:
            output = output[idx_unsort]
        else:
            output = output[:, idx_unsort]
        
        return output, (h_n, c_n)

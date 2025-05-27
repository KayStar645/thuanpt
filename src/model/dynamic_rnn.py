# -*- coding: utf-8 -*-
# file: dynamic_rnn.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import numpy as np

class DynamicRNN(nn.Module):
    """RNN với khả năng xử lý chuỗi có độ dài thay đổi.
    
    Args:
        input_size (int): Kích thước vector đầu vào
        hidden_size (int): Kích thước vector ẩn
        num_layers (int): Số lớp RNN
        batch_first (bool): True nếu batch là chiều đầu tiên
        dropout (float): Tỷ lệ dropout
        bidirectional (bool): True nếu sử dụng RNN hai chiều
        rnn_type (str): Loại RNN ('lstm' hoặc 'gru')
    """
    
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True,
                 dropout=0.0, bidirectional=False, rnn_type='lstm'):
        super(DynamicRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=batch_first,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Không hỗ trợ loại RNN: {rnn_type}")
    
    def forward(self, x, lengths):
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor
            lengths (torch.Tensor): Độ dài thực tế của các chuỗi
            
        Returns:
            torch.Tensor: Output của RNN
        """
        # Pack sequence
        packed_x = rnn_utils.pack_padded_sequence(
            x, lengths.cpu(), batch_first=self.batch_first, enforce_sorted=False
        )
        
        # Forward pass
        packed_output, _ = self.rnn(packed_x)
        
        # Unpack sequence
        output, _ = rnn_utils.pad_packed_sequence(
            packed_output, batch_first=self.batch_first
        )
        
        return output

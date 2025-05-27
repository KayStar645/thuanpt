# -*- coding: utf-8 -*-
# file: attention.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score


class NoQueryAttention(Attention):
    '''q is a parameter'''
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', q_len=1, dropout=0):
        super(NoQueryAttention, self).__init__(embed_dim, hidden_dim, out_dim, n_head, score_function, dropout)
        self.q_len = q_len
        self.q = nn.Parameter(torch.Tensor(q_len, embed_dim))
        self.reset_q()

    def reset_q(self):
        stdv = 1. / math.sqrt(self.embed_dim)
        self.q.data.uniform_(-stdv, stdv)

    def forward(self, k, **kwargs):
        mb_size = k.shape[0]
        q = self.q.expand(mb_size, -1, -1)
        return super(NoQueryAttention, self).forward(k, q)

class KGAttention(nn.Module):
    """Knowledge Graph Attention Network.
    
    Áp dụng attention mechanism lên các vector knowledge graph để
    tìm ra các thông tin quan trọng nhất cho bài toán ABSA.
    
    Args:
        input_size (int): Kích thước vector đầu vào
        hidden_size (int): Kích thước vector ẩn
        dropout (float): Tỷ lệ dropout cho các layer
        attention_dropout (float): Tỷ lệ dropout cho attention weights
        num_heads (int): Số lượng attention heads
        use_scaled_dot_product (bool): Sử dụng scaled dot-product attention
        use_relative_pos (bool): Sử dụng relative positional encoding
        max_relative_pos (int): Khoảng cách tối đa cho relative positional encoding
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        num_heads: int = 4,
        use_scaled_dot_product: bool = True,
        use_relative_pos: bool = False,
        max_relative_pos: int = 16
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_scaled_dot_product = use_scaled_dot_product
        self.use_relative_pos = use_relative_pos
        
        # Kiểm tra kích thước
        assert hidden_size % num_heads == 0, "hidden_size phải chia hết cho num_heads"
        
        # Linear layers cho query, key, value
        self.q_linear = nn.Linear(input_size, hidden_size)
        self.k_linear = nn.Linear(input_size, hidden_size)
        self.v_linear = nn.Linear(input_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        # Relative positional encoding
        if use_relative_pos:
            self.max_relative_pos = max_relative_pos
            self.relative_attention_bias = nn.Parameter(
                torch.zeros(2 * max_relative_pos + 1, num_heads)
            )
            self._init_relative_pos_bias()
        
        # Khởi tạo trọng số
        self._init_weights()
        
        logger.info(
            f"Khởi tạo KGAttention với {num_heads} heads, "
            f"scaled_dot_product={use_scaled_dot_product}, "
            f"relative_pos={use_relative_pos}"
        )
    
    def _init_weights(self):
        """Khởi tạo trọng số cho các layer."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'linear' in name:
                    nn.init.xavier_uniform_(param)
                elif 'layer_norm' in name:
                    nn.init.ones_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def _init_relative_pos_bias(self):
        """Khởi tạo trọng số cho relative positional encoding."""
        nn.init.normal_(self.relative_attention_bias, mean=0.0, std=0.02)
    
    def _get_relative_positions(self, length: int) -> torch.Tensor:
        """Tính toán relative positions.
        
        Args:
            length: Độ dài của sequence
            
        Returns:
            Tensor chứa relative positions [length, length]
        """
        range_vec = torch.arange(length, device=self.relative_attention_bias.device)
        relative_pos_mat = range_vec[None, :] - range_vec[:, None]
        relative_pos_mat = torch.clamp(
            relative_pos_mat,
            -self.max_relative_pos,
            self.max_relative_pos
        )
        relative_pos_mat = relative_pos_mat + self.max_relative_pos
        return relative_pos_mat
    
    def _reshape_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor cho multi-head attention.
        
        Args:
            x: Tensor đầu vào [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor đã reshape [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _reshape_from_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor từ multi-head attention về dạng ban đầu.
        
        Args:
            x: Tensor đầu vào [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Tensor đã reshape [batch_size, seq_len, hidden_size]
        """
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
    
    def _compute_attention_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        relative_pos: torch.Tensor = None
    ) -> torch.Tensor:
        """Tính toán attention scores.
        
        Args:
            q: Query tensor [batch_size, num_heads, seq_len, head_dim]
            k: Key tensor [batch_size, num_heads, seq_len, head_dim]
            relative_pos: Relative positions [seq_len, seq_len]
            
        Returns:
            Attention scores [batch_size, num_heads, seq_len, seq_len]
        """
        # Tính dot product
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        # Áp dụng scaling nếu cần
        if self.use_scaled_dot_product:
            scores = scores / math.sqrt(self.head_dim)
        
        # Thêm relative positional bias nếu cần
        if self.use_relative_pos and relative_pos is not None:
            relative_attention_bias = self.relative_attention_bias[
                relative_pos.view(-1)
            ].view(relative_pos.shape + (-1,))
            relative_attention_bias = relative_attention_bias.permute(2, 0, 1)
            scores = scores + relative_attention_bias.unsqueeze(0)
        
        return scores
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass của KGAttention.
        
        Args:
            x: Tensor đầu vào [batch_size, seq_len, input_size]
            mask: Boolean attention mask [batch_size, seq_len]
            
        Returns:
            Tensor đã được attention [batch_size, seq_len, hidden_size]
        """
        # Log shapes
        logger.debug(f"Input shape: {x.shape}")
        if mask is not None:
            logger.debug(f"Mask shape: {mask.shape}")
            # Đảm bảo mask là boolean
            if not mask.dtype == torch.bool:
                mask = mask.bool()
                logger.warning("Converting attention mask to boolean type")
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Linear projections
        q = self.q_linear(x)  # [batch_size, seq_len, hidden_size]
        k = self.k_linear(x)  # [batch_size, seq_len, hidden_size]
        v = self.v_linear(x)  # [batch_size, seq_len, hidden_size]
        
        # Reshape cho multi-head attention
        q = self._reshape_for_heads(q)  # [batch_size, num_heads, seq_len, head_dim]
        k = self._reshape_for_heads(k)  # [batch_size, num_heads, seq_len, head_dim]
        v = self._reshape_for_heads(v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Tính relative positions nếu cần
        relative_pos = None
        if self.use_relative_pos:
            relative_pos = self._get_relative_positions(x.size(1))
        
        # Tính attention scores
        scores = self._compute_attention_scores(q, k, relative_pos)
        
        # Áp dụng mask nếu có
        if mask is not None:
            # Chuyển mask thành [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
            logger.debug("Applied attention mask")
        
        # Softmax để có attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attention_dropout(attn_weights)
        
        # Áp dụng attention weights
        context = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape về dạng ban đầu
        context = self._reshape_from_heads(context)  # [batch_size, seq_len, hidden_size]
        
        # Output projection
        output = self.out_proj(context)
        output = self.dropout(output)
        
        # Residual connection
        output = output + x
        
        logger.debug(f"Output shape: {output.shape}")
        return output

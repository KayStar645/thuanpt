# squeeze_embedding.py
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import logging
from typing import Optional, Tuple, Union, Dict, Any

logger = logging.getLogger(__name__)

class SqueezeEmbedding(nn.Module):
    """Module nén embedding với xử lý padding tối ưu.
    
    Module này thực hiện:
    - Nén và giải nén embeddings hiệu quả
    - Xử lý padding tokens
    - Tối ưu bộ nhớ với packed sequences
    - Hỗ trợ batch processing
    
    Args:
        batch_first (bool, optional): Nếu True, batch dimension ở vị trí đầu tiên. Mặc định là True
        padding_idx (int, optional): Index của padding token. Mặc định là 0
        max_len (int, optional): Độ dài tối đa của chuỗi. Nếu None, không giới hạn. Mặc định là None
        use_mask (bool, optional): Có sử dụng mask để xử lý padding hay không. Mặc định là True
    """
    def __init__(
        self,
        batch_first: bool = True,
        padding_idx: int = 0,
        max_len: Optional[int] = None,
        use_mask: bool = True
    ):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.use_mask = use_mask
        
        logger.info(f"Khởi tạo SqueezeEmbedding: batch_first={batch_first}, "
                   f"padding_idx={padding_idx}, max_len={max_len}, "
                   f"use_mask={use_mask}")

    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        return_mask: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass của SqueezeEmbedding.
        
        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, seq_len, embedding_dim)
            mask (Optional[torch.Tensor]): Attention mask có kích thước (batch_size, seq_len)
                True cho các token thực tế, False cho padding. Nếu None, sẽ tự tạo mask
            return_mask (bool, optional): Có trả về mask đã xử lý hay không. Mặc định là False
            
        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - Nếu return_mask=False: tensor đã được nén
                - Nếu return_mask=True: (tensor đã được nén, mask đã xử lý)
        """
        # Kiểm tra và xử lý max_len
        if self.max_len is not None and x.size(1) > self.max_len:
            x = x[:, :self.max_len]
            if mask is not None:
                mask = mask[:, :self.max_len]
        
        # Tạo mask nếu không được cung cấp
        if mask is None and self.use_mask:
            if self.batch_first:
                mask = (x.sum(dim=-1) != self.padding_idx)
            else:
                mask = (x.sum(dim=0) != self.padding_idx)
        
        # Tính độ dài thực tế của mỗi chuỗi
        if mask is not None:
            lengths = mask.sum(dim=1).long()
        else:
            if self.batch_first:
                lengths = (x.sum(dim=-1) != self.padding_idx).sum(dim=1).long()
            else:
                lengths = (x.sum(dim=0) != self.padding_idx).sum(dim=0).long()
        
        # Kiểm tra lengths
        if lengths.max() == 0:
            logger.warning("All sequences have length 0!")
            if return_mask and mask is not None:
                return x, mask
            return x
        
        # Sắp xếp lại chuỗi theo độ dài giảm dần
        lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)
        _, original_indices = torch.sort(sorted_indices)
        
        # Đảm bảo sorted_indices là vector
        sorted_indices = sorted_indices.view(-1)
        original_indices = original_indices.view(-1)
        
        # Sắp xếp lại input và mask
        if self.batch_first:
            x = x[sorted_indices]
            if mask is not None:
                mask = mask[sorted_indices]
        else:
            x = x[:, sorted_indices]
            if mask is not None:
                mask = mask[:, sorted_indices]
        
        # Pack padded sequence
        packed_x = rnn_utils.pack_padded_sequence(
            x, lengths_sorted.cpu(), batch_first=self.batch_first
        )
        
        # Unpack để khôi phục kích thước ban đầu
        x_padded, _ = rnn_utils.pad_packed_sequence(
            packed_x,
            batch_first=self.batch_first,
            total_length=x.size(1)
        )
        
        # Khôi phục lại thứ tự ban đầu
        if self.batch_first:
            x_padded = x_padded[original_indices]
            if mask is not None:
                mask = mask[original_indices]
        else:
            x_padded = x_padded[:, original_indices]
            if mask is not None:
                mask = mask[:, original_indices]
        
        # Áp dụng mask nếu cần
        if self.use_mask and mask is not None:
            # Đảm bảo mask có cùng kích thước với x_padded
            if mask.size(1) != x_padded.size(1):
                if mask.size(1) > x_padded.size(1):
                    mask = mask[:, :x_padded.size(1)]
                else:
                    pad_size = x_padded.size(1) - mask.size(1)
                    mask = torch.nn.functional.pad(mask, (0, pad_size), value=False)
            
            # Áp dụng mask
            x_padded = x_padded * mask.unsqueeze(-1)
        
        return (x_padded, mask) if return_mask and mask is not None else x_padded
    
    def get_config(self) -> Dict[str, Any]:
        """Lấy cấu hình của module.
        
        Returns:
            Dict[str, Any]: Dictionary chứa các tham số cấu hình
        """
        return {
            'batch_first': self.batch_first,
            'padding_idx': self.padding_idx,
            'max_len': self.max_len,
            'use_mask': self.use_mask
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'SqueezeEmbedding':
        """Tạo instance từ cấu hình.
        
        Args:
            config (Dict[str, Any]): Dictionary chứa các tham số cấu hình
            
        Returns:
            SqueezeEmbedding: Instance mới được tạo từ cấu hình
        """
        return cls(**config)

if __name__ == "__main__":
    # Test code
    batch_size = 2
    seq_len = 10
    embedding_dim = 512
    
    # Tạo input tensor và mask
    x = torch.randn(batch_size, seq_len, embedding_dim)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[0, 5:] = False  # Một số token padding
    
    # Khởi tạo model
    squeeze = SqueezeEmbedding(
        batch_first=True,
        padding_idx=0,
        max_len=seq_len,
        use_mask=True
    )
    
    # Forward pass
    output, output_mask = squeeze(x, mask, return_mask=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mask shape: {output_mask.shape}")
    print(f"Model config: {squeeze.get_config()}")
    
    # Test với mask=None
    output_no_mask = squeeze(x)
    print(f"Output without mask shape: {output_no_mask.shape}")

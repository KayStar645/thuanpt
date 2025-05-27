# squeeze_embedding.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class SqueezeEmbedding(nn.Module):
    """Lớp nén embedding để xử lý padding hiệu quả hơn.
    
    Args:
        batch_first (bool): Nếu True, batch dimension sẽ ở vị trí đầu tiên
    """
    def __init__(self, batch_first: bool = True):
        super(SqueezeEmbedding, self).__init__()
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass của SqueezeEmbedding.
        
        Args:
            x (torch.Tensor): Tensor đầu vào có kích thước (batch_size, seq_len, embedding_dim)
            mask (torch.Tensor): Attention mask có kích thước (batch_size, seq_len)
                True cho các token thực tế, False cho padding
            
        Returns:
            torch.Tensor: Tensor đã được nén, chỉ giữ lại các token thực tế
        """
        # Log kích thước đầu vào
        logger.debug(f"Input shapes - x: {x.shape}, mask: {mask.shape}")
        
        # Tính độ dài thực tế của mỗi chuỗi từ mask
        x_len = mask.sum(dim=1).long()
        logger.debug(f"Sequence lengths: {x_len}")
        
        # Sắp xếp lại chuỗi theo độ dài giảm dần
        x_len_sorted, idx_sort = torch.sort(x_len, descending=True)
        _, idx_unsort = torch.sort(idx_sort)
        
        if self.batch_first:
            x = x[idx_sort]
            mask = mask[idx_sort]
        else:
            x = x[:, idx_sort]
            mask = mask[:, idx_sort]
            
        logger.debug(f"After sorting - x: {x.shape}, mask: {mask.shape}")
        
        # Nén tensor bằng cách loại bỏ padding
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            x, x_len_sorted.cpu(), batch_first=self.batch_first
        )
        
        # Khôi phục lại thứ tự ban đầu và giữ nguyên kích thước
        x_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(
            x_packed, 
            batch_first=self.batch_first,
            total_length=x.size(1)  # Giữ nguyên độ dài ban đầu
        )
        
        logger.debug(f"After pad_packed_sequence - x_padded: {x_padded.shape}")
        
        if self.batch_first:
            x_padded = x_padded[idx_unsort]
            mask = mask[idx_unsort]
        else:
            x_padded = x_padded[:, idx_unsort]
            mask = mask[:, idx_unsort]
            
        logger.debug(f"After unsorting - x_padded: {x_padded.shape}, mask: {mask.shape}")
        
        # Đảm bảo mask có cùng kích thước với x_padded
        if mask.size(1) != x_padded.size(1):
            logger.warning(f"Mask size {mask.size(1)} != x_padded size {x_padded.size(1)}")
            # Cắt mask nếu cần
            if mask.size(1) > x_padded.size(1):
                mask = mask[:, :x_padded.size(1)]
            # Pad mask nếu cần
            else:
                pad_size = x_padded.size(1) - mask.size(1)
                mask = torch.nn.functional.pad(mask, (0, pad_size), value=False)
        
        # Áp dụng mask để đảm bảo padding tokens có giá trị 0
        x_padded = x_padded * mask.unsqueeze(-1)
        
        logger.debug(f"Final output shape: {x_padded.shape}")
        return x_padded

"""
Module dataset cho mô hình ABSA.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Hàm gộp các mẫu trong batch.
    
    Args:
        batch (List[Dict[str, torch.Tensor]]): Danh sách các mẫu trong batch
        
    Returns:
        Dict[str, torch.Tensor]: Batch đã được padding
    """
    # Tìm độ dài lớn nhất trong batch, giới hạn bởi max_length
    max_len = min(
        max(x['input_ids'].size(0) for x in batch),
        512  # Giới hạn độ dài tối đa
    )
    
    # Khởi tạo tensors cho batch
    batch_size = len(batch)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.zeros(batch_size, max_len, dtype=torch.long)
    ids = []
    
    # Padding cho từng mẫu
    for i, sample in enumerate(batch):
        # Lấy độ dài thực tế của sequence
        seq_len = min(sample['input_ids'].size(0), max_len)
        
        # Cắt sequence nếu cần
        input_ids_seq = sample['input_ids'][:seq_len]
        attention_mask_seq = sample['attention_mask'][:seq_len]
        labels_seq = sample['labels'][:seq_len]
        
        # Gán vào tensor batch
        input_ids[i, :seq_len] = input_ids_seq
        attention_mask[i, :seq_len] = attention_mask_seq
        labels[i, :seq_len] = labels_seq
        
        # Đánh dấu padding bằng attention mask
        attention_mask[i, seq_len:] = 0
        
        ids.append(sample['id'])
    
    # Log thông tin về batch
    logger.debug(f"Batch shape - input_ids: {input_ids.shape}, labels: {labels.shape}")
    logger.debug(f"Sample lengths: {[x['input_ids'].size(0) for x in batch]}")
    
    # Trả về dict với các tensor và list riêng biệt
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'ids': ids  # Giữ nguyên dạng list
    }

class ABSADataset(Dataset):
    """Dataset cho bài toán ABSA.
    
    Attributes:
        data (List[Dict]): Danh sách các mẫu dữ liệu
        tokenizer: Tokenizer của PhoBERT
        max_length (int): Độ dài tối đa của sequence
        label2id (Dict[str, int]): Mapping từ nhãn sang id
        id2label (Dict[int, str]): Mapping từ id sang nhãn
    """
    
    def __init__(
        self,
        data_file: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        label2id: Optional[Dict[str, int]] = None
    ):
        """Khởi tạo dataset.
        
        Args:
            data_file (str): Đường dẫn file dữ liệu đã xử lý
            tokenizer (PreTrainedTokenizer): Tokenizer đã được khởi tạo
            max_length (int): Độ dài tối đa của sequence
            label2id (Optional[Dict[str, int]]): Mapping từ nhãn sang id
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Đọc dữ liệu
        self.data = []
        skipped_samples = 0
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    
                    # Kiểm tra tính nhất quán của dữ liệu
                    if not (len(sample['input_ids']) == len(sample['attention_mask']) == len(sample['labels'])):
                        logger.error(
                            f"Inconsistent sequence lengths in sample {sample['id']}: "
                            f"input_ids: {len(sample['input_ids'])}, "
                            f"attention_mask: {len(sample['attention_mask'])}, "
                            f"labels: {len(sample['labels'])}"
                        )
                        skipped_samples += 1
                        continue
                    
                    # Kiểm tra và cắt sequence nếu cần
                    if len(sample['input_ids']) > max_length:
                        logger.warning(
                            f"Sequence length {len(sample['input_ids'])} exceeds max_length {max_length}. "
                            f"Truncating to {max_length}."
                        )
                        sample['input_ids'] = sample['input_ids'][:max_length]
                        sample['attention_mask'] = sample['attention_mask'][:max_length]
                        sample['labels'] = sample['labels'][:max_length]
                    
                    # Đảm bảo tất cả các trường có cùng độ dài
                    min_len = min(len(sample['input_ids']), len(sample['attention_mask']), len(sample['labels']))
                    sample['input_ids'] = sample['input_ids'][:min_len]
                    sample['attention_mask'] = sample['attention_mask'][:min_len]
                    sample['labels'] = sample['labels'][:min_len]
                        
                    self.data.append(sample)
        
        logger.info(f"Đọc được {len(self.data)} mẫu từ file {data_file}")
        if skipped_samples > 0:
            logger.warning(f"Đã bỏ qua {skipped_samples} mẫu do không nhất quán")
        
        # Khởi tạo label mapping
        if label2id is None:
            self.label2id = {
                'O': 2,
                'B-POS': 3,
                'I-POS': 4,
                'B-NEG': 5,
                'I-NEG': 6,
                'B-NEU': 7,
                'I-NEU': 8,
                'START': 0,
                'END': 1
            }
        else:
            self.label2id = label2id
        
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def __len__(self) -> int:
        """Trả về số lượng mẫu trong dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Lấy một mẫu dữ liệu.
        
        Args:
            idx (int): Chỉ số của mẫu
            
        Returns:
            Dict[str, torch.Tensor]: Mẫu dữ liệu với các trường:
                - input_ids: Token ids
                - attention_mask: Attention mask
                - labels: Nhãn CRF
                - id: ID của mẫu
        """
        sample = self.data[idx]
        
        # Chuyển đổi sang tensor và đảm bảo kiểu dữ liệu
        input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
        attention_mask = torch.tensor(sample['attention_mask'], dtype=torch.long)
        labels = torch.tensor(sample['labels'], dtype=torch.long)
        
        # Kiểm tra kích thước
        assert len(input_ids) == len(attention_mask) == len(labels), \
            f"Inconsistent lengths in sample {sample['id']}"
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'id': sample['id']  # Giữ nguyên dạng string
        }

def create_dataloader(
    dataset: ABSADataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Tạo dataloader cho dataset.
    
    Args:
        dataset (ABSADataset): Dataset đã được khởi tạo
        batch_size (int): Kích thước batch
        shuffle (bool): Có shuffle dữ liệu không
        num_workers (int): Số worker cho dataloader
        
    Returns:
        DataLoader: Dataloader cho dataset
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn  # Sử dụng hàm collate_fn tùy chỉnh
    ) 
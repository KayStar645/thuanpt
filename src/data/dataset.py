"""
Module dataset cho mô hình ABSA.
"""

import json
from typing import Dict, List, Any, Optional
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoTokenizer
import logging

logger = logging.getLogger(__name__)

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
        tokenizer_name: str = "vinai/phobert-base",
        max_length: int = 512,
        label2id: Optional[Dict[str, int]] = None
    ):
        """Khởi tạo dataset.
        
        Args:
            data_file (str): Đường dẫn file dữ liệu đã xử lý
            tokenizer_name (str): Tên model tokenizer
            max_length (int): Độ dài tối đa của sequence
            label2id (Optional[Dict[str, int]]): Mapping từ nhãn sang id
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Đọc dữ liệu
        self.data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        logger.info(f"Đọc được {len(self.data)} mẫu từ file {data_file}")
        
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
        
        # Chuyển đổi sang tensor
        input_ids = torch.tensor(sample['input_ids'])
        attention_mask = torch.tensor(sample['attention_mask'])
        labels = torch.tensor(sample['labels'])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'id': sample['id']
        }

def create_dataloader(
    data_file: str,
    tokenizer_name: str = "vinai/phobert-base",
    max_length: int = 512,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    label2id: Optional[Dict[str, int]] = None
) -> DataLoader:
    """Tạo dataloader cho dataset.
    
    Args:
        data_file (str): Đường dẫn file dữ liệu
        tokenizer_name (str): Tên model tokenizer
        max_length (int): Độ dài tối đa của sequence
        batch_size (int): Kích thước batch
        shuffle (bool): Có shuffle dữ liệu không
        num_workers (int): Số worker cho dataloader
        label2id (Optional[Dict[str, int]]): Mapping từ nhãn sang id
        
    Returns:
        DataLoader: Dataloader cho dataset
    """
    dataset = ABSADataset(
        data_file=data_file,
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        label2id=label2id
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ) 
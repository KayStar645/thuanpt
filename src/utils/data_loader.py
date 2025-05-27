"""
Module chứa các lớp xử lý dữ liệu cho ABSA.
"""

import json
import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABSADataset(Dataset):
    """Dataset cho bài toán ABSA.
    
    Args:
        data_dir (str): Đường dẫn đến thư mục chứa dữ liệu
        tokenizer: Tokenizer để tokenize text
        max_length (int): Độ dài tối đa của sequence
    """
    
    def __init__(self, data_dir, tokenizer, max_length=128):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Label mapping cho sentiment
        self.sentiment_map = {
            "POSITIVE": 1,
            "NEGATIVE": 3,
            "NEUTRAL": 5
        }
        
        # Load dữ liệu
        self.data = self._load_data()
        logger.info(f"Đã load {len(self.data)} mẫu dữ liệu")
    
    def _convert_to_bio_labels(self, text, aspects):
        """Chuyển đổi dữ liệu dạng [start, end, sentiment] sang dạng BIO tagging.
        
        Args:
            text (str): Văn bản gốc
            aspects (list): Danh sách các aspect dạng [start, end, sentiment]
            
        Returns:
            list: Danh sách các label theo định dạng BIO
        """
        # Khởi tạo tất cả labels là "O"
        labels = ["O"] * len(text)
        
        # Đánh dấu các aspect theo định dạng BIO
        for start, end, sentiment in aspects:
            # Đánh dấu token đầu tiên là B-
            labels[start] = f"B-{sentiment}"
            # Đánh dấu các token còn lại là I-
            for i in range(start + 1, end):
                labels[i] = f"I-{sentiment}"
        
        return labels
    
    def _load_data(self):
        """Load dữ liệu từ file JSONL."""
        data = []
        train_file = f"{self.data_dir}/train.jsonl"
        logger.info(f"Đang đọc dữ liệu từ {train_file}")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    # Chuyển đổi labels sang định dạng BIO
                    bio_labels = self._convert_to_bio_labels(item['text'], item['labels'])
                    # Chuyển đổi BIO labels sang ID
                    item['labels'] = [self.sentiment_map[label.split('-')[1]] if label != "O" else 0 
                                    for label in bio_labels]
                    data.append(item)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý dòng: {line.strip()}")
                    logger.error(f"Chi tiết lỗi: {str(e)}")
                    raise
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Lấy một mẫu dữ liệu.
        
        Returns:
            tuple: (input_embeds, attention_mask, labels)
                - input_embeds: Tensor chứa embeddings [seq_len, hidden_size]
                - attention_mask: Tensor mask [seq_len]
                - labels: Tensor nhãn [seq_len]
        """
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Lấy input_ids và attention_mask
        input_ids = encoding['input_ids'].squeeze(0)  # [seq_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [seq_len]
        
        # Chuyển đổi labels thành tensor
        labels = torch.tensor(item['labels'], dtype=torch.long)
        
        # Đảm bảo labels có cùng độ dài với input_ids
        if len(labels) < self.max_length:
            # Padding labels với -100 (ignore_index)
            padding = torch.full((self.max_length - len(labels),), -100, dtype=torch.long)
            labels = torch.cat([labels, padding])
        else:
            # Cắt labels nếu dài hơn max_length
            labels = labels[:self.max_length]
        
        # Tạo input embeddings từ input_ids
        with torch.no_grad():
            input_embeds = self.tokenizer.convert_ids_to_tokens(input_ids)
            input_embeds = torch.tensor(
                [self.tokenizer.convert_tokens_to_ids(t) for t in input_embeds],
                dtype=torch.long
            )
        
        return input_embeds, attention_mask, labels 
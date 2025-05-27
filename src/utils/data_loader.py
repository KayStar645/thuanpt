"""
Module xử lý dữ liệu cho mô hình ABSA.
"""

import os
import json
import logging
import torch
from torch.utils.data import Dataset

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABSADataset(Dataset):
    """Dataset cho bài toán Aspect-Based Sentiment Analysis (ABSA).
    
    Args:
        data_dir (str): Thư mục chứa dữ liệu
        tokenizer: Tokenizer của PhoBERT
        max_length (int): Độ dài tối đa của chuỗi
    """
    
    def __init__(self, data_dir, tokenizer, max_length=128):
        self.samples = []
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Đọc và xử lý dữ liệu
        train_file = os.path.join(data_dir, "train.jsonl")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Không tìm thấy file {train_file}")
        
        logger.info(f"Đang đọc dữ liệu từ {train_file}")
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["text"]
                
                # Tokenize văn bản
                inputs = tokenizer(
                    text,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Chuyển đổi labels
                labels = torch.full_like(
                    inputs["input_ids"],
                    -100,  # Ignore index
                    dtype=torch.long
                )
                
                # Map labels cho các aspect terms
                for start, end, sentiment in obj["labels"]:
                    # Tìm các token tương ứng với span [start, end]
                    span_text = text[start:end]
                    span_tokens = tokenizer.tokenize(span_text)
                    
                    # Tìm vị trí của span tokens trong tokens
                    for i in range(len(inputs["input_ids"][0])):
                        if i + len(span_tokens) > len(inputs["input_ids"][0]):
                            break
                        
                        current_tokens = tokenizer.convert_ids_to_tokens(
                            inputs["input_ids"][0][i:i + len(span_tokens)]
                        )
                        if current_tokens == span_tokens:
                            # Gán nhãn B- cho token đầu tiên
                            labels[0, i] = self._get_label_id(f"B-{sentiment}")
                            # Gán nhãn I- cho các token còn lại
                            for j in range(1, len(span_tokens)):
                                labels[0, i + j] = self._get_label_id(f"I-{sentiment}")
                            break
                
                # Lưu sample
                self.samples.append({
                    "input_ids": inputs["input_ids"][0],
                    "attention_mask": inputs["attention_mask"][0],
                    "labels": labels[0]
                })
        
        logger.info(f"Đã load {len(self.samples)} mẫu dữ liệu")
    
    def _get_label_id(self, label):
        """Chuyển đổi nhãn thành ID.
        
        Args:
            label (str): Nhãn dạng text (ví dụ: "B-POSITIVE")
            
        Returns:
            int: ID của nhãn
        """
        label_map = {
            "O": 0,
            "B-POSITIVE": 1, "I-POSITIVE": 2,
            "B-NEGATIVE": 3, "I-NEGATIVE": 4,
            "B-NEUTRAL": 5, "I-NEUTRAL": 6
        }
        return label_map.get(label, 0)  # Mặc định là O nếu không tìm thấy
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx] 
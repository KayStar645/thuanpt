"""
Module chứa các lớp xử lý dữ liệu cho ABSA.
"""

import json
import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer
from typing import Dict, List, Optional, Tuple

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABSADataset(Dataset):
    """Dataset cho bài toán ABSA.
    
    Args:
        data_dir (str): Đường dẫn đến thư mục chứa dữ liệu
        tokenizer: Tokenizer để tokenize text
        max_length (int): Độ dài tối đa của sequence
        bert_model_name (str): Tên mô hình BERT để tạo embeddings
    """
    
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizer, max_length: int = 512, bert_model_name: str = "vinai/phobert-base"):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load BERT model để tạo embeddings
        self.bert_model = AutoModel.from_pretrained(bert_model_name)
        self.bert_model.eval()  # Chuyển sang chế độ evaluation
        
        # Label mapping cho sentiment
        self.sentiment_map = {
            "POSITIVE": 1,
            "NEGATIVE": 3,
            "NEUTRAL": 5
        }
        
        # Load và preprocess dữ liệu
        self.data = self._load_and_preprocess_data()
        logger.info(f"Đã load và preprocess {len(self.data)} mẫu dữ liệu")
    
    def _convert_to_bio_labels(self, text: str, aspects: List[Tuple[int, int, str]]) -> List[str]:
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
            if start >= len(text) or end > len(text):
                logger.warning(f"Invalid aspect span: start={start}, end={end}, text_len={len(text)}")
                continue
            # Đánh dấu token đầu tiên là B-
            labels[start] = f"B-{sentiment}"
            # Đánh dấu các token còn lại là I-
            for i in range(start + 1, end):
                labels[i] = f"I-{sentiment}"
        
        return labels

    def _preprocess_text_and_labels(self, text: str, labels: List[str]) -> Tuple[List[int], List[int], List[int], int]:
        """Preprocess text và labels để đảm bảo độ dài phù hợp.
        
        Args:
            text (str): Văn bản gốc
            labels (List[str]): Labels dạng BIO
            
        Returns:
            Tuple[List[int], List[int], List[int], int]: 
                - input_ids: Token ids
                - attention_mask: Attention mask
                - labels: Labels đã chuyển đổi sang số
                - actual_length: Độ dài thực tế của sequence
        """
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Lấy độ dài thực tế (không tính padding)
        actual_length = (encoding['input_ids'][0] != self.tokenizer.pad_token_id).sum().item()
        
        # Chuyển đổi labels sang số và pad/truncate
        label_ids = []
        for label in labels[:actual_length]:
            if label == "O":
                label_ids.append(2)  # O tag
            elif label.startswith("B-"):
                label_ids.append(self.sentiment_map.get(label[2:], 2))
            elif label.startswith("I-"):
                label_ids.append(self.sentiment_map.get(label[2:], 2))
            else:
                label_ids.append(2)  # Unknown tag -> O
        
        # Pad labels với O tag (2)
        if len(label_ids) < self.max_length:
            label_ids.extend([2] * (self.max_length - len(label_ids)))
        
        return (
            encoding['input_ids'][0].tolist(),
            encoding['attention_mask'][0].tolist(),
            label_ids,
            actual_length
        )
    
    def _load_and_preprocess_data(self) -> List[Dict]:
        """Load và preprocess dữ liệu từ file JSONL."""
        data = []
        train_file = f"{self.data_dir}/train.jsonl"
        logger.info(f"Đang đọc và preprocess dữ liệu từ {train_file}")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line)
                    text = item['text']
                    aspects = item.get('aspects', [])
                    
                    # Chuyển đổi sang BIO labels
                    bio_labels = self._convert_to_bio_labels(text, aspects)
                    
                    # Preprocess text và labels
                    input_ids, attention_mask, label_ids, actual_length = self._preprocess_text_and_labels(
                        text, bio_labels
                    )
                    
                    # Validate độ dài
                    assert len(input_ids) == self.max_length, \
                        f"input_ids length {len(input_ids)} != {self.max_length}"
                    assert len(attention_mask) == self.max_length, \
                        f"attention_mask length {len(attention_mask)} != {self.max_length}"
                    assert len(label_ids) == self.max_length, \
                        f"labels length {len(label_ids)} != {self.max_length}"
                    
                    # Lưu thông tin đã xử lý
                    processed_item = {
                        'id': item['id'],
                        'text': text,
                        'input_ids': input_ids,
                        'attention_mask': attention_mask,
                        'labels': label_ids,
                        'actual_length': actual_length
                    }
                    
                    data.append(processed_item)
                    
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý dòng {line_num}: {str(e)}")
                    logger.error(f"Text: {text[:100]}...")
                    continue
        
        logger.info(f"Đã load và preprocess {len(data)} mẫu dữ liệu")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Lấy một mẫu từ dataset.
        
        Args:
            idx: Index của mẫu cần lấy
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary chứa các tensor:
                - input_ids: Token ids [max_length]
                - attention_mask: Attention mask [max_length]
                - labels: Labels [max_length]
                - id: ID của mẫu
        """
        try:
            item = self.data[idx]
            
            # Convert lists to tensors
            input_ids = torch.tensor(item['input_ids'], dtype=torch.long)
            attention_mask = torch.tensor(item['attention_mask'], dtype=torch.long)
            labels = torch.tensor(item['labels'], dtype=torch.long)
            
            # Validate tensor sizes
            assert input_ids.shape[0] == self.max_length, \
                f"input_ids shape {input_ids.shape} != ({self.max_length},)"
            assert attention_mask.shape[0] == self.max_length, \
                f"attention_mask shape {attention_mask.shape} != ({self.max_length},)"
            assert labels.shape[0] == self.max_length, \
                f"labels shape {labels.shape} != ({self.max_length},)"
            
            # Validate padding
            pad_mask = (input_ids == self.tokenizer.pad_token_id)
            assert torch.all(labels[pad_mask] == 2), \
                "Labels should be 2 (O tag) for padding tokens"
            
            return {
                'id': item['id'],
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {str(e)}")
            if 'item' in locals():
                logger.error(f"Text: {item.get('text', '')[:100]}...")
                logger.error(f"Labels length: {len(item.get('labels', []))}")
                logger.error(f"Actual length: {item.get('actual_length', 'N/A')}")
            raise e
    
    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Gộp các mẫu thành một batch.
        
        Args:
            batch: List các dictionary chứa tensors
            
        Returns:
            Dict[str, torch.Tensor]: Batch đã được gộp
        """
        try:
            # Validate batch size
            batch_size = len(batch)
            if batch_size == 0:
                raise ValueError("Empty batch")
            
            # Tách tensor fields và non-tensor fields
            tensor_fields = ['input_ids', 'attention_mask', 'labels']
            non_tensor_fields = ['id']
            
            # Stack tensors
            tensor_batch = {}
            for field in tensor_fields:
                tensors = [item[field] for item in batch]
                # Validate tensor shapes trước khi stack
                for i, tensor in enumerate(tensors):
                    if tensor.shape[0] != self.max_length:
                        raise ValueError(
                            f"Tensor {field} at index {i} has length {tensor.shape[0]}, "
                            f"expected {self.max_length}"
                        )
                tensor_batch[field] = torch.stack(tensors)
            
            # Keep non-tensor fields as lists
            non_tensor_batch = {
                field: [item[field] for item in batch]
                for field in non_tensor_fields
            }
            
            # Validate final batch shapes
            for field in tensor_fields:
                tensor = tensor_batch[field]
                assert tensor.shape == (batch_size, self.max_length), \
                    f"{field} shape {tensor.shape} != ({batch_size}, {self.max_length})"
            
            return {**tensor_batch, **non_tensor_batch}
            
        except Exception as e:
            logger.error(f"Error in collate_fn: {str(e)}")
            logger.error(f"Batch size: {len(batch)}")
            for i, item in enumerate(batch):
                logger.error(f"Item {i} shapes:")
                for k, v in item.items():
                    if isinstance(v, torch.Tensor):
                        logger.error(f"- {k}: {v.shape}")
            raise e 
"""
Module chứa các lớp xử lý dữ liệu cho ABSA.
"""

import json
import logging
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer

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
    
    def __init__(self, data_dir, tokenizer, max_length=512, bert_model_name="vinai/phobert-base"):
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
                    # Đảm bảo labels có độ dài bằng max_length
                    if len(item['labels']) < self.max_length:
                        item['labels'].extend([2] * (self.max_length - len(item['labels'])))  # Padding với O tag
                    else:
                        item['labels'] = item['labels'][:self.max_length]  # Cắt bớt nếu dài hơn
                    data.append(item)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý dòng: {line.strip()}")
                    logger.error(f"Chi tiết lỗi: {str(e)}")
                    continue
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Lấy một mẫu dữ liệu.
        
        Returns:
            dict: Dictionary chứa các tensor:
                - input_ids: Token ids [max_length]
                - attention_mask: Attention mask [max_length]
                - labels: Labels [max_length]
                - id: ID của mẫu
        """
        item = self.data[idx]
        
        # Tokenize text với padding và truncation
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Lấy các tensors và đảm bảo chúng có shape [max_length]
        input_ids = encoding['input_ids'].squeeze(0)  # [max_length]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [max_length]
        
        # Chuyển đổi labels thành tensor và đảm bảo độ dài
        labels = torch.tensor(item['labels'], dtype=torch.long)
        if len(labels) < self.max_length:
            labels = torch.nn.functional.pad(labels, (0, self.max_length - len(labels)), value=2)
        else:
            labels = labels[:self.max_length]
        
        # Đảm bảo tất cả tensors có cùng độ dài
        assert len(input_ids) == self.max_length, f"input_ids length {len(input_ids)} != {self.max_length}"
        assert len(attention_mask) == self.max_length, f"attention_mask length {len(attention_mask)} != {self.max_length}"
        assert len(labels) == self.max_length, f"labels length {len(labels)} != {self.max_length}"
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'id': item['id']
        }
    
    def collate_fn(self, batch):
        """Hàm để gộp các mẫu thành batch.
        
        Args:
            batch (list): Danh sách các mẫu
            
        Returns:
            dict: Dictionary chứa batched tensors và non-tensor fields
        """
        # Tách tensor fields và non-tensor fields
        tensor_fields = ['input_ids', 'attention_mask', 'labels']
        non_tensor_fields = ['id']
        
        # Stack các tensors và đảm bảo chúng có cùng shape
        tensor_batch = {}
        for field in tensor_fields:
            tensors = [item[field] for item in batch]
            # Kiểm tra shape của mỗi tensor
            for i, tensor in enumerate(tensors):
                if tensor.shape[0] != self.max_length:
                    raise ValueError(
                        f"Tensor {field} at index {i} has length {tensor.shape[0]}, "
                        f"expected {self.max_length}"
                    )
            tensor_batch[field] = torch.stack(tensors)
        
        # Giữ nguyên các non-tensor fields
        non_tensor_batch = {
            field: [item[field] for item in batch]
            for field in non_tensor_fields
        }
        
        # Kết hợp cả hai
        return {**tensor_batch, **non_tensor_batch} 
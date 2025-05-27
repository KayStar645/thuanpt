"""
Module dự đoán kết quả ABSA cho văn bản đầu vào.
Chứa các hàm và lớp cần thiết cho việc dự đoán aspect và sentiment.
"""

import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Import từ các module trong project
from src.model import PhoBERT_CRF_KGAN
from src.utils import VietnameseProcessor

# Định nghĩa các đường dẫn tương đối
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "model")
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "src", "embeddings")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "data", "predictions")

# Định nghĩa các nhãn cho bài toán ABSA
LABEL_MAP = {
    "START": 0,  # Nhãn bắt đầu chuỗi
    "END": 1,    # Nhãn kết thúc chuỗi
    "O": 2,      # Nhãn Outside (không phải aspect)
    "B-POSITIVE": 3, "I-POSITIVE": 4,  # Nhãn cho aspect tích cực
    "B-NEGATIVE": 5, "I-NEGATIVE": 6,  # Nhãn cho aspect tiêu cực
    "B-NEUTRAL": 7, "I-NEUTRAL": 8     # Nhãn cho aspect trung tính
}

# Định nghĩa ánh xạ ngược từ ID về nhãn
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

class Predictor:
    """Lớp dự đoán aspect và sentiment cho văn bản đầu vào.
    
    Attributes:
        model: Mô hình PhoBERT-CRF-KGAN đã được huấn luyện
        tokenizer: Tokenizer của PhoBERT
        bert_model: Mô hình BERT để lấy embeddings
        kg_vectors: Vectơ knowledge graph
        device: Thiết bị tính toán (CPU/GPU)
        processor: Bộ xử lý văn bản tiếng Việt
    """
    def __init__(self, model_path, kg_path, device=None):
        """Khởi tạo predictor.
        
        Args:
            model_path: Đường dẫn đến file mô hình đã huấn luyện
            kg_path: Đường dẫn đến file vectơ knowledge graph
            device: Thiết bị tính toán (CPU/GPU)
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")
        
        # Khởi tạo tokenizer và mô hình BERT
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.bert_model = AutoModel.from_pretrained("vinai/phobert-base").to(self.device)
        self.bert_model.eval()
        
        # Load knowledge graph vectors
        with open(kg_path, "r", encoding="utf-8") as f:
            self.kg_vectors = json.load(f)
        
        # Khởi tạo mô hình
        self.model = PhoBERT_CRF_KGAN(
            bert_model_name="vinai/phobert-base",
            num_labels=7,  # Không bao gồm START và END
            start_tag_id=LABEL_MAP["START"],
            end_tag_id=LABEL_MAP["END"],
            pad_tag_id=-100
        ).to(self.device)
        
        # Load trọng số đã huấn luyện
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Khởi tạo bộ xử lý văn bản
        self.processor = VietnameseProcessor()
    
    def preprocess_text(self, text):
        """Tiền xử lý văn bản đầu vào.
        
        Args:
            text: Văn bản cần xử lý
            
        Returns:
            tuple: (input_ids, attention_mask, tokens)
                - input_ids: ID của các token
                - attention_mask: Mask tensor
                - tokens: Danh sách các token
        """
        # Tokenize văn bản
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > 510:  # Trừ đi [CLS] và [SEP]
            tokens = tokens[:510]
        
        # Thêm special tokens
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Tạo attention mask
        attention_mask = [1] * len(input_ids)
        
        # Chuyển sang tensor
        input_ids = torch.tensor([input_ids], device=self.device)
        attention_mask = torch.tensor([attention_mask], device=self.device)
        
        return input_ids, attention_mask, tokens
    
    def get_embeddings(self, input_ids, attention_mask):
        """Lấy embeddings cho văn bản đầu vào.
        
        Args:
            input_ids: ID của các token
            attention_mask: Mask tensor
            
        Returns:
            torch.Tensor: Embeddings kết hợp từ BERT và KG
        """
        # Lấy BERT embeddings
        with torch.no_grad():
            outputs = self.bert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            bert_embeds = outputs.last_hidden_state
        
        # Tạo KG embeddings
        batch_size, seq_len = input_ids.shape
        kg_embeds = torch.zeros((batch_size, seq_len, 200), device=self.device)
        
        # Xử lý từng token
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        for i, token in enumerate(tokens):
            # Bỏ qua special tokens
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue
            
            # Tìm vector KG cho token
            token_key = token.lower().replace("▁", "")
            if token_key in self.kg_vectors:
                kg_embeds[0, i] = torch.tensor(self.kg_vectors[token_key], device=self.device)
        
        # Kết hợp BERT và KG embeddings
        return torch.cat([bert_embeds, kg_embeds], dim=-1)
    
    def predict(self, text):
        """Dự đoán aspect và sentiment cho văn bản đầu vào.
        
        Args:
            text: Văn bản cần dự đoán
            
        Returns:
            list: Danh sách các aspect và sentiment được dự đoán
                Mỗi phần tử là một dict với các trường:
                - text: Văn bản của aspect
                - start: Vị trí bắt đầu
                - end: Vị trí kết thúc
                - sentiment: Cảm xúc (POSITIVE/NEGATIVE/NEUTRAL)
        """
        # Tiền xử lý văn bản
        input_ids, attention_mask, tokens = self.preprocess_text(text)
        
        # Lấy embeddings
        embeddings = self.get_embeddings(input_ids, attention_mask)
        
        # Dự đoán
        with torch.no_grad():
            predictions = self.model(embeddings, attention_mask)
        
        # Xử lý kết quả dự đoán
        pred_labels = [ID_TO_LABEL[p] for p in predictions[0]]
        
        # Tìm các aspect và sentiment
        aspects = []
        current_aspect = None
        
        for i, (token, label) in enumerate(zip(tokens[1:-1], pred_labels[1:-1])):  # Bỏ qua [CLS] và [SEP]
            if label.startswith("B-"):
                # Bắt đầu aspect mới
                if current_aspect:
                    aspects.append(current_aspect)
                current_aspect = {
                    "text": token.replace("▁", ""),
                    "start": i,
                    "end": i + 1,
                    "sentiment": label[2:]  # Bỏ prefix "B-"
                }
            elif label.startswith("I-") and current_aspect:
                # Tiếp tục aspect hiện tại
                current_aspect["text"] += token.replace("▁", "")
                current_aspect["end"] = i + 1
        
        # Thêm aspect cuối cùng nếu có
        if current_aspect:
            aspects.append(current_aspect)
        
        return aspects
    
    def predict_batch(self, texts, batch_size=32):
        """Dự đoán cho một batch văn bản.
        
        Args:
            texts: Danh sách các văn bản cần dự đoán
            batch_size: Kích thước batch
            
        Returns:
            list: Danh sách kết quả dự đoán cho từng văn bản
        """
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Dự đoán"):
            batch_texts = texts[i:i + batch_size]
            batch_results = [self.predict(text) for text in batch_texts]
            results.extend(batch_results)
        return results
    
    def save_predictions(self, texts, output_file):
        """Lưu kết quả dự đoán vào file.
        
        Args:
            texts: Danh sách các văn bản cần dự đoán
            output_file: Tên file output
        """
        # Tạo thư mục output nếu chưa tồn tại
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Dự đoán và lưu kết quả
        predictions = self.predict_batch(texts)
        
        output_path = os.path.join(OUTPUT_DIR, output_file)
        with open(output_path, "w", encoding="utf-8") as f:
            for text, preds in zip(texts, predictions):
                result = {
                    "text": text,
                    "aspects": preds
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"Đã lưu kết quả dự đoán vào file: {output_path}")

def main():
    """Hàm chính để chạy dự đoán."""
    # Đường dẫn đến các file cần thiết
    model_path = os.path.join(MODEL_DIR, "best_model.pt")
    kg_path = os.path.join(EMBEDDING_DIR, "kg_vectors.json")
    
    # Khởi tạo predictor
    predictor = Predictor(model_path, kg_path)
    
    # Đọc dữ liệu test
    test_file = os.path.join(DATA_DIR, "processed", "test.jsonl")
    texts = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    
    # Dự đoán và lưu kết quả
    predictor.save_predictions(texts, "predictions.jsonl")

if __name__ == "__main__":
    main() 
"""
Module huấn luyện mô hình ABSA.
Chứa các hàm và lớp cần thiết cho việc huấn luyện mô hình PhoBERT-CRF-KGAN.
"""

import os
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, f1_score

# Import từ các module trong project
from src.model import PhoBERT_CRF_KGAN
from src.utils import VietnameseProcessor

# Định nghĩa các đường dẫn tương đối
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "model")
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "src", "embeddings")

# Định nghĩa các nhãn cho bài toán ABSA
LABEL_MAP = {
    "START": 0,  # Nhãn bắt đầu chuỗi
    "END": 1,    # Nhãn kết thúc chuỗi
    "O": 2,      # Nhãn Outside (không phải aspect)
    "B-POSITIVE": 3, "I-POSITIVE": 4,  # Nhãn cho aspect tích cực
    "B-NEGATIVE": 5, "I-NEGATIVE": 6,  # Nhãn cho aspect tiêu cực
    "B-NEUTRAL": 7, "I-NEUTRAL": 8     # Nhãn cho aspect trung tính
}

# Số lượng nhãn thực tế (không bao gồm START và END)
NUM_LABELS = 7  # O, B-POSITIVE, I-POSITIVE, B-NEGATIVE, I-NEGATIVE, B-NEUTRAL, I-NEUTRAL

# Tổng số nhãn (bao gồm cả START và END)
TOTAL_LABELS = len(LABEL_MAP)  # 9 nhãn

class ABSADataset(Dataset):
    """Dataset cho bài toán Aspect-Based Sentiment Analysis (ABSA).
    
    Lớp này xử lý dữ liệu đầu vào dạng JSONL và chuyển đổi thành định dạng
    phù hợp cho việc huấn luyện mô hình.
    
    Attributes:
        samples (list): Danh sách các mẫu dữ liệu đã được xử lý
        max_length (int): Độ dài tối đa của chuỗi đầu vào
        tokenizer: Tokenizer của PhoBERT
    """
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.samples = []
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Đọc và xử lý dữ liệu từ file JSONL
        jsonl_path = os.path.join(DATA_DIR, "processed", jsonl_path)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj["text"]
                
                # Tokenize văn bản với PhoBERT tokenizer
                tokens = tokenizer.tokenize(text)
                if len(tokens) > max_length - 2:  # Trừ đi [CLS] và [SEP]
                    tokens = tokens[:max_length - 2]
                
                # Thêm special tokens
                tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                
                # Tạo attention mask
                attention_mask = [1] * len(input_ids)
                
                # Padding
                padding_length = max_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
                attention_mask = attention_mask + [0] * padding_length
                
                # Chuyển sang tensor
                input_ids = torch.tensor(input_ids, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                
                # Khởi tạo labels với O (2)
                labels = torch.full_like(input_ids, LABEL_MAP["O"], dtype=torch.long)
                
                # Map labels cho các aspect terms
                for start, end, tag in obj["labels"]:
                    # Tìm các token tương ứng với span [start, end]
                    span_text = text[start:end]
                    span_tokens = tokenizer.tokenize(span_text)
                    
                    # Tìm vị trí của span tokens trong tokens (đã bao gồm [CLS])
                    for i in range(1, len(tokens) - len(span_tokens)):  # Bắt đầu từ 1 để bỏ qua [CLS]
                        if tokens[i:i + len(span_tokens)] == span_tokens:
                            # Gán nhãn B- cho token đầu tiên
                            labels[i] = torch.tensor(LABEL_MAP[f"B-{tag.upper()}"], dtype=torch.long)
                            # Gán nhãn I- cho các token còn lại
                            for j in range(1, len(span_tokens)):
                                labels[i + j] = torch.tensor(LABEL_MAP[f"I-{tag.upper()}"], dtype=torch.long)
                            break
                
                # Đánh dấu padding tokens trong labels
                labels = torch.where(attention_mask == 0, torch.tensor(-100, dtype=torch.long), labels)
                
                # Lưu sample
                self.samples.append({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch, tokenizer, bert_model, kg_vectors, device, max_length=512):
    """Hàm gom nhóm dữ liệu cho DataLoader.
    
    Args:
        batch: Batch dữ liệu đầu vào
        tokenizer: Tokenizer của PhoBERT
        bert_model: Mô hình BERT để lấy embeddings
        kg_vectors: Vectơ knowledge graph
        device: Thiết bị tính toán (CPU/GPU)
        max_length: Độ dài tối đa của chuỗi
        
    Returns:
        tuple: (combined_embeds, attention_mask, labels)
            - combined_embeds: Embeddings kết hợp từ BERT và KG
            - attention_mask: Mask tensor
            - labels: Nhãn đích
    """
    # Gom nhóm dữ liệu
    input_ids = torch.stack([item["input_ids"] for item in batch]).to(device)
    attention_mask = torch.stack([item["attention_mask"] for item in batch]).to(device)
    labels = torch.stack([item["labels"] for item in batch]).to(device)
    
    # Lấy BERT embeddings
    with torch.no_grad():
        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        bert_embeds = outputs.last_hidden_state
    
    # Tạo KG embeddings
    batch_size, seq_len = input_ids.shape
    kg_embeds = torch.zeros((batch_size, seq_len, 200), device=device)
    
    # Xử lý từng batch
    for b in range(batch_size):
        tokens = tokenizer.convert_ids_to_tokens(input_ids[b])
        for i, token in enumerate(tokens):
            # Bỏ qua special tokens
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                continue
                
            # Tìm vector KG cho token
            token_key = token.lower().replace("▁", "")
            if token_key in kg_vectors:
                kg_embeds[b, i] = torch.tensor(kg_vectors[token_key], device=device)
    
    # Kết hợp BERT và KG embeddings
    combined_embeds = torch.cat([bert_embeds, kg_embeds], dim=-1)
    
    return combined_embeds, attention_mask, labels

def train(model, dataloader, optimizer, device, num_epochs=10):
    """Huấn luyện mô hình.
    
    Args:
        model: Mô hình cần huấn luyện
        dataloader: DataLoader chứa dữ liệu training
        optimizer: Optimizer để cập nhật tham số
        device: Thiết bị tính toán (CPU/GPU)
        num_epochs: Số epoch huấn luyện
    """
    model.train()
    total_loss = 0
    best_loss = float('inf')
    
    # Thêm learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,  # Learning rate tối đa
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,  # 10% đầu tiên để warmup
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1e4  # Final lr = initial_lr/1e4
    )
    
    # Gradient clipping
    max_grad_norm = 1.0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (inputs_embeds, attn_mask, labels) in enumerate(progress_bar):
            # Forward pass
            loss = model(inputs_embeds, attn_mask, labels)
            
            # Kiểm tra loss hợp lệ
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss value: {loss.item()}")
                continue
                
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            # Update weights
            optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Cập nhật loss
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            # Cập nhật progress bar với learning rate hiện tại
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # Lưu mô hình tốt nhất
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(MODEL_DIR, 'best_model.pt'))
            
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

def evaluate(model, dataloader, device):
    """Đánh giá mô hình trên tập validation/test.
    
    Args:
        model: Mô hình cần đánh giá
        dataloader: DataLoader chứa dữ liệu validation/test
        device: Thiết bị tính toán (CPU/GPU)
        
    Returns:
        dict: Kết quả đánh giá bao gồm precision, recall, f1-score
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs_embeds, attn_mask, labels in tqdm(dataloader, desc="Evaluating"):
            # Forward pass
            predictions = model(inputs_embeds, attn_mask)
            
            # Lưu predictions và labels
            for pred, label, mask in zip(predictions, labels, attn_mask):
                # Chỉ lấy các token thực (không phải padding)
                valid_pred = [p for p, m in zip(pred, mask) if m == 1]
                valid_label = [l.item() for l, m in zip(label, mask) if m == 1]
                
                all_predictions.extend(valid_pred)
                all_labels.extend(valid_label)
    
    # Tính toán metrics
    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(2, TOTAL_LABELS)),  # Bỏ qua START và END
        target_names=[k for k, v in LABEL_MAP.items() if k not in ["START", "END"]],
        output_dict=True
    )
    
    return report

def main():
    """Hàm chính để huấn luyện và đánh giá mô hình."""
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Khởi tạo tokenizer và mô hình BERT
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    bert_model = AutoModel.from_pretrained("vinai/phobert-base").to(device)
    
    # Load knowledge graph vectors
    kg_path = os.path.join(EMBEDDING_DIR, "kg_vectors.json")
    with open(kg_path, "r", encoding="utf-8") as f:
        kg_vectors = json.load(f)
    
    # Tạo datasets
    train_dataset = ABSADataset("train.jsonl", tokenizer)
    val_dataset = ABSADataset("val.jsonl", tokenizer)
    
    # Tạo dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, bert_model, kg_vectors, device)
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, bert_model, kg_vectors, device)
    )
    
    # Khởi tạo mô hình
    model = PhoBERT_CRF_KGAN(
        bert_model_name="vinai/phobert-base",
        num_labels=NUM_LABELS,
        start_tag_id=LABEL_MAP["START"],
        end_tag_id=LABEL_MAP["END"],
        pad_tag_id=-100
    ).to(device)
    
    # Khởi tạo optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    # Huấn luyện mô hình
    train(model, train_dataloader, optimizer, device)
    
    # Đánh giá mô hình
    results = evaluate(model, val_dataloader, device)
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

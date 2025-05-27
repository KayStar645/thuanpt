"""
Module huấn luyện mô hình ABSA.
"""

import os
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from src.model import PhoBERT_CRF_KGAN
from src.utils.data_loader import ABSADataset
from src.utils.preprocess import preprocess_text

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(epochs=10, batch_size=32, learning_rate=1e-5):
    """Huấn luyện mô hình ABSA.
    
    Args:
        epochs (int): Số epoch huấn luyện
        batch_size (int): Kích thước batch
        learning_rate (float): Learning rate
    """
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Sử dụng device: {device}")
    
    # Khởi tạo tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Khởi tạo dataset và dataloader
    train_dataset = ABSADataset(
        data_dir="src/data/origin",
        tokenizer=tokenizer,
        max_length=128
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    # Khởi tạo model
    model = PhoBERT_CRF_KGAN(
        bert_model_name="vinai/phobert-base",
        num_labels=7,  # Số lượng nhãn trong tập dữ liệu
        device=device
    )
    
    # Huấn luyện model
    history = model.fit(
        train_dataloader=train_dataloader,
        num_epochs=epochs,
        learning_rate=learning_rate,
        checkpoint_dir="src/data/checkpoints"
    )
    
    # Lưu model
    model.save_pretrained("src/data/model/phobert_crf_kgan.pt")
    logger.info("Đã lưu model thành công")

if __name__ == "__main__":
    main()

"""
Module huấn luyện mô hình ABSA.
"""

import os
import logging
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer
from src.model import PhoBERT_CRF_KGAN
from src.utils.data_loader import ABSADataset
from src.utils.preprocess import preprocess_text

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    
    # Khởi tạo dataset
    full_dataset = ABSADataset(
        data_dir="src/data/origin",
        tokenizer=tokenizer,
        max_length=128
    )
    
    # Chia dataset thành train và validation (90-10)
    val_size = int(len(full_dataset) * 0.1)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Kích thước tập train: {len(train_dataset)}")
    logger.info(f"Kích thước tập validation: {len(val_dataset)}")
    
    # Khởi tạo dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Khởi tạo model
    model = PhoBERT_CRF_KGAN(
        bert_model_name="vinai/phobert-base",
        num_labels=7,  # Số lượng nhãn trong tập dữ liệu
        device=device,
        warmup_steps=len(train_dataloader) // 2  # Warmup trong nửa epoch đầu
    )
    
    # Tạo thư mục checkpoints nếu chưa tồn tại
    checkpoint_dir = "src/data/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Huấn luyện model
    logger.info("Bắt đầu huấn luyện...")
    history = model.fit(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,  # Thêm validation dataloader
        num_epochs=epochs,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir
    )
    
    # Lưu model
    model_dir = "src/data/model"
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(os.path.join(model_dir, "phobert_crf_kgan.pt"))
    logger.info("Đã lưu model thành công")
    
    # Log kết quả cuối cùng
    logger.info("Kết quả huấn luyện:")
    logger.info(f"Train loss cuối cùng: {history['train_loss'][-1]:.4f}")
    if 'val_loss' in history:
        logger.info(f"Validation loss cuối cùng: {history['val_loss'][-1]:.4f}")
        logger.info(f"Validation F1 cuối cùng: {history['val_f1'][-1]:.4f}")

if __name__ == "__main__":
    main()

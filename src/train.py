"""
Script training cho mô hình ABSA.
"""

import os
import yaml
import torch
import logging
import argparse
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
from tqdm import tqdm

from model.phoBERT_CRF_KGAN import PhoBERT_CRF_KGAN
from data.dataset import create_dataloader
from utils.metrics import compute_metrics
from utils.training import set_seed, save_model, load_model

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(
    model: PhoBERT_CRF_KGAN,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    config: dict
) -> float:
    """Training một epoch.
    
    Args:
        model: Model cần train
        train_loader: DataLoader cho dữ liệu training
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device để train
        config: Cấu hình training
        
    Returns:
        float: Loss trung bình của epoch
    """
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    for step, batch in enumerate(progress_bar):
        # Chuyển batch lên device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config['model']['max_grad_norm']
        )
        
        # Update weights
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/(step+1):.4f}"
        })
        
        # Log và lưu model
        if (step + 1) % config['training']['logging_steps'] == 0:
            logger.info(
                f"Step {step+1}: loss = {loss.item():.4f}, "
                f"avg_loss = {total_loss/(step+1):.4f}"
            )
        
        if (step + 1) % config['training']['save_steps'] == 0:
            save_model(
                model,
                optimizer,
                scheduler,
                step + 1,
                config['data']['output_dir']
            )
    
    return total_loss / len(train_loader)

def evaluate(
    model: PhoBERT_CRF_KGAN,
    eval_loader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    """Đánh giá model.
    
    Args:
        model: Model cần đánh giá
        eval_loader: DataLoader cho dữ liệu evaluation
        device: Device để evaluate
        
    Returns:
        dict: Các metrics đánh giá
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            # Chuyển batch lên device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            preds = outputs.predictions
            
            # Lưu predictions và labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Tính metrics
    metrics = compute_metrics(all_preds, all_labels)
    return metrics

def main():
    """Hàm chính để training model."""
    parser = argparse.ArgumentParser(description="Training mô hình ABSA")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Đường dẫn file cấu hình"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        required=True,
        help="File dữ liệu training"
    )
    parser.add_argument(
        "--val_file",
        type=str,
        required=True,
        help="File dữ liệu validation"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Thư mục lưu model"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    args = parser.parse_args()
    
    # Đọc cấu hình
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(args.seed)
    
    # Tạo thư mục output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Khởi tạo model
    device = torch.device(config['training']['device'])
    model = PhoBERT_CRF_KGAN(config['model']).to(device)
    
    # Tạo dataloader
    train_loader = create_dataloader(
        args.train_file,
        tokenizer_name=config['model']['bert_model_name'],
        max_length=config['training']['max_length'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    val_loader = create_dataloader(
        args.val_file,
        tokenizer_name=config['model']['bert_model_name'],
        max_length=config['training']['max_length'],
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Khởi tạo optimizer và scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['model']['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Training
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Training
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            config
        )
        logger.info(f"Training loss: {train_loss:.4f}")
        
        # Evaluation
        metrics = evaluate(model, val_loader, device)
        logger.info(f"Validation metrics: {metrics}")
        
        # Lưu model tốt nhất
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_model(
                model,
                optimizer,
                scheduler,
                epoch + 1,
                output_dir,
                is_best=True
            )
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['model']['early_stopping_patience']:
            logger.info("Early stopping triggered")
            break
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()

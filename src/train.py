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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from model.phoBERT_CRF_KGAN import PhoBERT_CRF_KGAN
from utils.data_loader import ABSADataset
from utils.metrics import compute_metrics
from utils.training import set_seed, save_model, load_model

def setup_logging(logging_config: dict) -> None:
    """Thiết lập logging cho training.
    
    Args:
        logging_config (dict): Cấu hình logging từ file config
    """
    # Tạo thư mục logs nếu chưa tồn tại
    log_dir = os.path.dirname(logging_config['file'])
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Thiết lập logging
    logging.basicConfig(
        level=getattr(logging, logging_config['level']),
        format=logging_config['format'],
        handlers=[
            logging.FileHandler(logging_config['file']),
            logging.StreamHandler()
        ]
    )

# Thiết lập logging mặc định
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
        # Chuyển các tensor fields lên device
        tensor_fields = ['input_ids', 'attention_mask', 'labels']
        batch = {
            k: v.to(device) if k in tensor_fields else v
            for k, v in batch.items()
        }
        
        # Forward pass
        outputs = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
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
            # Chuyển các tensor fields lên device
            tensor_fields = ['input_ids', 'attention_mask', 'labels']
            batch = {
                k: v.to(device) if k in tensor_fields else v
                for k, v in batch.items()
            }
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            preds = outputs.predictions
            
            # Lưu predictions và labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
    
    # Tính metrics
    metrics = compute_metrics(all_preds, all_labels)
    return metrics

def main():
    """Hàm chính để training model."""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data')
    parser.add_argument('--val_file', type=str, required=True, help='Path to validation data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Convert string values to appropriate types
    # Model config
    model_config = config['model']
    model_config['dropout'] = float(model_config['dropout'])
    model_config['max_grad_norm'] = float(model_config['max_grad_norm'])
    model_config['warmup_steps'] = int(model_config['warmup_steps'])
    model_config['early_stopping_patience'] = int(model_config['early_stopping_patience'])
    model_config['num_labels'] = int(model_config['num_labels'])
    model_config['bert_hidden_size'] = int(model_config['bert_hidden_size'])
    model_config['kg_hidden_size'] = int(model_config['kg_hidden_size'])
    model_config['lstm_hidden_size'] = int(model_config['lstm_hidden_size'])
    model_config['lstm_num_layers'] = int(model_config['lstm_num_layers'])
    
    # Training config
    training_config = config['training']
    training_config['learning_rate'] = float(training_config['learning_rate'])
    training_config['weight_decay'] = float(training_config['weight_decay'])
    training_config['batch_size'] = int(training_config['batch_size'])
    training_config['num_epochs'] = int(training_config['num_epochs'])
    training_config['num_workers'] = int(training_config['num_workers'])
    training_config['logging_steps'] = int(training_config['logging_steps'])
    training_config['save_steps'] = int(training_config['save_steps'])
    training_config['eval_steps'] = int(training_config['eval_steps'])
    training_config['max_length'] = int(training_config['max_length'])
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    setup_logging(config['logging'])
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(config['training']['device'])
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model']['bert_model_name'])
    
    # Create datasets
    train_dataset = ABSADataset(
        data_dir=os.path.dirname(args.train_file),
        tokenizer=tokenizer,
        max_length=config['training']['max_length']
    )
    
    val_dataset = ABSADataset(
        data_dir=os.path.dirname(args.val_file),
        tokenizer=tokenizer,
        max_length=config['training']['max_length']
    )
    
    # Create dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=train_dataset.collate_fn
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=val_dataset.collate_fn
    )
    
    # Initialize model
    model = PhoBERT_CRF_KGAN(config['model']).to(device)
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Initialize scheduler
    num_training_steps = len(train_dataloader) * config['training']['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['model']['warmup_steps'],
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_f1 = 0
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        
        # Training
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            # Chuyển các tensor fields lên device
            tensor_fields = ['input_ids', 'attention_mask', 'labels']
            batch = {
                k: v.to(device) if k in tensor_fields else v
                for k, v in batch.items()
            }
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['model']['max_grad_norm']
            )
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            if (step + 1) % config['training']['logging_steps'] == 0:
                logger.info(
                    f"Step {step + 1}/{len(train_dataloader)} - "
                    f"Loss: {loss.item():.4f}"
                )
            
            # Save checkpoint
            if (step + 1) % config['training']['save_steps'] == 0:
                save_model(
                    model=model,
                    output_dir=args.output_dir,
                    step=step + 1,
                    epoch=epoch + 1
                )
            
            # Evaluation
            if (step + 1) % config['training']['eval_steps'] == 0:
                val_metrics = evaluate(model, val_dataloader, device)
                logger.info(f"Validation metrics: {val_metrics}")
                
                # Early stopping
                if val_metrics['f1'] > best_f1:
                    best_f1 = val_metrics['f1']
                    patience_counter = 0
                    # Save best model
                    save_model(
                        model=model,
                        output_dir=args.output_dir,
                        step=step + 1,
                        epoch=epoch + 1,
                        is_best=True
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= config['model']['early_stopping_patience']:
                        logger.info("Early stopping triggered")
                        return
        
        # End of epoch
        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} - Average loss: {avg_loss:.4f}")
        
        # Final evaluation
        val_metrics = evaluate(model, val_dataloader, device)
        logger.info(f"Final validation metrics: {val_metrics}")
        
        # Save final model
        save_model(
            model=model,
            output_dir=args.output_dir,
            step=len(train_dataloader),
            epoch=epoch + 1,
            is_final=True
        )
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()

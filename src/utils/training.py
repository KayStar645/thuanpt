"""
Module chứa các utility functions cho training.
"""

import torch
import random
import numpy as np
from pathlib import Path

def set_seed(seed: int):
    """Set random seed cho reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_model(model, optimizer, scheduler, step, output_dir, is_best=False):
    """Lưu checkpoint của model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': step
    }
    
    if is_best:
        torch.save(checkpoint, output_dir / 'best_model.pt')
    else:
        torch.save(checkpoint, output_dir / f'checkpoint-{step}.pt')

def load_model(model, optimizer, scheduler, checkpoint_path):
    """Load checkpoint của model."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step'] 
"""
Module tính toán các metrics cho model ABSA.
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(preds, labels):
    """Tính toán các metrics đánh giá.
    
    Args:
        preds: Predictions từ model
        labels: Ground truth labels
        
    Returns:
        dict: Các metrics (precision, recall, f1, accuracy)
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': acc
    } 
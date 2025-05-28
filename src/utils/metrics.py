"""
Module chứa các hàm tính toán metrics cho bài toán ABSA.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Union
from sklearn.metrics import precision_recall_fscore_support

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(predictions: List[List[int]], labels: List[List[int]]) -> Dict[str, float]:
    """Tính toán các metrics cho bài toán ABSA.
    
    Args:
        predictions: Danh sách các dự đoán cho mỗi mẫu
        labels: Danh sách các nhãn thực tế cho mỗi mẫu
        
    Returns:
        Dict[str, float]: Dictionary chứa các metrics:
            - precision: Precision trung bình
            - recall: Recall trung bình
            - f1: F1-score trung bình
            - precision_per_class: Precision cho từng class
            - recall_per_class: Recall cho từng class
            - f1_per_class: F1-score cho từng class
    """
    # Chuyển đổi predictions và labels thành mảng 1D
    flat_predictions = []
    flat_labels = []
    
    for pred, label in zip(predictions, labels):
        # Chỉ xét các token không phải padding (label != 2)
        mask = np.array(label) != 2
        flat_predictions.extend(np.array(pred)[mask])
        flat_labels.extend(np.array(label)[mask])
    
    # Tính toán metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels,
        flat_predictions,
        average='weighted',
        zero_division=0
    )
    
    # Tính toán metrics cho từng class
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        flat_labels,
        flat_predictions,
        average=None,
        zero_division=0
    )
    
    # Log chi tiết metrics
    logger.info("\nMetrics Summary:")
    logger.info(f"Overall Metrics:")
    logger.info(f"- Precision: {precision:.4f}")
    logger.info(f"- Recall: {recall:.4f}")
    logger.info(f"- F1-score: {f1:.4f}")
    
    logger.info("\nPer-class Metrics:")
    for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        logger.info(f"Class {i}:")
        logger.info(f"- Precision: {p:.4f}")
        logger.info(f"- Recall: {r:.4f}")
        logger.info(f"- F1-score: {f:.4f}")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist()
    }

def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics thành chuỗi để hiển thị.
    
    Args:
        metrics: Dictionary chứa các metrics
        
    Returns:
        str: Chuỗi metrics đã được format
    """
    return (
        f"Precision: {metrics['precision']:.4f}, "
        f"Recall: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}"
    ) 
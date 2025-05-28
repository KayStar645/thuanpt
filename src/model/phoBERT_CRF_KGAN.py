"""
Module chứa mô hình PhoBERT-CRF-KGAN cho bài toán ABSA.
Kết hợp PhoBERT, Knowledge Graph Attention Network và CRF để phân tích cảm xúc dựa trên khía cạnh.
"""

import logging
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import AutoModel, get_linear_schedule_with_warmup, AutoTokenizer
from TorchCRF import CRF
from sklearn.metrics import precision_recall_fscore_support
from typing import Optional, Tuple, List, Dict, Any, Union
from .attention import KGAttention
from .point_wise_feed_forward import PointWiseFeedForward
from .squeeze_embedding import SqueezeEmbedding
from .dynamic_rnn import DynamicRNN

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PhoBERT_CRF_KGAN(nn.Module):
    """
    Mô hình PhoBERT-CRF-KGAN cho bài toán ABSA.

    Kiến trúc:
    1. PhoBERT (fine-tune 4 layer cuối)
    2. Knowledge Graph Attention Network
    3. SqueezeEmbedding để loại bỏ padding
    4. Point-wise Feed Forward Network với LayerNorm và Residual
    5. BiLSTM (2 lớp) với LayerNorm và Residual
    6. CRF layer

    Args:
        config (dict): Cấu hình model
    """
    def __init__(self, config):
        """Khởi tạo model.
        
        Args:
            config (dict): Cấu hình model
        """
        super().__init__()
        
        # Lấy các tham số từ config
        bert_model_name = config.get('bert_model_name', 'vinai/phobert-base')
        num_labels = config.get('num_labels', 9)
        bert_hidden_size = config.get('bert_hidden_size', 768)
        kg_hidden_size = config.get('kg_hidden_size', 200)
        lstm_hidden_size = config.get('lstm_hidden_size', 256)
        lstm_num_layers = config.get('lstm_num_layers', 2)
        dropout = config.get('dropout', 0.1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Khởi tạo PhoBERT_CRF_KGAN trên {device}")
        
        # Khởi tạo BERT
        self.bert = AutoModel.from_pretrained(bert_model_name)
        
        # Các layer khác
        self.kg_projection = nn.Linear(bert_hidden_size, kg_hidden_size)
        self.lstm = nn.LSTM(
            input_size=bert_hidden_size + kg_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels)
        self.crf = CRF(num_labels)  # Bỏ batch_first
        
        # Lưu num_labels cho get_config
        self.num_labels = num_labels
        
        # Log kích thước tensor
        logger.info("Kích thước tensor:")
        logger.info(f"- BERT hidden size: {bert_hidden_size}")
        logger.info(f"- KG hidden size: {kg_hidden_size}")
        logger.info(f"- Combined size: {bert_hidden_size + kg_hidden_size}")
        logger.info(f"- BiLSTM output size: {lstm_hidden_size * 2}")
        logger.info(f"- Number of labels: {num_labels}")
        
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass.
        
        Args:
            input_ids: Token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size, seq_len] (optional)
            
        Returns:
            outputs: Model outputs
        """
        # BERT encoding
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = bert_outputs.last_hidden_state
        
        # KG projection
        kg_output = self.kg_projection(sequence_output)
        
        # Combine BERT and KG outputs
        combined = torch.cat([sequence_output, kg_output], dim=-1)
        
        # BiLSTM
        lstm_output, _ = self.lstm(combined)
        lstm_output = self.dropout(lstm_output)
        
        # Classification
        emissions = self.classifier(lstm_output)
        
        # Chuyển đổi batch dimension cho CRF
        emissions = emissions.transpose(0, 1)  # [seq_len, batch_size, num_labels]
        mask = attention_mask.bool().transpose(0, 1)  # [seq_len, batch_size]
        
        if labels is not None:
            # Training mode
            labels = labels.transpose(0, 1)  # [seq_len, batch_size]
            loss = -self.crf(emissions, labels, mask=mask)
            return type('Outputs', (), {
                'loss': loss,
                'logits': emissions.transpose(0, 1)  # Chuyển lại [batch_size, seq_len, num_labels]
            })
        else:
            # Inference mode
            predictions = self.crf.decode(emissions, mask=mask)
            # Chuyển predictions về dạng list of lists
            predictions = [pred for pred in zip(*predictions)]
            return type('Outputs', (), {
                'predictions': predictions,
                'logits': emissions.transpose(0, 1)  # Chuyển lại [batch_size, seq_len, num_labels]
            })

    def get_config(self) -> Dict[str, Any]:
        """Lấy cấu hình của model.
        
        Returns:
            Dict[str, Any]: Dictionary chứa các tham số cấu hình
        """
        return {
            'bert_model_name': self.bert.config._name_or_path,
            'num_labels': self.num_labels,
            'bert_hidden_size': self.bert.config.hidden_size,
            'kg_hidden_size': self.kg_projection.out_features,
            'lstm_hidden_size': self.lstm.hidden_size,
            'lstm_num_layers': self.lstm.num_layers,
            'dropout': self.dropout.p
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PhoBERT_CRF_KGAN':
        """Tạo instance từ cấu hình.
        
        Args:
            config (Dict[str, Any]): Dictionary chứa các tham số cấu hình
            
        Returns:
            PhoBERT_CRF_KGAN: Instance mới được tạo từ cấu hình
        """
        return cls(config)

if __name__ == "__main__":
    # Test code
    batch_size = 2
    seq_len = 10
    bert_hidden_size = 768
    kg_hidden_size = 200
    
    # Tạo input tensor và mask
    input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attention_mask[0, 5:] = False  # Một số token padding
    
    # Khởi tạo model
    model = PhoBERT_CRF_KGAN(
        config={
            'bert_model_name': 'vinai/phobert-base',
            'num_labels': 9,
            'bert_hidden_size': bert_hidden_size,
            'kg_hidden_size': kg_hidden_size,
            'lstm_hidden_size': bert_hidden_size + kg_hidden_size,
            'lstm_num_layers': 2,
            'dropout': 0.1
        }
    )
    
    # Forward pass (training)
    labels = torch.randint(0, 9, (batch_size, seq_len))
    outputs = model(input_ids, attention_mask, labels)
    print(f"Training loss: {outputs['loss'].item()}")
    
    # Forward pass (inference)
    outputs = model(input_ids, attention_mask)
    print(f"Number of predictions: {len(outputs['predictions'])}")
    print(f"First prediction length: {len(outputs['predictions'][0])}")
    print(f"Model config: {model.get_config()}")
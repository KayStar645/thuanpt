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
        bert_model_name (str): Tên mô hình BERT (mặc định: "vinai/phobert-base")
        num_labels (int): Số lượng nhãn (không bao gồm START và END)
        start_tag_id (int): ID của nhãn START
        end_tag_id (int): ID của nhãn END
        pad_tag_id (int): ID của nhãn padding (mặc định: -100)
        bert_hidden_size (int): Kích thước hidden của BERT (mặc định: 768)
        kg_hidden_size (int): Kích thước hidden của KG (mặc định: 200)
        lstm_hidden_size (int): Kích thước hidden của LSTM (mặc định: 256)
        dropout (float): Tỷ lệ dropout (mặc định: 0.1)
        device (torch.device): Thiết bị tính toán (CPU/GPU)
        max_grad_norm (float): Giá trị tối đa cho gradient clipping (mặc định: 1.0)
        warmup_steps (int): Số bước warmup cho learning rate (mặc định: 0)
        early_stopping_patience (int): Số epoch chờ trước khi early stopping (mặc định: 3)
        layer_norm_eps (float): Epsilon cho layer normalization (mặc định: 1e-5)
        use_residual (bool): Có sử dụng residual connection hay không (mặc định: True)
    """
    def __init__(
        self,
        bert_model_name="vinai/phobert-base",
        num_labels=7,
        start_tag_id=0,
        end_tag_id=1,
        pad_tag_id=-100,
        bert_hidden_size=768,
        kg_hidden_size=200,
        lstm_hidden_size=256,
        dropout=0.1,
        device=None,
        max_grad_norm=1.0,
        warmup_steps=0,
        early_stopping_patience=3,
        layer_norm_eps=1e-5,
        use_residual=True
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Khởi tạo PhoBERT_CRF_KGAN trên {self.device}")

        # Lưu các tham số cấu hình
        self.bert_hidden_size = bert_hidden_size
        self.kg_hidden_size = kg_hidden_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_labels = num_labels
        self.start_tag_id = start_tag_id
        self.end_tag_id = end_tag_id
        self.pad_tag_id = pad_tag_id
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.early_stopping_patience = early_stopping_patience
        self.layer_norm_eps = layer_norm_eps
        self.use_residual = use_residual

        # Tính toán kích thước tensor
        self.combined_size = bert_hidden_size + kg_hidden_size  # 768 + 200 = 968
        self.bilstm_output_size = lstm_hidden_size * 2  # 256 * 2 = 512

        logger.info(f"Kích thước tensor:")
        logger.info(f"- BERT hidden size: {bert_hidden_size}")
        logger.info(f"- KG hidden size: {kg_hidden_size}")
        logger.info(f"- Combined size: {self.combined_size}")
        logger.info(f"- BiLSTM output size: {self.bilstm_output_size}")
        logger.info(f"- Number of labels: {num_labels + 2}")

        # 1. PhoBERT
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        
        # Layer normalization cho BERT output
        self.bert_layer_norm = nn.LayerNorm(bert_hidden_size, eps=layer_norm_eps)

        # 2. KG projection + KG attention
        self.kg_projection = nn.Linear(bert_hidden_size, kg_hidden_size)
        self.kg_layer_norm = nn.LayerNorm(kg_hidden_size, eps=layer_norm_eps)
        self.kg_attention = KGAttention(
            input_size=kg_hidden_size,
            hidden_size=kg_hidden_size,
            dropout=dropout,
            attention_dropout=dropout
        )

        # 3. SqueezeEmbedding với các tính năng mới
        self.squeeze_embedding = SqueezeEmbedding(
            batch_first=True,
            padding_idx=pad_tag_id,
            max_len=512,  # Độ dài tối đa của PhoBERT
            use_mask=True
        )

        # 4. FFN với LayerNorm và Residual
        self.point_wise_ffn = PointWiseFeedForward(
            d_model=self.combined_size,
            d_ff=lstm_hidden_size * 4,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            use_residual=use_residual
        )

        # 5. BiLSTM với LayerNorm và Residual
        self.bilstm = DynamicRNN(
            input_size=self.combined_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            use_residual=use_residual
        )

        # 6. Final layer normalization
        self.final_layer_norm = nn.LayerNorm(self.bilstm_output_size, eps=layer_norm_eps)

        # 7. Dropout + Classifier + CRF
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bilstm_output_size, num_labels + 2)
        self.crf = CRF(num_labels + 2)

        # Khởi tạo trọng số
        self._init_weights()
        logger.info("Khởi tạo mô hình thành công")

    def _init_weights(self):
        """Khởi tạo trọng số cho các layer."""
        # BERT layers (chỉ fine-tune 4 layer cuối)
        for name, param in self.bert.named_parameters():
            if 'layer.11' in name or 'layer.10' in name or 'layer.9' in name or 'layer.8' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # BiLSTM layers
        for name, param in self.bilstm.named_parameters():
            if param.dim() > 1:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Linear layers
        nn.init.xavier_uniform_(self.kg_projection.weight)
        nn.init.zeros_(self.kg_projection.bias)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _validate_shapes(self, tensor, expected_shape, name):
        """Kiểm tra shape của tensor."""
        actual_shape = tensor.shape
        if actual_shape != expected_shape:
            raise ValueError(
                f"Shape không khớp ở {name}. "
                f"Expected: {expected_shape}, Got: {actual_shape}"
            )

    def forward(
        self, 
        input_embeds: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, List[List[int]]]:
        """Forward pass của mô hình.
        
        Args:
            input_embeds (torch.Tensor): BERT embeddings [batch_size, seq_len, bert_hidden_size]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            labels (Optional[torch.Tensor]): Ground truth labels [batch_size, seq_len]
            
        Returns:
            Union[torch.Tensor, List[List[int]]]: 
                - Nếu training: loss tensor
                - Nếu inference: danh sách các nhãn dự đoán
        """
        batch_size, seq_len, _ = input_embeds.shape

        # Validate input shapes
        self._validate_shapes(
            input_embeds, 
            (batch_size, seq_len, self.bert_hidden_size),
            "input_embeds"
        )
        self._validate_shapes(
            attention_mask,
            (batch_size, seq_len),
            "attention_mask"
        )

        normalized_mask = attention_mask.bool()

        # 1. BERT embeddings với layer normalization
        bert_output = self.bert_layer_norm(input_embeds)
        self._validate_shapes(bert_output, (batch_size, seq_len, self.bert_hidden_size), "bert_output")

        # 2. KG attention với layer normalization
        kg_embeds = self.kg_projection(bert_output)
        kg_embeds = self.kg_layer_norm(kg_embeds)
        self._validate_shapes(kg_embeds, (batch_size, seq_len, self.kg_hidden_size), "kg_embeds")
        
        kg_attended = self.kg_attention(kg_embeds, normalized_mask)
        self._validate_shapes(kg_attended, (batch_size, seq_len, self.kg_hidden_size), "kg_attended")

        # 3. Combine embeddings
        combined_embeds = torch.cat([bert_output, kg_attended], dim=-1)
        self._validate_shapes(combined_embeds, (batch_size, seq_len, self.combined_size), "combined_embeds")

        # 4. Squeeze embeddings
        squeezed_embeds = self.squeeze_embedding(combined_embeds, normalized_mask)
        self._validate_shapes(squeezed_embeds, (batch_size, seq_len, self.combined_size), "squeezed_embeds")

        # 5. FFN với LayerNorm và Residual
        ffn_output = self.point_wise_ffn(squeezed_embeds)
        self._validate_shapes(ffn_output, (batch_size, seq_len, self.combined_size), "ffn_output")

        # 6. BiLSTM với LayerNorm và Residual
        lstm_output, _ = self.bilstm(ffn_output, normalized_mask)
        self._validate_shapes(lstm_output, (batch_size, seq_len, self.bilstm_output_size), "lstm_output")

        # 7. Final layer normalization
        lstm_output = self.final_layer_norm(lstm_output)

        # 8. Dropout + Classifier
        lstm_output = self.dropout(lstm_output)
        emissions = self.classifier(lstm_output)
        self._validate_shapes(emissions, (batch_size, seq_len, self.num_labels + 2), "emissions")

        # 9. CRF
        if labels is not None:
            self._validate_shapes(labels, (batch_size, seq_len), "labels")
            labels = labels.clone()
            labels[labels == self.pad_tag_id] = self.end_tag_id
            emissions = emissions.transpose(0, 1)  # (seq_len, batch_size, num_labels + 2)
            labels = labels.transpose(0, 1)  # (seq_len, batch_size)
            mask = normalized_mask.transpose(0, 1)  # (seq_len, batch_size)
            loss = self.crf(emissions, labels, mask=mask)
            return loss.mean()
        else:
            emissions = emissions.transpose(0, 1)  # (seq_len, batch_size, num_labels + 2)
            mask = normalized_mask.transpose(0, 1)  # (seq_len, batch_size)
            predictions = self.crf.viterbi_decode(emissions, mask=mask)
            predictions = [pred for pred in zip(*predictions)]
            seq_lengths = normalized_mask.sum(dim=1).tolist()
            return [pred[:l] for pred, l in zip(predictions, seq_lengths)]

    def get_config(self) -> Dict[str, Any]:
        """Lấy cấu hình của model.
        
        Returns:
            Dict[str, Any]: Dictionary chứa các tham số cấu hình
        """
        return {
            'bert_model_name': self.bert.config._name_or_path,
            'num_labels': self.num_labels,
            'start_tag_id': self.start_tag_id,
            'end_tag_id': self.end_tag_id,
            'pad_tag_id': self.pad_tag_id,
            'bert_hidden_size': self.bert_hidden_size,
            'kg_hidden_size': self.kg_hidden_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'dropout': self.dropout.p,
            'max_grad_norm': self.max_grad_norm,
            'warmup_steps': self.warmup_steps,
            'early_stopping_patience': self.early_stopping_patience,
            'layer_norm_eps': self.layer_norm_eps,
            'use_residual': self.use_residual
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'PhoBERT_CRF_KGAN':
        """Tạo instance từ cấu hình.
        
        Args:
            config (Dict[str, Any]): Dictionary chứa các tham số cấu hình
            
        Returns:
            PhoBERT_CRF_KGAN: Instance mới được tạo từ cấu hình
        """
        return cls(**config)

if __name__ == "__main__":
    # Test code
    batch_size = 2
    seq_len = 10
    bert_hidden_size = 768
    kg_hidden_size = 200
    
    # Tạo input tensor và mask
    input_embeds = torch.randn(batch_size, seq_len, bert_hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    attention_mask[0, 5:] = False  # Một số token padding
    
    # Khởi tạo model
    model = PhoBERT_CRF_KGAN(
        bert_hidden_size=bert_hidden_size,
        kg_hidden_size=kg_hidden_size,
        num_labels=7
    )
    
    # Forward pass (training)
    labels = torch.randint(0, 7, (batch_size, seq_len))
    loss = model(input_embeds, attention_mask, labels)
    print(f"Training loss: {loss.item()}")
    
    # Forward pass (inference)
    predictions = model(input_embeds, attention_mask)
    print(f"Number of predictions: {len(predictions)}")
    print(f"First prediction length: {len(predictions[0])}")
    print(f"Model config: {model.get_config()}")
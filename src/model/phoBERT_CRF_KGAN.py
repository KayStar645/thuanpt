import torch
import torch.nn as nn
from transformers import AutoModel
from model.crf import CRF
from model.dynamic_rnn import DynamicLSTM
from model.point_wise_feed_forward import PointWiseFeedForward
from model.squeeze_embedding import SqueezeEmbedding

class PhoBERT_CRF_KGAN(nn.Module):
    """Model kết hợp PhoBERT, CRF và Knowledge Graph Attention Network cho bài toán ABSA.
    
    Args:
        bert_model_name (str): Tên hoặc đường dẫn của model BERT
        num_labels (int): Số lượng nhãn thực tế (không bao gồm START và END)
        start_tag_id (int): ID của nhãn bắt đầu
        end_tag_id (int): ID của nhãn kết thúc
        pad_tag_id (int): ID của nhãn padding
        hidden_size (int): Kích thước của vector ẩn
        kg_dim (int): Kích thước của vector knowledge graph
        dropout (float): Tỷ lệ dropout
x        lstm_hidden_size (int): Kích thước của hidden layer trong BiLSTM
        lstm_num_layers (int): Số lớp trong BiLSTM
        ff_dim (int): Kích thước của hidden layer trong Point-wise Feed Forward
    """
    def __init__(
        self,
        bert_model_name: str,
        num_labels: int,
        start_tag_id: int,
        end_tag_id: int,
        pad_tag_id: int,
        hidden_size: int = 768,
        kg_dim: int = 200,
        dropout: float = 0.1,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 2,
        ff_dim: int = 3072
    ):
        super(PhoBERT_CRF_KGAN, self).__init__()
        
        # Load BERT model và đóng băng các layer đầu
        self.bert = AutoModel.from_pretrained(bert_model_name)
        for param in list(self.bert.parameters())[:-4]:  # Chỉ fine-tune 4 layer cuối
            param.requires_grad = False
        
        # Squeeze embedding để xử lý padding hiệu quả
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        
        # Tổng kích thước vector sau khi kết hợp BERT và KG
        self.combined_size = hidden_size + kg_dim
        
        # Layer chuyển đổi với Point-wise Feed Forward
        self.transform = PointWiseFeedForward(
            d_model=self.combined_size,
            d_ff=ff_dim,
            dropout=dropout
        )
        
        # BiLSTM để xử lý chuỗi
        self.bilstm = DynamicLSTM(
            input_size=self.combined_size,
            hidden_size=lstm_hidden_size // 2,  # Chia 2 vì là bidirectional
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # Layer phân loại với khởi tạo trọng số
        self.classifier = nn.Linear(lstm_hidden_size, num_labels + 2)
        self._init_classifier_weights()
        
        # CRF layer với tổng số nhãn
        self.crf = CRF(
            tagset_size=num_labels + 2,
            start_tag_id=start_tag_id,
            end_tag_id=end_tag_id,
            pad_tag_id=pad_tag_id
        )
        
        # Knowledge Graph Attention với khởi tạo trọng số
        self.kg_attention = nn.Sequential(
            nn.Linear(kg_dim, hidden_size),
            nn.LayerNorm(hidden_size),  # Thêm LayerNorm
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
        self._init_kg_attention_weights()
        
        self.dropout = nn.Dropout(dropout)
        
        # Khởi tạo trọng số cho toàn bộ model
        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số cho các layer của model."""
        # Khởi tạo trọng số cho BERT (nếu cần)
        for name, param in self.bert.named_parameters():
            if 'layer.11' in name or 'layer.10' in name:  # Chỉ khởi tạo lại cho 2 layer cuối
                if param.dim() > 1:  # Chỉ khởi tạo cho tensor 2 chiều trở lên
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Khởi tạo trọng số cho BiLSTM
        for name, param in self.bilstm.named_parameters():
            if param.dim() > 1:  # Chỉ khởi tạo cho tensor 2 chiều trở lên
                if 'weight' in name:
                    nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Khởi tạo trọng số cho transform layer
        for name, param in self.transform.named_parameters():
            if param.dim() > 1:  # Chỉ khởi tạo cho tensor 2 chiều trở lên
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def _init_classifier_weights(self):
        """Khởi tạo trọng số cho classifier layer."""
        if self.classifier.weight.dim() > 1:
            nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _init_kg_attention_weights(self):
        """Khởi tạo trọng số cho KG attention layer."""
        for layer in self.kg_attention:
            if isinstance(layer, nn.Linear):
                if layer.weight.dim() > 1:
                    nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, input_embeds, attention_mask, labels=None):
        """Forward pass của model.
        
        Args:
            input_embeds (torch.Tensor): Vector đầu vào kết hợp từ BERT và KG
            attention_mask (torch.Tensor): Mask tensor
            labels (torch.Tensor, optional): Nhãn đích. Nếu None, model sẽ decode.
            
        Returns:
            Nếu labels được cung cấp: loss (torch.Tensor) - giá trị trung bình qua batch
            Nếu labels là None: predictions (List[List[int]])
        """
        # Tính độ dài thực tế của mỗi chuỗi
        seq_lengths = attention_mask.sum(dim=1)
        
        # Tách BERT embeddings và KG embeddings
        bert_embeds = input_embeds[:, :, :768]
        kg_embeds = input_embeds[:, :, 768:]
        
        # Áp dụng Knowledge Graph Attention với residual connection
        kg_attention_weights = self.kg_attention(kg_embeds)
        kg_attention_weights = torch.softmax(kg_attention_weights, dim=1)
        kg_attended = kg_embeds * kg_attention_weights + kg_embeds  # Residual connection
        
        # Kết hợp BERT và KG embeddings
        combined = torch.cat([bert_embeds, kg_attended], dim=-1)
        
        # Nén embedding để xử lý padding hiệu quả
        squeezed = self.squeeze_embedding(combined, seq_lengths)
        
        # Áp dụng Point-wise Feed Forward với residual connection
        transformed = self.transform(squeezed) + squeezed
        
        # Áp dụng BiLSTM
        lstm_out, _ = self.bilstm(transformed, seq_lengths)
        
        # Dropout và phân loại
        lstm_out = self.dropout(lstm_out)
        emissions = self.classifier(lstm_out)
        
        # Đảm bảo kích thước khớp nhau
        if labels is not None:
            if emissions.size(1) != labels.size(1):
                min_len = min(emissions.size(1), labels.size(1))
                emissions = emissions[:, :min_len, :]
                labels = labels[:, :min_len]
                attention_mask = attention_mask[:, :min_len]
        
        # Áp dụng CRF
        if labels is not None:
            # Training mode: tính loss
            batch_loss = self.crf(emissions, labels, attention_mask)
            # Tính trung bình loss qua batch
            loss = batch_loss.mean()
            return loss
        else:
            # Inference mode: decode chuỗi nhãn có xác suất cao nhất
            predictions = self.crf.decode(emissions, attention_mask)
            return predictions

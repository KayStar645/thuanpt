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
from .attention import KGAttention
from .feedforward import PointWiseFeedForward
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
    4. Point-wise Feed Forward Network
    5. BiLSTM
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
        early_stopping_patience=3
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

        # 2. KG projection + KG attention
        self.kg_projection = nn.Linear(bert_hidden_size, kg_hidden_size)
        self.kg_attention = KGAttention(
            input_size=kg_hidden_size,
            hidden_size=kg_hidden_size,
            dropout=dropout,
            attention_dropout=dropout
        )

        # 3. SqueezeEmbedding
        self.squeeze_embedding = SqueezeEmbedding()

        # 4. FFN
        self.point_wise_ffn = PointWiseFeedForward(
            d_model=self.combined_size,
            d_ff=lstm_hidden_size * 4,
            dropout=dropout
        )

        # 5. BiLSTM
        self.bilstm = DynamicRNN(
            input_size=self.combined_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Dropout + Classifier + CRF
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bilstm_output_size, num_labels + 2)
        self.crf = CRF(num_labels + 2)

        # Khởi tạo trọng số
        self._init_weights()
        logger.info("Khởi tạo mô hình thành công")

    def _init_weights(self):
        for name, param in self.bert.named_parameters():
            if 'layer.11' in name or 'layer.10' in name:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        for name, param in self.bilstm.named_parameters():
            if param.dim() > 1:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
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

    def forward(self, input_embeds, attention_mask, labels=None):
        """Forward pass của mô hình.
        
        Args:
            input_embeds (torch.Tensor): BERT embeddings [batch_size, seq_len, bert_hidden_size]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            labels (torch.Tensor, optional): Ground truth labels [batch_size, seq_len]
            
        Returns:
            Union[torch.Tensor, List[List[int]]]: 
                - Nếu training: loss tensor
                - Nếu inference: danh sách các nhãn dự đoán
        """
        batch_size, seq_len, _ = input_embeds.shape
        logger.debug(f"Input shapes - batch_size: {batch_size}, seq_len: {seq_len}")

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

        # 1. BERT embeddings (đã có sẵn)
        bert_output = input_embeds  # (batch_size, seq_len, bert_hidden_size)
        logger.debug(f"BERT output shape: {bert_output.shape}")

        # 2. KG attention
        kg_embeds = self.kg_projection(bert_output)  # (batch_size, seq_len, kg_hidden_size)
        self._validate_shapes(kg_embeds, (batch_size, seq_len, self.kg_hidden_size), "kg_embeds")
        
        kg_attended = self.kg_attention(kg_embeds, normalized_mask)  # (batch_size, seq_len, kg_hidden_size)
        self._validate_shapes(kg_attended, (batch_size, seq_len, self.kg_hidden_size), "kg_attended")
        logger.debug(f"KG attended shape: {kg_attended.shape}")

        # 3. Combine embeddings
        combined_embeds = torch.cat([bert_output, kg_attended], dim=-1)  # (batch_size, seq_len, combined_size)
        self._validate_shapes(combined_embeds, (batch_size, seq_len, self.combined_size), "combined_embeds")
        logger.debug(f"Combined embeddings shape: {combined_embeds.shape}")

        # 4. Squeeze embeddings
        squeezed_embeds = self.squeeze_embedding(combined_embeds, normalized_mask)
        self._validate_shapes(squeezed_embeds, (batch_size, seq_len, self.combined_size), "squeezed_embeds")
        logger.debug(f"Squeezed embeddings shape: {squeezed_embeds.shape}")

        # 5. FFN
        ffn_output = self.point_wise_ffn(squeezed_embeds)  # (batch_size, seq_len, combined_size)
        self._validate_shapes(ffn_output, (batch_size, seq_len, self.combined_size), "ffn_output")
        logger.debug(f"FFN output shape: {ffn_output.shape}")

        # 6. BiLSTM
        lstm_output, _ = self.bilstm(ffn_output, normalized_mask)  # (batch_size, seq_len, bilstm_output_size)
        self._validate_shapes(lstm_output, (batch_size, seq_len, self.bilstm_output_size), "lstm_output")
        logger.debug(f"LSTM output shape: {lstm_output.shape}")

        # 7. Dropout
        lstm_output = self.dropout(lstm_output)
        self._validate_shapes(lstm_output, (batch_size, seq_len, self.bilstm_output_size), "lstm_output after dropout")

        # 8. Classifier
        emissions = self.classifier(lstm_output)  # (batch_size, seq_len, num_labels + 2)
        self._validate_shapes(emissions, (batch_size, seq_len, self.num_labels + 2), "emissions")
        logger.debug(f"Emissions shape: {emissions.shape}")

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

    def fit(
        self,
        train_dataloader,
        val_dataloader=None,
        num_epochs=10,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=None,
        checkpoint_dir=None
    ):
        """Huấn luyện mô hình.
        
        Args:
            train_dataloader: DataLoader cho training
            val_dataloader: DataLoader cho validation (optional)
            num_epochs: Số epoch huấn luyện
            learning_rate: Learning rate
            weight_decay: Weight decay cho optimizer
            warmup_steps: Số bước warmup (nếu None, dùng self.warmup_steps)
            checkpoint_dir: Thư mục lưu checkpoint (optional)
            
        Returns:
            dict: Lịch sử huấn luyện
        """
        # Khởi tạo optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Khởi tạo scheduler
        if warmup_steps is None:
            warmup_steps = self.warmup_steps
        
        if warmup_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=len(train_dataloader) * num_epochs
            )
        else:
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=2,
                verbose=True
            )
        
        # Khởi tạo early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Lịch sử huấn luyện
        history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_metrics = self.train_epoch(train_dataloader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            history['train_metrics'].append(train_metrics)
            
            # Log training metrics
            logger.info(
                f"Training - Loss: {train_loss:.4f}, "
                f"F1: {train_metrics['avg_f1']:.4f}, "
                f"Precision: {train_metrics['avg_precision']:.4f}, "
                f"Recall: {train_metrics['avg_recall']:.4f}"
            )
            
            # Validation
            if val_dataloader is not None:
                val_loss, val_metrics = self.evaluate(val_dataloader)
                history['val_loss'].append(val_loss)
                history['val_metrics'].append(val_metrics)
                
                # Log validation metrics
                logger.info(
                    f"Validation - Loss: {val_loss:.4f}, "
                    f"F1: {val_metrics['avg_f1']:.4f}, "
                    f"Precision: {val_metrics['avg_precision']:.4f}, "
                    f"Recall: {val_metrics['avg_recall']:.4f}"
                )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.state_dict()
                    
                    # Lưu checkpoint
                    if checkpoint_dir is not None:
                        self.save_pretrained(f"{checkpoint_dir}/best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break
                
                # Update learning rate (nếu dùng ReduceLROnPlateau)
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
        
        # Load best model
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
        
        return history

    def train_epoch(self, dataloader, optimizer, scheduler=None):
        """Huấn luyện một epoch.
        
        Args:
            dataloader: DataLoader chứa dữ liệu training
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            
        Returns:
            tuple: (loss trung bình, metrics)
                - loss: float
                - metrics: dict chứa precision, recall, f1 cho từng class
        """
        self.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch_idx, (input_embeds, attention_mask, labels) in enumerate(dataloader):
            # Debug log input shapes
            if batch_idx == 0:
                logger.debug(f"Batch input shapes - embeds: {input_embeds.shape}, mask: {attention_mask.shape}, labels: {labels.shape}")
                logger.debug(f"Unique labels in first batch: {torch.unique(labels).tolist()}")
            
            # Chuyển dữ liệu lên device
            input_embeds = input_embeds.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            loss = self(input_embeds, attention_mask, labels)
            
            # Debug log loss
            if batch_idx == 0:
                logger.debug(f"First batch loss: {loss.item():.4f}")
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Debug log gradients
            if batch_idx == 0:
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        logger.debug(f"Gradients for {name} - mean: {param.grad.mean().item():.4f}, std: {param.grad.std().item():.4f}")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            
            # Update weights
            optimizer.step()
            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()
            
            # Cập nhật loss
            total_loss += loss.item()
            
            # Lưu predictions và labels cho metrics
            with torch.no_grad():
                predictions = self(input_embeds, attention_mask)
                normalized_mask = attention_mask.bool()
                for pred, label, m in zip(predictions, labels, normalized_mask):
                    valid_pred = [p for p, mask_val in zip(pred, m) if mask_val]
                    valid_label = [l.item() for l, mask_val in zip(label, m) if mask_val]
                    all_predictions.extend(valid_pred)
                    all_labels.extend(valid_label)
                
                # Debug logging for predictions
                if batch_idx % 10 == 0:
                    logger.debug(f"Batch {batch_idx} - Sample predictions: {valid_pred[:5]}")
                    logger.debug(f"Batch {batch_idx} - Sample labels: {valid_label[:5]}")
                    logger.debug(f"Batch {batch_idx} - Unique predictions: {set(valid_pred)}")
                    logger.debug(f"Batch {batch_idx} - Unique labels: {set(valid_label)}")
                    logger.debug(f"Batch {batch_idx} - Prediction distribution: {torch.bincount(torch.tensor(valid_pred)).tolist()}")
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                # Tính metrics cho batch hiện tại với zero_division=0
                batch_precision, batch_recall, batch_f1, _ = precision_recall_fscore_support(
                    all_labels[-len(valid_label):],
                    all_predictions[-len(valid_pred):],
                    average='weighted',
                    zero_division=0
                )
                logger.info(
                    f"Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"F1: {batch_f1:.4f}, "
                    f"Precision: {batch_precision:.4f}, "
                    f"Recall: {batch_recall:.4f}"
                )
        
        # Tính metrics cho toàn bộ epoch với zero_division=0
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average=None,
            labels=range(self.num_labels),
            zero_division=0
        )
        
        # Tính metrics trung bình (weighted) với zero_division=0
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted',
            zero_division=0
        )
        
        metrics = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1
        }
        
        return total_loss / len(dataloader), metrics

    def evaluate(self, dataloader):
        """Đánh giá mô hình.
        
        Args:
            dataloader: DataLoader chứa dữ liệu validation/test
            
        Returns:
            tuple: (loss trung bình, metrics)
                - loss: float
                - metrics: dict chứa precision, recall, f1 cho từng class
        """
        self.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for input_embeds, attention_mask, labels in dataloader:
                # Chuyển dữ liệu lên device
                input_embeds = input_embeds.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                loss = self(input_embeds, attention_mask, labels)
                predictions = self(input_embeds, attention_mask)
                
                # Cập nhật loss
                total_loss += loss.item()
                
                # Lưu predictions và labels với mask đã chuẩn hóa
                normalized_mask = attention_mask.bool()
                for pred, label, m in zip(predictions, labels, normalized_mask):
                    valid_pred = [p for p, mask_val in zip(pred, m) if mask_val]
                    valid_label = [l.item() for l, mask_val in zip(label, m) if mask_val]
                    all_predictions.extend(valid_pred)
                    all_labels.extend(valid_label)
        
        # Tính metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average=None,
            labels=range(self.num_labels)
        )
        
        # Tính metrics trung bình (weighted)
        avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average='weighted'
        )
        
        metrics = {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_f1': avg_f1
        }
        
        return total_loss / len(dataloader), metrics
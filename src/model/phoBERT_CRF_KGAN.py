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

        # Log số lượng nhãn
        logger.info(f"Số lượng nhãn (không bao gồm START/END): {num_labels}")
        logger.info(f"Tổng số nhãn (bao gồm START/END): {num_labels + 2}")

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
            d_model=bert_hidden_size + kg_hidden_size,
            d_ff=lstm_hidden_size * 4,
            dropout=dropout
        )

        # 5. BiLSTM (256 = 128 mỗi chiều)
        self.bilstm = DynamicRNN(
            input_size=bert_hidden_size + kg_hidden_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # Dropout + Classifier + CRF
        self.dropout = nn.Dropout(dropout)
        # Classifier output size should match number of labels (including START/END)
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels + 2)  # *2 for bidirectional
        # Initialize CRF with proper number of tags
        self.crf = CRF(num_labels + 2)  # TorchCRF doesn't support batch_first parameter
        
        # Initialize CRF transitions with small values
        with torch.no_grad():
            self.crf.transitions.data *= 0.1
            # Make transitions to START and END tags more likely
            self.crf.transitions.data[:, self.start_tag_id] *= 0.1
            self.crf.transitions.data[self.end_tag_id, :] *= 0.1

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
        logger.debug(f"Input embeddings shape: {input_embeds.shape}")
        logger.debug(f"Attention mask shape: {attention_mask.shape}")

        # 1. Chuẩn hóa attention mask
        normalized_mask = attention_mask.bool()

        # 2. Tạo KG embeddings và tính attention
        bert_embeds = input_embeds
        kg_embeds = self.kg_projection(bert_embeds)
        kg_attended = self.kg_attention(kg_embeds, normalized_mask)

        # 3. Kết hợp và squeeze
        combined_embeds = torch.cat([bert_embeds, kg_attended], dim=-1)
        squeezed_embeds = self.squeeze_embedding(combined_embeds, normalized_mask)

        # 4. FFN + BiLSTM + Dropout + Linear
        ffn_output = self.point_wise_ffn(squeezed_embeds)
        lstm_output, _ = self.bilstm(ffn_output, normalized_mask)
        lstm_output = self.dropout(lstm_output)
        emissions = self.classifier(lstm_output)

        # Debug log emissions
        logger.debug(f"Emissions shape: {emissions.shape}")
        logger.debug(f"Emissions min/max/mean: {emissions.min().item():.4f}/{emissions.max().item():.4f}/{emissions.mean().item():.4f}")

        # 5. Tính loss hoặc decode
        if labels is not None:
            labels = labels.clone()
            labels[labels == self.pad_tag_id] = self.end_tag_id
            
            # Debug log labels
            logger.debug(f"Labels shape: {labels.shape}")
            logger.debug(f"Unique labels: {torch.unique(labels).tolist()}")
            
            # CRF expects (seq_len, batch_size, num_tags) for emissions
            # and (seq_len, batch_size) for labels
            emissions = emissions.transpose(0, 1)  # (batch, seq, tags) -> (seq, batch, tags)
            labels = labels.transpose(0, 1)  # (batch, seq) -> (seq, batch)
            mask = normalized_mask.transpose(0, 1)  # (batch, seq) -> (seq, batch)
            
            # CRF returns negative log likelihood
            loss = self.crf(emissions, labels, mask=mask)
            # Take mean of loss across batch
            loss = loss.mean()
            logger.debug(f"Raw CRF loss: {loss.item():.4f}")
            return loss
        else:
            # CRF expects (seq_len, batch_size, num_tags) for emissions
            emissions = emissions.transpose(0, 1)  # (batch, seq, tags) -> (seq, batch, tags)
            mask = normalized_mask.transpose(0, 1)  # (batch, seq) -> (seq, batch)
            
            # Use viterbi_decode for predictions
            predictions = self.crf.viterbi_decode(emissions, mask=mask)
            # Convert predictions back to batch-first format
            predictions = [pred for pred in zip(*predictions)]  # Transpose back to (batch, seq)
            seq_lengths = normalized_mask.sum(dim=1).tolist()
            predictions = [pred[:length] for pred, length in zip(predictions, seq_lengths)]
            
            # Debug log predictions
            logger.debug(f"Number of sequences decoded: {len(predictions)}")
            if predictions:
                logger.debug(f"Sample prediction lengths: {[len(p) for p in predictions[:5]]}")
                logger.debug(f"Sample predictions: {predictions[:5]}")
            
            return predictions

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
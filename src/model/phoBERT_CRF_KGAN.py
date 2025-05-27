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
    """Mô hình PhoBERT-CRF-KGAN cho bài toán ABSA.
    
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
        
        # Lưu các tham số
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
        
        # Load BERT
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        
        # 2. Knowledge Graph Attention Network
        self.kg_attention = KGAttention(
            input_size=kg_hidden_size,
            hidden_size=kg_hidden_size,
            dropout=dropout,
            attention_dropout=dropout  # Thêm attention dropout
        )
        
        # 3. SqueezeEmbedding để loại bỏ padding
        self.squeeze_embedding = SqueezeEmbedding()
        
        # 4. Point-wise Feed Forward Network
        self.point_wise_ffn = PointWiseFeedForward(
            d_model=bert_hidden_size,
            d_ff=lstm_hidden_size * 4,
            dropout=dropout
        )
        
        # 5. BiLSTM
        self.bilstm = DynamicRNN(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 6. CRF layer
        self.crf = CRF(num_labels + 2)  # +2 cho START và END
        self.start_tag_id = start_tag_id
        self.end_tag_id = end_tag_id
        self.pad_tag_id = pad_tag_id
        
        # Classifier
        self.classifier = nn.Linear(lstm_hidden_size * 2, num_labels + 2)  # *2 cho bidirectional
        
        # Khởi tạo trọng số
        self._init_weights()
        
        logger.info("Khởi tạo mô hình thành công")
    
    def _init_weights(self):
        """Khởi tạo trọng số cho các layer."""
        # Khởi tạo trọng số cho BERT (nếu cần)
        for name, param in self.bert.named_parameters():
            if 'layer.11' in name or 'layer.10' in name:  # Chỉ khởi tạo lại cho 2 layer cuối
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Khởi tạo trọng số cho BiLSTM
        for name, param in self.bilstm.named_parameters():
            if param.dim() > 1:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Khởi tạo trọng số cho classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def build_input_embeds(self, bert_embeds, kg_embeds, attention_mask):
        """Xây dựng input embeddings từ BERT và KG.
        
        Args:
            bert_embeds (torch.Tensor): BERT embeddings [batch_size, seq_len, bert_hidden_size]
            kg_embeds (torch.Tensor): KG embeddings [batch_size, seq_len, kg_hidden_size]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            
        Returns:
            tuple: (combined_embeds, normalized_mask)
                - combined_embeds: Tensor kết hợp [batch_size, seq_len, bert_hidden_size + kg_hidden_size]
                - normalized_mask: Boolean mask đã chuẩn hóa [batch_size, seq_len]
        """
        # Log shapes
        logger.debug(f"BERT embeddings shape: {bert_embeds.shape}")
        logger.debug(f"KG embeddings shape: {kg_embeds.shape}")
        logger.debug(f"Attention mask shape: {attention_mask.shape}")
        
        # Chuẩn hóa attention mask
        normalized_mask = attention_mask.bool()
        
        # Áp dụng KG attention với mask đã chuẩn hóa
        kg_attended = self.kg_attention(kg_embeds, normalized_mask)
        
        # Kết hợp BERT và KG embeddings
        combined_embeds = torch.cat([bert_embeds, kg_attended], dim=-1)
        logger.debug(f"Combined embeddings shape: {combined_embeds.shape}")
        
        return combined_embeds, normalized_mask
    
    def forward(self, input_embeds, attention_mask, labels=None):
        """Forward pass của mô hình.
        
        Args:
            input_embeds (torch.Tensor): Combined embeddings [batch_size, seq_len, bert_hidden_size + kg_hidden_size]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            labels (torch.Tensor, optional): Ground truth labels [batch_size, seq_len]
            
        Returns:
            Union[torch.Tensor, List[List[int]]]: 
                - Nếu training: loss tensor
                - Nếu inference: danh sách các nhãn dự đoán
        """
        # Log shapes
        logger.debug(f"Input embeddings shape: {input_embeds.shape}")
        logger.debug(f"Attention mask shape: {attention_mask.shape}")
        
        # Tách BERT và KG embeddings
        bert_embeds = input_embeds[:, :, :self.bert_hidden_size]
        kg_embeds = input_embeds[:, :, self.bert_hidden_size:]
        
        # Xây dựng input embeddings và chuẩn hóa mask
        combined_embeds, normalized_mask = self.build_input_embeds(bert_embeds, kg_embeds, attention_mask)
        
        # Áp dụng SqueezeEmbedding với mask đã chuẩn hóa
        squeezed_embeds = self.squeeze_embedding(combined_embeds, normalized_mask)
        
        # Point-wise Feed Forward
        ffn_output = self.point_wise_ffn(squeezed_embeds)
        
        # BiLSTM với mask đã chuẩn hóa
        lstm_output, _ = self.bilstm(ffn_output, normalized_mask)
        
        # Dropout
        lstm_output = self.dropout(lstm_output)
        
        # Classifier
        emissions = self.classifier(lstm_output)
        
        # Chuyển đổi tensor để phù hợp với CRF (seq_len, batch_size, num_tags)
        emissions = emissions.transpose(0, 1)
        
        if labels is not None:
            # Thêm START và END tags cho labels
            labels = labels.clone()
            labels[labels == self.pad_tag_id] = self.end_tag_id
            labels = torch.cat([
                torch.full((labels.size(0), 1), self.start_tag_id, device=labels.device),
                labels
            ], dim=1)
            
            # Chuyển đổi labels và mask
            labels = labels.transpose(0, 1)
            mask = normalized_mask.transpose(0, 1)
            
            # Training mode: tính loss với mask đã chuẩn hóa
            loss = self.crf.neg_log_likelihood(emissions, labels, mask=mask)
            logger.debug(f"Training loss: {loss.item():.4f}")
            return loss
        else:
            # Inference mode: decode labels với mask đã chuẩn hóa
            mask = normalized_mask.transpose(0, 1)
            predictions = self.crf.decode(emissions, mask=mask)
            
            # Chuyển đổi lại predictions về dạng batch_first
            predictions = [pred for pred in zip(*predictions)]
            
            # Loại bỏ START và END tags
            predictions = [pred[1:-1] for pred in predictions]
            
            # Cắt predictions theo độ dài thực tế của mỗi câu
            seq_lengths = normalized_mask.sum(dim=1).tolist()
            predictions = [pred[:length] for pred, length in zip(predictions, seq_lengths)]
            
            logger.debug(f"Decoded {len(predictions)} sequences")
            return predictions
    
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
            # Chuyển dữ liệu lên device
            input_embeds = input_embeds.to(self.device)
            attention_mask = attention_mask.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            loss = self(input_embeds, attention_mask, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
            
            # Update weights
            optimizer.step()
            if scheduler is not None:
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
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                # Tính metrics cho batch hiện tại
                batch_precision, batch_recall, batch_f1, _ = precision_recall_fscore_support(
                    all_labels[-len(valid_label):],
                    all_predictions[-len(valid_pred):],
                    average='weighted'
                )
                logger.info(
                    f"Batch {batch_idx + 1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.4f}, "
                    f"F1: {batch_f1:.4f}, "
                    f"Precision: {batch_precision:.4f}, "
                    f"Recall: {batch_recall:.4f}"
                )
        
        # Tính metrics cho toàn bộ epoch
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
    
    def save_pretrained(self, path):
        """Lưu mô hình.
        
        Args:
            path (str): Đường dẫn để lưu mô hình
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'bert_hidden_size': self.bert_hidden_size,
            'kg_hidden_size': self.kg_hidden_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'num_labels': self.num_labels,
            'start_tag_id': self.start_tag_id,
            'end_tag_id': self.end_tag_id,
            'pad_tag_id': self.pad_tag_id,
            'max_grad_norm': self.max_grad_norm,
            'warmup_steps': self.warmup_steps,
            'early_stopping_patience': self.early_stopping_patience
        }, path)
        logger.info(f"Đã lưu mô hình tại {path}")
    
    @classmethod
    def from_pretrained(cls, path, device=None):
        """Load mô hình đã lưu.
        
        Args:
            path (str): Đường dẫn đến file mô hình đã lưu
            device (torch.device, optional): Thiết bị tính toán
            
        Returns:
            PhoBERT_CRF_KGAN: Mô hình đã được load
        """
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            bert_hidden_size=checkpoint['bert_hidden_size'],
            kg_hidden_size=checkpoint['kg_hidden_size'],
            lstm_hidden_size=checkpoint['lstm_hidden_size'],
            num_labels=checkpoint['num_labels'],
            start_tag_id=checkpoint['start_tag_id'],
            end_tag_id=checkpoint['end_tag_id'],
            pad_tag_id=checkpoint['pad_tag_id'],
            max_grad_norm=checkpoint.get('max_grad_norm', 1.0),
            warmup_steps=checkpoint.get('warmup_steps', 0),
            early_stopping_patience=checkpoint.get('early_stopping_patience', 3),
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Đã load mô hình từ {path}")
        
        return model

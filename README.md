# PhoBERT-CRF-KGAN cho Aspect-Based Sentiment Analysis (ABSA)

Dự án này triển khai mô hình PhoBERT-CRF-KGAN cho bài toán phân tích cảm xúc dựa trên khía cạnh (ABSA) trong tiếng Việt. Mô hình kết hợp PhoBERT, Knowledge Graph Attention Network và Conditional Random Field để đạt hiệu quả cao trong việc nhận diện khía cạnh và phân tích cảm xúc.

## Cấu trúc dự án

```
.
├── src/
│   ├── model/
│   │   ├── __init__.py
│   │   ├── phoBERT_CRF_KGAN.py      # Mô hình chính
│   │   ├── attention.py             # Module attention cho KG
│   │   ├── point_wise_feed_forward.py # FFN với LayerNorm và Residual
│   │   ├── squeeze_embedding.py     # Xử lý padding
│   │   └── dynamic_rnn.py          # BiLSTM với LayerNorm và Residual
│   ├── data/
│   │   ├── __init__.py
│   │   ├── origin/                  # Dữ liệu gốc
│   │   ├── processed/               # Dữ liệu đã xử lý
│   │   ├── predictions/             # Kết quả dự đoán
│   │   ├── kg/                      # Knowledge Graph data
│   │   ├── dataset.py              # Dataset và DataLoader
│   │   └── preprocess.py           # Tiền xử lý dữ liệu
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── metrics.py              # Các hàm đánh giá
│   │   └── training.py             # Các hàm training
│   ├── kg/                         # Knowledge Graph modules
│   ├── embeddings/                 # Word embeddings
│   ├── train.py                    # Script training
│   ├── predict.py                  # Script inference
│   └── main.py                     # Script chính
├── configs/
│   └── config.yaml                 # Cấu hình mô hình
├── requirements.txt                # Các thư viện cần thiết
└── README.md                       # Tài liệu hướng dẫn
```

## Kiến trúc mô hình

Mô hình PhoBERT-CRF-KGAN bao gồm các thành phần chính sau:

1. **PhoBERT Encoder**
   - Sử dụng mô hình PhoBERT-base
   - Fine-tune 4 layer cuối
   - Output size: 768

2. **Knowledge Graph Attention Network**
   - Projection layer: 768 -> 200
   - KG Attention với layer normalization
   - Kết hợp thông tin từ knowledge graph

3. **SqueezeEmbedding**
   - Xử lý padding hiệu quả
   - Hỗ trợ mask và max length
   - Tối ưu hóa bộ nhớ

4. **Point-wise Feed Forward Network**
   - Layer normalization
   - Residual connection
   - Dropout để tránh overfitting

5. **BiLSTM (2 lớp)**
   - Layer normalization
   - Residual connection
   - Bidirectional để nắm bắt context hai chiều

6. **CRF Layer**
   - Modeling dependencies giữa các labels
   - Viterbi decoding cho inference
   - Hỗ trợ START/END tags

## Cài đặt

1. Tạo môi trường ảo (khuyến nghị):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
source venv/Scripts/activate     # Windows
```

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## Cấu hình

Các tham số chính có thể được cấu hình trong `configs/config.yaml`:

```yaml
model:
  bert_model_name: "vinai/phobert-base"
  num_labels: 7
  bert_hidden_size: 768
  kg_hidden_size: 200
  lstm_hidden_size: 256
  dropout: 0.1
  max_grad_norm: 1.0
  warmup_steps: 0
  early_stopping_patience: 3

training:
  batch_size: 32
  learning_rate: 1e-5
  weight_decay: 0.01
  num_epochs: 10
  device: "cuda"  # hoặc "cpu"
```

## Sử dụng

### Training

1. Chuẩn bị dữ liệu:
```bash
python src/utils/preprocess.py --input_file src/data/origin/train.json --output_file src/data/processed/train.jsonl
```

2. Training mô hình:
```bash
python src/train.py --config configs/config.yaml --train_file src/data/processed/train.jsonl --val_file src/data/processed/val.jsonl --output_dir models/
```

Các tham số training:
- `--config`: Đường dẫn đến file cấu hình
- `--train_file`: File dữ liệu training (trong thư mục src/data/processed/)
- `--val_file`: File dữ liệu validation (trong thư mục src/data/processed/)
- `--output_dir`: Thư mục lưu model
- `--seed`: Random seed (mặc định: 42)
- `--num_workers`: Số worker cho DataLoader (mặc định: 4)
- `--logging_steps`: Số bước giữa các lần log (mặc định: 100)

### Inference

```bash
python src/predict.py --model_dir models/best_model/ --input_file src/data/processed/test.jsonl --output_file src/data/predictions/predictions.jsonl
```

Các tham số inference:
- `--model_dir`: Thư mục chứa model đã train
- `--input_file`: File dữ liệu cần dự đoán
- `--output_file`: File kết quả dự đoán
- `--batch_size`: Batch size cho inference (mặc định: 32)

## Đánh giá

Mô hình được đánh giá trên các metrics:
- Precision, Recall, F1-score cho từng class
- Macro và weighted average
- Confusion matrix

Chạy đánh giá:
```bash
python src/evaluate.py --pred_file predictions.jsonl --gold_file data/test.jsonl
```

## Lưu ý

1. **Yêu cầu phần cứng**:
   - GPU với ít nhất 8GB VRAM
   - RAM: 16GB trở lên
   - Disk: 10GB trống

2. **Tối ưu hóa**:
   - Sử dụng gradient clipping để tránh exploding gradients
   - Layer normalization và residual connections để training ổn định
   - Early stopping và model checkpointing
   - Learning rate scheduling với warmup

3. **Xử lý dữ liệu**:
   - Tokenization với PhoBERT tokenizer
   - Padding và masking tự động
   - Data augmentation (tùy chọn)

## Citation

Nếu bạn sử dụng code này trong nghiên cứu, vui lòng trích dẫn:

```bibtex
@misc{phobert-crf-kgan-absa,
  author = {Your Name},
  title = {PhoBERT-CRF-KGAN for Vietnamese Aspect-Based Sentiment Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/phobert-crf-kgan-absa}
}
```
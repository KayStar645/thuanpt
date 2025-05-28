# Vietnamese ABSA Model

Mô hình Aspect-Based Sentiment Analysis (ABSA) cho tiếng Việt sử dụng PhoBERT, CRF và Knowledge Graph Attention Network.

## Cài đặt

1. Clone repository:
```bash
git clone <repository_url>
cd <repository_name>
```

2. Tạo môi trường ảo và cài đặt dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
.\venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Cấu trúc thư mục

```
.
├── configs/
│   └── config.yaml          # Cấu hình model và training
├── src/
│   ├── data/
│   │   ├── origin/         # Dữ liệu gốc
│   │   ├── processed/      # Dữ liệu đã xử lý
│   │   ├── predictions/    # Kết quả dự đoán
│   │   ├── dataset.py      # Dataset và DataLoader
│   │   └── preprocess.py   # Tiền xử lý dữ liệu
│   ├── model/
│   │   └── phoBERT_CRF_KGAN.py  # Model ABSA
│   ├── utils/
│   │   ├── metrics.py      # Metrics đánh giá
│   │   └── training.py     # Utilities cho training
│   └── train.py            # Script training
├── models/                  # Thư mục lưu model
├── logs/                    # Log files
├── requirements.txt         # Dependencies
└── README.md               # Tài liệu hướng dẫn
```

## Chuẩn bị dữ liệu

1. Đặt dữ liệu gốc vào thư mục `src/data/origin/` với định dạng JSONL:
```json
{
    "id": 1,
    "data": "Văn bản gốc",
    "label": [[start, end, "CATEGORY#ASPECT#SENTIMENT"], ...],
    "labels": "Tóm tắt nhãn"
}
```

2. Tiền xử lý dữ liệu:
```bash
python src/utils/preprocess.py \
    --input_file src/data/origin/train.jsonl \
    --output_file src/data/processed/train_new.jsonl
```

Các tham số tiền xử lý:
- `--input_file`: Đường dẫn file dữ liệu gốc (JSONL)
- `--output_file`: Đường dẫn file dữ liệu đã xử lý
- `--max_samples`: Số lượng mẫu tối đa cần xử lý (tùy chọn)

3. Tạo tập validation:
```bash
# Tách dữ liệu training thành train và validation
head -n 500 src/data/processed/train_new.jsonl > src/data/processed/val.jsonl
tail -n +501 src/data/processed/train_new.jsonl > src/data/processed/train.jsonl
```

## Training model

1. Kiểm tra và điều chỉnh cấu hình trong `configs/config.yaml`:
   - Model: kích thước hidden layers, dropout, ...
   - Training: batch size, learning rate, số epochs, ...
   - Data: đường dẫn files, thư mục output, ...

2. Training model:
```bash
python src/train.py \
    --config configs/config.yaml \
    --train_file src/data/processed/train.jsonl \
    --val_file src/data/processed/val.jsonl \
    --output_dir models \
    --seed 42
```

Các tham số training:
- `--config`: Đường dẫn file cấu hình
- `--train_file`: File dữ liệu training
- `--val_file`: File dữ liệu validation
- `--output_dir`: Thư mục lưu model
- `--seed`: Random seed (mặc định: 42)

## Kết quả

- Model được lưu trong thư mục `models/`
- Log training được lưu trong `logs/training.log`
- Kết quả dự đoán được lưu trong `src/data/predictions/`

## Metrics

Model được đánh giá dựa trên các metrics:
- Precision, Recall, F1-score cho từng nhãn
- F1-score trung bình cho tất cả nhãn
- Accuracy cho sentiment classification

## Yêu cầu hệ thống

- Python 3.8+
- CUDA-compatible GPU (khuyến nghị)
- RAM: 16GB+ (khuyến nghị)
- Disk space: 10GB+ cho model và dữ liệu

## License

[Thêm thông tin license]
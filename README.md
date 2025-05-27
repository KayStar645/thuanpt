# ABSA - Phân tích cảm xúc dựa trên khía cạnh

Dự án phân tích cảm xúc dựa trên khía cạnh (Aspect-Based Sentiment Analysis) cho tiếng Việt, sử dụng mô hình PhoBERT-CRF-KGAN.

## Cấu trúc thư mục

```
.
├── README.md
├── requirements.txt
└── src/
    ├── __init__.py
    ├── main.py
    ├── train.py
    ├── predict.py
    ├── data/
    │   ├── origin/      # Dữ liệu gốc
    │   ├── processed/   # Dữ liệu đã xử lý
    │   └── predictions/ # Kết quả dự đoán
    ├── model/
    │   ├── __init__.py
    │   ├── phoBERT_CRF_KGAN.py
    │   ├── crf.py
    │   ├── attention.py
    │   ├── dynamic_rnn.py
    │   ├── point_wise_feed_forward.py
    │   └── squeeze_embedding.py
    ├── utils/
    │   ├── __init__.py
    │   ├── vietnamese_processor.py
    │   ├── seq_utils.py
    │   └── convert_to_absa_jsonl.py
    ├── embeddings/
    │   └── generate_embeddings.py
    └── kg/
        └── generate_kg.py
```

## Cài đặt

1. Clone repository:
```bash
git clone <repository-url>
cd absa
```

2. Tạo môi trường ảo và cài đặt các thư viện:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
source venv\Scripts\activate     # Windows
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Sử dụng

### Huấn luyện mô hình

```bash
python src/main.py --mode train --epochs 2 --batch_size 4 --learning_rate 1e-5
```

### Dự đoán

```bash
python src/main.py --mode predict --input_file data/processed/test.jsonl --output_file predictions.jsonl
```

## Cấu trúc dữ liệu

### Định dạng input (JSONL)

Mỗi dòng là một JSON object với cấu trúc:
```json
{
    "text": "Văn bản cần phân tích",
    "labels": [
        [start, end, sentiment],
        ...
    ]
}
```

Trong đó:
- `start`: Vị trí bắt đầu của aspect
- `end`: Vị trí kết thúc của aspect
- `sentiment`: Cảm xúc (POSITIVE/NEGATIVE/NEUTRAL)

### Định dạng output (JSONL)

Mỗi dòng là một JSON object với cấu trúc:
```json
{
    "text": "Văn bản đã phân tích",
    "aspects": [
        {
            "text": "Nội dung aspect",
            "start": vị_trí_bắt_đầu,
            "end": vị_trí_kết_thúc,
            "sentiment": "POSITIVE/NEGATIVE/NEUTRAL"
        },
        ...
    ]
}
```

## Mô hình

Mô hình sử dụng kiến trúc PhoBERT-CRF-KGAN, kết hợp:
- PhoBERT: Mô hình ngôn ngữ tiếng Việt
- CRF: Conditional Random Field
- KGAN: Knowledge Graph Attention Network
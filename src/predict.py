"""
Module dự đoán cho mô hình ABSA.
"""

import os
import json
import logging
import torch
from transformers import AutoTokenizer
from src.model import PhoBERT_CRF_KGAN
from src.utils.preprocess import preprocess_text

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(input_file, output_file="predictions.jsonl"):
    """Thực hiện dự đoán trên dữ liệu mới.
    
    Args:
        input_file (str): Đường dẫn đến file input
        output_file (str): Đường dẫn để lưu kết quả
    """
    # Thiết lập device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Sử dụng device: {device}")
    
    # Khởi tạo tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Load model
    model_path = "src/data/model/phobert_crf_kgan.pt"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model tại {model_path}")
    
    model = PhoBERT_CRF_KGAN.from_pretrained(model_path, device=device)
    model.eval()
    
    # Đọc dữ liệu input
    with open(input_file, "r", encoding="utf-8") as f:
        input_data = [json.loads(line) for line in f]
    
    # Thực hiện dự đoán
    predictions = []
    with torch.no_grad():
        for item in input_data:
            # Tiền xử lý văn bản
            text = preprocess_text(item["text"])
            
            # Tokenize
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)
            
            # Lấy embeddings từ BERT
            bert_outputs = model.bert(**inputs)
            bert_embeds = bert_outputs.last_hidden_state
            
            # Tạo KG embeddings (giả lập)
            kg_embeds = torch.zeros(
                (1, bert_embeds.size(1), 200),
                device=device
            )
            
            # Kết hợp embeddings
            input_embeds = torch.cat([bert_embeds, kg_embeds], dim=-1)
            
            # Dự đoán
            pred_labels = model(input_embeds, inputs["attention_mask"])
            
            # Chuyển đổi nhãn thành aspect terms
            aspects = []
            current_aspect = None
            current_sentiment = None
            
            for i, (token, label) in enumerate(zip(tokenizer.tokenize(text), pred_labels[0])):
                if label.startswith("B-"):
                    if current_aspect is not None:
                        aspects.append({
                            "term": current_aspect,
                            "sentiment": current_sentiment
                        })
                    current_aspect = token
                    current_sentiment = label[2:].lower()
                elif label.startswith("I-") and current_aspect is not None:
                    current_aspect += " " + token
            
            if current_aspect is not None:
                aspects.append({
                    "term": current_aspect,
                    "sentiment": current_sentiment
                })
            
            # Lưu kết quả
            predictions.append({
                "text": text,
                "aspects": aspects
            })
    
    # Lưu kết quả
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    
    logger.info(f"Đã lưu kết quả dự đoán tại {output_file}")

if __name__ == "__main__":
    main() 
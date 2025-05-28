"""
Module tiền xử lý dữ liệu cho mô hình ABSA.
Chuyển đổi dữ liệu từ JSON sang JSONL và chuẩn bị cho training.
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from underthesea import word_tokenize
from transformers import AutoTokenizer
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Khởi tạo tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def preprocess_text(text: str) -> str:
    """Tiền xử lý văn bản tiếng Việt.
    
    Args:
        text (str): Văn bản đầu vào
        
    Returns:
        str: Văn bản đã được tiền xử lý
    """
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Tokenize tiếng Việt
    text = word_tokenize(text, format="text")
    
    return text

def create_crf_labels(
    tokens: List[str],
    aspects: List[Dict[str, Any]],
    max_length: int = 512
) -> List[int]:
    """Tạo labels cho CRF từ tokens và các khía cạnh.
    
    Args:
        tokens (List[str]): Danh sách các token
        aspects (List[Dict]): Danh sách các khía cạnh, mỗi khía cạnh có:
            - term: từ khóa khía cạnh
            - start: vị trí bắt đầu (token index)
            - end: vị trí kết thúc (token index)
            - labels: List[Dict] chứa thông tin về các nhãn:
                - category: loại khía cạnh (FOOD, SERVICE, AMBIENCE, etc.)
                - aspect: khía cạnh cụ thể (QUALITY, PRICE, etc.)
                - sentiment: cảm xúc (POSITIVE/NEGATIVE/NEUTRAL)
        max_length (int): Độ dài tối đa của sequence
        
    Returns:
        List[int]: Danh sách labels cho CRF
            - 0: START tag
            - 1: END tag
            - 2: O (không phải khía cạnh)
            - 3: B-POS (bắt đầu khía cạnh positive)
            - 4: I-POS (tiếp tục khía cạnh positive)
            - 5: B-NEG (bắt đầu khía cạnh negative)
            - 6: I-NEG (tiếp tục khía cạnh negative)
            - 7: B-NEU (bắt đầu khía cạnh neutral)
            - 8: I-NEU (tiếp tục khía cạnh neutral)
    """
    # Giới hạn độ dài sequence
    if len(tokens) > max_length - 2:  # -2 cho [CLS] và [SEP]
        tokens = tokens[:max_length-2]
    
    # Khởi tạo labels với O
    labels = [2] * len(tokens)
    
    # Đánh dấu các khía cạnh
    for aspect in aspects:
        start = aspect['start']
        end = aspect['end']
        
        # Kiểm tra vị trí hợp lệ
        if start >= len(tokens) or end > len(tokens):
            continue
        
        # Lấy sentiment phổ biến nhất từ các nhãn của khía cạnh
        sentiments = [label['sentiment'] for label in aspect['labels']]
        sentiment = max(set(sentiments), key=sentiments.count).upper()
        
        # Xác định label dựa trên sentiment
        if sentiment == 'POSITIVE':
            b_label, i_label = 3, 4  # B-POS, I-POS
        elif sentiment == 'NEGATIVE':
            b_label, i_label = 5, 6  # B-NEG, I-NEG
        else:  # NEUTRAL
            b_label, i_label = 7, 8  # B-NEU, I-NEU
        
        # Đánh dấu vị trí bắt đầu và kết thúc
        labels[start] = b_label
        for i in range(start + 1, end):
            labels[i] = i_label
    
    # Thêm START và END tags
    labels = [0] + labels + [1]
    
    return labels

def process_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Xử lý một mẫu dữ liệu.
    
    Args:
        sample (Dict): Mẫu dữ liệu từ file JSONL với format:
            {
                "id": int,  # ID của mẫu
                "data": str,  # văn bản gốc
                "label": List[List[int, int, str]],  # [[start, end, "CATEGORY#ASPECT#SENTIMENT"], ...]
                "labels": str  # "{CATEGORY#ASPECT, sentiment}, ..."
            }
        
    Returns:
        Dict: Mẫu dữ liệu đã xử lý với format:
            {
                "id": int,  # ID của mẫu
                "text": str,  # văn bản đã tiền xử lý
                "input_ids": List[int],  # token ids
                "attention_mask": List[int],  # attention mask
                "labels": List[int],  # nhãn CRF
                "aspects": List[Dict],  # thông tin khía cạnh
                "all_labels": List[Dict]  # tất cả các nhãn của mẫu
            }
    """
    # Tokenize văn bản trước
    text = sample['data']
    tokens = tokenizer.tokenize(text)
    
    # Tạo mapping từ vị trí ký tự sang vị trí token
    char_to_token = {}
    token_idx = 0
    char_idx = 0
    for token in tokens:
        # Bỏ qua các token đặc biệt như ##
        if token.startswith('##'):
            token = token[2:]
        # Cập nhật mapping cho mỗi ký tự trong token
        for _ in token:
            char_to_token[char_idx] = token_idx
            char_idx += 1
        token_idx += 1
    
    # Chuyển đổi labels thành định dạng aspects
    aspects = []
    all_labels = []  # lưu tất cả các nhãn của mẫu
    
    # Nhóm các nhãn theo vị trí (start, end)
    label_groups = {}
    for start, end, label in sample['label']:
        # Chuyển đổi vị trí ký tự sang vị trí token
        if start in char_to_token and end-1 in char_to_token:
            token_start = char_to_token[start]
            token_end = char_to_token[end-1] + 1  # +1 vì end là exclusive
            
            key = (token_start, token_end)
            if key not in label_groups:
                label_groups[key] = []
            label_groups[key].append(label)
    
    # Xử lý từng nhóm nhãn
    for (token_start, token_end), labels in label_groups.items():
        # Lấy text của khía cạnh từ tokens
        aspect_text = ''.join(tokens[token_start:token_end]).replace('##', '')
        
        # Lưu thông tin khía cạnh
        aspect = {
            'term': aspect_text,
            'start': token_start,
            'end': token_end,
            'labels': []  # danh sách các nhãn của khía cạnh này
        }
        
        # Phân tích từng nhãn
        for label in labels:
            category, aspect_type, sentiment = label.split('#')
            label_info = {
                'category': category,
                'aspect': aspect_type,
                'sentiment': sentiment
            }
            aspect['labels'].append(label_info)
            all_labels.append(label_info)
        
        aspects.append(aspect)
    
    # Tạo labels cho CRF
    labels = create_crf_labels(tokens, aspects)
    
    # Tạo input_ids và attention_mask
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    return {
        'id': sample['id'],
        'text': text,
        'input_ids': encoding['input_ids'].squeeze().tolist(),
        'attention_mask': encoding['attention_mask'].squeeze().tolist(),
        'labels': labels,
        'aspects': aspects,  # lưu lại thông tin khía cạnh với đầy đủ các nhãn
        'all_labels': all_labels  # lưu tất cả các nhãn của mẫu
    }

def convert_json_to_jsonl(
    input_file: str,
    output_file: str,
    max_samples: Optional[int] = None
) -> None:
    """Chuyển đổi dữ liệu từ JSON/JSONL sang JSONL.
    
    Args:
        input_file (str): Đường dẫn file JSON/JSONL đầu vào
        output_file (str): Đường dẫn file JSONL đầu ra
        max_samples (Optional[int]): Số lượng mẫu tối đa cần xử lý
    """
    # Đọc dữ liệu từ file JSON hoặc JSONL
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        if input_file.endswith('.json'):
            data = json.load(f)
        else:  # .jsonl
            for line in f:
                if line.strip():  # Bỏ qua dòng trống
                    data.append(json.loads(line))
    
    # Giới hạn số lượng mẫu nếu cần
    if max_samples is not None:
        data = data[:max_samples]
    
    logger.info(f"Đọc được {len(data)} mẫu từ file {input_file}")
    
    # Tạo thư mục output nếu chưa tồn tại
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Xử lý và ghi từng mẫu
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sample in enumerate(data):
            try:
                processed = process_sample(sample)
                f.write(json.dumps(processed, ensure_ascii=False) + '\n')
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Đã xử lý {i + 1} mẫu")
                    
            except Exception as e:
                logger.error(f"Lỗi khi xử lý mẫu {i}: {str(e)}")
                continue
    
    logger.info(f"Đã xử lý xong {len(data)} mẫu")
    logger.info(f"Kết quả được lưu tại: {output_file}")

def main():
    """Hàm chính để chạy tiền xử lý."""
    parser = argparse.ArgumentParser(description="Tiền xử lý dữ liệu ABSA")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Đường dẫn file JSON/JSONL đầu vào"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Đường dẫn file JSONL đầu ra"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Số lượng mẫu tối đa cần xử lý"
    )
    
    args = parser.parse_args()
    
    try:
        convert_json_to_jsonl(
            args.input_file,
            args.output_file,
            args.max_samples
        )
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
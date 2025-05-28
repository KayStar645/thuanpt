"""
Script để xử lý lại dữ liệu với độ dài labels đã được chuẩn hóa.
"""

import os
import sys
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer
from .preprocess import process_sample

# Thêm thư mục gốc vào PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def reprocess_jsonl(input_file: str, output_file: str):
    """Xử lý lại dữ liệu từ file JSONL với độ dài labels đã được chuẩn hóa.
    
    Args:
        input_file (str): Đường dẫn file JSONL đầu vào
        output_file (str): Đường dẫn file JSONL đầu ra
    """
    # Khởi tạo tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    
    # Tạo thư mục output nếu chưa tồn tại
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Đọc và xử lý từng mẫu
    processed_count = 0
    error_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            if not line.strip():
                continue
                
            try:
                # Đọc mẫu dữ liệu
                sample = json.loads(line)
                
                # Kiểm tra các trường bắt buộc
                required_fields = ['id', 'text', 'input_ids', 'attention_mask', 'labels', 'aspects', 'all_labels']
                missing_fields = [field for field in required_fields if field not in sample]
                if missing_fields:
                    logger.warning(f"Mẫu {sample.get('id', f'line {line_num}')} thiếu các trường: {', '.join(missing_fields)}")
                    skipped_count += 1
                    continue
                
                # Kiểm tra dữ liệu trống
                if not sample['text'].strip():
                    logger.warning(f"Mẫu {sample['id']} có dữ liệu trống")
                    skipped_count += 1
                    continue
                
                # Kiểm tra độ dài các trường
                if len(sample['input_ids']) != 512:
                    logger.warning(f"Mẫu {sample['id']} có độ dài input_ids không đúng: {len(sample['input_ids'])}")
                    skipped_count += 1
                    continue
                    
                if len(sample['attention_mask']) != 512:
                    logger.warning(f"Mẫu {sample['id']} có độ dài attention_mask không đúng: {len(sample['attention_mask'])}")
                    skipped_count += 1
                    continue
                
                # Kiểm tra và chuẩn hóa độ dài labels
                if len(sample['labels']) < 512:
                    # Padding với O tag (2)
                    sample['labels'].extend([2] * (512 - len(sample['labels'])))
                elif len(sample['labels']) > 512:
                    # Truncate nếu dài hơn 512
                    sample['labels'] = sample['labels'][:512]
                
                # Kiểm tra tính hợp lệ của aspects
                for aspect in sample['aspects']:
                    if not all(k in aspect for k in ['term', 'start', 'end', 'labels']):
                        logger.warning(f"Mẫu {sample['id']} có aspect không hợp lệ: {aspect}")
                        skipped_count += 1
                        continue
                
                # Ghi mẫu đã xử lý
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Đã xử lý {processed_count} mẫu")
                    
            except json.JSONDecodeError as e:
                logger.error(f"Lỗi JSON ở dòng {line_num}: {str(e)}")
                error_count += 1
                continue
            except Exception as e:
                logger.error(f"Lỗi khi xử lý mẫu {sample.get('id', f'line {line_num}')}: {str(e)}")
                error_count += 1
                continue
    
    logger.info(f"Đã xử lý xong {processed_count} mẫu")
    if skipped_count > 0:
        logger.warning(f"Có {skipped_count} mẫu bị bỏ qua do thiếu trường hoặc dữ liệu không hợp lệ")
    if error_count > 0:
        logger.warning(f"Có {error_count} mẫu bị lỗi trong quá trình xử lý")
    logger.info(f"Kết quả được lưu tại: {output_file}")

def main():
    """Hàm chính để chạy xử lý lại dữ liệu."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Xử lý lại dữ liệu ABSA với độ dài labels đã chuẩn hóa")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Đường dẫn file JSONL đầu vào"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Đường dẫn file JSONL đầu ra"
    )
    
    args = parser.parse_args()
    
    try:
        reprocess_jsonl(args.input_file, args.output_file)
    except Exception as e:
        logger.error(f"Lỗi khi xử lý dữ liệu: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
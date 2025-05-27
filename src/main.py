"""
Module chính cho ứng dụng ABSA.
Cho phép người dùng chọn giữa chế độ training và prediction.
"""

import os
import sys
import argparse
import logging

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train import main as train_main
from src.predict import main as predict_main

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ABSA Training and Prediction')
    parser.add_argument('--mode', required=True, choices=['train', 'predict'],
                      help='Chọn chế độ: train hoặc predict')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                      help='Số epoch training (mặc định: 10)')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size (mặc định: 32)')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                      help='Learning rate (mặc định: 1e-5)')
    
    # Prediction arguments
    parser.add_argument('--input_file', type=str,
                      help='Đường dẫn đến file input cho prediction')
    parser.add_argument('--output_file', type=str, default='predictions.jsonl',
                      help='Đường dẫn để lưu kết quả prediction (mặc định: predictions.jsonl)')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Tạo thư mục data nếu chưa tồn tại
    os.makedirs('src/data', exist_ok=True)
    
    if args.mode == 'train':
        train_main(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    else:  # predict mode
        if not args.input_file:
            raise ValueError("--input_file là bắt buộc trong chế độ predict")
        predict_main(
            input_file=args.input_file,
            output_file=args.output_file
        )

if __name__ == '__main__':
    main()

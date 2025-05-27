"""
Điểm vào chính cho ứng dụng ABSA.
Cho phép người dùng chọn chế độ huấn luyện hoặc dự đoán.
"""

import os
import argparse
from src.train import main as train_main
from src.predict import main as predict_main

def parse_args():
    """Phân tích tham số dòng lệnh.
    
    Returns:
        argparse.Namespace: Các tham số đã được phân tích
    """
    parser = argparse.ArgumentParser(description="ABSA - Phân tích cảm xúc dựa trên khía cạnh")
    
    # Tham số chung
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "predict"],
        required=True,
        help="Chế độ chạy: train (huấn luyện) hoặc predict (dự đoán)"
    )
    
    # Tham số cho chế độ huấn luyện
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Số epoch huấn luyện (mặc định: 10)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Kích thước batch (mặc định: 16)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (mặc định: 1e-5)"
    )
    
    # Tham số cho chế độ dự đoán
    parser.add_argument(
        "--input_file",
        type=str,
        help="Đường dẫn đến file input cho chế độ dự đoán"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.jsonl",
        help="Tên file output cho kết quả dự đoán (mặc định: predictions.jsonl)"
    )
    
    return parser.parse_args()

def main():
    """Hàm chính của ứng dụng."""
    # Phân tích tham số
    args = parse_args()
    
    # Kiểm tra thư mục dữ liệu
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Đã tạo thư mục dữ liệu: {data_dir}")
    
    # Chạy chế độ tương ứng
    if args.mode == "train":
        print("Bắt đầu huấn luyện mô hình...")
        train_main()
    else:  # predict
        if not args.input_file:
            print("Lỗi: Cần cung cấp file input cho chế độ dự đoán")
            return
        print("Bắt đầu dự đoán...")
        predict_main()

if __name__ == "__main__":
    main()

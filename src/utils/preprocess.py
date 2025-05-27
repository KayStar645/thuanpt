"""
Module tiền xử lý văn bản cho mô hình ABSA.
"""

import re
from underthesea import word_tokenize

def preprocess_text(text):
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
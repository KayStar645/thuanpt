import torch
import torch.nn as nn
from typing import List, Tuple

class CRF(nn.Module):
    """Lớp Conditional Random Field (CRF) cho bài toán gán nhãn chuỗi.
    
    Triển khai này hỗ trợ xử lý theo batch và xử lý các token padding.
    
    Tham số:
        tagset_size (int): Số lượng nhãn trong tập nhãn
        start_tag_id (int): ID của nhãn bắt đầu
        end_tag_id (int): ID của nhãn kết thúc
        pad_tag_id (int, optional): ID của nhãn padding. Mặc định là None.
    """
    def __init__(self, tagset_size: int, start_tag_id: int, end_tag_id: int, pad_tag_id: int = None):
        super(CRF, self).__init__()
        # Kiểm tra tính hợp lệ của các tham số đầu vào
        if not isinstance(tagset_size, int) or tagset_size <= 0:
            raise ValueError("tagset_size phải là số nguyên dương")
        if not isinstance(start_tag_id, int) or not isinstance(end_tag_id, int):
            raise ValueError("start_tag_id và end_tag_id phải là số nguyên")
        if start_tag_id >= tagset_size or end_tag_id >= tagset_size:
            raise ValueError("start_tag_id và end_tag_id phải nhỏ hơn tagset_size")
        if pad_tag_id is not None and (not isinstance(pad_tag_id, int) or pad_tag_id >= tagset_size):
            raise ValueError("pad_tag_id phải là số nguyên nhỏ hơn tagset_size")

        # Lưu trữ các tham số
        self.tagset_size = tagset_size
        self.start_tag_id = start_tag_id
        self.end_tag_id = end_tag_id
        self.pad_tag_id = pad_tag_id

        # Khởi tạo ma trận chuyển tiếp (transition matrix)
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        
        # Áp dụng các ràng buộc: không thể chuyển tiếp đến nhãn bắt đầu và từ nhãn kết thúc
        self.transitions.data[start_tag_id, :] = -10000  # Không thể chuyển đến nhãn bắt đầu
        self.transitions.data[:, end_tag_id] = -10000    # Không thể chuyển từ nhãn kết thúc
        
        # Nếu có nhãn padding, không cho phép chuyển tiếp đến/từ nhãn padding
        if pad_tag_id is not None:
            self.transitions.data[pad_tag_id, :] = -10000  # Không thể chuyển đến nhãn padding
            self.transitions.data[:, pad_tag_id] = -10000  # Không thể chuyển từ nhãn padding

    def forward(self, feats: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Tính toán loss (negative log likelihood).
        
        Tham số:
            feats (torch.Tensor): Điểm phát xạ (emission scores) có kích thước (batch_size, seq_len, tagset_size)
            tags (torch.Tensor): Nhãn đích có kích thước (batch_size, seq_len)
            mask (torch.Tensor): Tensor mask có kích thước (batch_size, seq_len), 1 biểu thị token hợp lệ
            
        Trả về:
            torch.Tensor: Giá trị loss (negative log likelihood)
        """
        # Kiểm tra kích thước batch và độ dài chuỗi
        if feats.size(0) != tags.size(0) or feats.size(0) != mask.size(0):
            raise ValueError("Kích thước batch phải khớp nhau")
        if feats.size(1) != tags.size(1) or feats.size(1) != mask.size(1):
            raise ValueError("Độ dài chuỗi phải khớp nhau")
            
        # Chuyển đổi mask thành bool
        mask = mask.bool()
        
        # Thay thế các giá trị padding (-100) bằng nhãn padding
        if self.pad_tag_id is not None:
            tags = torch.where(tags == -100, torch.tensor(self.pad_tag_id, device=tags.device), tags)
        
        log_likelihood = self._compute_log_likelihood(feats, tags, mask)
        return -log_likelihood

    def decode(self, feats: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Tìm chuỗi nhãn có xác suất cao nhất sử dụng thuật toán Viterbi.
        
        Tham số:
            feats (torch.Tensor): Điểm phát xạ có kích thước (batch_size, seq_len, tagset_size)
            mask (torch.Tensor): Tensor mask có kích thước (batch_size, seq_len)
            
        Trả về:
            List[List[int]]: Danh sách các chuỗi nhãn dự đoán
        """
        if feats.size(0) != mask.size(0) or feats.size(1) != mask.size(1):
            raise ValueError("Kích thước batch và độ dài chuỗi phải khớp nhau")
            
        # Chuyển đổi mask thành bool
        mask = mask.bool()
        return self._viterbi_decode(feats, mask)

    def _compute_log_likelihood(self, feats: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Tính toán log likelihood của chuỗi nhãn cho trước.
        
        Tham số:
            feats (torch.Tensor): Điểm phát xạ
            tags (torch.Tensor): Nhãn đích
            mask (torch.Tensor): Tensor mask
            
        Trả về:
            torch.Tensor: Điểm log likelihood
        """
        batch_size, seq_len, tagset_size = feats.size()

        # Khởi tạo tensor điểm
        score = torch.zeros(batch_size, device=feats.device)
        
        # Thêm nhãn bắt đầu vào đầu chuỗi tags
        tags = torch.cat([
            torch.full((batch_size, 1), self.start_tag_id, dtype=torch.long, device=feats.device),
            tags
        ], dim=1)

        # Tính điểm cho từng vị trí trong chuỗi
        for i in range(seq_len):
            current_tag = tags[:, i + 1]  # Nhãn hiện tại
            prev_tag = tags[:, i]         # Nhãn trước đó
            
            # Lấy điểm chuyển tiếp (transition score)
            transition_score = self.transitions[prev_tag, current_tag]
            
            # Lấy điểm phát xạ (emission score)
            emission_score = feats[:, i].gather(1, current_tag.unsqueeze(1)).squeeze(1)
            
            # Áp dụng mask và cộng dồn điểm
            score += (transition_score + emission_score) * mask[:, i]

        # Thêm điểm cho chuyển tiếp đến nhãn kết thúc
        last_tags = tags.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.transitions[last_tags, self.end_tag_id]

        # Tính hàm phân hoạch log (log partition function)
        log_partition = self._compute_log_partition_function(feats, mask)
        
        return score - log_partition

    def _compute_log_partition_function(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Tính toán hàm phân hoạch log (log partition function) sử dụng thuật toán forward.
        
        Tham số:
            feats (torch.Tensor): Điểm phát xạ
            mask (torch.Tensor): Tensor mask
            
        Trả về:
            torch.Tensor: Giá trị log partition function
        """
        batch_size, seq_len, tagset_size = feats.size()
        
        # Khởi tạo biến forward
        log_alpha = torch.full((batch_size, tagset_size), -10000.0, device=feats.device)
        log_alpha[:, self.start_tag_id] = 0  # Điểm ban đầu cho nhãn bắt đầu

        # Thuật toán forward
        for i in range(seq_len):
            # Mở rộng điểm phát xạ và điểm chuyển tiếp
            emit_score = feats[:, i].unsqueeze(2)
            transition_score = self.transitions.unsqueeze(0)
            log_alpha_expanded = log_alpha.unsqueeze(1)

            # Tính tổng điểm
            score = log_alpha_expanded + transition_score + emit_score
            log_alpha = torch.logsumexp(score, dim=2)

            # Áp dụng mask
            mask_i = mask[:, i].unsqueeze(1)
            log_alpha = log_alpha * mask_i + log_alpha * (~mask_i)

        # Thêm điểm chuyển tiếp đến nhãn kết thúc
        log_alpha += self.transitions[:, self.end_tag_id].unsqueeze(0)
        return torch.logsumexp(log_alpha, dim=1)

    def _viterbi_decode(self, feats: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Thuật toán Viterbi để tìm chuỗi nhãn có xác suất cao nhất.
        
        Tham số:
            feats (torch.Tensor): Điểm phát xạ
            mask (torch.Tensor): Tensor mask
            
        Trả về:
            List[List[int]]: Danh sách các chuỗi nhãn dự đoán
        """
        batch_size, seq_len, tagset_size = feats.size()
        backpointers = []  # Lưu trữ các con trỏ ngược để truy vết

        # Khởi tạo biến Viterbi
        init_vvars = torch.full((batch_size, tagset_size), -10000.0, device=feats.device)
        init_vvars[:, self.start_tag_id] = 0  # Điểm ban đầu cho nhãn bắt đầu
        forward_var = init_vvars

        # Bước tiến (forward pass)
        for i in range(seq_len):
            # Mở rộng biến forward
            next_tag_var = forward_var.unsqueeze(1) + self.transitions.unsqueeze(0)
            
            # Tìm điểm cao nhất và con trỏ ngược
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=2)
            
            # Cộng điểm phát xạ và áp dụng mask
            forward_var = viterbivars_t + feats[:, i]
            mask_i = mask[:, i].unsqueeze(1)
            forward_var = forward_var * mask_i + forward_var * (~mask_i)
            
            backpointers.append(bptrs_t)

        # Thêm điểm chuyển tiếp đến nhãn kết thúc
        terminal_var = forward_var + self.transitions[:, self.end_tag_id].unsqueeze(0)
        best_tag_scores, best_last_tags = torch.max(terminal_var, dim=1)

        # Truy vết ngược để tìm đường đi tốt nhất
        best_paths = []
        for i in range(batch_size):
            # Lấy độ dài chuỗi cho batch item này
            seq_len_i = int(mask[i].sum())
            
            # Khởi tạo đường đi với nhãn cuối cùng tốt nhất
            path = [best_last_tags[i].item()]
            
            # Truy vết ngược qua các con trỏ
            for bptrs_t in reversed(backpointers[:seq_len_i]):
                path.insert(0, bptrs_t[i][path[0]].item())
                
            best_paths.append(path)

        return best_paths

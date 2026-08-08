"""
hyena_study — mã nguồn đồ án cuối kỳ môn Xử lý ngôn ngữ tự nhiên (Nhóm 8).

Bài báo nền: Poli et al., "Hyena Hierarchy: Towards Larger Convolutional
Language Models", ICML 2023 (Oral). arXiv 2302.10866 / PMLR v202.

QUY ƯỚC QUAN TRỌNG CỦA REPO NÀY:
  1. Mọi công thức lấy từ paper đều phải ghi rõ số hiệu (Def 3.1, eq.(7),
     Algorithm 1-3, ...) ngay tại chỗ cài đặt.
  2. Mọi lựa chọn KHÔNG có trong paper phải được đánh dấu là quyết định của nhóm
     (xem các ghi chú (I1)-(I5) trong models/hyena.py).
  3. Huấn luyện chỉ chạy trên GPU đám mây (Kaggle). Máy cá nhân chỉ dùng để chạy
     unit test trên CPU.
"""

__version__ = "0.1.0"

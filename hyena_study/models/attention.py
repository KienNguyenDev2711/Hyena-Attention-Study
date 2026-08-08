"""
Self-attention nhân quả — đường cơ sở (baseline) để so sánh với Hyena.

VÌ SAO CÓ HAI CÀI ĐẶT (`sdpa` và `naive`):

Paper Hyena so sánh tốc độ với **FlashAttention**, và tự ghi rõ trong chú thích:
"FlashAttention is already 2-4x faster than a standard attention implementation
in PyTorch." Các con số tăng tốc được trích trong paper là:
  - 5x so với dense self-attention ở L = 8192
  - 2x so với FlashAttention ở L = 8192
  - 100x so với FlashAttention ở L = 64k (attention thường trong PyTorch OOM)

Hệ quả: nếu chỉ so Hyena với một cài đặt attention ngây thơ (materialize ma trận
L×L) thì con số tăng tốc bị THỔI PHỒNG và bài báo cáo sẽ bị phản biện ngay. Ngược
lại, nếu chỉ so với SDPA thì lại không minh hoạ được bộ nhớ O(L²).

Do đó ta giữ cả hai và **luôn báo cáo cả hai** trong thí nghiệm hiệu năng:
  - `naive`: materialize ma trận điểm số L×L  -> bộ nhớ O(L²), minh hoạ rào cản.
  - `sdpa` : torch.nn.functional.scaled_dot_product_attention, PyTorch tự chọn
             kernel memory-efficient/flash  -> đường cơ sở MẠNH và CÔNG BẰNG.

Hai cài đặt phải cho kết quả trùng nhau trong sai số số học — có unit test kiểm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    d_model: int = 256
    n_heads: int = 8
    dropout: float = 0.0
    impl: str = "sdpa"          # "sdpa" | "naive"


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention nhân quả, chữ ký (B, L, D) -> (B, L, D).

    Cùng chữ ký với `HyenaOperator` để hai toán tử thay thế nhau được trong cùng
    một khối — đúng tinh thần "drop-in replacement" của paper.
    """

    def __init__(self, cfg: AttentionConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError(f"d_model={cfg.d_model} không chia hết cho n_heads={cfg.n_heads}")
        if cfg.impl not in ("sdpa", "naive"):
            raise ValueError(f"impl không hợp lệ: {cfg.impl}")
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        B, L, D = u.shape
        q, k, v = self.qkv(u).split(D, dim=-1)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # (B, H, L, dh)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        if self.cfg.impl == "sdpa":
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Cài đặt ngây thơ: materialize ma trận L×L -> bộ nhớ O(L²)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # (B, H, L, L)
            mask = torch.ones(L, L, dtype=torch.bool, device=u.device).tril()
            att = att.masked_fill(~mask, float("-inf"))
            att = att.softmax(dim=-1)
            if self.training and self.dropout_p > 0:
                att = F.dropout(att, p=self.dropout_p)
            y = att @ v                                                # (B, H, L, dh)

        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.resid_dropout(self.out_proj(y))

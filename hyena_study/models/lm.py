"""
Khung mô hình ngôn ngữ dùng chung cho Hyena, Transformer và mô hình LAI (hybrid).

NGUYÊN TẮC THIẾT KẾ — SO SÁNH PHẢI CÔNG BẰNG:

Toàn bộ mô hình chỉ khác nhau ở **toán tử trộn token** (token-mixing operator).
Mọi thứ còn lại giữ NGUYÊN VẸN như nhau: embedding, pre-LayerNorm, khối MLP,
kết nối residual, weight tying, cách khởi tạo, lịch learning rate.

Nếu để mô hình khác nhau ở nhiều chỗ cùng lúc thì không thể quy chênh lệch PPL
cho toán tử được nữa — đây là lỗi thiết kế thí nghiệm nghiêm trọng và là câu hỏi
phản biện gần như chắc chắn sẽ bị hỏi.

VỀ POSITIONAL EMBEDDING (một điểm gây nhiễu tinh vi):
  - Attention **bắt buộc** cần thông tin vị trí, tự nó hoán vị bất biến.
  - Hyena **không cần**: bộ lọc tích chập đã mang thông tin vị trí sẵn.
  Nếu cấp pos-emb cho Transformer mà không cấp cho Hyena thì hai mô hình khác
  nhau ở HAI biến, không phải một. Mặc định ở đây: cấp pos-emb học được cho CẢ
  HAI (`pos_emb="learned"`) để chỉ còn đúng một biến thay đổi. Cờ
  `pos_emb="none"` để chạy nhánh ablation kiểm tra Hyena có thật sự không cần.

CHUỖI LỚP (`layer_spec`): chuỗi ký tự, mỗi ký tự là một lớp
  'H' = HyenaOperator, 'A' = CausalSelfAttention
  Ví dụ: "HHHH" = Hyena thuần · "AAAA" = Transformer thuần · "HHHA" = lai.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .attention import AttentionConfig, CausalSelfAttention
from .hyena import HyenaConfig, HyenaFilterConfig, HyenaOperator

VALID_LAYERS = {"H", "A"}


@dataclass
class LMConfig:
    vocab_size: int = 16000
    d_model: int = 256
    layer_spec: str = "HHHH"
    max_seq_len: int = 512
    d_ff_mult: int = 4
    dropout: float = 0.1
    pos_emb: str = "learned"                 # "learned" | "none"
    tie_weights: bool = True
    # tham số riêng của từng toán tử
    n_heads: int = 8
    attn_impl: str = "sdpa"
    hyena_order: int = 2
    hyena_short_filter: int = 3
    hyena_filter: HyenaFilterConfig = field(default_factory=HyenaFilterConfig)
    hyena_skip_scale: bool = False

    def __post_init__(self) -> None:
        bad = set(self.layer_spec.upper()) - VALID_LAYERS
        if bad:
            raise ValueError(f"layer_spec chứa ký tự không hợp lệ: {sorted(bad)}; chỉ nhận H/A")
        if not self.layer_spec:
            raise ValueError("layer_spec rỗng")
        self.layer_spec = self.layer_spec.upper()
        if self.pos_emb not in ("learned", "none"):
            raise ValueError(f"pos_emb không hợp lệ: {self.pos_emb}")

    @property
    def n_layers(self) -> int:
        return len(self.layer_spec)

    @property
    def n_attn_layers(self) -> int:
        return self.layer_spec.count("A")

    @property
    def attn_ratio(self) -> float:
        return self.n_attn_layers / self.n_layers


class FeedForward(nn.Module):
    """MLP theo vị trí — GIỐNG HỆT nhau ở mọi biến thể mô hình."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Khối pre-LN: x = x + Op(LN(x)); x = x + MLP(LN(x)).

    `op` là toán tử trộn token — Hyena hoặc Attention, thay thế nhau được.
    """

    def __init__(self, op: nn.Module, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.op = op
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.op(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def _build_op(kind: str, cfg: LMConfig) -> nn.Module:
    if kind == "A":
        return CausalSelfAttention(AttentionConfig(
            d_model=cfg.d_model, n_heads=cfg.n_heads,
            dropout=cfg.dropout, impl=cfg.attn_impl,
        ))
    return HyenaOperator(
        HyenaConfig(
            d_model=cfg.d_model,
            order=cfg.hyena_order,
            short_filter_size=cfg.hyena_short_filter,
            skip_scale=cfg.hyena_skip_scale,
            dropout=cfg.dropout,
            filter=cfg.hyena_filter,
        ),
        max_seq_len=cfg.max_seq_len,
    )


class SequenceLM(nn.Module):
    """Mô hình ngôn ngữ nhân quả; kiến trúc do `cfg.layer_spec` quyết định."""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        d_ff = cfg.d_model * cfg.d_ff_mult

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = (
            nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.pos_emb == "learned" else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            Block(_build_op(kind, cfg), cfg.d_model, d_ff, cfg.dropout)
            for kind in cfg.layer_spec
        ])
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Không đụng vào HyenaFilter: nó đã có khởi tạo SIREN riêng, ghi đè sẽ hỏng.
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, L) chỉ số token -> logits (B, L, vocab)."""
        B, L = idx.shape
        if L > self.cfg.max_seq_len:
            raise ValueError(f"độ dài {L} vượt max_seq_len={self.cfg.max_seq_len}")

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(L, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm_f(x))

    # ── tiện ích báo cáo ────────────────────────────────────────────────────
    def num_parameters(self, trainable_only: bool = True, exclude_embedding: bool = False) -> int:
        params = [p for p in self.parameters() if p.requires_grad or not trainable_only]
        total = sum(p.numel() for p in params)
        if exclude_embedding:
            total -= self.tok_emb.weight.numel()
            if self.pos_emb is not None:
                total -= self.pos_emb.weight.numel()
        return total

    def param_breakdown(self) -> dict[str, int]:
        """Chia nhỏ số tham số theo thành phần — cần để chứng minh so sánh iso-param."""
        out: dict[str, int] = {}
        out["embedding"] = self.tok_emb.weight.numel()
        if self.pos_emb is not None:
            out["pos_embedding"] = self.pos_emb.weight.numel()
        mixer = mlp = 0
        for block in self.blocks:
            mixer += sum(p.numel() for p in block.op.parameters())
            mlp += sum(p.numel() for p in block.mlp.parameters())
        out["token_mixer"] = mixer
        out["mlp"] = mlp
        out["total"] = self.num_parameters()
        # lm_head dùng chung trọng số với embedding khi tie_weights=True
        out["tied_lm_head"] = int(self.cfg.tie_weights)
        return out


def build_model(cfg: LMConfig) -> SequenceLM:
    return SequenceLM(cfg)

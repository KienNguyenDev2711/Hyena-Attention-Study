"""
Toán tử Hyena — cài đặt bám sát Poli et al., ICML 2023 (arXiv 2302.10866).

TRUY VẾT NGUỒN (mọi công thức dưới đây đều dẫn về paper gốc):

  [Def 3.1] Order-N Hyena Operator, eq.(4):
        z^1_t     = v_t
        z^{n+1}_t = x^n_t * (h^n ⊛ z^n)_t ,  n = 1..N
        y_t       = z^{N+1}_t
      (⊛ = tích chập nhân quả dài; * = nhân từng phần tử)

  [Remark 3.1] Độ phức tạp thời gian: O(N · L · log2 L).

  [eq.(7), §3.3] Bộ lọc ngầm:
        h_t = Window(t) · (FFN ∘ PositionalEncoding)(t)
      - Window(t) = exp{-α t}; α **thay đổi theo từng kênh độc lập** để
        "regularize filters to be of different lengths" (Figure 3).
      - "In practice, we add a bias term to our window, so that the filters are
        not constrained to be zeros after a length determined by the decay rate."
      - FFN dùng **kích hoạt tuần hoàn tần số cao (sine)**, theo Romero et al.
        2021a, để khắc phục thiên lệch tần số thấp của mạng nơ-ron.

  [§3.3 Preserving causality] "If we use FFT-based convolution algorithms, all we
      need is to evaluate the filter at t = 0..L-1 and zero-pad the input and
      filter sequences to 2L-1 before taking the FFT."

  [Algorithm 1] Projection: Linear R^D -> R^{(N+1)D}, rồi DepthwiseConv1d (bộ lọc
      ngắn), rồi tách thành x^1..x^N, v.
  [Algorithm 2] HyenaFilter: t = PositionalEncoding(L); ĥ = FFN(t);
      reshape (N, D, L); h = ĥ · Window(t); tách thành h^1..h^N.
  [Algorithm 3] for n = 1..N: v <- x^n * FFTConv(h^n, v); return y = v.

--------------------------------------------------------------------------------
QUYẾT ĐỊNH CÀI ĐẶT (paper KHÔNG đặc tả — đây là lựa chọn của nhóm, phải khai báo
trong báo cáo, KHÔNG được trình bày như thể lấy từ paper):

  (I1) Dạng hàm PositionalEncoding cụ thể. Paper chỉ ghi "PositionalEncoding(t)".
       Ta dùng đặc trưng Fourier: [t, sin(2π f_k t), cos(2π f_k t)] với t ∈ [0,1]
       và f_k cách đều theo thang log. Xem `PositionalEncoding`.
  (II) t được chuẩn hoá về [0,1] theo L. Hệ quả: hình dạng bộ lọc gắn với L
       tương đối, nên bộ lọc học ở L=256 KHÔNG chuyển thẳng sang L=1024 được.
       Mỗi mô hình trong nghiên cứu này được huấn luyện ở một L cố định.
  (I3) α mặc định là hằng số cách đều theo kênh (khớp mô tả Figure 3), KHÔNG học.
       Cờ `learn_decay=True` cho phép học α — dùng làm nhánh ablation, không phải
       cấu hình gốc.
  (I4) `out_proj` (Linear D->D sau đệ quy) không xuất hiện trong Algorithm 3
       nhưng cần để toán tử ghép được vào khối residual; đây là quy ước chuẩn.
  (I5) `skip_scale` (số hạng D·v kiểu S4) mặc định TẮT để trung thành với eq.(4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Cấu hình
# -----------------------------------------------------------------------------
@dataclass
class HyenaFilterConfig:
    """Siêu tham số của bộ lọc ngầm (eq. 7)."""

    emb_dim: int = 33               # số chiều PositionalEncoding (lẻ: 1 + 2*n_freq)
    hidden: int = 64                # bề rộng FFN sinh bộ lọc
    n_layers: int = 3               # tổng số lớp Linear trong FFN (>= 2)
    w0: float = 10.0                # tần số của kích hoạt sine (SIREN/CKConv)
    use_sine: bool = True           # False -> dùng GELU (nhánh ablation E5b)
    use_window: bool = True         # False -> bỏ cửa sổ suy giảm (nhánh ablation E5a)
    alpha_min: float = 0.05         # α nhỏ  -> bộ lọc "dài"
    alpha_max: float = 6.0          # α lớn  -> bộ lọc "ngắn"
    learn_decay: bool = False       # True  -> học α (nhánh ablation E5c)
    normalize_filter: bool = False  # True  -> chuẩn hoá L1 bộ lọc theo từng kênh
    # ĐỀ XUẤT MỚI CỦA NHÓM (E4): thay vì trải đều α, đặt α theo thống kê hình thái
    # của ngôn ngữ đích. Truyền vào danh sách α dài đúng d_model.
    # None = giữ nguyên cách trải đều của paper (đường cơ sở).
    alpha_values: list[float] | None = None


@dataclass
class HyenaConfig:
    """Siêu tham số của toán tử Hyena."""

    d_model: int = 256
    order: int = 2                  # N trong Def 3.1 (N=1 ≈ GSS, N=2 ≈ H3 — Remark 3.2)
    short_filter_size: int = 3      # bộ lọc ngắn trong Algorithm 1
    skip_scale: bool = False        # xem (I5)
    dropout: float = 0.0
    filter: HyenaFilterConfig = field(default_factory=HyenaFilterConfig)


# -----------------------------------------------------------------------------
# Thành phần
# -----------------------------------------------------------------------------
class Sine(nn.Module):
    """Kích hoạt tuần hoàn y = sin(w0 · x) — §3.3, theo Romero et al. 2021a."""

    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)

    def extra_repr(self) -> str:
        return f"w0={self.w0}"


class PositionalEncoding(nn.Module):
    """t ∈ {0..L-1} -> đặc trưng Fourier. Xem ghi chú (I1), (I2).

    Trả về tensor (L, emb_dim_effective) với
        emb_dim_effective = 1 + 2 · n_freq,
    gồm [t_chuẩn_hoá, sin(2π f_k t), cos(2π f_k t)].
    """

    def __init__(self, emb_dim: int = 33, max_seq_len: int = 8192):
        super().__init__()
        if emb_dim < 3:
            raise ValueError("emb_dim phải >= 3 (một kênh t + ít nhất một cặp sin/cos)")
        self.n_freq = (emb_dim - 1) // 2
        self.out_dim = 1 + 2 * self.n_freq
        self.max_seq_len = max_seq_len
        # Tần số cách đều theo thang log: bắt phổ từ rất chậm tới rất nhanh.
        freqs = torch.logspace(0.0, math.log10(max_seq_len / 2.0), self.n_freq)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # (I2): chuẩn hoá t về [0, 1]
        t = torch.linspace(0.0, 1.0, L, device=device, dtype=torch.float32)  # (L,)
        ang = 2.0 * math.pi * t[:, None] * self.freqs.to(device)[None, :]    # (L, n_freq)
        pe = torch.cat([t[:, None], torch.sin(ang), torch.cos(ang)], dim=-1)  # (L, out_dim)
        return pe.to(dtype)


class HyenaFilter(nn.Module):
    """Bộ lọc dài tham số hoá ngầm — eq.(7) và Algorithm 2.

    forward(L) -> h có shape (N, D, L): N bộ lọc, mỗi bộ lọc D kênh, dài L.
    """

    def __init__(self, d_model: int, order: int, cfg: HyenaFilterConfig,
                 max_seq_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        self.order = order
        self.cfg = cfg

        self.pos_enc = PositionalEncoding(cfg.emb_dim, max_seq_len)
        act = Sine(cfg.w0) if cfg.use_sine else nn.GELU()

        if cfg.n_layers < 2:
            raise ValueError("FFN sinh bộ lọc cần >= 2 lớp Linear")
        layers: list[nn.Module] = [nn.Linear(self.pos_enc.out_dim, cfg.hidden), act]
        for _ in range(cfg.n_layers - 2):
            layers += [nn.Linear(cfg.hidden, cfg.hidden),
                       Sine(cfg.w0) if cfg.use_sine else nn.GELU()]
        layers += [nn.Linear(cfg.hidden, order * d_model, bias=False)]
        self.ffn = nn.Sequential(*layers)

        # Window(t) = exp(-α t) + bias.  α biến thiên theo kênh — Figure 3, ghi chú (I3).
        if cfg.alpha_values is not None:
            # Nhánh ĐỀ XUẤT MỚI: α do thống kê corpus quyết định (đề cương §6.2).
            if len(cfg.alpha_values) != d_model:
                raise ValueError(
                    f"alpha_values có {len(cfg.alpha_values)} phần tử nhưng d_model={d_model}; "
                    "phải khớp đúng số kênh"
                )
            alpha = torch.tensor(cfg.alpha_values, dtype=torch.float32)
            if (alpha <= 0).any():
                raise ValueError("alpha_values phải dương (độ dài hiệu dụng = L/α)")
        else:
            alpha = torch.linspace(cfg.alpha_min, cfg.alpha_max, d_model)  # (D,)
        if cfg.learn_decay:
            # tham số hoá qua log để giữ α > 0
            self.log_alpha = nn.Parameter(torch.log(alpha))
            self.register_buffer("alpha_const", torch.empty(0), persistent=False)
        else:
            self.register_buffer("alpha_const", alpha, persistent=False)
            self.log_alpha = None
        self.window_bias = nn.Parameter(torch.zeros(d_model))

        self._init_siren()

    def _init_siren(self) -> None:
        """Khởi tạo kiểu SIREN (Sitzmann et al. 2020) khi dùng kích hoạt sine.

        Không khởi tạo đúng cách thì mạng sine rất dễ phân kỳ — đây là lỗi kinh
        điển khi cài lại loại bộ lọc này.
        """
        if not self.cfg.use_sine:
            return
        linears = [m for m in self.ffn if isinstance(m, nn.Linear)]
        with torch.no_grad():
            first, *rest = linears
            fan_in = first.in_features
            first.weight.uniform_(-1.0 / fan_in, 1.0 / fan_in)
            if first.bias is not None:
                first.bias.zero_()
            for m in rest:
                fan_in = m.in_features
                bound = math.sqrt(6.0 / fan_in) / self.cfg.w0
                m.weight.uniform_(-bound, bound)
                if m.bias is not None:
                    m.bias.zero_()

    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha) if self.log_alpha is not None else self.alpha_const

    def window(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Window(t) = exp(-α t) + bias, shape (D, L)."""
        t = torch.linspace(0.0, 1.0, L, device=device, dtype=torch.float32)  # (L,)
        a = self.alpha().to(device=device, dtype=torch.float32)              # (D,)
        w = torch.exp(-a[:, None] * t[None, :])                              # (D, L)
        w = w + self.window_bias.to(device=device, dtype=torch.float32)[:, None]
        return w.to(dtype)

    def forward(self, L: int, device: torch.device | None = None,
                dtype: torch.dtype | None = None) -> torch.Tensor:
        device = device or self.window_bias.device
        dtype = dtype or self.window_bias.dtype

        # Algorithm 2, bước 1-3
        t = self.pos_enc(L, device, dtype)                 # (L, emb)
        h = self.ffn(t)                                    # (L, N·D)
        h = h.view(L, self.order, self.d_model).permute(1, 2, 0)  # (N, D, L)

        # Algorithm 2, bước 4
        if self.cfg.use_window:
            h = h * self.window(L, device, dtype)[None, :, :]   # broadcast (1,D,L)

        if self.cfg.normalize_filter:
            h = h / (h.abs().sum(dim=-1, keepdim=True) + 1e-6)
        return h                                           # (N, D, L)


# -----------------------------------------------------------------------------
# Tích chập nhân quả qua FFT
# -----------------------------------------------------------------------------
def causal_fft_conv(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Tích chập nhân quả (h ⊛ v) tính trong miền Fourier.

    Args:
        h: (D, L)      — đáp ứng xung của bộ lọc, đánh giá tại t = 0..L-1
        v: (B, D, L)   — tín hiệu vào

    Returns:
        (B, D, L)

    Nhân quả được bảo đảm bằng cách đệm 0 tới >= 2L-1 rồi cắt lấy L phần tử đầu
    (§3.3 "Preserving causality"). Ta dùng n = 2L (>= 2L-1).

    Nửa độ chính xác (fp16/bf16) được NÂNG lên float32 trước khi biến đổi: FFT
    dưới autocast fp16 vừa kém chính xác vừa không ổn định — đây là lỗi thầm lặng
    hay gặp khi bật AMP trên Kaggle. float32/float64 được giữ nguyên, nhờ vậy
    unit test đối chiếu ở float64 mới có ý nghĩa.
    """
    L = v.shape[-1]
    if h.shape[-1] != L:
        raise ValueError(f"độ dài bộ lọc {h.shape[-1]} != độ dài chuỗi {L}")
    n = 2 * L
    compute_dtype = torch.float32 if v.dtype in (torch.float16, torch.bfloat16) else v.dtype
    with torch.autocast(device_type=v.device.type, enabled=False):
        hf = torch.fft.rfft(h.to(compute_dtype), n=n)          # (D, n//2+1)
        vf = torch.fft.rfft(v.to(compute_dtype), n=n)          # (B, D, n//2+1)
        y = torch.fft.irfft(hf[None, :, :] * vf, n=n)[..., :L]
    return y.to(v.dtype)


def causal_direct_conv(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Tích chập nhân quả tính trực tiếp trong miền thời gian — O(L²).

    CHỈ dùng để kiểm chứng `causal_fft_conv` trong unit test; quá chậm để huấn
    luyện. Định nghĩa: y[b,d,t] = Σ_{s=0..t} h[d,s] · v[b,d,t-s].
    """
    B, D, L = v.shape
    # F.conv1d là tương quan chéo -> lật bộ lọc để thành tích chập thật sự.
    h_flip = torch.flip(h, dims=[-1]).unsqueeze(1)      # (D, 1, L)
    v_pad = F.pad(v, (L - 1, 0))                        # đệm trái -> nhân quả
    return F.conv1d(v_pad, h_flip, groups=D)


# -----------------------------------------------------------------------------
# Toán tử Hyena
# -----------------------------------------------------------------------------
class HyenaOperator(nn.Module):
    """Toán tử Hyena bậc N — Def 3.1 + Algorithm 1/2/3.

    Thay thế trực tiếp (drop-in) cho self-attention nhân quả: cùng chữ ký
    (B, L, D) -> (B, L, D).
    """

    def __init__(self, cfg: HyenaConfig, max_seq_len: int = 8192):
        super().__init__()
        self.cfg = cfg
        D, N = cfg.d_model, cfg.order
        if N < 1:
            raise ValueError("order N phải >= 1")

        # Algorithm 1, bước 1
        self.in_proj = nn.Linear(D, (N + 1) * D, bias=False)
        # Algorithm 1, bước 2 — depthwise, đệm trái để giữ nhân quả
        k = cfg.short_filter_size
        self.short_filter = nn.Conv1d(
            (N + 1) * D, (N + 1) * D, kernel_size=k, groups=(N + 1) * D, padding=k - 1
        )
        self.filter = HyenaFilter(D, N, cfg.filter, max_seq_len)
        self.out_proj = nn.Linear(D, D)                       # ghi chú (I4)
        self.dropout = nn.Dropout(cfg.dropout)
        # ghi chú (I5)
        self.skip = nn.Parameter(torch.zeros(N, D)) if cfg.skip_scale else None

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, L, D) -> (B, L, D)."""
        B, L, D = u.shape
        N = self.cfg.order

        # -- Algorithm 1: Projection ------------------------------------------
        z = self.in_proj(u).transpose(1, 2)                   # (B, (N+1)D, L)
        z = self.short_filter(z)[..., :L]                     # cắt phần đệm phải -> nhân quả
        parts = z.split(D, dim=1)                             # (N+1) × (B, D, L)
        xs, v = parts[:N], parts[N]                           # x^1..x^N, v

        # -- Algorithm 2: bộ lọc ngầm ----------------------------------------
        h = self.filter(L, device=u.device, dtype=u.dtype)    # (N, D, L)

        # -- Algorithm 3: đệ quy eq.(4) --------------------------------------
        for n in range(N):
            y = causal_fft_conv(h[n], v)
            if self.skip is not None:
                y = y + v * self.skip[n][None, :, None]
            v = xs[n] * y                                     # x^n ⊙ (h^n ⊛ z^n)

        out = self.out_proj(v.transpose(1, 2))                # (B, L, D)
        return self.dropout(out)

"""
Unit test KIỂM CHỨNG TÍNH ĐÚNG ĐẮN của cài đặt.

Vì sao bộ test này quan trọng hơn nó trông có vẻ:

Một mô hình ngôn ngữ nhân quả cài SAI (rò rỉ thông tin từ tương lai) vẫn huấn
luyện trơn tru và cho perplexity ĐẸP BẤT THƯỜNG. Không có test, ta không thể phân
biệt "Hyena tốt" với "Hyena gian lận". Đây là dạng lỗi phổ biến nhất khi tự cài
lại các toán tử tích chập dài, và cũng là thứ đầu tiên một người phản biện nghiêm
túc sẽ hỏi: *"làm sao bạn biết cài đặt của bạn đúng?"*

Bộ test trả lời câu hỏi đó bằng bằng chứng chứ không bằng lời hứa:
  T1. FFTConv trùng khớp tích chập trực tiếp O(L²) tính trong miền thời gian.
  T2. Toán tử Hyena nhân quả — kiểm bằng đạo hàm (gradient), không phải bằng mắt.
  T3. Attention nhân quả, và hai cài đặt sdpa/naive cho cùng kết quả.
  T4. Toàn mô hình nhân quả: đổi token ở vị trí t không làm đổi logit ở vị trí < t.
  T5. Mọi biến thể (bậc N, các nhánh ablation, mô hình lai) chạy tiến/lùi không NaN.
  T6. Hyena thật sự có tham số dưới tuyến tính theo L (điểm bán hàng chính của paper).

Chạy: python -m pytest tests/ -v      (hoặc: python tests/test_models.py)
Chạy trên CPU, vài giây. KHÔNG cần GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyena_study.models import (  # noqa: E402
    AttentionConfig,
    CausalSelfAttention,
    HyenaConfig,
    HyenaFilterConfig,
    HyenaOperator,
    LMConfig,
    SequenceLM,
    causal_direct_conv,
    causal_fft_conv,
)

torch.manual_seed(0)
DEV = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# T1 — FFTConv phải trùng tích chập trực tiếp
# ─────────────────────────────────────────────────────────────────────────────
def test_fftconv_matches_direct_convolution():
    """FFT nhanh, nhưng nhanh mà sai thì vô nghĩa. Đối chiếu ở float64."""
    B, D, L = 2, 5, 64
    h = torch.randn(D, L, dtype=torch.float64)
    v = torch.randn(B, D, L, dtype=torch.float64)

    y_fft = causal_fft_conv(h, v)
    y_dir = causal_direct_conv(h, v)

    err = (y_fft - y_dir).abs().max().item()
    assert y_fft.shape == (B, D, L)
    assert err < 1e-9, f"FFTConv lệch tích chập trực tiếp: sai số tối đa {err:.3e}"
    return err


# Ngưỡng rò rỉ TƯƠNG ĐỐI cho các toán tử dựa trên FFT.
#
# VÌ SAO KHÔNG ĐÒI BẰNG 0 TUYỆT ĐỐI:
# Attention đạt nhân quả bằng cách che (mask) — các số hạng tương lai bị nhân với
# 0 nên gradient về đúng 0 tuyệt đối. Hyena đạt nhân quả bằng cách đệm 0 rồi đi
# qua FFT -> nhân -> IFFT. Về mặt toán học kết quả nhân quả tuyệt đối, nhưng
# FFT là phép biến đổi số học trên số dấu phẩy động, nên vòng FFT/IFFT để lại
# nhiễu làm tròn cỡ epsilon máy (float64 ≈ 2.2e-16) ở vùng lẽ ra phải bằng 0.
#
# Vì vậy phép kiểm ĐÚNG là: rò rỉ phải nhỏ hơn tín hiệu hợp lệ nhiều bậc độ lớn.
# Đòi bằng 0 tuyệt đối là hiểu sai bản chất số học, còn bỏ qua hoàn toàn thì
# không phát hiện được rò rỉ thật. Ngưỡng 1e-10 nằm giữa: cách epsilon máy khoảng
# 6 bậc (đủ chỗ cho nhiễu tích luỹ), và cách một rò rỉ THẬT (tỉ lệ ~1e-1..1e-3)
# ít nhất 7 bậc.
FFT_LEAK_TOL = 1e-10


def test_fftconv_is_causal_by_construction():
    """Xung đơn vị đặt ở vị trí p chỉ được ảnh hưởng các vị trí >= p.

    Cho v = xung đơn vị tại vị trí p thì (h ⊛ v)[t] = h[t-p] khi t >= p, và phải
    bằng 0 khi t < p.
    """
    D, L, p = 3, 32, 10
    h = torch.randn(D, L, dtype=torch.float64)
    v = torch.zeros(1, D, L, dtype=torch.float64)
    v[0, :, p] = 1.0

    y = causal_fft_conv(h, v)[0]                       # (D, L)
    leak = y[:, :p].abs().max().item()
    signal = y[:, p:].abs().max().item()
    ratio = leak / signal
    assert ratio < FFT_LEAK_TOL, f"rò rỉ tương lai thật sự trong FFTConv: tỉ lệ {ratio:.3e}"
    # phần từ p trở đi phải đúng bằng h dịch đi p
    shifted_err = (y[:, p:] - h[:, : L - p]).abs().max().item()
    assert shifted_err < 1e-9, f"đáp ứng xung sai: {shifted_err:.3e}"
    return ratio


# ─────────────────────────────────────────────────────────────────────────────
# Tiện ích: kiểm nhân quả bằng gradient
# ─────────────────────────────────────────────────────────────────────────────
def _future_grad_ratio(op: torch.nn.Module, B: int, L: int, D: int, t: int) -> float:
    """Tỉ lệ rò rỉ = max|∂y_t/∂u_s| với s > t   chia cho   max|∂y_t/∂u_s| với s <= t.

    Toán tử nhân quả  <=>  tử số bằng 0 về mặt toán học. Trên số dấu phẩy động,
    tỉ lệ này phải nằm ở mức nhiễu làm tròn (xem FFT_LEAK_TOL).

    Dùng gradient là cách kiểm CHẶT NHẤT: nó quét toàn bộ đồ thị tính toán, không
    phụ thuộc vào việc ta có nghĩ ra được ca kiểm thử hiểm hay không.
    """
    op = op.double().eval()
    u = torch.randn(B, L, D, dtype=torch.float64, requires_grad=True)
    y = op(u)
    y[:, t, :].sum().backward()
    assert u.grad is not None
    future = u.grad[:, t + 1 :, :].abs().max().item()
    past = u.grad[:, : t + 1, :].abs().max().item()
    assert past > 0, "gradient quá khứ bằng 0 — phép kiểm vô nghĩa, toán tử có thể đã chết"
    return future / past


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Hyena nhân quả
# ─────────────────────────────────────────────────────────────────────────────
def test_hyena_operator_is_causal():
    D, L = 16, 32
    ratios = {}
    for order in (1, 2, 3):
        op = HyenaOperator(HyenaConfig(d_model=D, order=order, dropout=0.0), max_seq_len=L)
        r = _future_grad_ratio(op, B=2, L=L, D=D, t=L // 2)
        assert r < FFT_LEAK_TOL, f"Hyena bậc N={order} rò rỉ tương lai THẬT: tỉ lệ {r:.3e}"
        ratios[f"N={order}"] = r
    return max(ratios.values())


def test_hyena_short_filter_is_causal():
    """Bộ lọc ngắn (Algorithm 1 bước 2) là chỗ RẤT dễ cài sai thành phi nhân quả.

    Conv1d với padding đối xứng rồi quên cắt phần đệm phải sẽ làm rò rỉ tương lai
    mà loss vẫn giảm bình thường — lỗi im lặng điển hình.
    """
    D, L = 8, 24
    worst = 0.0
    for k in (3, 5, 7):
        op = HyenaOperator(
            HyenaConfig(d_model=D, order=2, short_filter_size=k, dropout=0.0), max_seq_len=L
        )
        r = _future_grad_ratio(op, B=2, L=L, D=D, t=L // 2)
        assert r < FFT_LEAK_TOL, f"bộ lọc ngắn k={k} rò rỉ tương lai THẬT: tỉ lệ {r:.3e}"
        worst = max(worst, r)
    return worst


def test_broken_short_filter_is_detected():
    """PHÉP KIỂM NGƯỢC: cố tình làm hỏng tính nhân quả, test PHẢI bắt được.

    Nếu không có phép kiểm này thì cả bộ test trên chỉ chứng minh "không có gì
    kêu", chứ chưa chứng minh nó CÓ KHẢ NĂNG kêu. Ở đây ta bỏ bước cắt phần đệm
    phải của bộ lọc ngắn — đúng lỗi kinh điển — và xác nhận tỉ lệ rò rỉ vọt lên
    nhiều bậc so với ngưỡng.
    """
    D, L, k = 8, 24, 3

    class LeakyHyena(HyenaOperator):
        def forward(self, u):                       # noqa: D102
            B, L_, D_ = u.shape
            N = self.cfg.order
            z = self.in_proj(u).transpose(1, 2)
            z = self.short_filter(z)[..., k - 1 : k - 1 + L_]   # LỖI: cắt lệch -> nhìn tương lai
            parts = z.split(D_, dim=1)
            xs, v = parts[:N], parts[N]
            h = self.filter(L_, device=u.device, dtype=u.dtype)
            for n in range(N):
                v = xs[n] * causal_fft_conv(h[n], v)
            return self.out_proj(v.transpose(1, 2))

    op = LeakyHyena(HyenaConfig(d_model=D, order=2, short_filter_size=k, dropout=0.0),
                    max_seq_len=L)
    r = _future_grad_ratio(op, B=2, L=L, D=D, t=L // 2)
    assert r > 1e-6, (
        f"phép kiểm nhân quả KHÔNG phát hiện được lỗi cố ý (tỉ lệ {r:.3e}) — "
        "bộ test này không đáng tin"
    )
    return r


# ─────────────────────────────────────────────────────────────────────────────
# T3 — Attention nhân quả và hai cài đặt khớp nhau
# ─────────────────────────────────────────────────────────────────────────────
def test_attention_is_causal():
    """Attention che bằng mask nên phải nhân quả TUYỆT ĐỐI — đúng 0, không dung sai.

    Đây là điểm đối chứng cho ngưỡng nới ở phía Hyena: nếu Hyena buộc phải có
    dung sai còn attention thì không, ta biết dung sai đó đến từ FFT chứ không
    phải từ việc bộ test bị làm cho dễ dãi.
    """
    D, L = 16, 32
    for impl in ("sdpa", "naive"):
        op = CausalSelfAttention(AttentionConfig(d_model=D, n_heads=4, dropout=0.0, impl=impl))
        r = _future_grad_ratio(op, B=2, L=L, D=D, t=L // 2)
        assert r == 0.0, f"attention ({impl}) rò rỉ tương lai: tỉ lệ {r:.3e}"
    return True


def test_attention_implementations_agree():
    """sdpa và naive phải cho cùng đầu ra — nếu lệch thì mọi so sánh tốc độ đều vô nghĩa."""
    D, L, B = 32, 16, 2
    torch.manual_seed(1)
    a = CausalSelfAttention(AttentionConfig(d_model=D, n_heads=4, dropout=0.0, impl="sdpa"))
    b = CausalSelfAttention(AttentionConfig(d_model=D, n_heads=4, dropout=0.0, impl="naive"))
    b.load_state_dict(a.state_dict())
    a, b = a.double().eval(), b.double().eval()

    u = torch.randn(B, L, D, dtype=torch.float64)
    with torch.no_grad():
        err = (a(u) - b(u)).abs().max().item()
    assert err < 1e-10, f"hai cài đặt attention lệch nhau: {err:.3e}"
    return err


# ─────────────────────────────────────────────────────────────────────────────
# T4 — Toàn mô hình nhân quả (kiểm trên token rời rạc)
# ─────────────────────────────────────────────────────────────────────────────
def test_full_model_is_causal_under_token_perturbation():
    """Đổi token ở vị trí t thì logits tại mọi vị trí < t phải KHÔNG đổi.

    Đây là phép kiểm sát với thực tế huấn luyện nhất: nó thao tác trên đầu vào
    rời rạc thật, đi qua embedding, mọi khối, và cả lm_head.
    """
    V, L, t = 50, 24, 12
    worst: dict[str, float] = {}
    for spec in ("HHHH", "AAAA", "HAHA", "HHHA"):
        cfg = LMConfig(
            vocab_size=V, d_model=32, layer_spec=spec, max_seq_len=L,
            dropout=0.0, n_heads=4, hyena_order=2,
        )
        model = SequenceLM(cfg).double().eval()

        idx = torch.randint(0, V, (1, L))
        idx2 = idx.clone()
        idx2[0, t] = (idx[0, t].item() + 7) % V          # đổi đúng một token
        assert idx2[0, t] != idx[0, t]

        with torch.no_grad():
            log1, log2 = model(idx), model(idx2)
            d = (log1[0, :t] - log2[0, :t]).abs().max().item()
            # so với mức thay đổi HỢP LỆ ở vị trí t trở đi
            legit = (log1[0, t:] - log2[0, t:]).abs().max().item()
        ratio = d / max(legit, 1e-12)
        assert ratio < FFT_LEAK_TOL, (
            f"[{spec}] logit quá khứ đổi thật khi sửa token tương lai: tỉ lệ {ratio:.3e}"
        )
        worst[spec] = ratio
    return max(worst.values())


# ─────────────────────────────────────────────────────────────────────────────
# T5 — Mọi biến thể chạy được, không NaN
# ─────────────────────────────────────────────────────────────────────────────
def test_all_variants_forward_backward_clean():
    V, L, B = 40, 32, 2
    variants = {
        "hyena_thuan": dict(layer_spec="HHHH"),
        "transformer_thuan": dict(layer_spec="AAAA"),
        "lai_1_attn_cuoi": dict(layer_spec="HHHA"),
        "lai_xen_ke": dict(layer_spec="HAHA"),
        "bac_N1": dict(layer_spec="HHHH", hyena_order=1),
        "bac_N3": dict(layer_spec="HHHH", hyena_order=3),
        "khong_pos_emb": dict(layer_spec="HHHH", pos_emb="none"),
        "ablation_bo_window": dict(
            layer_spec="HHHH", hyena_filter=HyenaFilterConfig(use_window=False)),
        "ablation_bo_sine": dict(
            layer_spec="HHHH", hyena_filter=HyenaFilterConfig(use_sine=False)),
        "ablation_hoc_decay": dict(
            layer_spec="HHHH", hyena_filter=HyenaFilterConfig(learn_decay=True)),
        "ablation_chuan_hoa_loc": dict(
            layer_spec="HHHH", hyena_filter=HyenaFilterConfig(normalize_filter=True)),
    }
    report = {}
    for name, kw in variants.items():
        cfg = LMConfig(vocab_size=V, d_model=32, max_seq_len=L, dropout=0.0, n_heads=4, **kw)
        model = SequenceLM(cfg)
        idx = torch.randint(0, V, (B, L))
        logits = model(idx)
        assert logits.shape == (B, L, V), f"[{name}] shape sai: {tuple(logits.shape)}"
        assert torch.isfinite(logits).all(), f"[{name}] logits có NaN/Inf ngay lúc khởi tạo"

        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, V), idx[:, 1:].reshape(-1)
        )
        loss.backward()
        gmax = max(p.grad.abs().max().item() for p in model.parameters() if p.grad is not None)
        assert torch.isfinite(torch.tensor(gmax)), f"[{name}] gradient NaN/Inf"
        report[name] = (model.num_parameters(), loss.item(), gmax)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# T6 — Tham số dưới tuyến tính theo L (luận điểm cốt lõi của paper)
# ─────────────────────────────────────────────────────────────────────────────
def test_hyena_param_count_independent_of_seq_len():
    """Bộ lọc dài SINH ra từ FFN nên số tham số KHÔNG tăng theo L.

    Đây chính là "decoupling of filter length and parameter cost" (§3.3). Nếu số
    tham số tăng theo L thì ta đã cài thành bộ lọc tường minh (explicit), tức là
    cài sai bản chất của Hyena.
    """
    counts = {}
    for L in (128, 512, 2048, 8192):
        op = HyenaOperator(HyenaConfig(d_model=64, order=2), max_seq_len=L)
        counts[L] = sum(p.numel() for p in op.parameters())
    unique = set(counts.values())
    assert len(unique) == 1, f"số tham số Hyena thay đổi theo L: {counts}"
    return counts


def test_filter_length_scales_with_L():
    """Ngược lại, ĐỘ DÀI bộ lọc sinh ra phải bằng đúng L."""
    f_cfg = HyenaFilterConfig()
    from hyena_study.models import HyenaFilter
    filt = HyenaFilter(d_model=16, order=2, cfg=f_cfg, max_seq_len=4096)
    for L in (64, 256, 1024):
        h = filt(L, device=DEV, dtype=torch.float32)
        assert h.shape == (2, 16, L), f"shape bộ lọc sai ở L={L}: {tuple(h.shape)}"
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Bộ chạy độc lập (không cần pytest)
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    tests = [
        ("T1a FFTConv == tích chập trực tiếp", test_fftconv_matches_direct_convolution),
        ("T1b FFTConv nhân quả (đáp ứng xung)", test_fftconv_is_causal_by_construction),
        ("T2a Hyena nhân quả (N=1,2,3)", test_hyena_operator_is_causal),
        ("T2b Bộ lọc ngắn nhân quả (k=3,5,7)", test_hyena_short_filter_is_causal),
        ("T2c Phép kiểm ngược: bắt được lỗi cố ý", test_broken_short_filter_is_detected),
        ("T3a Attention nhân quả (sdpa/naive)", test_attention_is_causal),
        ("T3b sdpa == naive", test_attention_implementations_agree),
        ("T4  Toàn mô hình nhân quả", test_full_model_is_causal_under_token_perturbation),
        ("T5  Mọi biến thể chạy sạch", test_all_variants_forward_backward_clean),
        ("T6a Tham số Hyena độc lập với L", test_hyena_param_count_independent_of_seq_len),
        ("T6b Độ dài bộ lọc bằng L", test_filter_length_scales_with_L),
    ]
    n_fail = 0
    for name, fn in tests:
        try:
            out = fn()
            extra = ""
            if isinstance(out, float):
                extra = f"  (sai số {out:.2e})"
            elif isinstance(out, dict) and all(isinstance(v, int) for v in out.values()):
                extra = f"  {out}"
            print(f"  [PASS] {name}{extra}")
        except AssertionError as exc:
            n_fail += 1
            print(f"  [FAIL] {name}\n         {exc}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [LỖI ] {name}\n         {type(exc).__name__}: {exc}")

    print()
    if n_fail:
        print(f"==> {n_fail}/{len(tests)} test THẤT BẠI")
    else:
        print(f"==> Toàn bộ {len(tests)} test ĐẠT")

    if n_fail == 0:
        rep = test_all_variants_forward_backward_clean()
        print("\nSố tham số & loss khởi tạo của từng biến thể (d_model=32, vocab=40):")
        print(f"  {'biến thể':<24} {'#tham số':>10} {'loss':>8}   (ln 40 = 3.689)")
        for name, (n, loss, _) in rep.items():
            print(f"  {name:<24} {n:>10,} {loss:>8.3f}")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())

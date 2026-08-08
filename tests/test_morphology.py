"""
Kiểm định CÔNG CỤ ĐO trước khi đem nó đo dữ liệu thật.

Nguyên tắc nghiên cứu: một phép đo chưa được hiệu chuẩn trên dữ liệu có đáp án
biết trước thì không dùng để kết luận được. Nếu ta đem thẳng bộ đo thông tin
tương hỗ áp lên corpus tiếng Việt rồi thấy một đường cong đẹp, ta **không có cách
nào biết** đường cong đó phản ánh cấu trúc ngôn ngữ hay chỉ là hiện vật (artifact)
của ước lượng.

Nên ở đây ta dựng các chuỗi nhân tạo mà ta BIẾT TRƯỚC câu trả lời, rồi kiểm tra
bộ đo có trả về đúng câu trả lời đó không:

  M1. Chuỗi độc lập hoàn toàn -> thông tin thật = 0. Bộ đo phải trả về ~0 SAU
      hiệu chỉnh, trong khi ước lượng thô phải dương rõ rệt. Đây là bằng chứng
      trực tiếp rằng độ chệch có thật và cơ chế hiệu chỉnh hoạt động.
  M2. Chuỗi có chu kỳ sao chép k -> thông tin phải ĐỘT BIẾN đúng tại d = k.
  M3. Chuỗi tương quan tầm ngắn -> thông tin phải giảm đơn điệu theo d.
  M4. Ánh xạ I(d) -> alpha phải cấp nhiều kênh hơn cho vùng tập trung thông tin.

Chạy: python tests/test_morphology.py   (CPU, không cần Internet)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyena_study.morphology import (  # noqa: E402
    alphas_from_mi,
    logspaced_alphas,
    mutual_information_decay,
    uniform_alphas,
)

RNG = np.random.default_rng(12345)
V = 60          # cỡ từ điển nhân tạo
N = 300_000     # số token
TOP_K = 50


# -----------------------------------------------------------------------------
# Sinh dữ liệu nhân tạo có đáp án biết trước
# -----------------------------------------------------------------------------
def _iid_sequence(n: int = N, v: int = V) -> np.ndarray:
    """Không có cấu trúc gì. Thông tin tương hỗ THẬT = 0 tại mọi khoảng cách."""
    return RNG.integers(0, v, size=n, dtype=np.int32)


def _periodic_copy(n: int = N, v: int = V, period: int = 8, p: float = 0.9) -> np.ndarray:
    """x_t = x_{t-period} với xác suất p, còn lại là ngẫu nhiên.

    Thông tin tương hỗ phải đột biến tại d = period (và bội của nó), gần 0 ở các
    khoảng cách khác.
    """
    x = RNG.integers(0, v, size=n, dtype=np.int32)
    for t in range(period, n):
        if RNG.random() < p:
            x[t] = x[t - period]
    return x


def _markov_decay(n: int = N, v: int = V, p_stay: float = 0.85) -> np.ndarray:
    """x_t = x_{t-1} với xác suất p_stay. Tương quan giảm theo hàm mũ với d."""
    x = np.empty(n, dtype=np.int32)
    x[0] = RNG.integers(0, v)
    rand = RNG.random(n)
    fresh = RNG.integers(0, v, size=n, dtype=np.int32)
    for t in range(1, n):
        x[t] = x[t - 1] if rand[t] < p_stay else fresh[t]
    return x


# -----------------------------------------------------------------------------
# M1 — Hiệu chỉnh độ chệch có thật sự hoạt động không
# -----------------------------------------------------------------------------
def test_bias_correction_zeroes_out_independent_sequence():
    """Chuỗi độc lập: ước lượng THÔ phải dương (đó là độ chệch), hiệu chỉnh phải khử nó.

    Đây là test quan trọng nhất của cả module. Không có nó, ta không có bằng
    chứng nào rằng đường cong I(d) đo trên tiếng Việt không phải chỉ là độ chệch.
    """
    x = _iid_sequence()
    lags = np.array([1, 2, 4, 8, 16, 32])
    lags_out, mi, base = mutual_information_decay(x, top_k=TOP_K, lags=lags, seed=0)

    assert (base > 0).all(), (
        "đường nền xáo trộn không dương — nghĩa là không có độ chệch để khử, "
        "trái với lý thuyết; nghi ngờ bộ đo hỏng"
    )
    raw = mi + base
    assert (raw > 0).all(), "ước lượng thô phải dương do độ chệch mẫu hữu hạn"

    # Sau hiệu chỉnh, tín hiệu phải nhỏ hơn độ chệch nhiều lần
    worst = float(np.max(np.abs(mi) / base))
    assert worst < 0.25, (
        f"sau hiệu chỉnh vẫn còn tín hiệu giả bằng {worst:.2%} độ chệch trên chuỗi "
        "ĐỘC LẬP — cơ chế hiệu chỉnh không đạt"
    )
    return worst


# -----------------------------------------------------------------------------
# M2 — Bộ đo có tìm đúng cấu trúc đã biết không
# -----------------------------------------------------------------------------
def test_detects_known_periodicity():
    """Chuỗi sao chép chu kỳ 8: thông tin phải đột biến đúng tại d = 8."""
    period = 8
    x = _periodic_copy(period=period)
    lags = np.arange(1, 18)
    _, mi, base = mutual_information_decay(x, top_k=TOP_K, lags=lags, seed=0)

    peak_at = int(lags[int(np.argmax(mi))])
    assert peak_at == period, (
        f"bộ đo chỉ ra chu kỳ {peak_at} trong khi sự thật là {period} — "
        "công cụ đo không đáng tin"
    )
    # đỉnh phải nổi bật rõ so với các khoảng cách lân cận không liên quan
    neighbours = mi[(lags != period) & (lags != 2 * period) & (lags != period // 2)]
    ratio = float(mi.max() / max(np.abs(neighbours).max(), 1e-12))
    assert ratio > 10, f"đỉnh không nổi bật (chỉ gấp {ratio:.1f}x nền xung quanh)"
    return ratio


def test_detects_monotonic_decay():
    """Chuỗi Markov: thông tin phải giảm đơn điệu theo khoảng cách."""
    x = _markov_decay()
    lags = np.array([1, 2, 4, 8, 16, 32, 64])
    _, mi, _ = mutual_information_decay(x, top_k=TOP_K, lags=lags, seed=0)

    assert mi[0] > 0, "không phát hiện được tương quan lân cận trong chuỗi Markov"
    drops = np.diff(mi)
    assert (drops <= 1e-6).all(), f"I(d) không giảm đơn điệu: {mi}"
    assert mi[-1] < 0.05 * mi[0], (
        f"thông tin ở d=64 chưa tắt: {mi[-1]:.4f} so với d=1 là {mi[0]:.4f}"
    )
    return float(mi[0])


# -----------------------------------------------------------------------------
# M3 — Ánh xạ sang alpha
# -----------------------------------------------------------------------------
def test_alpha_mapping_concentrates_where_information_is():
    """Nơi nào tập trung thông tin thì nơi đó phải được cấp nhiều kênh hơn.

    Dựng I(d) giả có toàn bộ khối lượng nằm ở khoảng cách ngắn, rồi kiểm tra phần
    lớn kênh nhận được độ dài hiệu dụng ngắn.
    """
    lags = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    mi = np.array([1.0, 0.8, 0.5, 0.2, 0.05, 0.01, 0.0, 0.0, 0.0, 0.0])

    spec = alphas_from_mi(lags, mi, d_model=64, seq_len=512)
    eff = np.array(spec.effective_lengths)

    assert len(spec.alpha) == 64
    assert all(a > 0 for a in spec.alpha), "alpha phải dương"
    assert (eff >= 1).all() and (eff <= 512).all(), "độ dài hiệu dụng ra ngoài [1, L]"
    assert float(np.median(eff)) < 8, (
        f"trung vị độ dài hiệu dụng = {np.median(eff):.1f}, quá dài so với phân bố "
        "thông tin tập trung ở khoảng cách ngắn"
    )
    # so với đường cơ sở trải đều: đề xuất phải cấp NHIỀU kênh ngắn hơn hẳn
    base_eff = np.array(uniform_alphas(64, seq_len=512).effective_lengths)
    assert (eff < 8).sum() > (base_eff < 8).sum(), (
        "phân bố đề xuất không hề tập trung kênh ngắn hơn đường cơ sở — "
        "vậy thì nó không khác gì bản gốc"
    )
    return float(np.median(eff))


def test_fair_baseline_actually_covers_short_range():
    """Doi chung CONG BANG phai phu ca vung khoang cach ngan.

    Ly do co test nay: doi chung 1 (`uniform_alphas`) voi khoang alpha nhom chon
    dat 100% so kenh ra xa hon 64 token, trong khi phep do E0-b cho thay 80-90%
    thong tin nam trong d <= 34 token. Dem de xuat di so voi mot cau hinh nhu the
    la so voi HINH NON: thang cung khong chung minh duoc gi.

    Test nay chot rang `logspaced_alphas` khong mac loi do, de moi so sanh trong
    bao cao deu co mot doi chung dang tin.
    """
    L, D = 512, 256
    fair = np.array(logspaced_alphas(D, seq_len=L).effective_lengths)
    weak = np.array(uniform_alphas(D, seq_len=L).effective_lengths)

    assert (weak <= 64).sum() == 0, (
        "doi chung 1 le ra phai dat toan bo kenh o vung xa - neu khong thi "
        "ly do ton tai cua doi chung 2 da thay doi, phai xem lai"
    )
    assert (fair <= 4).sum() >= D // 8, (
        f"doi chung cong bang chi co {(fair <= 4).sum()}/{D} kenh <= 4 token, "
        "chua phu duoc vung thong tin tap trung"
    )
    assert fair.min() <= 1.5 and fair.max() >= L * 0.9, (
        f"doi chung cong bang khong trai het thang do: [{fair.min():.1f}, {fair.max():.1f}]"
    )
    return float(np.median(fair))


def test_alpha_mapping_refuses_degenerate_input():
    """Nếu thông tin đo được toàn <= 0, phải TỪ CHỐI sinh alpha, không được đoán bừa.

    Sinh ra một file alpha từ phép đo vô nghĩa rồi huấn luyện cả tuần là kịch bản
    tệ nhất có thể xảy ra với đóng góp này.
    """
    lags = np.array([1, 2, 4, 8])
    for bad in (np.zeros(4), -np.ones(4)):
        try:
            alphas_from_mi(lags, bad, d_model=16, seq_len=128)
        except ValueError:
            continue
        raise AssertionError("chấp nhận đầu vào vô nghĩa mà không báo lỗi")
    return True


def test_alpha_spec_plugs_into_model():
    """alpha sinh ra phải cắm thẳng được vào mô hình và chạy được."""
    import torch
    from hyena_study.models import HyenaFilterConfig, LMConfig, SequenceLM

    lags = np.array([1, 2, 4, 8, 16, 32])
    mi = np.array([1.0, 0.6, 0.3, 0.1, 0.02, 0.0])
    spec = alphas_from_mi(lags, mi, d_model=32, seq_len=64)

    model = SequenceLM(LMConfig(
        vocab_size=50, d_model=32, layer_spec="HHHH", max_seq_len=64,
        dropout=0.0, hyena_filter=HyenaFilterConfig(alpha_values=spec.alpha),
    ))
    out = model(torch.randint(0, 50, (2, 64)))
    assert torch.isfinite(out).all(), "mô hình dùng alpha từ corpus sinh ra NaN"
    return True


# -----------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("M1  Hiệu chỉnh khử được độ chệch", test_bias_correction_zeroes_out_independent_sequence),
        ("M2a Tìm đúng chu kỳ đã biết (k=8)", test_detects_known_periodicity),
        ("M2b Phát hiện suy giảm đơn điệu", test_detects_monotonic_decay),
        ("M3a Alpha tập trung đúng chỗ", test_alpha_mapping_concentrates_where_information_is),
        ("M3b Đối chứng công bằng phủ vùng ngắn", test_fair_baseline_actually_covers_short_range),
        ("M3c Từ chối phép đo vô nghĩa", test_alpha_mapping_refuses_degenerate_input),
        ("M3d Alpha cắm được vào mô hình", test_alpha_spec_plugs_into_model),
    ]
    n_fail = 0
    for name, fn in tests:
        try:
            out = fn()
            extra = f"  ({out:.4f})" if isinstance(out, float) else ""
            print(f"  [PASS] {name}{extra}")
        except AssertionError as exc:
            n_fail += 1
            print(f"  [FAIL] {name}\n         {exc}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [LỖI ] {name}\n         {type(exc).__name__}: {exc}")
    print()
    print(f"==> {'Toàn bộ ' + str(len(tests)) + ' test ĐẠT' if not n_fail else str(n_fail) + '/' + str(len(tests)) + ' THẤT BẠI'}")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())

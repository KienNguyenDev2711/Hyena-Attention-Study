"""
E0-b — Đo cấu trúc tầm xa của corpus, rồi sinh phân bố suy giảm α cho bộ lọc Hyena.

ĐÂY LÀ NỀN TẢNG CỦA ĐÓNG GÓP MỚI (đề cương §6). Đọc kỹ phần lập luận trước khi sửa.

--------------------------------------------------------------------------------
VẤN ĐỀ

Trong bộ lọc Hyena (eq. 7), kênh thứ i có cửa sổ exp(-α_i · t). Với t chuẩn hoá về
[0,1] trên L vị trí, kênh đó suy giảm còn 1/e tại token thứ ≈ L/α_i — gọi là **độ
dài hiệu dụng** ℓ_i của kênh.

Paper đặt {α_i} **trải đều** theo kênh (Figure 3), một lựa chọn thuần trực giác,
không gắn với ngôn ngữ nào. Câu hỏi của nhóm: nếu đặt {ℓ_i} phủ đúng nơi thông
tin dự đoán THỰC SỰ nằm trong tiếng Việt thì có tốt hơn không?

--------------------------------------------------------------------------------
VÌ SAO ĐO THÔNG TIN TƯƠNG HỖ, KHÔNG ĐO ĐỘ DÀI TỪ GHÉP

Phương án hiển nhiên là chạy bộ tách từ tiếng Việt rồi lấy phân bố số âm tiết mỗi
từ. Nhóm **bác bỏ** phương án đó vì ba lý do, xếp theo mức nghiêm trọng:

1. **Nó phá hỏng nhánh đối chứng tiếng Anh.** Nếu tiếng Việt đo bằng bộ tách từ
   còn tiếng Anh đo bằng thứ khác, hai nhánh không còn so sánh được: mọi khác
   biệt quan sát được đều có thể quy cho việc dùng hai công cụ đo khác nhau. Toàn
   bộ giá trị của thiết kế đối chứng sụp đổ.
2. **Nó đo sai đại lượng.** Độ dài từ ghép chỉ là đại lượng thay thế (proxy). Thứ
   bộ lọc thực sự cần biết là: *thông tin dự đoán nằm ở khoảng cách bao xa?* Ta
   đo thẳng đại lượng đó thay vì đo cái gần đúng với nó.
3. Phụ thuộc thư viện ngoài, có thể hỏng trên Kaggle.

Thay vào đó ta đo **thông tin tương hỗ giữa hai token cách nhau d**:

    I(d) = Σ_{x,y} p_d(x,y) · log[ p_d(x,y) / (p(x)·p(y)) ]

Cùng một công thức, cùng một cách ước lượng, áp cho cả hai ngôn ngữ.

--------------------------------------------------------------------------------
CẠM BẪY THỐNG KÊ ĐÃ ĐƯỢC XỬ LÝ

Ước lượng hợp lý cực đại của thông tin tương hỗ **luôn chệch DƯƠNG** trên mẫu hữu
hạn: hai chuỗi hoàn toàn độc lập vẫn cho I > 0 chỉ vì trùng khớp ngẫu nhiên. Độ
chệch tăng theo bình phương cỡ từ điển và giảm theo cỡ mẫu — với vocab 16k thì độ
chệch có thể lớn hơn cả tín hiệu thật.

Không xử lý điều này sẽ dẫn tới kết luận "tiếng Việt có phụ thuộc tầm xa mạnh hơn
tiếng Anh" trong khi thực chất chỉ là *corpus tiếng Việt nhỏ hơn nên chệch nhiều
hơn*. Đó là loại sai lầm khiến cả đóng góp mất giá trị.

Cách xử lý: **trừ đường nền xáo trộn (shuffled baseline)**. Xáo ngẫu nhiên corpus
phá huỷ mọi cấu trúc thật, nên thông tin tương hỗ đo được trên bản xáo chính là độ
chệch. Ta báo cáo I_hiệu_chỉnh(d) = I_thô(d) − I_xáo(d).

Ngoài ra ta thô hoá từ điển xuống K token thường gặp nhất (phần còn lại gộp thành
một ký hiệu "khác") để giảm độ chệch. Thô hoá làm thông tin đo được **thấp hơn**
giá trị thật — tức là ước lượng THẬN TRỌNG, hướng sai lệch an toàn.

--------------------------------------------------------------------------------
Chạy:
    python -m hyena_study.morphology --lang vi --out alpha_vi.json
    python -m hyena_study.morphology --lang en --out alpha_en.json
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# -----------------------------------------------------------------------------
# Ước lượng thông tin tương hỗ
# -----------------------------------------------------------------------------
def _coarse_grain(tokens: np.ndarray, top_k: int) -> tuple[np.ndarray, int]:
    """Giữ K token thường gặp nhất, gộp phần còn lại vào một ký hiệu 'khác'.

    Trả về (chuỗi đã ánh xạ, cỡ bảng chữ cái). Ký hiệu 'khác' mang chỉ số K.
    """
    counts = np.bincount(tokens)
    keep = np.argsort(counts)[::-1][:top_k]
    mapping = np.full(counts.shape[0], top_k, dtype=np.int32)   # mặc định = 'khác'
    mapping[keep] = np.arange(len(keep), dtype=np.int32)
    return mapping[tokens], top_k + 1


def _mi_at_lag(seq: np.ndarray, alphabet: int, lag: int) -> float:
    """Ước lượng hợp lý cực đại (plug-in) của I(X_t ; X_{t+lag}), đơn vị nat."""
    if lag <= 0 or lag >= len(seq):
        raise ValueError(f"lag phải nằm trong (0, {len(seq)}), nhận {lag}")
    x, y = seq[:-lag], seq[lag:]
    n = x.size

    joint = np.bincount(x.astype(np.int64) * alphabet + y.astype(np.int64),
                        minlength=alphabet * alphabet).astype(np.float64)
    joint /= n
    joint = joint.reshape(alphabet, alphabet)

    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    outer = px[:, None] * py[None, :]

    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / outer[nz])))


def mutual_information_decay(
    tokens: np.ndarray,
    max_lag: int = 512,
    top_k: int = 1000,
    n_shuffle: int = 1,
    lags: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Đo I(d) đã hiệu chỉnh độ chệch, với d chạy trên tập khoảng cách `lags`.

    Returns:
        (lags, I_hiệu_chỉnh, I_đường_nền) — đơn vị nat.

    `I_đường_nền` PHẢI được vẽ kèm trong báo cáo. Nếu I_hiệu_chỉnh tụt xuống mức
    của đường nền tại khoảng cách nào đó, nghĩa là ở khoảng cách đó ta **không còn
    đo được gì** — không được diễn giải phần đuôi đó như tín hiệu thật.
    """
    seq, alphabet = _coarse_grain(tokens, top_k)
    if lags is None:
        # Lấy mẫu theo thang log: cấu trúc ngôn ngữ thay đổi nhanh ở khoảng cách
        # ngắn, chậm ở khoảng cách dài. Lấy mẫu đều theo tuyến tính sẽ lãng phí
        # phần lớn công sức vào vùng đuôi phẳng.
        lags = np.unique(np.round(np.logspace(0, np.log10(max_lag), 40)).astype(int))
        lags = lags[(lags >= 1) & (lags < len(seq))]

    rng = np.random.default_rng(seed)
    shuffled = [rng.permutation(seq) for _ in range(n_shuffle)]

    raw = np.array([_mi_at_lag(seq, alphabet, int(d)) for d in lags])
    base = np.array([
        np.mean([_mi_at_lag(s, alphabet, int(d)) for s in shuffled]) for d in lags
    ])
    return lags, raw - base, base


# -----------------------------------------------------------------------------
# Ánh xạ I(d) -> phân bố α
# -----------------------------------------------------------------------------
@dataclass
class AlphaSpec:
    alpha: list[float]
    effective_lengths: list[float]
    seq_len: int
    d_model: int
    source: str
    notes: str


def alphas_from_mi(
    lags: np.ndarray,
    mi: np.ndarray,
    d_model: int,
    seq_len: int,
    min_len: float = 1.0,
) -> AlphaSpec:
    """Biến đường cong I(d) thành d_model giá trị α.

    Ý tưởng: coi I(d) là "lượng thông tin dự đoán nằm tại khoảng cách d". Chuẩn
    hoá thành một phân bố xác suất trên khoảng cách, rồi lấy d_model phân vị cách
    đều của phân bố đó làm tập độ dài hiệu dụng {ℓ_i}. Cuối cùng α_i = L / ℓ_i.

    Hệ quả trực giác: nơi nào tập trung nhiều thông tin thì nơi đó được cấp nhiều
    kênh hơn — thay vì trải đều mù quáng như bản gốc.

    Ta cắt bỏ phần I(d) < 0 (đó là nhiễu ước lượng, không phải thông tin âm).
    """
    w = np.clip(mi, 0.0, None)
    if w.sum() <= 0:
        raise ValueError(
            "Toàn bộ thông tin tương hỗ sau hiệu chỉnh đều <= 0. Corpus quá nhỏ "
            "hoặc top_k quá lớn. KHÔNG được sinh alpha từ dữ liệu này."
        )
    w = w / w.sum()

    cdf = np.cumsum(w)
    # phân vị cách đều, tránh hai đầu mút để không lấy trúng đuôi nhiễu
    q = (np.arange(d_model) + 0.5) / d_model
    eff = np.interp(q, cdf, lags.astype(float))
    eff = np.clip(eff, min_len, float(seq_len))

    alpha = np.clip(seq_len / eff, 1e-3, 1e4)
    return AlphaSpec(
        alpha=[float(a) for a in alpha],
        effective_lengths=[float(e) for e in eff],
        seq_len=seq_len,
        d_model=d_model,
        source="mutual_information_decay",
        notes=(
            "alpha_i = seq_len / ell_i; {ell_i} = các phân vị cách đều của phân bố "
            "I(d) đã trừ đường nền xáo trộn. Sinh bởi hyena_study/morphology.py."
        ),
    )


def uniform_alphas(d_model: int, alpha_min: float = 0.05,
                   alpha_max: float = 6.0, seq_len: int = 512) -> AlphaSpec:
    """Đường cơ sở 1: alpha trải đều tuyến tính (cách đọc sát chữ của Figure 3).

    CẢNH BÁO PHẢI ĐỌC TRƯỚC KHI DÙNG LÀM ĐỐI CHỨNG:
    Khoảng [alpha_min, alpha_max] = [0.05, 6.0] là lựa chọn của NHÓM, paper KHÔNG
    quy định khoảng này. Với seq_len=512, khoảng đó cho độ dài hiệu dụng từ 85
    tới 10240 token, nghĩa là 100% số kênh nằm xa hơn 64 token.

    Phép đo E0-b trên corpus cho thấy 80-90% thông tin dự đoán nằm trong d <= 34
    token. Vậy đường cơ sở này đặt toàn bộ kênh ra ngoài vùng có thông tin, và
    đem đề xuất đi so với nó là so với một HÌNH NỘM: thắng cũng không chứng minh
    được điều gì.

    Vì vậy phải luôn báo cáo kèm `logspaced_alphas` (đường cơ sở 2). Kết luận
    khoa học chỉ được rút ra từ so sánh với đường cơ sở 2.
    """
    alpha = np.linspace(alpha_min, alpha_max, d_model)
    return AlphaSpec(
        alpha=[float(a) for a in alpha],
        effective_lengths=[float(seq_len / a) for a in alpha],
        seq_len=seq_len, d_model=d_model,
        source="uniform_linspace",
        notes=("Doi chung 1: alpha trai deu tuyen tinh trong [%.3f, %.3f]. "
               "Khoang nay do nhom chon, paper khong quy dinh."
               % (alpha_min, alpha_max)),
    )


def logspaced_alphas(d_model: int, seq_len: int = 512,
                     min_len: float = 1.0) -> AlphaSpec:
    """Đường cơ sở 2 (ĐỐI CHỨNG CÔNG BẰNG): độ dài hiệu dụng cách đều theo thang log.

    Đây là lựa chọn "không biết gì về ngôn ngữ" nhưng HỢP LÝ: phủ đều mọi thang
    khoảng cách từ 1 token tới trọn chuỗi, thay vì dồn hết vào vùng xa như đường
    cơ sở 1.

    Đây mới là đối chứng mà đề xuất phải vượt qua. Nếu `corpus` chỉ hơn được
    `uniform` mà KHÔNG hơn được `logspace`, kết luận trung thực là: cải thiện đến
    từ việc chọn khoảng alpha hợp lý hơn, KHÔNG phải từ việc thích ứng ngôn ngữ.
    """
    eff = np.logspace(np.log10(min_len), np.log10(seq_len), d_model)
    alpha = seq_len / eff
    return AlphaSpec(
        alpha=[float(a) for a in alpha],
        effective_lengths=[float(e) for e in eff],
        seq_len=seq_len, d_model=d_model,
        source="logspaced_effective_lengths",
        notes=("Doi chung 2 (cong bang): do dai hieu dung cach deu theo thang log "
               "tu %.1f toi %d token. Khong dung thong tin ngon ngu nao."
               % (min_len, seq_len)),
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="E0-b: đo suy giảm thông tin tương hỗ, sinh alpha cho bộ lọc Hyena")
    p.add_argument("--lang", default="vi", choices=["vi", "en"])
    p.add_argument("--tokenizer", default="syllable", choices=["syllable", "bpe"])
    p.add_argument("--n_docs", type=int, default=20000)
    p.add_argument("--vocab_size", type=int, default=16000)
    p.add_argument("--max_tokens", type=int, default=20_000_000,
                   help="số token dùng để đo; PHẢI đặt BẰNG NHAU giữa vi và en")
    p.add_argument("--top_k", type=int, default=1000,
                   help="cỡ bảng chữ cái sau thô hoá; càng lớn càng chệch")
    p.add_argument("--max_lag", type=int, default=512)
    p.add_argument("--n_shuffle", type=int, default=1)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--data_seed", type=int, default=0)
    p.add_argument("--cache_dir", default=None,
                   help="thu muc cache cua HuggingFace (khong phai cache token)")
    p.add_argument("--token_cache", default="data_cache",
                   help="thu muc cache dong token da xu ly, dung chung voi train.py")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--out", default=None, help="đường dẫn JSON alpha; mặc định alpha_<lang>.json")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args(argv)

    from .data import cached_token_stream

    print(f"[E0-b/{args.lang}] nạp {args.n_docs} bài ...")
    # Dung chung cache voi train.py: chay lai E0-b voi top_k khac khong phai
    # token hoa lai corpus tu dau.
    train_ids, _, _, tok, stats = cached_token_stream(
        lang=args.lang, tokenizer=args.tokenizer, vocab_size=args.vocab_size,
        n_docs=args.n_docs, data_seed=args.data_seed,
        max_tokens=args.max_tokens, cache_root=args.token_cache,
        use_cache=not args.no_cache, hf_cache_dir=args.cache_dir,
    )
    print(f"[E0-b/{args.lang}] đo trên {len(train_ids):,} token "
          f"(vocab {tok.vocab_size:,}, thô hoá còn {args.top_k})")

    lags, mi, base = mutual_information_decay(
        train_ids, max_lag=args.max_lag, top_k=args.top_k,
        n_shuffle=args.n_shuffle, seed=args.data_seed,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Ten file PHAI chua top_k: do chech ty le voi top_k^2 nen hai lan chay khac
    # top_k cho ra hai phep do KHAC HAN. Neu ten file bo qua tham so nay thi lan
    # chay sau am tham de mat ket qua lan truoc - da xay ra that ngay 2026-08-08.
    mi_path = out_dir / f"E0b_mi_decay_{args.lang}_{args.tokenizer}_k{args.top_k}.csv"
    with mi_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lag", "mi_corrected_nats", "mi_baseline_nats", "signal_to_bias"])
        for d, m, b in zip(lags, mi, base):
            w.writerow([int(d), f"{m:.8f}", f"{b:.8f}", f"{(m / b) if b > 0 else float('nan'):.4f}"])

    print(f"\n  {'d':>6} {'I(d) nats':>12} {'đường nền':>12} {'tín hiệu/chệch':>15}")
    for d, m, b in zip(lags, mi, base):
        flag = "" if (b <= 0 or m / b > 1.0) else "   <-- dưới mức nhiễu"
        print(f"  {int(d):>6} {m:>12.6f} {b:>12.6f} "
              f"{(m/b if b > 0 else float('nan')):>15.2f}{flag}")

    spec = alphas_from_mi(lags, mi, args.d_model, args.seq_len)
    out_path = Path(args.out or f"alpha_{args.lang}.json")
    payload = {
        "alpha": spec.alpha,
        "effective_lengths": spec.effective_lengths,
        "source": f"{spec.source} · lang={args.lang} · tokenizer={args.tokenizer}",
        "notes": spec.notes,
        "measurement": {
            "n_tokens_measured": int(len(train_ids)),
            "top_k": args.top_k,
            "max_lag": args.max_lag,
            "vocab_size": int(tok.vocab_size),
            "corpus": {"lang": stats.lang, "n_docs": stats.n_docs,
                       "chars_per_token": stats.chars_per_token,
                       "unk_rate": stats.unk_rate},
        },
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    eff = np.array(spec.effective_lengths)
    print(f"\n[E0-b/{args.lang}] độ dài hiệu dụng của {args.d_model} kênh: "
          f"min {eff.min():.1f} · trung vị {np.median(eff):.1f} · max {eff.max():.1f} token")
    print(f"[E0-b/{args.lang}] đã ghi {out_path} và {mi_path}")
    print("\n Trước khi dùng: đối chiếu cột 'tín hiệu/chệch'. Ở khoảng cách nào "
          "tỉ số này xuống gần 1 thì phép đo tại đó KHÔNG còn ý nghĩa — phải nêu "
          "giới hạn này trong báo cáo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

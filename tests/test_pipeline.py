"""
Test đường ống (pipeline) đầu-cuối — chạy được trên CPU, KHÔNG cần Internet.

Bộ test ở `test_models.py` chứng minh toán tử cài ĐÚNG. Bộ này chứng minh mô hình
HỌC ĐƯỢC: dữ liệu -> token hoá -> dataset -> vòng huấn luyện -> loss giảm.

Vì sao cần tách riêng: một mô hình nhân quả hoàn hảo nhưng nối dây sai (lệch nhãn,
learning rate chết, mask hỏng) vẫn qua được test tính đúng đắn mà không bao giờ
học được gì. Chỗ hay hỏng nhất là lệch một vị trí giữa đầu vào và nhãn.

Phép kiểm cốt lõi: cho mô hình học thuộc (overfit) một corpus tí hon. Nếu đường
ống nối đúng, loss phải tụt sâu dưới mức đoán đều là ln(V). Nếu nối sai, loss sẽ
mắc kẹt quanh ln(V).

Chạy: python tests/test_pipeline.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyena_study.data import (  # noqa: E402
    LMWindowDataset,
    SyllableTokenizer,
    build_token_stream,
    normalize_text,
)
from hyena_study.models import HyenaFilterConfig, LMConfig, SequenceLM  # noqa: E402
from hyena_study.train import lr_at, set_seed  # noqa: E402


# Corpus tí hon, có cấu trúc lặp lại để mô hình nhỏ học thuộc được nhanh.
# Cố ý dùng tiếng Việt CÓ DẤU để đồng thời kiểm tra chuẩn hoá NFC không làm hỏng
# âm tiết.
TOY_SENTENCES = [
    "trường đại học công nghệ thông tin là một trường thành viên",
    "sinh viên cao học nghiên cứu xử lý ngôn ngữ tự nhiên",
    "mô hình ngôn ngữ tích chập dài thay thế cơ chế chú ý",
    "bộ lọc tham số hoá ngầm được sinh ra bởi một mạng nhỏ",
]


def _toy_texts(n_repeat: int = 400) -> list[str]:
    return [normalize_text(s) for s in TOY_SENTENCES * n_repeat]


# -----------------------------------------------------------------------------
# P1 — Token hoá
# -----------------------------------------------------------------------------
def test_syllable_tokenizer_roundtrip():
    """Token hoá âm tiết phải khôi phục lại được văn bản, dấu tiếng Việt còn nguyên."""
    texts = _toy_texts(2)
    tok = SyllableTokenizer.train(texts, vocab_size=200)
    s = texts[0]
    ids = tok.encode(s)
    back = tok.decode(ids)
    assert back == s, f"mất mát khi mã hoá/giải mã:\n  gốc : {s}\n  khôi: {back}"
    assert "ườ" in back, "dấu tiếng Việt bị hỏng sau khi qua bộ token hoá"
    return len(tok.vocab)


def test_nfc_normalisation_unifies_encodings():
    """Cùng một chữ viết ở NFC và NFD phải cho CÙNG chuỗi token.

    Không chuẩn hoá thì "ế" tổ hợp sẵn và "ế" ghép từ e + dấu là hai chuỗi ký tự
    khác nhau -> phình từ điển và tạo khác biệt giả giữa các lần chạy.
    """
    import unicodedata
    s_nfc = "kiến thức tiếng việt"
    s_nfd = unicodedata.normalize("NFD", s_nfc)
    assert s_nfc != s_nfd, "hai dạng phải khác nhau ở mức chuỗi ký tự thô"
    assert normalize_text(s_nfc) == normalize_text(s_nfd), "chuẩn hoá NFC không hợp nhất được"
    return True


def test_syllable_sequences_are_longer_than_bpe_conceptually():
    """Kiểm chứng cơ chế của giả thuyết H2 ở quy mô nhỏ.

    Ở đây ta chưa dùng BPE thật (cần thư viện `tokenizers`), mà kiểm tính chất nền
    tảng: số token theo âm tiết phải BẰNG số đơn vị tách theo khoảng trắng/dấu
    câu. Đây là điều khiến chuỗi tiếng Việt dài ra — chứng minh đầy đủ cho H2 sẽ
    do thí nghiệm E2 trên Kaggle thực hiện.
    """
    s = normalize_text("trường đại học công nghệ thông tin")
    tok = SyllableTokenizer.train([s], vocab_size=100)
    n_syllable = len(tok.encode(s))
    assert n_syllable == 7, f"kỳ vọng 7 âm tiết, nhận {n_syllable}"
    return n_syllable


# -----------------------------------------------------------------------------
# P2 — Dataset
# -----------------------------------------------------------------------------
def test_dataset_input_label_alignment():
    """Nhãn phải là đầu vào DỊCH ĐI ĐÚNG MỘT vị trí.

    Lệch một vị trí là lỗi phổ biến nhất của mọi LM tự cài, và nó không làm mô
    hình đổ vỡ — chỉ làm kết quả sai một cách âm thầm.
    """
    tokens = np.arange(100, dtype=np.int32)
    ds = LMWindowDataset(tokens, seq_len=10)
    x, y = ds[0]
    assert len(x) == len(y) == 10
    assert (y[:-1] == x[1:]).all(), "đầu vào và nhãn không lệch đúng 1 vị trí"
    assert x[0] == 0 and y[0] == 1, f"cửa sổ đầu sai: x0={x[0]} y0={y[0]}"
    x2, _ = ds[1]
    assert x2[0] == 10, "các cửa sổ bị chồng lấn (phải KHÔNG chồng lấn)"
    return len(ds)


def test_token_budget_is_respected():
    """`max_tokens` phải cắt tập train đúng ngân sách — nền tảng của so sánh công bằng."""
    texts = _toy_texts(200)
    budget = 5000
    train, _, _, _, stats = build_token_stream(
        texts, tokenizer_kind="syllable", vocab_size=300, max_tokens=budget
    )
    assert len(train) == budget, f"ngân sách token không được tôn trọng: {len(train)} != {budget}"
    assert stats.n_tokens_train == budget
    return len(train)


# -----------------------------------------------------------------------------
# P3 — Mô hình học được thật
# -----------------------------------------------------------------------------
def _overfit(layer_spec: str, steps: int = 120, **cfg_kw) -> tuple[float, float, int]:
    """Cho mô hình học thuộc corpus tí hon; trả về (loss đầu, loss cuối, vocab)."""
    set_seed(0)
    texts = _toy_texts(150)
    train_ids, _, _, tok, _ = build_token_stream(
        texts, tokenizer_kind="syllable", vocab_size=100, val_frac=0.0, test_frac=0.0
    )
    seq_len = 32
    ds = LMWindowDataset(train_ids, seq_len)

    cfg = LMConfig(vocab_size=tok.vocab_size, d_model=64, layer_spec=layer_spec,
                   max_seq_len=seq_len, dropout=0.0, n_heads=4, **cfg_kw)
    model = SequenceLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    batch = 8
    idx = np.arange(min(len(ds), 64))
    first_loss = last_loss = float("nan")
    for step in range(steps):
        pick = np.random.choice(idx, size=batch, replace=True)
        xb = torch.stack([torch.from_numpy(ds[i][0]) for i in pick])
        yb = torch.stack([torch.from_numpy(ds[i][1]) for i in pick])
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, 3e-3, warmup=10)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()
    return first_loss, last_loss, tok.vocab_size


def test_hyena_learns():
    first, last, V = _overfit("HHHH")
    uniform = math.log(V)
    assert last < first, f"Hyena không học: {first:.3f} -> {last:.3f}"
    assert last < 0.5 * uniform, (
        f"Hyena mắc kẹt quanh mức đoán đều (ln V = {uniform:.3f}): loss cuối {last:.3f}. "
        "Nghi ngờ nối sai đường ống chứ không phải mô hình yếu."
    )
    return last


def test_transformer_learns():
    first, last, V = _overfit("AAAA")
    assert last < 0.5 * math.log(V), f"Transformer mắc kẹt: {last:.3f}"
    return last


def test_hybrid_learns():
    first, last, V = _overfit("HHHA")
    assert last < 0.5 * math.log(V), f"Mô hình lai mắc kẹt: {last:.3f}"
    return last


def test_hyena_learns_without_positional_embedding():
    """Hyena về lý thuyết KHÔNG cần positional embedding — bộ lọc đã mang thông tin vị trí.

    Nếu bỏ pos-emb mà Hyena vẫn học được, ta có bằng chứng cho phát biểu đó, và
    biện minh được cho nhánh ablation trong đề cương.
    """
    first, last, V = _overfit("HHHH", pos_emb="none")
    assert last < 0.5 * math.log(V), (
        f"Hyena không học được khi bỏ pos-emb: {last:.3f} (ln V = {math.log(V):.3f})"
    )
    return last


# -----------------------------------------------------------------------------
# P4 — Đường ống của ĐỀ XUẤT MỚI (E4)
# -----------------------------------------------------------------------------
def test_corpus_alpha_pathway_works_and_validates():
    """alpha do corpus quyết định phải (a) dùng được, (b) từ chối đầu vào sai.

    Cờ dòng lệnh bị bỏ qua âm thầm là cách nhanh nhất tạo ra bảng kết quả sai mà
    không ai phát hiện — nên đường ống này phải kêu to khi bị dùng sai.
    """
    d_model = 64
    # (a) độ dài hiệu dụng mong muốn -> alpha, theo đề cương §6.2: alpha = L / ell
    L = 32
    eff_len = np.linspace(2, L, d_model)          # từ 2 token tới toàn chuỗi
    alphas = (L / eff_len).tolist()
    f_cfg = HyenaFilterConfig(alpha_values=alphas)
    cfg = LMConfig(vocab_size=50, d_model=d_model, layer_spec="HHHH",
                   max_seq_len=L, dropout=0.0, hyena_filter=f_cfg)
    model = SequenceLM(cfg)
    out = model(torch.randint(0, 50, (2, L)))
    assert torch.isfinite(out).all(), "nhánh alpha-corpus sinh ra NaN"

    # (b) sai số kênh -> phải báo lỗi, không được im lặng
    bad = HyenaFilterConfig(alpha_values=[1.0, 2.0])
    try:
        SequenceLM(LMConfig(vocab_size=50, d_model=d_model, layer_spec="HHHH",
                            max_seq_len=L, hyena_filter=bad))
    except ValueError:
        pass
    else:
        raise AssertionError("alpha_values sai độ dài mà KHÔNG báo lỗi — bẫy nguy hiểm")

    # (c) alpha âm -> phải báo lỗi
    neg = HyenaFilterConfig(alpha_values=[-1.0] * d_model)
    try:
        SequenceLM(LMConfig(vocab_size=50, d_model=d_model, layer_spec="HHHH",
                            max_seq_len=L, hyena_filter=neg))
    except ValueError:
        pass
    else:
        raise AssertionError("alpha âm mà KHÔNG báo lỗi")
    return True


def test_lr_schedule_shape():
    """Warmup phải tăng dần rồi cosine giảm — dùng chung cho MỌI mô hình."""
    total, warm, base = 1000, 100, 1e-3
    assert lr_at(0, total, base, warm) < base
    assert abs(lr_at(warm - 1, total, base, warm) - base) < 1e-12
    assert lr_at(total - 1, total, base, warm) < 0.2 * base
    seq = [lr_at(s, total, base, warm) for s in range(warm)]
    assert all(b >= a for a, b in zip(seq, seq[1:])), "warmup không tăng đơn điệu"
    return True


def test_max_train_tokens_flag_parses():
    """Cờ --max_train_tokens phải tồn tại, mặc định None (giữ hành vi cũ)."""
    from hyena_study.train import parse_args
    assert parse_args([]).max_train_tokens is None
    assert parse_args(["--max_train_tokens", "38250964"]).max_train_tokens == 38_250_964
    return True


def test_max_train_tokens_decoupled_from_budget():
    """Cắt corpus phải TÁCH RỜI khỏi đếm bước huấn luyện (T13).

    Không truyền cờ thì cắt theo token_budget như cũ — mọi kết quả E1--E4 cũ vẫn
    tái lập được. Truyền cờ thì corpus bị cắt theo cờ nhưng token_budget (quyết
    định số bước) giữ nguyên — đây chính là nhánh đối chứng "cùng số token":
    corpus EN cắt xuống đúng 38.250.964 token của VI, vẫn huấn luyện đủ 50M.
    """
    from hyena_study.train import parse_args, resolve_max_train_tokens
    a = parse_args(["--token_budget", "50000000"])
    assert resolve_max_train_tokens(a) == 50_000_000, "hành vi cũ bị đổi"
    a = parse_args(["--token_budget", "50000000", "--max_train_tokens", "38250964"])
    assert resolve_max_train_tokens(a) == 38_250_964, "cờ không có tác dụng"
    assert a.token_budget == 50_000_000, "cờ không được đụng vào token_budget"
    return True


# -----------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("P1a Token hoá âm tiết khứ hồi", test_syllable_tokenizer_roundtrip),
        ("P1b Chuẩn hoá NFC hợp nhất mã", test_nfc_normalisation_unifies_encodings),
        ("P1c Số âm tiết đúng như mong đợi", test_syllable_sequences_are_longer_than_bpe_conceptually),
        ("P2a Đầu vào/nhãn lệch đúng 1", test_dataset_input_label_alignment),
        ("P2b Ngân sách token được tôn trọng", test_token_budget_is_respected),
        ("P3a Hyena học được", test_hyena_learns),
        ("P3b Transformer học được", test_transformer_learns),
        ("P3c Mô hình lai học được", test_hybrid_learns),
        ("P3d Hyena học được KHÔNG cần pos-emb", test_hyena_learns_without_positional_embedding),
        ("P4a Đường ống alpha-corpus (E4)", test_corpus_alpha_pathway_works_and_validates),
        ("P4b Lịch learning rate", test_lr_schedule_shape),
        ("P4c Cờ --max_train_tokens tồn tại", test_max_train_tokens_flag_parses),
        ("P4d Cắt corpus tách rời ngân sách bước", test_max_train_tokens_decoupled_from_budget),
    ]
    n_fail = 0
    for name, fn in tests:
        try:
            out = fn()
            extra = f"  ({out:.3f})" if isinstance(out, float) else (
                f"  ({out})" if isinstance(out, int) and not isinstance(out, bool) else "")
            print(f"  [PASS] {name}{extra}")
        except AssertionError as exc:
            n_fail += 1
            print(f"  [FAIL] {name}\n         {exc}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [LỖI ] {name}\n         {type(exc).__name__}: {exc}")

    print()
    print(f"==> {'Toàn bộ ' + str(len(tests)) + ' test ĐẠT' if not n_fail else str(n_fail) + '/' + str(len(tests)) + ' test THẤT BẠI'}")
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())

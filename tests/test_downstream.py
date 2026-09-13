"""
Test cho `hyena_study/downstream.py` (đánh giá ngoại tại trên UIT-VSFC).

NGUYÊN TẮC: mọi test ở đây phải chạy được KHÔNG CẦN MẠNG và KHÔNG CẦN GPU. Test
nào cần tải dữ liệu thật thì phải tự bỏ qua (skip) chứ không được làm đỏ cả bộ
test khi máy không có mạng.

Test quan trọng nhất là D3: nó kiểm chứng bằng SỐ cái giả định "đệm bên phải
không ảnh hưởng mô hình nhân quả". Nếu giả định đó sai thì mọi điểm số phân loại
đều sai theo, vì độ dài batch phụ thuộc vào việc câu nào rơi vào batch nào.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyena_study.data.corpus import PAD, UNK, SyllableTokenizer
from hyena_study.downstream import (
    TASKS,
    Classifier,
    compare_arms,
    encode_dataset,
    macro_f1,
    main,
    majority_baseline,
    make_batches,
    pool_hidden,
    summarize,
)
from hyena_study.models import HyenaFilterConfig, LMConfig, SequenceLM

VOCAB = ["<pad>", "<unk>", "<eos>"] + [f"w{i}" for i in range(40)]


def tiny_cfg(layer_spec: str) -> LMConfig:
    return LMConfig(vocab_size=len(VOCAB), d_model=32, layer_spec=layer_spec,
                    max_seq_len=64, n_heads=4, dropout=0.0,
                    hyena_filter=HyenaFilterConfig(hidden=32, n_layers=2))


def tiny_model(layer_spec: str, seed: int = 0) -> SequenceLM:
    torch.manual_seed(seed)
    m = SequenceLM(tiny_cfg(layer_spec))
    m.eval()
    return m


# -----------------------------------------------------------------------------
# D1 - cấu hình tác vụ
# -----------------------------------------------------------------------------
def test_d1_task_config_wellformed():
    assert set(TASKS) == {"sentiment", "topic"}
    assert TASKS["sentiment"]["names"] == ["negative", "neutral", "positive"]
    assert TASKS["topic"]["names"] == ["lecturer", "program", "facility", "others"]
    for t in TASKS.values():
        assert len(t["names"]) == len(set(t["names"]))


# -----------------------------------------------------------------------------
# D2 - gộp biểu diễn
# -----------------------------------------------------------------------------
def test_d2_pool_last_takes_real_last_token_not_padding():
    # (B=2, L=4, D=2); câu 1 dài 2, câu 2 dài 4
    h = torch.tensor([[[1., 1.], [2., 2.], [99., 99.], [99., 99.]],
                      [[3., 3.], [4., 4.], [5., 5.], [6., 6.]]])
    lens = torch.tensor([2, 4])
    got = pool_hidden(h, lens, "last")
    assert torch.allclose(got, torch.tensor([[2., 2.], [6., 6.]])), got
    # 99 là giá trị ở vị trí đệm: nếu lọt vào kết quả thì cài đặt sai
    assert not (got == 99.).any()


def test_d2b_pool_mean_ignores_padding():
    h = torch.tensor([[[1., 1.], [3., 3.], [99., 99.], [99., 99.]]])
    lens = torch.tensor([2])
    assert torch.allclose(pool_hidden(h, lens, "mean"), torch.tensor([[2., 2.]]))


def test_d2c_pool_rejects_unknown_mode():
    with pytest.raises(ValueError, match="pooling"):
        pool_hidden(torch.zeros(1, 2, 2), torch.tensor([1]), "cls")


# -----------------------------------------------------------------------------
# D3 - PHU THUOC DO DAI: cam bay rieng cua Hyena  (nhom test then chot)
# -----------------------------------------------------------------------------
def test_d3_attention_is_truly_padding_invariant():
    """Voi Attention, tru;c giac thong thuong dung: dem phai khong doi vi tri that."""
    m = tiny_model("AAAA")
    short = torch.tensor([[3, 7, 11, 5, 9]])
    padded = torch.cat([short, torch.full((1, 20), PAD, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a = m.hidden_states(short)
        b = m.hidden_states(padded)[:, :short.size(1), :]
    diff = (a - b).abs().max().item()
    assert diff < 1e-5, f"attention le ra phai bat bien voi dem, lech {diff:.3e}"


def test_d3b_hyena_is_NOT_padding_invariant_filter_depends_on_L():
    """PHAT HIEN, khong phai loi: bo loc Hyena tham so hoa tren t = linspace(0,1,L),
    nen doi L la co gian toan bo bo loc va doi ca dau ra tai vi tri THAT.

    Test nay CO Y khang dinh su khac biet ton tai. Neu mot ngay nao do no do vi
    'diff qua nho', nghia la cach tham so hoa bo loc da doi, va phan pad co dinh
    ben duoi co the khong con can thiet - phai xem lai chu khong sua test cho qua.
    """
    m = tiny_model("HHHH")
    short = torch.tensor([[3, 7, 11, 5, 9]])
    with torch.no_grad():
        a = m.hidden_states(short)
        one_pad = m.hidden_states(
            torch.cat([short, torch.full((1, 1), PAD, dtype=torch.long)], 1))[:, :5, :]
        many_pad = m.hidden_states(
            torch.cat([short, torch.full((1, 20), PAD, dtype=torch.long)], 1))[:, :5, :]
    d1 = (a - one_pad).abs().max().item()
    d20 = (a - many_pad).abs().max().item()
    assert d1 > 1e-4, f"chi mot token dem le ra phai lam doi ket qua, lech {d1:.3e}"
    assert d20 > d1, "dem cang nhieu thi lech cang lon"


@pytest.mark.parametrize("layer_spec", ["HHHH", "AAAA"])
def test_d3c_fixed_length_padding_removes_the_dependence(layer_spec):
    """CACH KHAC PHUC: dem moi chuoi ve cung mot do dai co dinh -> lech dung 0.

    Day la ly do `make_batches` luon nhan `pad_to` o duong chay that.
    """
    m = tiny_model(layer_spec)
    L = m.cfg.max_seq_len
    x1 = torch.full((1, L), PAD, dtype=torch.long)
    x1[0, :5] = torch.tensor([3, 7, 11, 5, 9])
    x2 = torch.full((1, L), PAD, dtype=torch.long)
    x2[0, :5] = torch.tensor([3, 7, 11, 5, 9])
    with torch.no_grad():
        d = (m.hidden_states(x1)[:, :5] - m.hidden_states(x2)[:, :5]).abs().max().item()
    assert d == 0.0, f"{layer_spec}: cung L ma van lech {d:.3e}"


@pytest.mark.parametrize("layer_spec", ["HHHH", "AAAA"])
def test_d3d_pooled_vector_invariant_to_batch_mates_when_pad_to_fixed(layer_spec):
    """He qua thuc te: voi pad_to co dinh, vecto gop cua mot cau KHONG doi khi no
    nam chung batch voi cau dai hon. Khong co tinh chat nay thi diem so phu thuoc
    cach chia batch, va so sanh hai kien truc mat y nghia."""
    m = tiny_model(layer_spec)
    clf = Classifier(m, n_classes=3, pooling="last", freeze=True).eval()
    L = m.cfg.max_seq_len

    solo = torch.full((1, L), PAD, dtype=torch.long)
    solo[0, :3] = torch.tensor([3, 7, 11])
    both = torch.full((2, L), PAD, dtype=torch.long)
    both[0, :3] = torch.tensor([3, 7, 11])
    both[1, :6] = torch.tensor([4, 4, 4, 4, 4, 4])
    with torch.no_grad():
        a = clf(solo, torch.tensor([3]))
        b = clf(both, torch.tensor([3, 6]))
    diff = (a[0] - b[0]).abs().max().item()
    assert diff < 1e-6, f"{layer_spec}: ket qua phu thuoc ban cung batch, lech {diff:.3e}"


def test_d3e_make_batches_pads_every_row_to_pad_to():
    ids = [[5], [6, 7, 8], [9, 10]]
    x, lens, _ = next(iter(make_batches(ids, [0, 1, 2], 3, False, pad_to=16)))
    assert x.shape == (3, 16)
    assert lens.tolist() == [1, 3, 2]
    for r, L in enumerate(lens.tolist()):
        assert (x[r, L:] == PAD).all()


def test_d3f_make_batches_rejects_sequence_longer_than_pad_to():
    with pytest.raises(ValueError, match="pad_to"):
        list(make_batches([[1, 2, 3, 4]], [0], 1, False, pad_to=2))


# -----------------------------------------------------------------------------
# D4/D5 - đóng băng và tinh chỉnh
# -----------------------------------------------------------------------------
def _one_step(clf):
    x = torch.tensor([[3, 7, 11, PAD], [4, 5, 6, 8]])
    lens = torch.tensor([3, 4])
    y = torch.tensor([0, 1])
    opt = torch.optim.AdamW([p for p in clf.parameters() if p.requires_grad], lr=0.1)
    loss = torch.nn.functional.cross_entropy(clf(x, lens), y)
    opt.zero_grad(); loss.backward(); opt.step()


def test_d4_probe_leaves_backbone_bitwise_unchanged():
    m = tiny_model("HHHH")
    before = {k: v.clone() for k, v in m.state_dict().items()}
    clf = Classifier(m, 3, "last", freeze=True)
    assert all(not p.requires_grad for p in clf.backbone.parameters())
    _one_step(clf)
    for k, v in clf.backbone.state_dict().items():
        assert torch.equal(v, before[k]), f"probe đã sửa tham số thân: {k}"


def test_d5_finetune_actually_updates_backbone():
    m = tiny_model("HHHH")
    before = m.tok_emb.weight.clone()
    clf = Classifier(m, 3, "last", freeze=False)
    _one_step(clf)
    assert not torch.equal(clf.backbone.tok_emb.weight, before), \
        "finetune mà thân không đổi -> gradient không chảy vào thân"


def test_d5b_head_is_single_linear_layer():
    """Đầu phân loại phải đơn giản; đầu mạnh sẽ che mất chất lượng biểu diễn."""
    clf = Classifier(tiny_model("AAAA"), 4, "mean", freeze=True)
    assert isinstance(clf.head, torch.nn.Linear)
    assert clf.head.out_features == 4


# -----------------------------------------------------------------------------
# D6 - chỉ số
# -----------------------------------------------------------------------------
def test_d6_macro_f1_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    for n_classes in (3, 4):
        for _ in range(5):
            yt = rng.integers(0, n_classes, 200)
            yp = rng.integers(0, n_classes, 200)
            mine = macro_f1(yt, yp, n_classes)
            theirs = sk.f1_score(yt, yp, average="macro",
                                 labels=list(range(n_classes)), zero_division=0)
            assert abs(mine - theirs) < 1e-9, (mine, theirs)


def test_d6b_macro_f1_penalises_ignoring_a_rare_class():
    """Vì sao macro-F1 là chỉ số chính: đoán toàn lớp đông vẫn có accuracy cao."""
    yt = np.array([0] * 96 + [1] * 4)
    yp = np.zeros(100, dtype=int)
    assert (yp == yt).mean() == 0.96
    assert macro_f1(yt, yp, 2) < 0.5


def test_d7_majority_baseline():
    out = majority_baseline([0, 0, 0, 1], [0, 0, 1, 1], 2)
    assert out["predicted_class"] == 0
    assert out["accuracy"] == 0.5
    assert out["macro_f1"] == macro_f1(np.array([0, 0, 1, 1]), np.zeros(4, int), 2)


# -----------------------------------------------------------------------------
# D8 - tạo batch
# -----------------------------------------------------------------------------
def test_d8_batches_pad_right_and_report_true_lengths():
    ids = [[5], [6, 7, 8], [9, 10]]
    batches = list(make_batches(ids, [0, 1, 2], batch_size=3, shuffle=False))
    x, lens, y = batches[0]
    assert x.shape == (3, 3)
    assert lens.tolist() == [1, 3, 2]
    assert y.tolist() == [0, 1, 2]
    assert x[0].tolist() == [5, PAD, PAD]      # đệm nằm BÊN PHẢI
    for r, L in enumerate(lens.tolist()):
        assert (x[r, L:] == PAD).all()


def test_d8b_shuffle_is_seed_reproducible_and_requires_rng():
    ids, y = [[i] for i in range(10)], list(range(10))
    a = [b[2].tolist() for b in make_batches(ids, y, 4, True, np.random.default_rng(0))]
    b = [b[2].tolist() for b in make_batches(ids, y, 4, True, np.random.default_rng(0))]
    c = [b[2].tolist() for b in make_batches(ids, y, 4, True, np.random.default_rng(1))]
    assert a == b and a != c
    with pytest.raises(ValueError, match="rng"):
        list(make_batches(ids, y, 4, shuffle=True))


def test_d8c_every_example_appears_exactly_once():
    ids, y = [[i] for i in range(37)], list(range(37))
    seen = [v for b in make_batches(ids, y, 8, True, np.random.default_rng(3))
            for v in b[2].tolist()]
    assert sorted(seen) == list(range(37))


# -----------------------------------------------------------------------------
# D9 - token hoá + thống kê
# -----------------------------------------------------------------------------
def test_d9_encode_reports_oov_truncation_and_empty():
    tok = SyllableTokenizer(VOCAB)
    sents = ["w0 w1 w2", "tuhoantoanla khong co", "w3 w4 w5 w6 w7", "   "]
    ids, y, st = encode_dataset(sents, [0, 1, 0, 1], tok, max_len=3)
    assert st["n_examples"] == 4
    assert st["n_empty"] == 1                      # câu chỉ có khoảng trắng
    assert st["n_truncated"] == 1                  # câu 5 token bị cắt còn 3
    assert all(len(i) <= 3 for i in ids)
    assert st["oov_rate"] > 0                      # câu 2 toàn từ ngoài từ điển
    assert st["len_max"] == 3
    assert y == [0, 1, 0, 1]


def test_d9b_encode_never_yields_empty_sequence():
    """Chuỗi rỗng sẽ làm vỡ pooling (lens-1 = -1). Phải chặn từ khâu token hoá."""
    tok = SyllableTokenizer(VOCAB)
    ids, _, _ = encode_dataset(["", "  ", "..."], [0, 0, 0], tok, max_len=8)
    assert all(len(i) >= 1 for i in ids)
    assert ids[0] == [UNK]


# -----------------------------------------------------------------------------
# D10 - tổng hợp và so sánh
# -----------------------------------------------------------------------------
def test_d10_summarize_uses_t_distribution_and_refuses_to_invent_ci():
    one = summarize([{"m": 0.5}], "m")
    assert one["ci95"] is None and one["std"] is None    # n=1: không bịa ra 0
    three = summarize([{"m": 0.1}, {"m": 0.2}, {"m": 0.3}], "m")
    assert abs(three["mean"] - 0.2) < 1e-12
    expected = 4.303 * float(np.std([0.1, 0.2, 0.3], ddof=1)) / np.sqrt(3)
    assert abs(three["ci95"] - expected) < 1e-12


def test_d10b_overlapping_intervals_give_no_conclusion():
    a = {"name": "Hyena", "runs": [{"m": v} for v in (0.50, 0.52, 0.54)]}
    b = {"name": "Transformer", "runs": [{"m": v} for v in (0.51, 0.53, 0.55)]}
    out = compare_arms(a, b, "m")
    assert out["separated"] is False
    assert "chồng lấn" in out["verdict"]
    assert "tốt hơn" not in out["verdict"]


def test_d10c_separated_intervals_state_the_scale_caveat():
    a = {"name": "Hyena", "runs": [{"m": v} for v in (0.90, 0.901, 0.902)]}
    b = {"name": "Transformer", "runs": [{"m": v} for v in (0.50, 0.501, 0.502)]}
    out = compare_arms(a, b, "m")
    assert out["separated"] is True
    assert "Hyena" in out["verdict"]
    # cấm phát biểu trần trụi "Hyena tốt hơn Transformer"
    assert "cùng quy mô này" in out["verdict"]


# -----------------------------------------------------------------------------
# D11 - lm.py: hidden_states nhất quán với forward
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("layer_spec", ["HHHH", "AAAA"])
def test_d11_hidden_states_then_head_equals_forward(layer_spec):
    m = tiny_model(layer_spec)
    x = torch.tensor([[3, 7, 11, 5]])
    with torch.no_grad():
        assert torch.equal(m.lm_head(m.hidden_states(x)), m(x))


# -----------------------------------------------------------------------------
# D12 - đường dây đầu-cuối (không cần mạng)
# -----------------------------------------------------------------------------
def test_d12_smoke_pipeline_writes_valid_json(tmp_path, monkeypatch):
    """Chạy trọn `main(--smoke)` với checkpoint tí hon tự tạo. Đây là test bắt
    được lỗi ráp nối — loại lỗi đã làm hỏng bốn lần chạy Kaggle trước đây."""
    from hyena_study.checkpoint import save_checkpoint

    tok = SyllableTokenizer(VOCAB)
    paths = []
    for spec in ("HHHH", "AAAA"):
        cfg = tiny_cfg(spec)
        model = SequenceLM(cfg)
        p = tmp_path / f"tiny_{spec}.pt"
        save_checkpoint(p, model=model, cfg=cfg, tok=tok,
                        tokenizer_kind="syllable", run_name=f"TINY_{spec}",
                        metrics={"test_ppl": 123.0})
        paths.append(str(p))

    out = tmp_path / "ds.json"
    rc = main(["--smoke", "--ckpt", paths[0], "--compare", paths[1],
               "--task", "sentiment", "--mode", "probe", "--pooling", "last",
               "--out", str(out), "--device", "cpu", "--batch_size", "16"])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["smoke"] is True
    assert payload["dataset"]["repo"] == "uitnlp/vietnamese_students_feedback"
    arms = {r["arm"] for r in payload["runs"]}
    assert arms == {"Hyena-pretrained", "Transformer-pretrained",
                    "Hyena-random", "Transformer-random"}, arms
    for r in payload["runs"]:
        assert 0.0 <= r["test_macro_f1"] <= 1.0
        assert 0.0 <= r["test_accuracy"] <= 1.0
        assert r["majority_test_accuracy"] > 0
    assert payload["comparisons"], "thiếu phần so sánh hai nhánh tiền huấn luyện"


def test_d13_missing_checkpoint_fails_loudly(tmp_path, capsys):
    rc = main(["--ckpt", str(tmp_path / "khong_ton_tai.pt"), "--compare", "",
               "--out", str(tmp_path / "x.json")])
    assert rc == 1
    assert "KHONG THAY CHECKPOINT" in capsys.readouterr().out


# -----------------------------------------------------------------------------
# D14 - dữ liệu thật (bỏ qua nếu không có mạng / chưa tải)
# -----------------------------------------------------------------------------
def test_d14_real_vsfc_split_sizes_if_available():
    """Chốt số dòng của UIT-VSFC. Nếu bản trên HuggingFace bị đổi, test này đỏ và
    mọi con số trong báo cáo phải được xem lại."""
    from hyena_study.downstream import load_vsfc
    try:
        data = load_vsfc("data_cache/vsfc", "sentiment")
    except Exception as exc:                      # noqa: BLE001
        pytest.skip(f"không tải được VSFC: {type(exc).__name__}")
    assert len(data["train"][0]) == 11426
    assert len(data["validation"][0]) == 1583
    assert len(data["test"][0]) == 3166
    assert set(data["train"][1]) == {0, 1, 2}

# -----------------------------------------------------------------------------
# D15 - bo nho dem dac trung
# -----------------------------------------------------------------------------
def _tiny_ckpt(tmp_path, spec):
    from hyena_study.checkpoint import save_checkpoint
    from hyena_study.data.corpus import SyllableTokenizer
    cfg = tiny_cfg(spec)
    pth = tmp_path / f"c_{spec}.pt"
    save_checkpoint(pth, model=SequenceLM(cfg), cfg=cfg,
                    tok=SyllableTokenizer(VOCAB), tokenizer_kind="syllable",
                    run_name=f"C_{spec}", metrics={"test_ppl": 1.0})
    return str(pth)


def _mini_data(n_classes=3, n=48):
    rng = np.random.default_rng(0)
    out = {}
    for split, m in zip(("train", "validation", "test"), (n, n // 2, n // 2)):
        y = rng.integers(0, n_classes, size=m).tolist()
        out[split] = ([" ".join(["w" + str(c), "w9", "w8"]) for c in y], y)
    return out


def test_d15_random_control_seeds_do_not_share_cached_features(tmp_path):
    """Than 'random' khac nhau o moi seed. Neu bo nho dem tra ve cung dac trung
    cho ca ba seed thi do lech cua duong doi chung bi hep lai MOT CACH GIA TAO -
    va ket luan 'tien huan luyen co ich' se dua tren khoang tin cay sai."""
    from hyena_study.checkpoint import load_checkpoint
    from hyena_study.downstream import run_one

    ck = _tiny_ckpt(tmp_path, "HHHH")
    _, tok, _ = load_checkpoint(ck, device="cpu")
    data = _mini_data()
    cache: dict = {}
    kw = dict(data=data, tok=tok, ckpt=ck, task="sentiment", mode="probe",
              pooling="last", epochs=2, batch_size=16, lr=1e-2,
              weight_decay=0.0, grad_clip=1.0, max_len=16, device="cpu",
              pad_to=16, feature_cache=cache, log=False)

    for seed in (0, 1, 2):
        run_one(init="random", seed=seed, **kw)
    assert len(cache) == 3, f"ba seed ngau nhien phai cho ba muc dem, co {len(cache)}"

    before = len(cache)
    for seed in (0, 1, 2):
        run_one(init="pretrained", seed=seed, **kw)
    assert len(cache) == before + 1, \
        "than da tien huan luyen la co dinh -> ba seed phai DUNG CHUNG mot muc dem"


def test_d15b_cached_probe_matches_uncached(tmp_path):
    """Dung bo nho dem phai cho ket qua GIONG HET khong dung - neu khong thi toi
    uu hoa nay da lam sai lech ket qua."""
    from hyena_study.checkpoint import load_checkpoint
    from hyena_study.downstream import run_one

    ck = _tiny_ckpt(tmp_path, "AAAA")
    _, tok, _ = load_checkpoint(ck, device="cpu")
    data = _mini_data()
    kw = dict(data=data, tok=tok, ckpt=ck, init="pretrained", task="sentiment",
              mode="probe", pooling="last", seed=0, epochs=3, batch_size=16,
              lr=1e-2, weight_decay=0.0, grad_clip=1.0, max_len=16,
              device="cpu", pad_to=16, log=False)
    a = run_one(feature_cache=None, **kw)
    b = run_one(feature_cache={}, **kw)
    for k in ("test_macro_f1", "test_accuracy", "val_macro_f1"):
        assert a[k] == b[k], f"{k}: co dem {b[k]} != khong dem {a[k]}"

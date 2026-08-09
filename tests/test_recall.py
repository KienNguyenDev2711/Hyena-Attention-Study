"""
Kiem dinh tac vu ASSOCIATIVE RECALL truoc khi dung no de ket luan bat cu dieu gi.

Neu bo qua buoc nay, mot ket qua 0% se hoan toan mo ho: khong the phan biet
"mo hinh khong nho duoc" voi "tac vu bi cai sai nen khong ai giai duoc". Ma ket
qua 0% lai chinh la thu ta dang di tim (de chung minh bo loc qua ngan lam mat kha
nang tam xa), nen no PHAI dien giai duoc mot cach chac chan.

  R1. Tac vu DUNG DINH NGHIA: gia tri dung luon truy nguoc duoc tu chuoi.
  R2. Anh xa khoa -> gia tri don tri trong tung chuoi (khong mau thuan).
  R3. Khoang cach truy van tinh dung.
  R4. Muc doan mo dung bang 1/V (moc so sanh bat buoc cua moi bang ket qua).
  R5. TAC VU GIAI DUOC: mot mo hinh attention nho hoc duoc no vuot xa muc doan mo.

Chay: python tests/test_recall.py   (CPU, khong can Internet)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyena_study.data.synthetic import (  # noqa: E402
    RecallConfig,
    build_recall_dataset,
    chance_accuracy,
    make_recall_split,
    query_distance,
)
from hyena_study.models import LMConfig, SequenceLM  # noqa: E402


class SkipTest(Exception):
    """Test bi bo qua co chu dich (khong phai that bai)."""


def _heavy_allowed() -> bool:
    """Cho phep chay cac test CO HUAN LUYEN THUC SU hay khong.

    QUY TAC CUA DU AN: moi viec huan luyen chay tren Kaggle GPU, KHONG chay tren
    may ca nhan. Cac test nhe (kiem tinh dung dan, hinh dang du lieu) van chay
    duoc o dau cung vi chi ton vai giay; nhung test nay huan luyen that su hang
    nghin buoc gradient va se lam nong may.

    Nen mac dinh no CHI chay khi:
      - co GPU (tuc dang o tren Kaggle), hoac
      - nguoi dung ep bang co --heavy.

    De mac dinh la "chay" thi som muon cung co nguoi vo tinh nau CPU may minh -
    da xay ra that ngay 2026-08-08.
    """
    return torch.cuda.is_available() or "--heavy" in sys.argv


# -----------------------------------------------------------------------------
# R1/R2 - tac vu co dung dinh nghia khong
# -----------------------------------------------------------------------------
def test_target_is_recoverable_from_sequence():
    """Gia tri dung phai truy nguoc duoc: tim khoa truy van trong chuoi, lay
    token ngay sau no. Neu khong khop thi tac vu KHONG GIAI DUOC."""
    cfg = RecallConfig(vocab_size=15, seq_len=41, n_train=0, n_val=0, n_test=0)
    rng = np.random.default_rng(0)
    x, y = make_recall_split(cfg, 500, rng)

    assert x.shape[1] == 2 * cfg.n_pairs + 1
    for i in range(len(x)):
        q = x[i, -1]
        pos = [p for p in range(0, 2 * cfg.n_pairs, 2) if x[i, p] == q]
        assert pos, f"mau {i}: khoa truy van khong he xuat hien truoc do"
        vals = {int(x[i, p + 1]) for p in pos}
        assert len(vals) == 1, f"mau {i}: cung mot khoa di kem {len(vals)} gia tri khac nhau"
        assert vals.pop() == int(y[i]), f"mau {i}: nhan khong khop gia tri trong chuoi"
    return len(x)


def test_mapping_is_single_valued_within_sequence():
    """Trong cung mot chuoi, moi khoa chi duoc ung voi dung mot gia tri."""
    cfg = RecallConfig(vocab_size=8, seq_len=61)
    rng = np.random.default_rng(1)
    x, _ = make_recall_split(cfg, 300, rng)
    for i in range(len(x)):
        seen: dict[int, int] = {}
        for p in range(0, 2 * cfg.n_pairs, 2):
            k, v = int(x[i, p]), int(x[i, p + 1])
            if k in seen:
                assert seen[k] == v, f"mau {i}: khoa {k} ung voi ca {seen[k]} va {v}"
            seen[k] = v
    return True


def test_query_distance_is_correct():
    """Khoang cach phai dung bang so token tu truy van nguoc ve lan xuat hien cuoi."""
    cfg = RecallConfig(vocab_size=10, seq_len=21)
    rng = np.random.default_rng(2)
    x, _ = make_recall_split(cfg, 200, rng)
    dist = query_distance(x)

    assert (dist > 0).all(), "co mau khong tim thay khoa truy van"
    for i in range(len(x)):
        pos = (len(x[i]) - 1) - dist[i]
        assert x[i, pos] == x[i, -1], f"mau {i}: khoang cach chi sai vi tri"
        # phai la lan xuat hien CUOI CUNG
        later = [p for p in range(int(pos) + 2, 2 * cfg.n_pairs, 2) if x[i, p] == x[i, -1]]
        assert not later, f"mau {i}: van con lan xuat hien gan hon"
    return float(dist.mean())


def test_chance_accuracy_matches_empirical():
    """Muc doan mo ly thuyet 1/V phai khop voi thuc nghiem."""
    cfg = RecallConfig(vocab_size=20, seq_len=41)
    rng = np.random.default_rng(3)
    _, y = make_recall_split(cfg, 20000, rng)
    counts = np.bincount(y, minlength=cfg.vocab_size) / len(y)
    assert abs(counts.max() - chance_accuracy(cfg)) < 0.01, (
        f"nhan khong phan bo deu: max {counts.max():.4f} vs ly thuyet "
        f"{chance_accuracy(cfg):.4f} - muc doan mo se sai"
    )
    return chance_accuracy(cfg)


def test_splits_have_right_shapes():
    cfg = RecallConfig(vocab_size=12, seq_len=33, n_train=100, n_val=50, n_test=40)
    data = build_recall_dataset(cfg)
    L = 2 * cfg.n_pairs + 1
    for name, n in (("train", 100), ("val", 50), ("test", 40)):
        x, y = data[name]
        assert x.shape == (n, L), f"{name}: shape x sai {x.shape}"
        assert y.shape == (n,), f"{name}: shape y sai {y.shape}"
        assert x.max() < cfg.vocab_size and y.max() < cfg.vocab_size
    return True


def test_filler_isolates_distance_from_task_difficulty():
    """Token dem phai keo dai khoang cach MA KHONG doi do kho cua phep tra cuu.

    Day la ly do ky thuat de co `n_pairs_fixed`. Neu de so cap tang theo do dai
    chuoi thi moi lan tang L ta doi dong thoi ca khoang cach LAN so cap phai ghi
    nho, nen khong quy duoc ket qua cho truc nao. Da mac dung loi do ngay
    2026-08-08 va lam hong ca mot lan chay Kaggle.
    """
    P, V = 8, 10
    rng = np.random.default_rng(0)
    lengths = [2 * P + 1, 2 * P + 33, 2 * P + 97]
    means = []
    for L in lengths:
        cfg = RecallConfig(vocab_size=V, seq_len=L, n_pairs_fixed=P)
        assert cfg.n_pairs == P, "so cap phai giu nguyen khi keo dai chuoi"
        assert cfg.n_filler == L - 2 * P - 1
        x, y = make_recall_split(cfg, 300, rng)
        assert x.shape[1] == L, f"do dai chuoi sai: {x.shape[1]} != {L}"

        # tac vu van dung dinh nghia du co dem
        for i in range(len(x)):
            q = x[i, -1]
            pos = [p for p in range(0, 2 * P, 2) if x[i, p] == q]
            assert pos, f"mau {i}: khoa truy van bien mat"
            assert int(x[i, pos[-1] + 1]) == int(y[i]), f"mau {i}: nhan sai"

        d = query_distance(x, n_pairs=P)
        assert (d > 0).all(), f"L={L}: co mau khong tinh duoc khoang cach"
        means.append(float(d.mean()))

    # khoang cach phai tang gan dung bang phan dem them vao
    for (L1, m1), (L2, m2) in zip(zip(lengths, means), zip(lengths[1:], means[1:])):
        grew, expected = m2 - m1, L2 - L1
        assert abs(grew - expected) < 1.0, (
            f"chen them {expected} token dem nhung khoang cach chi tang {grew:.1f}"
        )
    return means[-1]


# -----------------------------------------------------------------------------
# R5 - tac vu CO GIAI DUOC khong
# -----------------------------------------------------------------------------
def test_task_is_actually_solvable_by_attention():
    """Mot mo hinh attention nho phai giai duoc tac vu nay vuot xa muc doan mo.

    Day la test quan trong nhat cua ca file. Muc tieu cua E6 la phat hien mo hinh
    KHONG nho duoc xa; ket qua do chi dien giai duoc neu ta da chung minh tac vu
    tu no giai duoc. Attention la moc doi chieu vi no truy cap truc tiep moi vi
    tri, dung ra phai giai gon tac vu nay.

    TEST NANG - CHI CHAY TREN GPU (Kaggle). Xem `_heavy_allowed`.
    """
    if not _heavy_allowed():
        raise SkipTest(
            "test nang (huan luyen thuc su) - bo qua tren may khong co GPU.\n"
            "         Chay tren Kaggle GPU, hoac ep chay bang: "
            "python tests/test_recall.py --heavy"
        )

    torch.manual_seed(0)
    np.random.seed(0)

    # CONG THUC HUAN LUYEN DUOC CHOT BANG THUC NGHIEM, khong phai doan.
    # Quet tren Kaggle T4 ngay 2026-08-08 (notebooks/kaggle_E6_recall.ipynb):
    #   vocab=10, L=33, AA,   744 buoc, CO warmup    -> 0,194  (doan mo 0,100)
    #   vocab=10, L=33, AA, 18720 buoc, CO warmup    -> 1,000
    #   vocab=10, L=33, AA, 18720 buoc, KHONG warmup -> 0,187
    #
    # Hai dong cuoi chi khac nhau DUNG MOT bien: lich warmup. Ban dau bo test
    # nay chi copy so epoch va learning rate ma bo quen warmup, nen no that bai
    # o 0,187 trong khi cung ngan sach do dat 1,000 o notebook. AdamW voi
    # lr=1e-3 ngay tu buoc 0 pha vo cau truc tra cuu truoc khi no kip hinh thanh.
    EPOCHS, N_TRAIN = 60, 20000            # = 18720 buoc, ~90 giay tren T4
    WARMUP_FRAC = 0.05
    cfg = RecallConfig(vocab_size=10, seq_len=33, n_train=N_TRAIN, n_val=1000,
                       n_test=1000, seed=0)
    data = build_recall_dataset(cfg)
    xtr, ytr = data["train"]
    xte, yte = data["test"]
    L = xtr.shape[1]

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceLM(LMConfig(
        vocab_size=cfg.vocab_size, d_model=64, layer_spec="AA",
        max_seq_len=L, dropout=0.0, n_heads=4,
    )).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    bs = 64
    total_steps = EPOCHS * (len(xtr) // bs)
    warmup = max(int(total_steps * WARMUP_FRAC), 1)
    step = 0
    for epoch in range(EPOCHS):
        perm = np.random.permutation(len(xtr))
        for i in range(0, len(xtr) - bs + 1, bs):
            idx = perm[i:i + bs]
            xb = torch.from_numpy(xtr[idx]).to(dev)
            yb = torch.from_numpy(ytr[idx]).to(dev)
            for g in opt.param_groups:      # warmup - BAT BUOC, xem ghi chu tren
                g["lr"] = 1e-3 * min(1.0, (step + 1) / warmup)
            step += 1
            loss = F.cross_entropy(model(xb)[:, -1, :], yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xte).to(dev))[:, -1, :].argmax(-1).cpu().numpy()
    acc = float((pred == yte).mean())
    chance = chance_accuracy(cfg)

    assert acc > chance + 0.30, (
        f"attention chi dat {acc:.3f} so voi muc doan mo {chance:.3f}. "
        "Tac vu co the bi cai sai hoac ngan sach huan luyen qua nho - KHONG duoc "
        "dung E6 de ket luan ve kha nang tam xa cho toi khi moc doi chieu nay dat."
    )
    return acc


def test_unique_keys_spreads_query_distance():
    """Voi unique_keys=True, khoang cach truy van phai trai deu tren toan chuoi.

    Day la ly do ky thuat de co lua chon do. Khi lay khoa CO HOAN LAI tu bang chu
    cai nho, chuoi cang dai thi moi khoa cang xuat hien nhieu lan, nen ban sao gan
    nhat cang GAN - tang do dai chuoi lai lam GIAM khoang cach can nho. Do la do
    sai dai luong khi muon danh gia kha nang tam xa.
    """
    P = 24
    L = 2 * P + 1
    rng = np.random.default_rng(0)

    rep = make_recall_split(RecallConfig(vocab_size=10, seq_len=L), 400, rng)[0]
    uniq = make_recall_split(
        RecallConfig(vocab_size=64, seq_len=L, unique_keys=True), 400, rng)[0]

    d_rep = query_distance(rep)
    d_uniq = query_distance(uniq)
    assert (d_uniq > 0).all(), "unique_keys: co mau khong tim thay khoa truy van"

    # Kiem bang GIA TRI LY THUYET chu khong bang mot ty le tuy tien:
    # moi khoa xuat hien dung mot lan tai vi tri chan 0,2,...,2P-2, truy van chon
    # deu trong so do, nen khoang cach nhan cac gia tri 2,4,...,2P deu nhau
    # => trung binh = P + 1.
    expected = P + 1
    assert abs(d_uniq.mean() - expected) < 3.0, (
        f"unique_keys: khoang cach trung binh {d_uniq.mean():.1f}, "
        f"ly thuyet {expected} - phan bo khong deu nhu mong doi"
    )
    assert d_uniq.max() >= 2 * P - 2, (
        f"unique_keys: khoang cach xa nhat chi {d_uniq.max()}, chua phu het chuoi"
    )
    # va phai dai hon han so voi lay co hoan lai (do la ca ly do ton tai cua co nay)
    assert d_uniq.mean() > 1.5 * d_rep.mean(), (
        f"unique_keys khong trai duoc khoang cach: trung binh {d_uniq.mean():.1f} "
        f"so voi {d_rep.mean():.1f} khi lay co hoan lai"
    )
    # moi khoa dung mot lan
    for i in range(len(uniq)):
        keys = uniq[i, 0:2 * P:2]
        assert len(set(keys.tolist())) == P, f"mau {i}: khoa bi lap du unique_keys=True"
    return float(d_uniq.mean())


# -----------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("R1  Gia tri dung truy nguoc duoc", test_target_is_recoverable_from_sequence),
        ("R2  Anh xa don tri trong chuoi", test_mapping_is_single_valued_within_sequence),
        ("R3  Khoang cach truy van dung", test_query_distance_is_correct),
        ("R4  Muc doan mo = 1/V", test_chance_accuracy_matches_empirical),
        ("R4b Shape cac tap du lieu", test_splits_have_right_shapes),
        ("R4c unique_keys trai deu khoang cach", test_unique_keys_spreads_query_distance),
        ("R4d Token dem co lap duoc khoang cach", test_filler_isolates_distance_from_task_difficulty),
        ("R5  TAC VU GIAI DUOC (attention)", test_task_is_actually_solvable_by_attention),
    ]
    n_fail = n_skip = 0
    for name, fn in tests:
        try:
            out = fn()
            extra = f"  ({out:.4f})" if isinstance(out, float) else (
                f"  ({out})" if isinstance(out, int) and not isinstance(out, bool) else "")
            print(f"  [PASS] {name}{extra}")
        except SkipTest as exc:
            n_skip += 1
            print(f"  [BO QUA] {name}\n         {exc}")
        except AssertionError as exc:
            n_fail += 1
            print(f"  [FAIL] {name}\n         {exc}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [LOI ] {name}\n         {type(exc).__name__}: {exc}")
    print()
    ran = len(tests) - n_skip
    if n_fail:
        print(f"==> {n_fail}/{ran} test THAT BAI" + (f", {n_skip} bo qua" if n_skip else ""))
    else:
        print(f"==> Toan bo {ran} test da chay deu DAT"
              + (f", {n_skip} bo qua (can GPU)" if n_skip else ""))
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())

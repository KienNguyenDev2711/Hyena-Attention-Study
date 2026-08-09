"""
Test DUONG DAY cua script thi nghiem E6 - loai test da thieu suot ba lan hong.

LY DO TON TAI:

Ba lan chay Kaggle truoc deu hong, va ca ba loi deu nam trong o notebook:
  1. `solvable[-1]` lay cau hinh de nhat trong khi comment ghi "kho nhat"
  2. bo quen lich warmup khi bung ngan sach tu notebook sang test
  3. quet do dai dat vocab = n_pairs -> doi ba truc kho cung luc

Khong test nao bat duoc chung, vi code trong o notebook KHONG duoc test chay
qua. Sau khi dua logic ra module `hyena_study/e6_recall.py`, bo test nay chay
toan bo ba pha o che do --smoke trong vai giay va se bat ngay cac loi kieu:
tham so truyen sai, ham doi ten, chon sai phan tu, ghi file hong.

Test nay KHONG kiem tinh dung dan khoa hoc (ngan sach smoke qua nho de mo hinh
hoc duoc gi). No chi tra loi mot cau: "script co chay tron tu dau den cuoi
va sinh ra dung cac file khong?"

Chay: python tests/test_e6_script.py    (CPU, vai giay, khong can GPU)
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyena_study.e6_recall import main, run_one, solved  # noqa: E402


class _Cfg:
    """Cau hinh toi thieu de goi run_one truc tiep."""

    vocab_size, n_pairs, d_model, n_heads = 8, 4, 16, 4
    steps, n_train, n_test, batch_size, lr = 10, 128, 32, 32, 1e-3


def test_run_one_returns_expected_fields():
    """run_one phai tra ve du cac truong ma cac pha sau dung den.

    Thieu mot truong se lam ca script chet giua chung sau khi da tieu GPU.
    """
    r = run_one(_Cfg(), n_filler=0, layers="AA")
    need = {"n_filler", "seq_len", "layers", "seed", "accuracy", "chance",
            "acc_near", "acc_far", "dist_mean", "dist_max", "steps", "seconds"}
    missing = need - set(r)
    assert not missing, f"run_one thieu truong: {sorted(missing)}"
    assert r["seq_len"] == 2 * _Cfg.n_pairs + 1
    assert 0.0 <= r["accuracy"] <= 1.0
    assert r["chance"] == 1.0 / _Cfg.vocab_size
    return r["accuracy"]


def test_filler_changes_only_length_not_pair_count():
    """Them token dem phai keo dai chuoi va khoang cach, KHONG doi so cap.

    Day chinh la loi da lam hong lan chay thu hai: so cap tang theo do dai nen
    khong quy duoc ket qua cho truc nao.
    """
    a = run_one(_Cfg(), n_filler=0, layers="AA")
    b = run_one(_Cfg(), n_filler=16, layers="AA")
    assert b["seq_len"] - a["seq_len"] == 16, "do dai khong tang dung phan dem"
    assert b["dist_mean"] - a["dist_mean"] > 15.0, (
        f"khoang cach chi tang {b['dist_mean'] - a['dist_mean']:.1f} khi chen 16 dem"
    )
    assert a["chance"] == b["chance"], "muc doan mo doi => do kho da doi theo"
    return b["dist_mean"] - a["dist_mean"]


def test_solved_threshold():
    assert solved({"accuracy": 0.5, "chance": 0.1})
    assert not solved({"accuracy": 0.35, "chance": 0.1})
    assert not solved({"accuracy": 0.1, "chance": 0.1})
    return True


def test_scout_picks_the_hardest_not_the_last():
    """Chon muc kho nhat = do dem LON NHAT, khong phai phan tu cuoi danh sach.

    Lan chay dau tien hong dung vi cho nay: `solvable[-1]` lay phan tu cuoi theo
    thu tu duyet, vo tinh trung cau hinh de nhat, khien ca bon cau hinh bao hoa.
    """
    rows = [{"n_filler": 0, "solvable": True},
            {"n_filler": 96, "solvable": True},
            {"n_filler": 32, "solvable": True}]
    ok = [r for r in rows if r["solvable"]]
    target = max(ok, key=lambda r: r["n_filler"])
    assert target["n_filler"] == 96, (
        f"chon nham muc {target['n_filler']} thay vi 96 - dung lai loi cu"
    )
    assert ok[-1]["n_filler"] != 96, "vi du phai co phan tu cuoi KHAC muc kho nhat"
    return True


def test_full_script_runs_end_to_end():
    """Chay ca ba pha o che do smoke va kiem cac file ket qua duoc sinh ra.

    Day la test quan trong nhat: no thuc thi dung duong ma nguoi dung se chay
    tren Kaggle, chi khac ngan sach.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        code = main(["--smoke", "--skip_phase0", "--out_dir", str(tmp),
                     "--mi_csv", str(tmp / "khong_ton_tai.csv")])
        assert code == 0, f"script tra ve ma loi {code}"

        for name in ("E6_scout_attention.csv", "E6_recall_comparison.csv",
                     "E6_meta.json"):
            f = tmp / name
            assert f.exists(), f"thieu file ket qua {name}"
            assert f.stat().st_size > 0, f"file {name} rong"

        with (tmp / "E6_recall_comparison.csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert rows, "bang so sanh rong"
        names = {r["config"] for r in rows}
        # khong co file MI thi cau hinh corpus bi bo qua co chu dich
        assert "hyena_corpus" not in names, (
            "phai bo qua corpus khi thieu file MI, thay vi chet giua chung"
        )
        assert {"attention", "hyena_uniform", "hyena_logspace"} <= names, (
            f"thieu cau hinh trong bang: {sorted(names)}"
        )
        return len(rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_configs_filter_selects_and_rejects():
    """--configs phai loc dung cau hinh, va BAO LOI khi ten sai.

    Neu ten sai bi bo qua am tham thi nguoi dung tuong da chay cau hinh minh
    muon trong khi thuc te khong chay gi - dung loai loi im lang da lam hong
    ba lan chay truoc.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        code = main(["--smoke", "--skip_phase0", "--out_dir", str(tmp),
                     "--mi_csv", str(tmp / "khong_ton_tai.csv"),
                     "--configs", "hyena_uniform"])
        assert code == 0
        with (tmp / "E6_recall_comparison.csv").open(encoding="utf-8") as fh:
            names = {r["config"] for r in csv.DictReader(fh)}
        assert names == {"hyena_uniform"}, f"loc sai, con lai: {sorted(names)}"

        # ten sai phai bao loi to, khong duoc im lang
        try:
            main(["--smoke", "--skip_phase0", "--out_dir", str(tmp),
                  "--mi_csv", str(tmp / "khong_ton_tai.csv"),
                  "--configs", "hyena_khong_ton_tai"])
        except SystemExit:
            pass
        else:
            raise AssertionError("ten cau hinh sai ma khong bao loi")
        return len(names)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_phase0_runs_and_writes():
    """Pha 0 phai chay va ghi file, ke ca khi ngan sach smoke khong du de dat."""
    tmp = Path(tempfile.mkdtemp())
    try:
        main(["--smoke", "--out_dir", str(tmp),
              "--mi_csv", str(tmp / "khong_ton_tai.csv")])
        f = tmp / "E6_phase0_warmup.csv"
        assert f.exists(), "pha 0 khong ghi file ket qua"
        with f.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2, f"pha 0 phai co dung 2 dong, dang co {len(rows)}"
        assert {r["warmup"] for r in rows} == {"co (5%)", "khong"}
        return len(rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------------
def main_runner() -> int:
    tests = [
        ("S1 run_one tra ve du truong", test_run_one_returns_expected_fields),
        ("S2 Dem chi doi do dai, khong doi do kho", test_filler_changes_only_length_not_pair_count),
        ("S3 Nguong 'giai duoc'", test_solved_threshold),
        ("S4 Chon muc KHO NHAT, khong phai cuoi danh sach", test_scout_picks_the_hardest_not_the_last),
        ("S5 Ca script chay tron tu dau den cuoi", test_full_script_runs_end_to_end),
        ("S6 Pha 0 chay va ghi file", test_phase0_runs_and_writes),
        ("S7 --configs loc dung va bao loi ten sai", test_configs_filter_selects_and_rejects),
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
            print(f"  [LOI ] {name}\n         {type(exc).__name__}: {exc}")
    print()
    print("==> " + (f"Toan bo {len(tests)} test DAT" if not n_fail
                    else f"{n_fail}/{len(tests)} test THAT BAI"))
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main_runner())

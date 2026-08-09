"""
Test script tong hop ket qua, tap trung vao cai bay thong ke nguy hiem nhat.

E6 da cho thay du lieu that co the LUONG CUC: `hyena_corpus` chay 5 seed ra
0,092 · 0,141 · 0,922 · 0,981 · 0,987. Hai cum tach han, khong co gia tri trung
gian. Voi phan bo do, cau "trung binh 0,62 +/- 0,46" la NOI DOI: khong lan chay
nao dat gan 0,62.

Neu script lang le in ra trung binh do, no se di thang vao bao cao va khong ai
phat hien. Nen phan lon bo test nay danh cho `detect_bimodal`:
  A1. Nhan ra du lieu E6 THAT la luong cuc.
  A2. KHONG bao dong gia tren du lieu binh thuong (PPL sat nhau).
  A3. Tra ve ty le thanh cong dung.
  A4. Khong ket luan luong cuc khi qua it quan sat.
  A5. Khoang tin cay dung phan phoi t, khong phai chuan tac.

Chay: python tests/test_analyze.py    (CPU, vai giay)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyena_study.analyze import (  # noqa: E402
    detect_bimodal,
    group_key,
    latex_table,
    load_lm_runs,
    main,
    mean_ci,
    summarise,
)

# So lieu THAT do tren Kaggle T4, 2026-08-08 va 2026-08-09
E6_CORPUS = [0.092, 0.141, 0.922, 0.981, 0.987]


# -----------------------------------------------------------------------------
def test_detects_the_real_bimodal_case():
    """Phai nhan ra du lieu E6 that la luong cuc va tra ve ty le thanh cong dung."""
    b = detect_bimodal(E6_CORPUS)
    assert b["bimodal"], f"khong nhan ra luong cuc tren du lieu E6 that: {b}"
    assert b["cluster_low"]["n"] == 2, f"cum thap phai co 2 diem: {b['cluster_low']}"
    assert b["cluster_high"]["n"] == 3, f"cum cao phai co 3 diem: {b['cluster_high']}"
    assert abs(b["success_rate"] - 3 / 5) < 1e-9, f"ty le thanh cong sai: {b}"
    assert abs(b["mean_when_success"] - 0.963333) < 1e-4, (
        f"gia tri khi thanh cong sai: {b['mean_when_success']}"
    )
    return b["success_rate"]


def test_does_not_cry_wolf_on_normal_data():
    """KHONG duoc bao luong cuc tren du lieu binh thuong.

    Bao dong gia con nguy hiem hon bo sot: nguoi viet se mat long tin vao canh
    bao va bo qua ca truong hop that.
    """
    cases = {
        "PPL sat nhau": [170.4, 172.1, 175.3, 171.8],
        "PPL hoi tan": [168.0, 175.0, 181.0, 177.0, 172.0],
        "do chinh xac cao deu": [0.981, 0.987, 0.978, 0.990],
        "gan nhu bang nhau": [1.0, 1.0, 0.999, 1.0],
    }
    for name, vals in cases.items():
        b = detect_bimodal(vals)
        assert not b["bimodal"], f"bao dong gia tren '{name}': {b['reason']}"
    return len(cases)


def test_refuses_to_judge_with_too_few_points():
    """Voi 2-3 quan sat thi khoang trong nao cung co ve lon. Phai tu choi ket luan."""
    b = detect_bimodal([0.1, 0.95])
    assert not b["bimodal"], "ket luan luong cuc chi voi 2 diem"
    assert "quan sat" in b["reason"], f"khong noi ro ly do tu choi: {b['reason']}"
    # nhung van phai canh bao phan tan lon
    assert b.get("high_variance"), "khong canh bao phan tan lon voi 2 diem cach xa"

    b3 = detect_bimodal([0.1, 0.12, 0.95])
    assert not b3["bimodal"], "ket luan luong cuc chi voi 3 diem"
    return True


def test_confidence_interval_uses_t_distribution():
    """Voi n nho phai dung phan phoi t; dung chuan tac se cho khoang hep gia tao."""
    vals = [100.0, 110.0, 120.0]           # TB 110, do lech mau 10
    st = mean_ci(vals)
    assert st["n"] == 3
    assert abs(st["mean"] - 110.0) < 1e-9
    assert abs(st["std"] - 10.0) < 1e-9
    # t(2 bac tu do) = 4,303 => nua khoang = 4,303 * 10 / sqrt(3) = 24,84
    half = st["ci_high"] - st["mean"]
    assert abs(half - 24.844) < 0.05, f"nua khoang = {half:.3f}, ky vong 24,844"
    # neu dung chuan tac se ra 1,96*10/sqrt(3) = 11,32, hep hon nhieu
    assert half > 20.0, "khoang tin cay qua hep, nghi ngo dung phan phoi chuan tac"
    return half


def test_single_run_reports_no_interval():
    """Mot lan chay thi khong co khoang tin cay. Khong duoc bia ra so 0."""
    st = mean_ci([123.4])
    assert st["n"] == 1
    assert st["mean"] == 123.4
    assert st["std"] != st["std"], "n=1 ma van dua ra do lech"       # NaN
    assert st["ci_low"] != st["ci_low"], "n=1 ma van dua ra khoang tin cay"
    return True


# -----------------------------------------------------------------------------
def _fake_run(tmp: Path, name: str, ppl: float, **cfg) -> None:
    base = {"lang": "vi", "tokenizer": "syllable", "layers": "HHHH", "seed": 0,
            "decay_mode": "uniform", "hyena_order": 2, "no_window": False,
            "no_sine": False, "pos_emb": "learned"}
    base.update(cfg)
    (tmp / f"{name}.json").write_text(json.dumps({
        "run_name": name, "test_ppl": ppl, "test_loss": 5.0,
        "tokens_seen": 50_000_000, "wall_time_s": 540.0, "peak_mem_mb": 2303.0,
        "params": {"total": 7553280}, "config": base,
    }, ensure_ascii=False), encoding="utf-8")


def test_grouping_puts_seeds_together_and_keeps_variants_apart():
    """Cac lan chay chi khac seed phai vao chung nhom; khac cau hinh thi phai tach.

    Gom nham hai cau hinh vao mot nhom se tao ra mot "do lech" gia khong lo va
    lam bang ket qua vo nghia.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        for s in (0, 1, 2):
            _fake_run(tmp, f"E1_vi_HHHH_s{s}", 170.0 + s, seed=s)
        for s in (0, 1, 2):
            _fake_run(tmp, f"E1_vi_AAAA_s{s}", 180.0 + s, seed=s, layers="AAAA")
        _fake_run(tmp, "E3_no_window_s0", 200.0, no_window=True)

        rows = summarise(load_lm_runs(tmp))
        assert len(rows) == 3, f"phai co 3 nhom, dang co {len(rows)}: " \
                               f"{[r['key'] for r in rows]}"
        by = {(r["tag"], r["layers"], r["ablation"]): r for r in rows}
        assert by[("E1", "HHHH", "-")]["n"] == 3
        assert by[("E1", "AAAA", "-")]["n"] == 3
        assert by[("E3", "HHHH", "no_window")]["n"] == 1
        assert abs(by[("E1", "HHHH", "-")]["mean"] - 171.0) < 1e-9
        return len(rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ablation_variants_do_not_collapse_together():
    """Hai ablation khac nhau phai la hai nhom, khong duoc gop lam mot."""
    tmp = Path(tempfile.mkdtemp())
    try:
        _fake_run(tmp, "E3_no_window_s0", 200.0, no_window=True)
        _fake_run(tmp, "E3_no_sine_s0", 190.0, no_sine=True)
        _fake_run(tmp, "E3_order1_s0", 195.0, hyena_order=1)
        _fake_run(tmp, "E3_no_posemb_s0", 210.0, pos_emb="none")
        rows = summarise(load_lm_runs(tmp))
        abl = {r["ablation"] for r in rows}
        assert abl == {"no_window", "no_sine", "order1", "no_posemb"}, (
            f"ablation bi gop nham: {abl}"
        )
        return len(rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_latex_table_has_no_em_dash():
    """Bang LaTeX khong duoc chua dau gach dai (yeu cau cua nguoi huong dan)."""
    tmp = Path(tempfile.mkdtemp())
    try:
        for s in (0, 1, 2):
            _fake_run(tmp, f"E1_vi_HHHH_s{s}", 170.0 + s, seed=s)
        rows = summarise(load_lm_runs(tmp))
        tex = latex_table(rows, "E1", "Bang thu.", "tab:thu")
        assert "---" not in tex, "bang LaTeX co chua dau gach dai (---)"
        assert r"\begin{tabular}" in tex and r"\end{table}" in tex
        assert "170" in tex or "171" in tex, "khong thay so lieu trong bang"
        return len(tex)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bimodal_group_is_flagged_in_latex():
    """Nhom luong cuc phai duoc danh dau trong bang de nguoi doc khong hieu nham."""
    tmp = Path(tempfile.mkdtemp())
    try:
        for s, v in enumerate([100.0, 105.0, 900.0, 950.0]):
            _fake_run(tmp, f"E4_corpus_s{s}", v, seed=s, decay_mode="corpus")
        rows = summarise(load_lm_runs(tmp))
        assert rows[0]["bimodal"]["bimodal"], "khong nhan ra nhom luong cuc"
        tex = latex_table(rows, "E4", "Bang thu.", "tab:thu")
        assert "dagger" in tex, "bang khong danh dau nhom luong cuc"
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_runs_end_to_end_on_empty_results():
    """Chay tren thu muc rong phai bao nhe nhang, khong duoc no."""
    tmp = Path(tempfile.mkdtemp())
    try:
        code = main(["--results", str(tmp)])
        assert code == 0, f"tra ve ma loi {code} tren thu muc rong"
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------------
def main_runner() -> int:
    tests = [
        ("A1 Nhan ra luong cuc tren du lieu E6 that", test_detects_the_real_bimodal_case),
        ("A2 Khong bao dong gia tren du lieu thuong", test_does_not_cry_wolf_on_normal_data),
        ("A3 Tu choi ket luan khi qua it diem", test_refuses_to_judge_with_too_few_points),
        ("A4 Khoang tin cay dung phan phoi t", test_confidence_interval_uses_t_distribution),
        ("A5 n=1 khong bia ra khoang tin cay", test_single_run_reports_no_interval),
        ("A6 Gom seed dung, tach cau hinh dung", test_grouping_puts_seeds_together_and_keeps_variants_apart),
        ("A7 Ablation khong bi gop nham", test_ablation_variants_do_not_collapse_together),
        ("A8 Bang LaTeX khong co gach dai", test_latex_table_has_no_em_dash),
        ("A9 Danh dau nhom luong cuc trong bang", test_bimodal_group_is_flagged_in_latex),
        ("A10 Chay duoc tren thu muc rong", test_runs_end_to_end_on_empty_results),
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

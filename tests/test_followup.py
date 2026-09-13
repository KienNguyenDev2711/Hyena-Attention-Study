"""
Test DUONG DAY cho thi nghiem bo sung `hyena_study/followup.py`.

Quy tac rut ra tu bon lan chay Kaggle hong: logic thi nghiem phai co test chay
qua TRUOC khi dot GPU. Bo nay kiem ba loai loi da tung xay ra:

  1. Co dong lenh sai: moi lan chay moi phai TAI LAP DUNG cau hinh cua file ket
     qua goc, chi khac o cac khoa duoc phep (seed, run_name, alpha_file...).
     Mac dinh cua train.py khac cau hinh bao cao, nen day la loi rat de mac.
  2. Tron corpus: dau van tay lech thi phai chuyen sang ke hoach doi chung va
     bo ablation seed 2, khong duoc im lang chay tiep.
  3. Phien chet giua chung: chay lai phai bo qua lan da xong.

Them kiem dinh Welch/Holm tu cai, doi chieu voi so scipy va so trong bao cao.

Chay: python tests/test_followup.py    (CPU, khoang mot phut)
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hyena_study.analyze import (  # noqa: E402
    followup_tests,
    holm_adjust,
    t_two_sided_p,
    welch_test,
)
from hyena_study.followup import (  # noqa: E402
    FINGERPRINT_FLOAT,
    FINGERPRINT_INT,
    LANGS,
    build_plan,
    fingerprint_mismatches,
    main,
)
from hyena_study.train import parse_args  # noqa: E402

RESULTS = ROOT / "results"


def _ppl(prefix: str) -> list[float]:
    return [json.loads(f.read_text(encoding="utf-8"))["test_ppl"]
            for f in sorted(RESULTS.glob(f"{prefix}_s[0-9].json"))]


# -----------------------------------------------------------------------------
def test_every_run_reproduces_reference_config():
    """Moi lan chay phai trung cau hinh file goc, tru cac khoa duoc phep."""
    checked = 0
    for lang in LANGS:
        for matched in (True, False):
            for spec in build_plan(lang, matched):
                got = vars(parse_args(spec.argv + ["--run_name", spec.name,
                                                   "--out_dir", "x", "--token_cache", "y"]))
                ref = json.loads((ROOT / spec.reference).read_text(encoding="utf-8"))["config"]
                for k, v in ref.items():
                    if k in spec.differs_in or k not in got:
                        continue
                    assert got[k] == v, f"{spec.name}: {k} = {got[k]!r}, file goc {v!r}"
                    checked += 1
                assert got["seed"] == int(spec.name.rsplit("_s", 1)[1]), spec.name
                if spec.kind == "swap":
                    other = LANGS[lang]["other"]
                    assert got["alpha_file"] == LANGS[other]["own_alpha"], spec.name
                    assert got["alpha_file"] != ref["alpha_file"], "hoan doi ma alpha khong doi"
    return checked


def test_plan_shape():
    names = lambda plan: [s.name for s in plan]  # noqa: E731
    vi_ok = build_plan("vi", True)
    assert names(vi_ok)[:3] == [f"E4x_vi_alphaen_s{s}" for s in (0, 1, 2)]
    assert sorted(n for n in names(vi_ok) if n.startswith("E3_")) == sorted(
        f"E3_{t}_s2" for t in ("no_window", "no_sine", "order1", "order3", "no_posemb"))
    vi_bad = build_plan("vi", False)
    assert names(vi_bad) == ["E4c_vi_alphavi_s0", "E4x_vi_alphaen_s0",
                             "E4c_vi_alphavi_s1", "E4x_vi_alphaen_s1",
                             "E4c_vi_alphavi_s2", "E4x_vi_alphaen_s2"]
    assert names(build_plan("en", True)) == [f"E4x_en_alphavi_s{s}" for s in (0, 1, 2)]
    assert len(build_plan("en", False)) == 6
    return len(vi_ok)


def test_fingerprint():
    ref = json.loads((RESULTS / "E1_vi_HHHH_s0.json").read_text(encoding="utf-8"))["corpus"]
    assert fingerprint_mismatches(dict(ref), ref) == []
    bumped = dict(ref, n_tokens_train=ref["n_tokens_train"] + 1)
    assert any("n_tokens_train" in m for m in fingerprint_mismatches(bumped, ref))
    near = dict(ref, unk_rate=ref["unk_rate"] * (1 + 1e-13))
    assert fingerprint_mismatches(near, ref) == []
    far = dict(ref, unk_rate=ref["unk_rate"] * (1 + 1e-6))
    assert fingerprint_mismatches(far, ref), "lech 1e-6 phai bi bat"
    for lang in LANGS:
        c = json.loads((ROOT / LANGS[lang]["reference"]).read_text(encoding="utf-8"))["corpus"]
        missing = [k for k in FINGERPRINT_INT + FINGERPRINT_FLOAT if k not in c]
        assert not missing, f"file tham chieu {lang} thieu truong {missing}"
    return True


def test_smoke_matched_runs_swap_and_ablation():
    tmp = Path(tempfile.mkdtemp())
    try:
        assert main(["--lang", "vi", "--smoke", "--out_dir", str(tmp)]) == 0
        for n in ["E4x_vi_alphaen_s0", "E3_no_window_s2", "E3_no_sine_s2",
                  "E3_order1_s2", "E3_order3_s2", "E3_no_posemb_s2"]:
            assert (tmp / f"{n}.json").exists(), f"thieu {n}.json"
        man = json.loads((tmp / "followup_vi_manifest.json").read_text(encoding="utf-8"))
        assert man["corpus_matches"] is True and len(man["runs"]) == 6
        cfg = json.loads((tmp / "E4x_vi_alphaen_s0.json").read_text(encoding="utf-8"))["config"]
        assert cfg["alpha_file"].replace("\\", "/").endswith("results/alpha_en_bpe500.json")
        return len(man["runs"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_smoke_mismatch_switches_to_control_and_skips_ablation():
    tmp = Path(tempfile.mkdtemp())
    try:
        fake = tmp / "ref.json"
        ref = json.loads((RESULTS / "E1_vi_HHHH_s0.json").read_text(encoding="utf-8"))
        fake.write_text(json.dumps(ref), encoding="utf-8")   # corpus that khac corpus do choi
        assert main(["--lang", "vi", "--smoke", "--out_dir", str(tmp),
                     "--reference", str(fake)]) == 0
        man = json.loads((tmp / "followup_vi_manifest.json").read_text(encoding="utf-8"))
        assert man["corpus_matches"] is False and man["mismatches"]
        assert [r["name"] for r in man["runs"]] == ["E4c_vi_alphavi_s0", "E4x_vi_alphaen_s0"]
        assert not list(tmp.glob("E3_*.json")), "corpus lech ma van chay ablation seed 2"
        assert "skipped" in man
        return len(man["mismatches"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_prepare_only_exit_code_on_mismatch():
    tmp = Path(tempfile.mkdtemp())
    try:
        fake = tmp / "ref.json"
        fake.write_text((RESULTS / "E4_corpus_en_s0.json").read_text(encoding="utf-8"),
                        encoding="utf-8")
        code = main(["--lang", "en", "--smoke", "--prepare_only", "--out_dir", str(tmp),
                     "--reference", str(fake)])
        assert code == 2, f"lech dau van tay phai tra ma 2, nhan {code}"
        assert not list(tmp.glob("E4*.json")), "prepare_only khong duoc huan luyen"
        return code
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_skips_finished_runs():
    tmp = Path(tempfile.mkdtemp())
    try:
        assert main(["--lang", "en", "--smoke", "--out_dir", str(tmp)]) == 0
        f = tmp / "E4x_en_alphavi_s0.json"
        stamp = f.stat().st_mtime_ns
        assert main(["--lang", "en", "--smoke", "--out_dir", str(tmp)]) == 0
        man = json.loads((tmp / "followup_en_manifest.json").read_text(encoding="utf-8"))
        assert [r["status"] for r in man["runs"]] == ["skipped_existing"]
        assert f.stat().st_mtime_ns == stamp, "lan chay da xong bi huan luyen lai"
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_dry_run_prints_without_loading():
    assert main(["--lang", "vi", "--dry_run"]) == 0
    return True


# -----------------------------------------------------------------------------
def test_t_distribution_matches_scipy_reference():
    # Gia tri tham chieu sinh bang scipy.stats.t.sf(t, df) * 2 ngay 2026-09-13.
    for t, df, ref in ((10.813, 2.77, 0.002398385828012341),
                       (2.5, 1.3, 0.1932807671561534),
                       (0.408, 3.99, 0.7042178381992278)):
        got = t_two_sided_p(t, df)
        assert abs(got - ref) < 1e-9, f"t={t} df={df}: {got} != {ref}"
    return True


def test_welch_reproduces_report_numbers():
    r = welch_test(_ppl("E4_logspace_vi"), _ppl("E4_corpus_vi"))
    assert abs(r["diff"] - 0.285) < 1e-3 and abs(r["p"] - 0.0243) < 5e-4, r
    assert abs(r["df"] - 3.14) < 0.01, r
    assert r["ci_low"] < r["diff"] < r["ci_high"]
    r = welch_test(_ppl("E1_vi_HHHH"), _ppl("E4_logspace_vi"))
    assert abs(r["p"] - 0.0024) < 5e-5, r
    return r["p"]


def test_holm_on_ablation_matches_reference():
    res = followup_tests([RESULTS])
    got = {r["name"]: r["p_holm"] for r in res["ablation"]}
    ref = {"no_window": 0.0059, "order1": 0.0242, "no_sine": 0.0473,
           "order3": 0.3448, "no_posemb": 0.0242}
    for k, v in ref.items():
        assert abs(got[k] - v) < 5e-4, f"{k}: {got[k]:.4f} != {v}"
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    return len(got)


def test_alpha_robustness_numbers_in_report():
    """Muc 5.3 trich cac so nay; doi code anh xa ma quen sua bao cao thi test nay do."""
    from hyena_study.analyze import alpha_robustness_metrics
    r = alpha_robustness_metrics(RESULTS)
    ref = {"width_vi_median": 90.54, "width_en_median": 160.45,
           "width_vi_en_rel_den_en_pct": 36.17, "width_vi_en_rel_den_vi_pct": 81.08,
           "width_vi_en_symmetric_pct": 48.93, "width_vi_logspace_rel_den_logspace_pct": 163.30,
           "k1000_vi_en_rel_den_en_pct": 12.9}
    for k, v in ref.items():
        assert abs(r[k] - v) < 0.06, f"{k}: {r[k]:.3f} != {v}"
    return True


# -----------------------------------------------------------------------------
def main_runner() -> int:
    tests = [
        ("F1 Moi lan chay tai lap dung cau hinh file goc", test_every_run_reproduces_reference_config),
        ("F2 Hinh dang ke hoach khop/lech", test_plan_shape),
        ("F3 Dau van tay corpus", test_fingerprint),
        ("F4 Smoke corpus khop: swap + ablation", test_smoke_matched_runs_swap_and_ablation),
        ("F5 Smoke corpus lech: doi chung, bo ablation", test_smoke_mismatch_switches_to_control_and_skips_ablation),
        ("F6 --prepare_only tra ma 2 khi lech", test_prepare_only_exit_code_on_mismatch),
        ("F7 Chay lai bo qua lan da xong", test_resume_skips_finished_runs),
        ("F8 --dry_run", test_dry_run_prints_without_loading),
        ("F9 Phan phoi t khop scipy", test_t_distribution_matches_scipy_reference),
        ("F10 Welch tai lap so trong bao cao", test_welch_reproduces_report_numbers),
        ("F11 Holm cho ablation", test_holm_on_ablation_matches_reference),
        ("F12 So do ben VI-EN trong bao cao", test_alpha_robustness_numbers_in_report),
    ]
    n_fail = 0
    for name, fn in tests:
        try:
            out = fn()
            extra = f"  ({out})" if not isinstance(out, bool) and out is not None else ""
            print(f"  [PASS] {name}{extra}")
        except AssertionError as exc:
            n_fail += 1
            print(f"  [FAIL] {name}\n         {exc}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [LOI ] {name}\n         {type(exc).__name__}: {exc}")
    print("\n==> " + (f"Toan bo {len(tests)} test DAT" if not n_fail
                      else f"{n_fail}/{len(tests)} test THAT BAI"))
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main_runner())

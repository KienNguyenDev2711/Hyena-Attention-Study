"""
Tong hop ket qua thi nghiem thanh bang LaTeX san de dan vao bao cao.

VI SAO VIET SCRIPT THAY VI TONG HOP TAY:

Chep tay 40 con so tu 40 file JSON vao bang LaTeX la con duong ngan nhat dan
den mot bang sai ma khong ai phat hien. Script doc thang tu file ket qua nen
moi so trong bao cao deu truy nguoc duoc ve mot file cu the.

BAY THONG KE QUAN TRONG NHAT MA SCRIPT NAY XU LY:

E6 cho thay du lieu co the LUONG CUC (bimodal). `hyena_corpus` chay 5 seed cho
ra 0,092 · 0,141 · 0,922 · 0,981 · 0,987: hai cum tach hoan toan, khong co gia
tri trung gian. Voi phan bo nhu vay, "trung binh 0,62 +/- 0,46" la mot cau NOI
DOI: khong lan chay nao dat gan 0,62 ca.

Dai luong dung phai la TY LE THANH CONG va DO CHINH XAC KHI THANH CONG. Script
tu phat hien dang phan bo nay va doi cach bao cao, thay vi lang le in ra mot
trung binh vo nghia.

Chay:
    python -m hyena_study.analyze                 # tat ca
    python -m hyena_study.analyze --latex         # chi in bang LaTeX
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

# Nguong phat hien luong cuc. Xem `detect_bimodal` de biet ly do tung nguong.
MIN_N_FOR_BIMODAL = 4
GAP_OVER_RANGE = 0.5
MIN_CV = 0.25


# -----------------------------------------------------------------------------
# Thong ke
# -----------------------------------------------------------------------------
def mean_ci(values: list[float], conf: float = 0.95) -> dict:
    """Trung binh, do lech chuan mau, va khoang tin cay theo phan phoi t.

    Dung phan phoi t chu khong phai chuan tac: voi 2-3 seed thi xap xi chuan tac
    cho khoang tin cay hep gia tao, khien chenh lech trong nhu co y nghia trong
    khi khong phai.
    """
    a = np.asarray(values, dtype=float)
    n = len(a)
    m = float(a.mean())
    if n < 2:
        return {"n": n, "mean": m, "std": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    s = float(a.std(ddof=1))
    # gia tri t hai phia cho muc 95%, tra san de khong phai phu thuoc scipy
    t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
           8: 2.365, 9: 2.306, 10: 2.262}
    t = t95.get(n, 1.96)
    half = t * s / math.sqrt(n)
    return {"n": n, "mean": m, "std": s, "ci_low": m - half, "ci_high": m + half}


def detect_bimodal(values: list[float]) -> dict:
    """Phat hien phan bo tach thanh hai cum roi han.

    Quy tac (co y de don gian va giai thich duoc, khong dung kiem dinh nang):
      1. Can it nhat 4 quan sat. Voi 2-3 diem thi "khoang trong" nao cung co ve
         lon, moi ket luan deu la doc bua ngau nhien.
      2. Khoang trong lon nhat giua hai gia tri lien tiep phai chiem hon mot nua
         toan bo bien do.
      3. He so bien thien (do lech / trung binh) phai vuot 0,25, tuc phan tan
         that su lon chu khong phai dao dong nho.

    Ca ba dieu kien cung dung thi bao cao trung binh la sai lech.

    GIOI HAN: quy tac nay khong phai kiem dinh thong ke chinh thuc. No chi de
    canh bao nguoi viet bao cao dung nham dai luong. Voi n nho no co the bo sot
    (thieu do nhay), nhung it khi bao dong gia.
    """
    a = np.sort(np.asarray(values, dtype=float))
    n = len(a)
    out = {"bimodal": False, "n": n, "reason": ""}
    if n < MIN_N_FOR_BIMODAL:
        out["reason"] = f"chi co {n} quan sat, can >= {MIN_N_FOR_BIMODAL}"
        # van canh bao neu phan tan qua lon
        if n >= 2:
            m, s = float(a.mean()), float(a.std(ddof=1))
            if abs(m) > 1e-12 and s / abs(m) > MIN_CV:
                out["reason"] += f"; phan tan rat lon (CV={s/abs(m):.2f})"
                out["high_variance"] = True
        return out

    gaps = np.diff(a)
    rng = float(a[-1] - a[0])
    if rng <= 1e-12:
        out["reason"] = "moi gia tri gan nhu bang nhau"
        return out
    gi = int(np.argmax(gaps))
    gap = float(gaps[gi])
    m, s = float(a.mean()), float(a.std(ddof=1))
    cv = s / abs(m) if abs(m) > 1e-12 else float("inf")

    if gap / rng > GAP_OVER_RANGE and cv > MIN_CV:
        lo, hi = a[:gi + 1], a[gi + 1:]
        out.update({
            "bimodal": True,
            "gap": gap, "gap_over_range": gap / rng, "cv": cv,
            "cluster_low": {"n": len(lo), "mean": float(lo.mean())},
            "cluster_high": {"n": len(hi), "mean": float(hi.mean())},
            "success_rate": len(hi) / n,
            "mean_when_success": float(hi.mean()),
            "reason": (f"khoang trong {gap:.3f} chiem {gap/rng:.0%} bien do, "
                       f"CV={cv:.2f}"),
        })
    else:
        out["reason"] = f"khoang trong {gap/rng:.0%} bien do, CV={cv:.2f}"
    return out


# -----------------------------------------------------------------------------
# Doc ket qua
# -----------------------------------------------------------------------------
def load_lm_runs(results_dir: Path) -> list[dict]:
    """Doc moi file JSON do train.py sinh ra."""
    runs = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "test_ppl" not in d or "config" not in d:
            continue        # khong phai ket qua huan luyen LM
        c = d["config"]
        runs.append({
            "run_name": d.get("run_name", f.stem),
            "file": f.name,
            "tag": d.get("run_name", f.stem).split("_")[0],
            "lang": c.get("lang"), "tokenizer": c.get("tokenizer"),
            "layers": c.get("layers"), "seed": c.get("seed"),
            "decay_mode": c.get("decay_mode", "uniform"),
            "hyena_order": c.get("hyena_order"),
            "no_window": c.get("no_window", False),
            "no_sine": c.get("no_sine", False),
            "pos_emb": c.get("pos_emb", "learned"),
            "test_ppl": d["test_ppl"],
            "params": d.get("params", {}).get("total"),
            "tokens": d.get("tokens_seen"),
            "wall_s": d.get("wall_time_s"),
            "peak_mb": d.get("peak_mem_mb"),
        })
    return runs


def group_key(r: dict) -> tuple:
    """Khoa gom nhom: moi thu tru seed. Cac lan chay chung khoa chi khac seed."""
    abl = []
    if r["no_window"]:
        abl.append("no_window")
    if r["no_sine"]:
        abl.append("no_sine")
    if r["pos_emb"] == "none":
        abl.append("no_posemb")
    if r["hyena_order"] not in (None, 2):
        abl.append(f"order{r['hyena_order']}")
    return (r["tag"], r["lang"], r["tokenizer"], r["layers"],
            r["decay_mode"], "+".join(abl) or "-")


def summarise(runs: list[dict], metric: str = "test_ppl") -> list[dict]:
    groups = defaultdict(list)
    for r in runs:
        groups[group_key(r)].append(r)
    out = []
    for k, rs in sorted(groups.items()):
        vals = [r[metric] for r in rs]
        st = mean_ci(vals)
        bm = detect_bimodal(vals)
        out.append({"key": k, "tag": k[0], "lang": k[1], "tokenizer": k[2],
                    "layers": k[3], "decay_mode": k[4], "ablation": k[5],
                    "values": sorted(vals), "seeds": sorted(r["seed"] for r in rs),
                    "params": rs[0]["params"], "tokens": rs[0]["tokens"],
                    **st, "bimodal": bm})
    return out


# -----------------------------------------------------------------------------
# In bang
# -----------------------------------------------------------------------------
def print_console(rows: list[dict]) -> None:
    if not rows:
        print("  (chua co ket qua huan luyen LM nao trong results/)")
        return
    print(f"{'nhom':<44}{'n':>3}{'PPL TB':>10}{'do lech':>9}{'KTC 95%':>20}")
    print("-" * 86)
    for r in rows:
        name = f"{r['tag']} {r['lang']}/{r['tokenizer']}/{r['layers']}"
        if r["decay_mode"] != "uniform":
            name += f"/{r['decay_mode']}"
        if r["ablation"] != "-":
            name += f"/{r['ablation']}"
        ci = (f"[{r['ci_low']:.2f}, {r['ci_high']:.2f}]"
              if not math.isnan(r["ci_low"]) else "khong du seed")
        print(f"{name:<44}{r['n']:>3}{r['mean']:>10.3f}{r['std']:>9.3f}{ci:>20}")
        if r["bimodal"].get("bimodal"):
            b = r["bimodal"]
            print(f"{'':4}!! LUONG CUC: {b['reason']}")
            print(f"{'':4}   ty le thanh cong {b['success_rate']:.0%}, "
                  f"gia tri khi thanh cong {b['mean_when_success']:.3f}")
            print(f"{'':4}   => KHONG bao cao trung binh cho nhom nay")
        elif r["bimodal"].get("high_variance"):
            print(f"{'':4}!! phan tan lon: {r['bimodal']['reason']}")


def latex_table(rows: list[dict], tag: str, caption: str, label: str) -> str:
    """Sinh bang LaTeX cho mot thi nghiem. Khong dung dau gach dai."""
    sel = [r for r in rows if r["tag"] == tag]
    if not sel:
        return f"% (chua co ket qua cho {tag})\n"
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Cau hinh & Ngon ngu & $n$ & PPL & KTC 95\% \\", r"\midrule",
    ]
    for r in sel:
        name = r["layers"]
        if r["decay_mode"] != "uniform":
            name += f", {r['decay_mode']}"
        if r["ablation"] != "-":
            name += f", {r['ablation']}"
        ci = (f"[{r['ci_low']:.1f}, {r['ci_high']:.1f}]"
              if not math.isnan(r["ci_low"]) else r"n/a")
        mark = r"$^{\dagger}$" if r["bimodal"].get("bimodal") else ""
        lines.append(f"{name}{mark} & {r['lang']} & {r['n']} & "
                     f"{r['mean']:.2f} $\\pm$ {r['std']:.2f} & {ci} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              r"\end{table}"]
    if any(r["bimodal"].get("bimodal") for r in sel):
        lines.insert(-1, r"% $\dagger$: phan bo luong cuc, xem phan thao luan")
    return "\n".join(lines) + "\n"


def analyse_benchmark(results_dir: Path) -> None:
    """E5: bang tang toc theo do dai chuoi."""
    files = sorted(results_dir.glob("E5_benchmark_*.csv"))
    if not files:
        print("  (chua co ket qua E5)")
        return
    for f in files:
        with f.open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        by = {(r["op"], int(r["seq_len"])): r for r in rows}
        lens = sorted({int(r["seq_len"]) for r in rows})
        print(f"\n  {f.name}")
        print(f"  {'L':>7}{'hyena ms':>11}{'sdpa ms':>10}{'naive ms':>11}"
              f"{'vs sdpa':>10}{'vs naive':>10}")
        for L in lens:
            h = by.get(("hyena", L))
            cells = [f"{L:>7}"]
            for op in ("hyena", "sdpa", "naive"):
                r = by.get((op, L))
                cells.append(f"{'OOM':>10}" if (r and r["status"] != "ok")
                             else (f"{float(r['time_ms']):>10.2f}" if r else f"{'-':>10}"))
            for op in ("sdpa", "naive"):
                b = by.get((op, L))
                if h and b and h["status"] == "ok" and b["status"] == "ok":
                    cells.append(f"{float(b['time_ms'])/float(h['time_ms']):>9.2f}x")
                else:
                    cells.append(f"{'-':>10}")
            print("  " + "".join(cells))


def analyse_recall(results_dir: Path) -> None:
    """E6: gom moi file E6_recall_*.csv, gop theo cau hinh, canh bao luong cuc.

    Gom NHIEU file thay vi doc mot ten co dinh, vi hai ly do:

    1. Cac lan chay E6 co ngan sach huan luyen khac nhau nam o file khac nhau.
       Gop lai moi du seed de phat hien luong cuc: rieng 3 seed cua mot lan chay
       thi `detect_bimodal` tu choi ket luan, dung nhu no phai lam.
    2. Mot ten file co dinh tung gay hau qua that: ban E6 cu (tran bao hoa,
       chance=0,2) da duoc commit vao repo, nen moi lan Kaggle clone repo de
       chay thi nghiem khac, file cu do di theo va nam trong ket qua tai ve,
       khien nguoi doc tuong la so lieu moi nhat.

    Ta chi gop cac lan chay CO CUNG muc doan mo. Gop hai cau hinh tac vu khac
    nhau (vocab khac nhau) se tao ra mot phan bo gia luong cuc.
    """
    files = sorted(results_dir.glob("E6_recall_*.csv"))
    if not files:
        print("  (chua co ket qua E6)")
        return

    by = defaultdict(list)          # (chance, config) -> [accuracy]
    src = defaultdict(set)
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                key = (float(r["chance"]), r["config"])
                by[key].append(float(r["accuracy"]))
                src[key].add(f.name)

    for chance in sorted({k[0] for k in by}):
        print(f"\n  Muc doan mo {chance:.3f}")
        print(f"  {'cau hinh':<18}{'n':>3}{'TB':>8}{'do lech':>9}  ghi chu")
        for (ch, cfg), vals in sorted(by.items()):
            if ch != chance:
                continue
            st = mean_ci(vals)
            bm = detect_bimodal(vals)
            if bm.get("bimodal"):
                note = (f"LUONG CUC: thanh cong {bm['success_rate']:.0%}, "
                        f"khi thanh cong {bm['mean_when_success']:.3f} "
                        f"=> KHONG bao cao trung binh")
            elif bm.get("high_variance"):
                note = "phan tan lon, can them seed"
            else:
                note = ""
            print(f"  {cfg:<18}{st['n']:>3}{st['mean']:>8.3f}{st['std']:>9.3f}  {note}")
            if len(src[(ch, cfg)]) > 1:
                print(f"  {'':<18}   (gop tu {len(src[(ch, cfg)])} lan chay)")


# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tong hop ket qua thanh bang bao cao")
    p.add_argument("--results", default="results")
    p.add_argument("--latex", action="store_true", help="chi in bang LaTeX")
    p.add_argument("--out", default=None, help="ghi bang LaTeX ra file")
    args = p.parse_args(argv)

    d = Path(args.results)
    if not d.exists():
        print(f"Khong thay thu muc {d}")
        return 1

    runs = load_lm_runs(d)
    rows = summarise(runs)

    if not args.latex:
        print("=" * 86)
        print(f"KET QUA HUAN LUYEN LM  ({len(runs)} lan chay, {len(rows)} nhom)")
        print("=" * 86)
        print_console(rows)
        print("\n" + "=" * 86)
        print("E5: HIEU NANG THEO DO DAI CHUOI")
        print("=" * 86)
        analyse_benchmark(d)
        print("\n" + "=" * 86)
        print("E6: ASSOCIATIVE RECALL")
        print("=" * 86)
        analyse_recall(d)

    tex = []
    for tag, cap, lab in [
        ("E1", "Perplexity tren tap kiem tra, Hyena so voi Transformer.", "tab:e1"),
        ("E2", "Anh huong cua don vi token hoa tieng Viet.", "tab:e2"),
        ("E3", "Ablation cac thanh phan cua toan tu Hyena.", "tab:e3"),
        ("E4", "Khoi tao bo loc tu cau truc thong tin corpus.", "tab:e4"),
    ]:
        tex.append(latex_table(rows, tag, cap, lab))
    body = "\n".join(tex)

    if args.latex:
        print(body)
    if args.out:
        Path(args.out).write_text(body, encoding="utf-8")
        print(f"\nDa ghi bang LaTeX vao {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

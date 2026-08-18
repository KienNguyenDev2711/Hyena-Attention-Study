"""
Sinh hinh cho bao cao TU FILE KET QUA, khong ve tay.

Ve tay mot bieu do roi dan vao bao cao la cach nhanh nhat de co mot hinh khong
khop voi bang so lieu ben canh no. Moi hinh o day doc thang tu results/, nen
neu so lieu thay doi thi chi can chay lai.

Ba hinh, chon theo gia tri chu khong theo so luong (bao cao da co 5 bang, them
qua nhieu hinh se vuot gioi han 8 trang):

  H1. Suy giam thong tin tuong ho I(d), tieng Viet so voi tieng Anh, kem nen
      nhieu. Day la nen tang cua phan dong gop va la phat hien dac thu tieng
      Viet duy nhat con dung vung.
  H2. Hieu nang theo do dai chuoi. Day la phan tai hien thanh cong nhat: Hyena
      vuot kernel attention 1,88 lan o L=8192.
  H3. Phan bo do dai hieu dung cua ba cach khoi tao. Hinh nay giai thich bang
      mot buc anh VI SAO duong co so `uniform` la hinh non.

Chay:
    python -m hyena_study.figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Kich thuoc mot cot cua template ACL, tinh theo inch.
COL_W = 3.2

# Mau va kieu net: phan biet duoc ca khi in den trang, vi bao cao co the duoc
# in ra giay.
VI_STYLE = dict(color="#C0392B", marker="o", linestyle="-", markersize=3)
EN_STYLE = dict(color="#2C6FBB", marker="s", linestyle="--", markersize=3)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def fig_mi_decay(results: Path, out: Path) -> Path | None:
    """H1: I(d) tieng Viet so voi tieng Anh, hai cach token hoa."""
    import matplotlib.pyplot as plt

    need = {
        "vi_syl": results / "E0b_mi_decay_vi_syllable.csv",
        "en_syl": results / "E0b_mi_decay_en_syllable.csv",
        "vi_bpe": results / "E0b_mi_decay_vi_bpe_k500.csv",
        "en_bpe": results / "E0b_mi_decay_en_bpe_k500.csv",
    }
    missing = [k for k, p in need.items() if not p.exists()]
    if missing:
        print(f"  H1 bo qua, thieu: {missing}")
        return None
    d = {k: _read_csv(p) for k, p in need.items()}

    fig, ax = plt.subplots(1, 2, figsize=(COL_W * 2, 2.4), sharey=True)
    for i, (tag, title) in enumerate([("syl", "Token hoá âm tiết"),
                                      ("bpe", "BPE, 0% ngoài từ điển")]):
        for lang, style, name in (("vi", VI_STYLE, "Tiếng Việt"),
                                  ("en", EN_STYLE, "Tiếng Anh")):
            rows = d[f"{lang}_{tag}"]
            x = [float(r["lag"]) for r in rows]
            y = [float(r["mi_corrected_nats"]) for r in rows]
            nz = [float(r["mi_baseline_nats"]) for r in rows]
            ax[i].plot(x, y, label=name, **style)
            ax[i].plot(x, nz, color=style["color"], alpha=0.35, lw=0.8,
                       linestyle=":", label=f"{name}, nền nhiễu")
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].set_xlabel("khoảng cách $d$ (token)", fontsize=8)
        ax[i].set_title(title, fontsize=9)
        ax[i].tick_params(labelsize=7)
        ax[i].grid(alpha=0.25, which="both", lw=0.4)
    ax[0].set_ylabel("$I(d)$ [nat]", fontsize=8)
    ax[0].legend(fontsize=6, loc="lower left")

    fig.tight_layout()
    p = out / "fig_mi_decay.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_efficiency(results: Path, out: Path) -> Path | None:
    """H2: tang toc va bo nho theo do dai chuoi."""
    import matplotlib.pyplot as plt

    fwd = results / "E5_benchmark_fp16_fwd.csv"
    bwd = results / "E5_benchmark_fp16_bwd.csv"
    if not fwd.exists():
        print("  H2 bo qua, thieu E5_benchmark_fp16_fwd.csv")
        return None

    def speedups(rows):
        by = {(r["op"], int(r["seq_len"])): r for r in rows}
        lens = sorted({int(r["seq_len"]) for r in rows})
        out_ = {"L": [], "sdpa": [], "naive": []}
        for L in lens:
            h = by.get(("hyena", L))
            if not h or h["status"] != "ok":
                continue
            out_["L"].append(L)
            for base in ("sdpa", "naive"):
                b = by.get((base, L))
                ok = b and b["status"] == "ok"
                out_[base].append(float(b["time_ms"]) / float(h["time_ms"]) if ok else np.nan)
        return out_

    fig, ax = plt.subplots(1, 2, figsize=(COL_W * 2, 2.6))

    # Chu giai dat DUOI hinh chu khong trong khung ve: dat trong khung se che
    # mat duong di len phia tren ben trai, dung cho phan quan trong nhat cua
    # hinh (diem Hyena vuot len tren muc 1).
    for path, name, ls in ((fwd, "lượt tiến", "-"), (bwd, "lượt tiến và lùi", "--")):
        if not path.exists():
            continue
        s = speedups(_read_csv(path))
        ax[0].plot(s["L"], s["sdpa"], marker="o", ms=3, ls=ls, color="#C0392B",
                   label=f"vs kernel tiết kiệm bộ nhớ, {name}")
        ax[0].plot(s["L"], s["naive"], marker="s", ms=3, ls=ls, color="#2C6FBB",
                   label=f"vs cài đặt ngây thơ, {name}")
    ax[0].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[0].text(2 ** 8.1, 1.15, "Hyena nhanh hơn từ đây trở lên", fontsize=5.5)
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("độ dài chuỗi $L$", fontsize=8)
    ax[0].set_ylabel("mức tăng tốc của Hyena", fontsize=8)
    ax[0].tick_params(labelsize=7)
    ax[0].grid(alpha=0.25, which="both", lw=0.4)
    speed_handles, speed_labels = ax[0].get_legend_handles_labels()

    rows = _read_csv(fwd)
    by = {(r["op"], int(r["seq_len"])): r for r in rows}
    lens = sorted({int(r["seq_len"]) for r in rows})
    for op, style, name in (("hyena", VI_STYLE, "Hyena"),
                            ("naive", EN_STYLE, "attention ngây thơ")):
        xs = [L for L in lens if by.get((op, L)) and by[(op, L)]["status"] == "ok"]
        ys = [float(by[(op, L)]["peak_mem_mb"]) for L in xs]
        ax[1].plot(xs, ys, label=name, **style)
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("độ dài chuỗi $L$", fontsize=8)
    ax[1].set_ylabel("bộ nhớ đỉnh (MB)", fontsize=8)
    ax[1].tick_params(labelsize=7)
    ax[1].grid(alpha=0.25, which="both", lw=0.4)
    ax[1].legend(fontsize=6, loc="upper left", frameon=False)
    # Mui ten tro vao DIEM CUOI CUNG con do duoc cua duong ngay tho, kem chu ghi
    # ro no het bo nho o buoc tiep theo. Tro vao toa do khong co du lieu se lam
    # nguoi doc tuong duong con keo dai toi do.
    last = max(L for L in lens if by.get(("naive", L)) and by[("naive", L)]["status"] == "ok")
    ax[1].annotate(f"hết bộ nhớ\nở $L$ = {last * 2}", xy=(last, float(by[("naive", last)]["peak_mem_mb"])),
                   fontsize=5.5, xytext=(last * 0.16, 6500), color="#2C6FBB",
                   arrowprops=dict(arrowstyle="->", color="#2C6FBB", lw=0.7))

    fig.tight_layout()
    # Chu giai cap HINH, dat duoi ca hai bang. Dat trong khung ve thi che mat
    # duong cong; dat duoi mot bang thi de len nhan truc x cua bang do.
    fig.subplots_adjust(bottom=0.34)
    fig.legend(speed_handles, speed_labels, fontsize=5.5, ncol=2,
               loc="lower center", bbox_to_anchor=(0.5, -0.02), frameon=False)
    p = out / "fig_efficiency.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_effective_lengths(results: Path, out: Path) -> Path | None:
    """H3: phan bo do dai hieu dung cua ba cach khoi tao."""
    import matplotlib.pyplot as plt

    from .morphology import logspaced_alphas, uniform_alphas

    vi = results / "alpha_vi_bpe500.json"
    if not vi.exists():
        print("  H3 bo qua, thieu alpha_vi_bpe500.json")
        return None
    L, D = 512, 256
    corpus = np.array(json.loads(vi.read_text(encoding="utf-8"))["effective_lengths"])
    logsp = np.array(logspaced_alphas(D, seq_len=L).effective_lengths)
    unif = np.array(uniform_alphas(D, seq_len=L).effective_lengths)

    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    bins = np.logspace(0, np.log10(L * 2), 28)
    for arr, name, c, hs in ((unif, "uniform", "#888888", "//"),
                             (logsp, "logspace", "#2C6FBB", "\\\\"),
                             (corpus, "corpus", "#C0392B", None)):
        ax.hist(arr, bins=bins, alpha=0.55, label=name, color=c, hatch=hs,
                edgecolor="white", lw=0.3)
    ax.axvline(34, color="k", ls=":", lw=1.0)
    # Chu dat BEN TRAI duong ke, can le phai: dat ben phai se bi chu giai che.
    ax.text(30, ax.get_ylim()[1] * 0.5, "80-90% thông tin\nnằm bên trái",
            fontsize=6, ha="right")
    ax.set_xscale("log")
    ax.set_xlabel("độ dài hiệu dụng của kênh (token)", fontsize=8)
    ax.set_ylabel("số kênh", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.2, lw=0.4)

    fig.tight_layout()
    p = out / "fig_effective_lengths.pdf"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sinh hinh cho bao cao tu file ket qua")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="report/figures")
    args = p.parse_args(argv)

    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    made = []
    for fn in (fig_mi_decay, fig_efficiency, fig_effective_lengths):
        r = fn(res, out)
        if r:
            made.append(r)
            print(f"  da sinh {r}")
    print(f"\n{len(made)}/3 hinh")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())

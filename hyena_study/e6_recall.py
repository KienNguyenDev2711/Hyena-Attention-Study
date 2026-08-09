"""
E6 - Associative recall: bo loc ngan co lam mat kha nang tam xa khong?

TAI SAO THI NGHIEM NAY LA MODULE CHU KHONG PHAI O NOTEBOOK:

Ba lan chay Kaggle truoc deu hong, va ca ba loi deu nam trong o notebook:
  1. `solvable[-1]` lay cau hinh DE NHAT trong khi comment ghi "kho nhat"
     -> ca bon cau hinh deu bao hoa o 1,000, khong phan biet duoc gi.
  2. Test R5 lam roi lich warmup khi bung ngan sach tu notebook sang
     -> cung 18720 buoc: co warmup 1,000, khong warmup 0,187.
  3. Quet do dai dat vocab = n_pairs -> tang dong thoi ba truc kho
     -> moi do dai deu ve muc doan mo, khong quy duoc cho truc nao.

Code trong o notebook KHONG co test nao chay qua. Dua ra module thi
`tests/test_e6_script.py` chay duoc che do --smoke trong vai giay va bat duoc
dung loai loi noi day. Do la ly do ky thuat duy nhat cua file nay.

CAU HOI KHOA HOC:

Bo loc suy tu corpus co do dai hieu dung trung vi 2,9 token. No CO THE giam
perplexity (vi thong tin trong van ban that su tap trung o khoang cach ngan)
trong khi da vut bo kha nang tam xa. Perplexity khong lo ra dieu do; associative
recall thi co.

THIET KE - co lap DUNG MOT bien:

    [k1 v1 k2 v2 ... kP vP] [dem dem ... dem] [truy van]
     <---- 2P token ---->    <---- F ---->     1 token

Giu nguyen phep tra cuu (cung so cap, cung vocab), chi keo dai khoang cach bang
token dem. Token dem lay tu chinh bang chu cai nen mo hinh khong nhan ra chung
bang mot ky hieu rieng - no buoc phai mang thong tin qua ca doan dem.

Chay:
    python -m hyena_study.e6_recall              # day du, tren Kaggle GPU
    python -m hyena_study.e6_recall --smoke      # kiem tra duong day, vai giay
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data.synthetic import (
    RecallConfig,
    build_recall_dataset,
    chance_accuracy,
    query_distance,
)
from .models import HyenaFilterConfig, LMConfig, SequenceLM

MARGIN = 0.30   # coi la "giai duoc" khi vuot muc doan mo bao nhieu


# -----------------------------------------------------------------------------
def run_one(cfg: argparse.Namespace, n_filler: int, layers: str,
            alpha: list[float] | None = None, seed: int = 0,
            warmup_frac: float = 0.05) -> dict:
    """Huan luyen mot cau hinh. Giua cac lan quet CHI `n_filler` thay doi."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    L = 2 * cfg.n_pairs + n_filler + 1
    task = RecallConfig(vocab_size=cfg.vocab_size, seq_len=L,
                        n_pairs_fixed=cfg.n_pairs, n_train=cfg.n_train,
                        n_val=10, n_test=cfg.n_test, seed=0)
    data = build_recall_dataset(task)
    xtr, ytr = data["train"]
    xte, yte = data["test"]

    model = SequenceLM(LMConfig(
        vocab_size=cfg.vocab_size, d_model=cfg.d_model, layer_spec=layers,
        max_seq_len=L, dropout=0.0, n_heads=cfg.n_heads,
        hyena_filter=HyenaFilterConfig(alpha_values=alpha),
    )).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    xtr_t = torch.from_numpy(xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    per_epoch = max(1, len(xtr) // cfg.batch_size)
    epochs = max(1, round(cfg.steps / per_epoch))
    total = epochs * per_epoch
    # warmup BAT BUOC: cung 18720 buoc, co warmup dat 1,000 con khong co chi
    # dat 0,187. Bo no di la lam hong thi nghiem mot cach am tham.
    warmup = max(int(total * warmup_frac), 1)
    step, t0 = 0, time.time()

    for _ in range(epochs):
        perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(xtr) - cfg.batch_size + 1, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            for g in opt.param_groups:
                g["lr"] = cfg.lr * min(1.0, (step + 1) / warmup)
            loss = F.cross_entropy(model(xtr_t[idx])[:, -1, :], ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xte).to(dev))[:, -1, :].argmax(-1).cpu().numpy()
    acc = float((pred == yte).mean())

    # do chinh xac tong the co the che giau viec mo hinh chi giai duoc cap o gan
    dist = query_distance(xte, n_pairs=cfg.n_pairs)
    half = np.median(dist)
    near = float((pred[dist <= half] == yte[dist <= half]).mean())
    far = float((pred[dist > half] == yte[dist > half]).mean())

    return {"n_filler": n_filler, "seq_len": L, "layers": layers, "seed": seed,
            "accuracy": acc, "chance": chance_accuracy(task),
            "acc_near": near, "acc_far": far,
            "dist_mean": float(dist.mean()), "dist_max": int(dist.max()),
            "steps": step, "seconds": time.time() - t0}


def solved(r: dict, margin: float = MARGIN) -> bool:
    """Co vuot muc doan mo du xa de coi la giai duoc khong.

    `margin` tach ra lam tham so vi che do --smoke can bo qua cong nay: voi 20
    buoc thi khong mo hinh nao hoc duoc gi, nen neu giu nguong that thi script
    dung o pha 1 va pha 2 KHONG BAO GIO duoc test chay qua - dung cai bay ma bo
    test nay sinh ra de tranh.
    """
    return r["accuracy"] > r["chance"] + margin


def _write(rows: list[dict], path: Path) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# -----------------------------------------------------------------------------
def phase0_warmup(cfg) -> tuple[bool, list[dict]]:
    """Kiem chung chinh gia thuyet warmup truoc khi dua vao no."""
    print("\n=== PHA 0: warmup co that su quyet dinh khong? ===")
    print(f"{'warmup':>10}{'acc':>9}{'doan mo':>10}{'giay':>7}")
    print("-" * 36)
    rows = []
    for wf, tag in ((0.05, "co (5%)"), (1e-9, "khong")):
        r = run_one(cfg, n_filler=0, layers="AA", warmup_frac=wf)
        r["warmup"] = tag
        rows.append(r)
        print(f"{tag:>10}{r['accuracy']:>9.3f}{r['chance']:>10.3f}{r['seconds']:>7.0f}")

    ok = solved(rows[0], cfg.margin)
    print()
    if ok and not solved(rows[1], cfg.margin):
        print("=> Gia thuyet DUNG: warmup la yeu to quyet dinh.")
    elif ok:
        print("=> Ca hai deu giai duoc: warmup KHONG phai nguyen nhan that su cua")
        print("   lan that bai truoc. Van chay tiep duoc, nhung phai tim lai")
        print("   nguyen nhan that su truoc khi viet vao bao cao.")
    else:
        print("=> CA HAI DEU THAT BAI. Gia thuyet sai. Dung lai, khong chay tiep.")
        print("   Nghi ngo tiep: khac biet giua ham trial() cu va run_one() moi")
        print("   (kich thuoc tap val, cach xao tron, thu tu sinh du lieu).")
    return ok, rows


def phase1_scout(cfg) -> tuple[dict | None, list[dict]]:
    """Chi attention: tim khoang cach xa nhat con giai duoc."""
    print("\n=== PHA 1: attention chiu duoc khoang cach bao xa? ===")
    print(f"{'dem':>6}{'L':>6}{'k/cach TB':>11}{'acc':>8}{'gan':>7}{'xa':>7}{'giay':>7}")
    print("-" * 52)
    rows = []
    for f in cfg.fillers:
        r = run_one(cfg, n_filler=f, layers="AA")
        r["config"] = "attention"
        r["solvable"] = solved(r, cfg.margin)
        rows.append(r)
        print(f"{f:>6}{r['seq_len']:>6}{r['dist_mean']:>11.1f}{r['accuracy']:>8.3f}"
              f"{r['acc_near']:>7.3f}{r['acc_far']:>7.3f}{r['seconds']:>7.0f}"
              + ("  giai duoc" if r["solvable"] else "  chua dat"))
        if not r["solvable"]:
            print("     -> dung quet, khoang cach xa hon chac chan cung that bai")
            break

    ok = [r for r in rows if r["solvable"]]
    if not ok:
        print("\n=> Khong muc nao giai duoc.")
        return None, rows
    target = max(ok, key=lambda r: r["n_filler"])
    print(f"\n=> Pha 2 dung n_filler={target['n_filler']} (L={target['seq_len']}, "
          f"khoang cach TB {target['dist_mean']:.0f}, attention {target['accuracy']:.3f})")
    if target["accuracy"] > 0.995:
        print("   CANH BAO: attention gan tuyet doi o muc nay. Neu ca 4 cau hinh")
        print("   cung bao hoa thi lai khong ket luan duoc - phai tang --fillers.")
    return target, rows


def phase2_compare(cfg, target: dict) -> list[dict]:
    """So sanh 4 cau hinh o dung khoang cach tim duoc o pha 1."""
    from .morphology import alphas_from_mi, logspaced_alphas

    L = target["seq_len"]
    print(f"\n=== PHA 2: so sanh 4 cau hinh o L={L} ===")

    # Alpha PHAI sinh lai cho dung d_model va do dai cua tac vu nay. Dung lai
    # file alpha sinh cho d_model=256 / L=512 se lam sai do dai hieu dung.
    mi_path = Path(cfg.mi_csv)
    corpus_alpha = None
    if mi_path.exists():
        import csv as _csv
        with mi_path.open(encoding="utf-8") as fh:
            rd = list(_csv.DictReader(fh))
        lags = np.array([float(r["lag"]) for r in rd])
        vals = np.array([float(r["mi_corrected_nats"]) for r in rd])
        corpus_alpha = alphas_from_mi(lags, vals, d_model=cfg.d_model, seq_len=L).alpha
    else:
        print(f"  CANH BAO: khong thay {mi_path}, bo qua cau hinh corpus.")
    logspace_alpha = logspaced_alphas(cfg.d_model, seq_len=L).alpha

    for nm, a in (("corpus", corpus_alpha), ("logspace", logspace_alpha)):
        if a is None:
            continue
        e = L / np.array(a)
        print(f"  {nm:<9} do dai hieu dung: trung vi {np.median(e):.1f}, "
              f"{(e <= 4).sum()}/{cfg.d_model} kenh <= 4 token, xa nhat {e.max():.0f}")

    configs = [("attention", "AA", None), ("hyena_uniform", "HH", None),
               ("hyena_logspace", "HH", logspace_alpha)]
    if corpus_alpha is not None:
        configs.append(("hyena_corpus", "HH", corpus_alpha))

    # Cho phep chi chay mot phan cau hinh. Dung khi chan doan: vi du kiem tra
    # Hyena da hoi tu chua thi khong can chay lai attention (da dat 1,000 voi
    # do lech 0), tiet kiem hon mot nua thoi gian GPU.
    if cfg.configs:
        want = set(cfg.configs)
        unknown = want - {c[0] for c in configs}
        if unknown:
            raise SystemExit(
                f"--configs khong hop le: {sorted(unknown)}. "
                f"Chon trong {sorted(c[0] for c in configs)}"
            )
        configs = [c for c in configs if c[0] in want]
        print(f"  chi chay: {', '.join(c[0] for c in configs)}")

    rows = []
    print(f"\n{'cau hinh':<16}{'seed':>5}{'acc':>8}{'gan':>7}{'xa':>7}{'giay':>7}")
    print("-" * 50)
    for name, spec, alpha in configs:
        for sd in range(cfg.seeds):
            r = run_one(cfg, target["n_filler"], spec, alpha=alpha, seed=sd)
            r["config"] = name
            rows.append(r)
            print(f"{name:<16}{sd:>5}{r['accuracy']:>8.4f}{r['acc_near']:>7.3f}"
                  f"{r['acc_far']:>7.3f}{r['seconds']:>7.0f}")

    print("\n" + "=" * 60)
    print(f"{'cau hinh':<16}{'acc TB':>9}{'do lech':>9}{'gan TB':>9}{'xa TB':>9}")
    for name in dict.fromkeys(r["config"] for r in rows):
        g = [r for r in rows if r["config"] == name]
        a = np.array([r["accuracy"] for r in g])
        print(f"{name:<16}{a.mean():>9.4f}{a.std():>9.4f}"
              f"{np.mean([r['acc_near'] for r in g]):>9.4f}"
              f"{np.mean([r['acc_far'] for r in g]):>9.4f}")
    print(f"\nDoan mo {rows[0]['chance']:.3f} | L={rows[0]['seq_len']} | "
          f"khoang cach TB {rows[0]['dist_mean']:.0f}")
    print("\nCACH DOC:")
    print("  * Moi cau hinh deu ~1,000 => lai bao hoa, KHONG ket luan gi,")
    print("    phai tang --fillers roi chay lai.")
    print("  * corpus co 'xa TB' thap hon logspace => khoi tao tu du lieu danh")
    print("    doi kha nang tam xa. Day la phat hien can tim.")
    print("  * Chenh lech nho hon do lech giua cac seed => nhieu, khong ket luan.")
    return rows


# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E6 - associative recall theo khoang cach")
    p.add_argument("--vocab_size", type=int, default=10)
    p.add_argument("--n_pairs", type=int, default=16)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=18720,
                   help="ngan sach da duoc chung minh du (attention dat 1,000)")
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--fillers", type=int, nargs="+", default=[0, 32, 96, 224, 480])
    p.add_argument("--mi_csv", default="results/E0b_mi_decay_vi_bpe_k500.csv")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--configs", nargs="+", default=None,
                   help="chi chay mot phan cau hinh o pha 2, vd: hyena_uniform "
                        "hyena_corpus. Mac dinh chay het.")
    p.add_argument("--margin", type=float, default=MARGIN,
                   help="vuot muc doan mo bao nhieu thi coi la giai duoc")
    p.add_argument("--skip_phase0", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="chay sieu nhanh de kiem duong day, KHONG dung ket qua")
    cfg = p.parse_args(argv)

    if cfg.smoke:
        # Che do kiem duong day: vai giay, ket qua VO NGHIA ve khoa hoc.
        cfg.steps, cfg.n_train, cfg.n_test = 20, 256, 64
        cfg.seeds, cfg.fillers = 1, [0, 8]
        cfg.n_pairs, cfg.d_model = 4, 16
        # Bo qua cong "giai duoc": voi 20 buoc khong mo hinh nao dat nguong that,
        # ma muc dich cua smoke la chay HET ca ba pha de test phu duoc duong day.
        cfg.margin = -1.0
        print("CHE DO SMOKE: chi kiem duong day chay duoc, KHONG dung ket qua.")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiet bi: {dev} | vocab={cfg.vocab_size}, {cfg.n_pairs} cap, "
          f"doan mo={1/cfg.vocab_size:.3f} | {cfg.steps} buoc")
    if dev == "cpu" and not cfg.smoke:
        print("!! CANH BAO: khong co GPU. Chay day du tren CPU se rat lau.")

    out = Path(cfg.out_dir)
    ok = True
    if not cfg.skip_phase0:
        ok, rows0 = phase0_warmup(cfg)
        _write(rows0, out / "E6_phase0_warmup.csv")
        if not ok:
            print("\nDUNG LAI o pha 0.")
            return 1

    target, rows1 = phase1_scout(cfg)
    _write(rows1, out / "E6_scout_attention.csv")
    if target is None:
        print("\nDUNG LAI o pha 1.")
        return 1

    rows2 = phase2_compare(cfg, target)
    _write(rows2, out / "E6_recall_comparison.csv")
    (out / "E6_meta.json").write_text(
        json.dumps({"config": vars(cfg), "target": target}, ensure_ascii=False,
                   indent=2, default=str), encoding="utf-8")
    print(f"\nDa ghi ket qua vao {out}/E6_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

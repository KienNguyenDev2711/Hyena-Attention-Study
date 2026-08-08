"""
E6 - Huan luyen va danh gia tren tac vu ASSOCIATIVE RECALL.

Muc dich: phat hien dieu ma perplexity KHONG the phat hien.

Phep do E0-b sinh ra bo loc co do dai hieu dung trung vi 2,9 token. Cau hinh do
co the giam PPL tren van ban tu nhien (vi thong tin that su tap trung o khoang
cach ngan) trong khi da vut bo kha nang tam xa. Tac vu nay dat cau tra loi o xa
va khong cho doan mo tu ngu canh cuc bo, nen no lo ra ngay diem yeu do.

Bao cao BAT BUOC kem hai thu:
  1. Muc doan mo 1/V - khong co no thi con so 20% khong the dien giai.
  2. Do chinh xac PHAN THEO KHOANG CACH truy van. Do chinh xac tong the co the
     che giau viec mo hinh chi giai duoc cac cap o gan.

Vi du chay:
    python -m hyena_study.recall --layers HHHH --vocab_size 20 --seq_len 128 --seed 0
    python -m hyena_study.recall --layers AAAA --vocab_size 20 --seq_len 128 --seed 0
    python -m hyena_study.recall --layers HHHH --decay_mode corpus --alpha_file results/alpha_vi_bpe500.json
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
from .train import get_device, lr_at, set_seed


@torch.no_grad()
def evaluate(model: SequenceLM, x: np.ndarray, y: np.ndarray, device: torch.device,
             batch_size: int = 128) -> tuple[float, float, np.ndarray]:
    """Tra ve (loss, do chinh xac, mang du doan)."""
    model.eval()
    preds, total_loss = [], 0.0
    for i in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[i:i + batch_size]).to(device)
        yb = torch.from_numpy(y[i:i + batch_size]).to(device)
        logits = model(xb)[:, -1, :]                 # chi lay vi tri cuoi
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        preds.append(logits.argmax(-1).cpu().numpy())
    model.train()
    pred = np.concatenate(preds)
    return total_loss / len(x), float((pred == y).mean()), pred


def accuracy_by_distance(pred: np.ndarray, y: np.ndarray, dist: np.ndarray,
                         n_bins: int = 5) -> list[dict]:
    """Do chinh xac phan theo khoang cach truy van.

    Day moi la bang quan trong: do chinh xac TONG THE co the cao chi nho cac cap
    nam gan, che giau viec mo hinh khong nho duoc gi o xa.
    """
    ok = dist > 0
    if ok.sum() == 0:
        return []
    edges = np.unique(np.percentile(dist[ok], np.linspace(0, 100, n_bins + 1)).astype(int))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (dist >= lo) & (dist <= hi)
        if m.sum() == 0:
            continue
        rows.append({
            "dist_min": int(lo), "dist_max": int(hi), "n": int(m.sum()),
            "accuracy": float((pred[m] == y[m]).mean()),
        })
    return rows


def run(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device()

    cfg_task = RecallConfig(
        vocab_size=args.vocab_size, seq_len=args.seq_len,
        n_train=args.n_train, n_val=args.n_val, n_test=args.n_test,
        seed=args.data_seed,
    )
    data = build_recall_dataset(cfg_task)
    xtr, ytr = data["train"]
    xva, yva = data["val"]
    xte, yte = data["test"]
    L = xtr.shape[1]

    run_name = args.run_name or (
        f"E6_{args.layers}_V{args.vocab_size}_L{L}_{args.decay_mode}_s{args.seed}"
    )
    chance = chance_accuracy(cfg_task)
    print(f"[{run_name}] {cfg_task.n_pairs} cap khoa-gia tri, chuoi dai {L}, "
          f"vocab {args.vocab_size}, doan mo = {chance:.1%}")

    alpha_values = None
    if args.decay_mode == "logspace":
        from .morphology import logspaced_alphas
        alpha_values = logspaced_alphas(args.d_model, L).alpha
    elif args.decay_mode == "corpus":
        if not args.alpha_file:
            raise SystemExit("--decay_mode corpus yeu cau --alpha_file")
        payload = json.loads(Path(args.alpha_file).read_text(encoding="utf-8"))
        src_len = payload.get("seq_len") or payload.get("measurement", {}).get("seq_len")
        alpha_values = payload["alpha"]
        if len(alpha_values) != args.d_model:
            raise SystemExit(
                f"alpha_file co {len(alpha_values)} kenh nhung d_model={args.d_model}"
            )
        # alpha duoc suy ra o mot seq_len khac thi do dai hieu dung se lech;
        # canh bao to thay vi am tham dung sai.
        if src_len and int(src_len) != L:
            print(f"  CANH BAO: alpha sinh cho seq_len={src_len} nhung tac vu dai {L}. "
                  f"Do dai hieu dung se bi co gian {L/int(src_len):.2f} lan.")

    model = SequenceLM(LMConfig(
        vocab_size=args.vocab_size, d_model=args.d_model, layer_spec=args.layers,
        max_seq_len=L, dropout=args.dropout, pos_emb=args.pos_emb,
        n_heads=args.n_heads, hyena_order=args.hyena_order,
        hyena_filter=HyenaFilterConfig(alpha_values=alpha_values),
    )).to(device)
    n_params = model.num_parameters()
    print(f"[{run_name}] {n_params:,} tham so")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(len(xtr) // args.batch_size, 1)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(int(total_steps * 0.05), 1)

    history, step = [], 0
    t0 = time.time()
    best_val = 0.0
    for epoch in range(args.epochs):
        perm = np.random.permutation(len(xtr))
        for i in range(0, len(xtr) - args.batch_size + 1, args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = torch.from_numpy(xtr[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.lr, warmup)
            loss = F.cross_entropy(model(xb)[:, -1, :], yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

        vl, va, _ = evaluate(model, xva, yva, device, args.batch_size)
        best_val = max(best_val, va)
        history.append({"epoch": epoch + 1, "val_loss": vl, "val_acc": va,
                        "elapsed_s": time.time() - t0})
        print(f"  epoch {epoch+1:>3}/{args.epochs} | val loss {vl:.4f} "
              f"| val acc {va:.3f} (doan mo {chance:.3f})")

    tl, ta, pred = evaluate(model, xte, yte, device, args.batch_size)
    dist = query_distance(xte)
    by_dist = accuracy_by_distance(pred, yte, dist)

    print(f"\n[{run_name}] TEST do chinh xac {ta:.4f} | doan mo {chance:.4f} "
          f"| tren muc doan mo {ta - chance:+.4f}")
    if by_dist:
        print(f"  {'khoang cach truy van':<22}{'n':>7}{'do chinh xac':>14}")
        for r in by_dist:
            print(f"  {str(r['dist_min']) + '-' + str(r['dist_max']):<22}"
                  f"{r['n']:>7}{r['accuracy']:>14.4f}")
        near, far = by_dist[0]["accuracy"], by_dist[-1]["accuracy"]
        print(f"\n  Chenh lech gan-xa: {near:.4f} -> {far:.4f} ({far - near:+.4f})")
        print("  Chenh lech am va lon = mo hinh chi giai duoc cac cap o GAN,")
        print("  tuc da mat kha nang tam xa du do chinh xac tong the co the van cao.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "run_name": run_name,
        "test_accuracy": ta, "test_loss": tl,
        "chance_accuracy": chance, "best_val_accuracy": best_val,
        "accuracy_by_distance": by_dist,
        "n_params": n_params, "seq_len": int(L),
        "wall_time_s": time.time() - t0,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "history": history,
        "config": vars(args),
    }
    (out_dir / f"{run_name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{run_name}] da ghi {out_dir / (run_name + '.json')}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E6 - associative recall")
    p.add_argument("--layers", default="HHHH")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--hyena_order", type=int, default=2)
    p.add_argument("--pos_emb", default="learned", choices=["learned", "none"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--decay_mode", default="uniform",
                   choices=["uniform", "logspace", "corpus"])
    p.add_argument("--alpha_file", default=None)

    p.add_argument("--vocab_size", type=int, default=20)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--n_train", type=int, default=20000)
    p.add_argument("--n_val", type=int, default=2000)
    p.add_argument("--n_test", type=int, default=2000)
    p.add_argument("--data_seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--run_name", default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

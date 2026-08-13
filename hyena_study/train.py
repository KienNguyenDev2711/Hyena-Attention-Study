"""
Vòng huấn luyện — dùng chung cho MỌI biến thể mô hình.

HAI QUYẾT ĐỊNH THIẾT KẾ QUAN TRỌNG:

1. **Huấn luyện theo NGÂN SÁCH TOKEN, không theo epoch.**
   Nếu so theo epoch, mô hình nào chạy nhanh hơn sẽ được nhiều lượt cập nhật hơn
   trong cùng thời gian thực, và bảng kết quả trở thành so sánh trộn lẫn giữa
   chất lượng kiến trúc và tốc độ cài đặt. Cố định số token đã tiêu thụ thì mọi
   mô hình nhìn thấy đúng cùng một lượng tín hiệu.

2. **Mọi siêu tham số ảnh hưởng tới kết quả đều được ghi vào file JSON tóm tắt.**
   Một con số trong báo cáo mà không truy ngược được về cấu hình sinh ra nó thì
   không dùng để kết luận được.

Ví dụ chạy (trên Kaggle GPU):
    python -m hyena_study.train --layers HHHH --lang vi --tokenizer syllable --seed 0
    python -m hyena_study.train --layers AAAA --lang vi --tokenizer syllable --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import LMWindowDataset, cached_token_stream, stats_to_dict
from .models import HyenaFilterConfig, LMConfig, SequenceLM


# -----------------------------------------------------------------------------
# Tái lập được (reproducibility)
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Cố định seed cho MỌI nguồn ngẫu nhiên.

    Không cố định đủ ba nguồn (python/numpy/torch) thì hai lần chạy "cùng seed"
    vẫn khác nhau, và toàn bộ phần phân tích độ lệch giữa các seed trở nên vô
    nghĩa.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_max_train_tokens(args: argparse.Namespace) -> int:
    """Số token TRAIN giữ lại sau khi cắt. Mặc định bằng token_budget (hành vi cũ).

    --max_train_tokens tách việc CẮT CORPUS khỏi việc ĐẾM BƯỚC: dùng cho nhánh
    đối chứng "cùng số token" (T13) — corpus EN bị cắt xuống đúng số token duy
    nhất của nhánh VI, nhưng tổng token huấn luyện vẫn theo token_budget nên số
    bước và số epoch tương đương giữa hai nhánh.
    """
    return args.token_budget if args.max_train_tokens is None else args.max_train_tokens


# -----------------------------------------------------------------------------
# Lịch learning rate
# -----------------------------------------------------------------------------
def lr_at(step: int, total: int, base_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    """Warmup tuyến tính rồi giảm theo cosine — giống hệt nhau cho mọi mô hình."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


# -----------------------------------------------------------------------------
# Đánh giá
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: SequenceLM, loader: DataLoader, device: torch.device,
             max_batches: int | None = None) -> tuple[float, float]:
    """Trả về (loss trung bình theo token, perplexity).

    Lấy trung bình có TRỌNG SỐ theo số token, không phải trung bình theo batch:
    batch cuối thường ngắn hơn, lấy trung bình đơn giản sẽ lệch.
    """
    model.eval()
    total_loss, total_tok = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               reduction="sum")
        total_loss += loss.item()
        total_tok += y.numel()
    model.train()
    mean_loss = total_loss / max(total_tok, 1)
    return mean_loss, math.exp(min(mean_loss, 20.0))


# -----------------------------------------------------------------------------
# Huấn luyện
# -----------------------------------------------------------------------------
def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device()
    if device.type != "cuda":
        print("!!! CẢNH BÁO: không thấy GPU. Kết quả huấn luyện thật PHẢI chạy "
              "trên Kaggle GPU; chạy CPU chỉ dùng để kiểm tra pipeline.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or (
        f"{args.layers}_{args.lang}_{args.tokenizer}_N{args.hyena_order}"
        f"_{args.decay_mode}_s{args.seed}"
    )

    # -- Dữ liệu -------------------------------------------------------------
    print(f"[{run_name}] nạp corpus {args.lang} ({args.n_docs} bài) ...")
    # Dùng cache dòng token: token hoá lại corpus ở MỖI lần chạy tốn khoảng 10
    # phút cho một lần chạy 50M token, nhân với ~36 lần chạy là ~6 giờ GPU đốt
    # vô ích. Cache khoá theo (lang, tokenizer, vocab, n_docs, data_seed) nên
    # mọi seed và mọi kiến trúc dùng chung một lần token hoá.
    train_ids, val_ids, test_ids, tok, stats = cached_token_stream(
        lang=args.lang, tokenizer=args.tokenizer, vocab_size=args.vocab_size,
        n_docs=args.n_docs, data_seed=args.data_seed,
        max_tokens=resolve_max_train_tokens(args), cache_root=args.token_cache,
        use_cache=not args.no_cache, hf_cache_dir=args.cache_dir,
    )
    print(f"[{run_name}] token: train={len(train_ids):,} val={len(val_ids):,} "
          f"test={len(test_ids):,} | vocab={tok.vocab_size:,} "
          f"| unk={stats.unk_rate:.4f} | ký tự/token={stats.chars_per_token:.3f}")

    ds_tr = LMWindowDataset(train_ids, args.seq_len)
    ds_va = LMWindowDataset(val_ids, args.seq_len)
    ds_te = LMWindowDataset(test_ids, args.seq_len)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size)

    # -- Mô hình -------------------------------------------------------------
    alpha_values = None
    if args.decay_mode == "logspace":
        # Doi chung CONG BANG (xem morphology.logspaced_alphas): do dai hieu dung
        # cach deu theo thang log tu 1 toi seq_len. Khong dung thong tin ngon ngu.
        from .morphology import logspaced_alphas
        alpha_values = logspaced_alphas(args.d_model, args.seq_len).alpha
        print(f"[{run_name}] alpha = log-spaced 1..{args.seq_len} token (doi chung cong bang)")
    elif args.decay_mode == "corpus":
        # ĐỀ XUẤT MỚI (E4). Cố tình BẮT LỖI TO thay vì im lặng bỏ qua: một cờ
        # dòng lệnh không có tác dụng gì là cách nhanh nhất để tạo ra một bảng
        # kết quả sai mà không ai phát hiện.
        if not args.alpha_file:
            raise SystemExit(
                "--decay_mode corpus yêu cầu --alpha_file <đường dẫn JSON>.\n"
                "File đó do bước E0-b sinh ra: đo phân bố độ dài từ ghép trên chính "
                "corpus huấn luyện rồi ánh xạ sang alpha theo đề cương §6.2.\n"
                "Định dạng: {\"alpha\": [a1, ..., a_dmodel], \"source\": \"...\"}"
            )
        payload = json.loads(Path(args.alpha_file).read_text(encoding="utf-8"))
        alpha_values = payload["alpha"]
        print(f"[{run_name}] dùng alpha từ corpus: {args.alpha_file} "
              f"({len(alpha_values)} kênh, nguồn: {payload.get('source', 'không ghi')})")

    f_cfg = HyenaFilterConfig(
        use_window=not args.no_window,
        use_sine=not args.no_sine,
        learn_decay=args.learn_decay,
        normalize_filter=args.normalize_filter,
        alpha_values=alpha_values,
    )
    cfg = LMConfig(
        vocab_size=tok.vocab_size, d_model=args.d_model, layer_spec=args.layers,
        max_seq_len=args.seq_len, dropout=args.dropout, pos_emb=args.pos_emb,
        n_heads=args.n_heads, attn_impl=args.attn_impl,
        hyena_order=args.hyena_order, hyena_filter=f_cfg,
    )
    model = SequenceLM(cfg).to(device)
    breakdown = model.param_breakdown()
    print(f"[{run_name}] tham số: {breakdown['total']:,} "
          f"(mixer {breakdown['token_mixer']:,} · mlp {breakdown['mlp']:,} "
          f"· emb {breakdown['embedding']:,})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    tokens_per_step = args.batch_size * args.seq_len
    total_steps = args.token_budget // tokens_per_step
    warmup = max(int(total_steps * args.warmup_frac), 1)
    print(f"[{run_name}] {total_steps:,} bước × {tokens_per_step:,} token/bước "
          f"= {total_steps * tokens_per_step:,} token · warmup {warmup:,}")

    history: list[dict] = []
    step, seen_tokens = 0, 0
    t0 = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    done = False
    while not done:
        for x, y in dl_tr:
            if step >= total_steps:
                done = True
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, args.lr, warmup)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            step += 1
            seen_tokens += tokens_per_step

            if step % args.log_every == 0 or step == total_steps:
                el = time.time() - t0
                print(f"  b{step:>6}/{total_steps} | loss {loss.item():.4f} "
                      f"| |g| {gnorm:.2f} | {seen_tokens/1e6:.1f}M token "
                      f"| {el:.0f}s | {seen_tokens/max(el,1e-9)/1e3:.1f}k tok/s")

            if step % args.eval_every == 0 or step == total_steps:
                vl, vppl = evaluate(model, dl_va, device, args.eval_batches)
                history.append({
                    "step": step, "tokens": seen_tokens,
                    "train_loss": loss.item(), "val_loss": vl, "val_ppl": vppl,
                    "lr": opt.param_groups[0]["lr"],
                    "elapsed_s": time.time() - t0,
                })
                print(f"  --> val loss {vl:.4f} | val PPL {vppl:.2f}")

    # -- Đánh giá cuối trên tập TEST -----------------------------------------
    test_loss, test_ppl = evaluate(model, dl_te, device)
    wall = time.time() - t0
    peak_mem = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    print(f"[{run_name}] XONG · test PPL {test_ppl:.3f} · {wall:.0f}s "
          f"· đỉnh bộ nhớ {peak_mem:.0f} MB")

    summary = {
        "run_name": run_name,
        "test_loss": test_loss, "test_ppl": test_ppl,
        "final_val_ppl": history[-1]["val_ppl"] if history else None,
        "wall_time_s": wall, "peak_mem_mb": peak_mem,
        "tokens_seen": seen_tokens, "steps": step,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "params": breakdown,
        "corpus": stats_to_dict(stats),
        "config": {**vars(args), "layer_spec": cfg.layer_spec},
        "torch_version": torch.__version__,
    }
    (out_dir / f"{run_name}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if history:
        import csv
        with (out_dir / f"{run_name}_history.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
    print(f"[{run_name}] đã ghi kết quả vào {out_dir}")
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Huấn luyện LM Hyena/Transformer/lai")

    g = p.add_argument_group("kiến trúc")
    g.add_argument("--layers", default="HHHH",
                   help="chuỗi lớp: H=Hyena, A=Attention. VD: HHHH, AAAA, HHHA, HAHA")
    g.add_argument("--d_model", type=int, default=256)
    g.add_argument("--n_heads", type=int, default=8)
    g.add_argument("--hyena_order", type=int, default=2, help="bậc N (Def 3.1)")
    g.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "naive"])
    g.add_argument("--pos_emb", default="learned", choices=["learned", "none"])
    g.add_argument("--dropout", type=float, default=0.1)

    g = p.add_argument_group("ablation bộ lọc")
    g.add_argument("--no_window", action="store_true", help="bỏ cửa sổ suy giảm (E3a)")
    g.add_argument("--no_sine", action="store_true", help="dùng GELU thay sine (E3b)")
    g.add_argument("--learn_decay", action="store_true", help="học alpha (E3c)")
    g.add_argument("--normalize_filter", action="store_true")
    g.add_argument("--decay_mode", default="uniform",
                   choices=["uniform", "logspace", "corpus"],
                   help="uniform = doi chung 1 (alpha trai deu, khoang do nhom chon); "
                        "logspace = doi chung 2 CONG BANG (do dai hieu dung cach deu "
                        "theo log); corpus = de xuat moi cua nhom (E4). "
                        "Ket luan khoa hoc phai so voi logspace, khong phai uniform.")
    g.add_argument("--alpha_file", default=None,
                   help="JSON chứa alpha đo từ corpus; BẮT BUỘC khi --decay_mode corpus")

    g = p.add_argument_group("dữ liệu")
    g.add_argument("--lang", default="vi", choices=["vi", "en"])
    g.add_argument("--tokenizer", default="syllable", choices=["syllable", "bpe"])
    g.add_argument("--vocab_size", type=int, default=16000)
    g.add_argument("--n_docs", type=int, default=20000)
    g.add_argument("--seq_len", type=int, default=512)
    g.add_argument("--cache_dir", default=None,
                   help="thư mục cache của HuggingFace (không phải cache token)")
    g.add_argument("--token_cache", default="data_cache",
                   help="thư mục cache dòng token đã xử lý; dùng chung cho mọi "
                        "seed và mọi kiến trúc, tiết kiệm ~10 phút mỗi lần chạy")
    g.add_argument("--no_cache", action="store_true",
                   help="bỏ qua cache token, token hoá lại từ đầu")
    g.add_argument("--data_seed", type=int, default=0,
                   help="seed chọn tài liệu — GIỮ CỐ ĐỊNH giữa các lần chạy so sánh")
    g.add_argument("--max_train_tokens", type=int, default=None,
                   help="cắt tập TRAIN xuống đúng số token này, ĐỘC LẬP với "
                        "--token_budget (số bước vẫn tính theo budget). Dùng cho "
                        "nhánh đối chứng cùng số token (T13). Bỏ trống = cắt "
                        "theo token_budget như cũ.")

    g = p.add_argument_group("tối ưu")
    g.add_argument("--token_budget", type=int, default=40_000_000,
                   help="tổng số token huấn luyện — cố định để so sánh công bằng")
    g.add_argument("--batch_size", type=int, default=16)
    g.add_argument("--lr", type=float, default=3e-4)
    g.add_argument("--weight_decay", type=float, default=0.1)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--warmup_frac", type=float, default=0.05)
    g.add_argument("--no_amp", action="store_true")

    g = p.add_argument_group("chạy")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out_dir", default="results")
    g.add_argument("--run_name", default=None)
    g.add_argument("--log_every", type=int, default=100)
    g.add_argument("--eval_every", type=int, default=500)
    g.add_argument("--eval_batches", type=int, default=50)

    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())

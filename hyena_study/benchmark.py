"""
E5 — Đo thời gian và bộ nhớ theo độ dài chuỗi. KHÔNG huấn luyện, chỉ đo.

BA ĐIỀU PHẢI LÀM ĐÚNG, NẾU KHÔNG SỐ ĐO SẼ VÔ NGHĨA:

1. **Phải có ĐỦ BA đường cơ sở.** Paper tự ghi *"FlashAttention is already 2-4x
   faster than a standard attention implementation in PyTorch"*, và các con số
   tăng tốc mà paper công bố (2× ở 8K, 100× ở 64k) là so với **FlashAttention**,
   không phải so với attention ngây thơ. Chỉ so Hyena với `naive` là thổi phồng
   kết quả gấp mấy lần. Ta đo cả `naive`, `sdpa` và `hyena` rồi báo cáo cả ba.

2. **Phải đồng bộ CUDA trước khi bấm giờ.** Lệnh GPU chạy bất đồng bộ; không gọi
   `torch.cuda.synchronize()` thì ta đang đo tốc độ *xếp hàng lệnh*, không phải
   tốc độ tính toán. Đây là lỗi làm hỏng phần lớn các bảng benchmark tự làm.

3. **Phải khởi động trước (warm-up).** Lần gọi đầu tiên gánh chi phí biên dịch
   kernel, cấp phát bộ nhớ, chọn thuật toán cuFFT. Tính cả lần đó vào là bôi bẩn
   số đo.

Chạy:
    python -m hyena_study.benchmark --lengths 256 512 1024 2048 4096 8192
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .models import (
    AttentionConfig,
    CausalSelfAttention,
    HyenaConfig,
    HyenaOperator,
)


def build_operator(kind: str, d_model: int, n_heads: int, order: int,
                   max_seq_len: int) -> torch.nn.Module:
    if kind == "hyena":
        return HyenaOperator(HyenaConfig(d_model=d_model, order=order, dropout=0.0),
                             max_seq_len=max_seq_len)
    return CausalSelfAttention(AttentionConfig(d_model=d_model, n_heads=n_heads,
                                              dropout=0.0, impl=kind))


def measure(op: torch.nn.Module, B: int, L: int, D: int, device: torch.device,
            n_warmup: int, n_iter: int, backward: bool, dtype: torch.dtype) -> dict:
    """Đo một cấu hình. Trả về None-an-toàn bằng cách ném OOM lên trên để bên gọi bắt."""
    op = op.to(device=device, dtype=dtype).eval()
    x = torch.randn(B, L, D, device=device, dtype=dtype, requires_grad=backward)

    def one_pass() -> None:
        if backward:
            y = op(x)
            y.sum().backward()
            op.zero_grad(set_to_none=True)
            if x.grad is not None:
                x.grad = None
        else:
            with torch.no_grad():
                op(x)

    for _ in range(n_warmup):                       # điểm 3
        one_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()                    # điểm 2
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        one_pass()
    if device.type == "cuda":
        torch.cuda.synchronize()                    # điểm 2
    elapsed = time.perf_counter() - t0

    ms = elapsed / n_iter * 1000.0
    peak = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else float("nan")
    return {
        "time_ms": ms,
        "peak_mem_mb": peak,
        "throughput_tok_s": B * L / (ms / 1000.0),
    }


def run(args: argparse.Namespace) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("!!! CẢNH BÁO: đang đo trên CPU. Số liệu hiệu năng đưa vào báo cáo "
              "BẮT BUỘC phải đo trên Kaggle GPU — đặc tính tỉ lệ trên CPU khác hẳn.")
    dtype = {"fp32": torch.float32, "fp16": torch.float16,
             "bf16": torch.bfloat16}[args.dtype]

    rows: list[dict] = []
    for L in args.lengths:
        for kind in args.ops:
            op = build_operator(kind, args.d_model, args.n_heads, args.order, max_seq_len=L)
            try:
                res = measure(op, args.batch_size, L, args.d_model, device,
                              args.warmup, args.iters, args.backward, dtype)
                status = "ok"
            except torch.cuda.OutOfMemoryError:
                # OOM là một KẾT QUẢ, không phải sự cố: chính nó chứng minh rào
                # cản bộ nhớ O(L²). Paper cũng ghi nhận attention thường OOM ở 64k.
                torch.cuda.empty_cache()
                res = {"time_ms": float("nan"), "peak_mem_mb": float("nan"),
                       "throughput_tok_s": float("nan")}
                status = "OOM"
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                torch.cuda.empty_cache()
                res = {"time_ms": float("nan"), "peak_mem_mb": float("nan"),
                       "throughput_tok_s": float("nan")}
                status = "OOM"

            row = {"op": kind, "seq_len": L, "batch_size": args.batch_size,
                   "d_model": args.d_model, "dtype": args.dtype,
                   "backward": args.backward, "status": status, **res}
            rows.append(row)
            if status == "OOM":
                print(f"  L={L:>6} {kind:>6}: OOM (đây là kết quả có ý nghĩa, không phải lỗi)")
            else:
                print(f"  L={L:>6} {kind:>6}: {res['time_ms']:8.2f} ms "
                      f"| {res['peak_mem_mb']:8.1f} MB "
                      f"| {res['throughput_tok_s']/1e3:8.1f}k tok/s")
            del op
            if device.type == "cuda":
                torch.cuda.empty_cache()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"E5_benchmark_{args.dtype}{'_bwd' if args.backward else '_fwd'}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "config": vars(args),
    }
    (out_dir / "E5_benchmark_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # Bảng tăng tốc — luôn nêu RÕ đang so với đường cơ sở nào
    print("\nTăng tốc của Hyena (chỉ tính các cấu hình cả hai đều chạy được):")
    print(f"  {'L':>7} {'vs naive':>10} {'vs sdpa':>10}")
    by = {(r["op"], r["seq_len"]): r for r in rows}
    for L in args.lengths:
        h = by.get(("hyena", L))
        parts = []
        for base in ("naive", "sdpa"):
            b = by.get((base, L))
            if h and b and h["status"] == "ok" and b["status"] == "ok":
                parts.append(f"{b['time_ms'] / h['time_ms']:9.2f}x")
            elif b and b["status"] == "OOM":
                parts.append("  base OOM")
            else:
                parts.append("        --")
        print(f"  {L:>7} {parts[0]:>10} {parts[1]:>10}")

    print(f"\nĐã ghi: {csv_path}")
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E5 — benchmark thời gian & bộ nhớ theo độ dài chuỗi")
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[256, 512, 1024, 2048, 4096, 8192])
    p.add_argument("--ops", nargs="+", default=["hyena", "sdpa", "naive"],
                   choices=["hyena", "sdpa", "naive"])
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--order", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--backward", action="store_true",
                   help="đo cả lượt lùi (sát với huấn luyện hơn)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--out_dir", default="results")
    return p.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

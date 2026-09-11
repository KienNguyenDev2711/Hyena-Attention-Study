"""
Giao diện web cho demo sinh văn bản. Chạy MÔ HÌNH THẬT ngay trên máy, không gọi
dịch vụ nào bên ngoài.

VÌ SAO LÀ MÁY CHỦ CỤC BỘ CHỨ KHÔNG PHẢI TRANG WEB TĨNH:

Trọng số mỗi mô hình nặng khoảng 44 MB. Không thể nhét vào một trang web tĩnh để
chạy trong trình duyệt, và cũng không nên giả lập. Trang tĩnh chỉ trưng được các
mẫu đã sinh sẵn. Ở đây trình duyệt chỉ làm giao diện, còn mỗi lần bấm "Sinh tiếp"
là một lượt suy diễn PyTorch thật chạy trong tiến trình này.

CỐ Ý KHÔNG DÙNG THƯ VIỆN NGOÀI. Chỉ `http.server` của thư viện chuẩn. Lúc báo cáo
trước hội đồng mà phải cài thêm gói, hoặc phòng họp không có mạng, là hỏng buổi
trình bày. Cách này chạy được offline hoàn toàn.

PHẠM VI AN TOÀN: máy chủ này chỉ dành cho demo trên máy cá nhân. Mặc định chỉ lắng
nghe trên 127.0.0.1, không xác thực, không giới hạn tần suất. KHÔNG mở ra mạng
công cộng.

Ví dụ chạy:
    python -m hyena_study.serve
    python -m hyena_study.serve --port 8080 --no_browser
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import webbrowser
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .checkpoint import load_checkpoint
from .generate import generate_text

UI_DIR = Path(__file__).resolve().parent / "webui"
MAX_BODY = 64 * 1024          # prompt dài nhất chấp nhận, chặn thân yêu cầu khổng lồ
MAX_NEW_TOKENS = 200          # trần cứng, tránh một lần bấm nhầm treo máy chủ


# -----------------------------------------------------------------------------
# Nạp mô hình
# -----------------------------------------------------------------------------
class ModelBundle:
    """Một mô hình đã nạp sẵn, kèm siêu dữ liệu để hiển thị."""

    def __init__(self, key: str, name: str, path: Path, device: str):
        self.key, self.name, self.path = key, name, Path(path)
        t0 = time.time()
        self.model, self.tok, self.meta = load_checkpoint(self.path, device=device)
        self.load_s = time.time() - t0
        self.lock = threading.Lock()   # một mô hình chỉ phục vụ một lượt tại một thời điểm

    @property
    def info(self) -> dict:
        m = self.meta.get("metrics", {})
        return {
            "key": self.key,
            "name": self.name,
            "file": self.path.name,
            "run_name": self.meta.get("run_name"),
            "layer_spec": self.meta.get("lm_config", {}).get("layer_spec"),
            "test_ppl": m.get("test_ppl"),
            "params_total": m.get("params_total"),
            "tokens_seen": m.get("tokens_seen"),
            "steps": m.get("steps"),
            "vocab_size": self.tok.vocab_size,
            "max_seq_len": self.meta.get("lm_config", {}).get("max_seq_len"),
        }


def load_bundles(ckpt: str, compare: str | None, device: str) -> list[ModelBundle]:
    """Nạp một hoặc hai checkpoint. Tên hiển thị suy từ layer_spec, không đoán theo tên tệp."""
    specs = [("a", ckpt)] + ([("b", compare)] if compare else [])
    out = []
    for key, path in specs:
        b = ModelBundle(key, "…", Path(path), device)
        spec = (b.meta.get("lm_config", {}).get("layer_spec") or "").upper()
        if spec and set(spec) == {"H"}:
            b.name = "Hyena"
        elif spec and set(spec) == {"A"}:
            b.name = "Transformer"
        else:
            b.name = f"Lai ({spec})" if spec else b.path.stem
        print(f"[nap] {b.path.name} | {spec} | vocab {b.tok.vocab_size:,} "
              f"| test PPL {b.info['test_ppl']} | {b.load_s:.1f}s")
        out.append(b)
    return out


# -----------------------------------------------------------------------------
# Kiểm tra tham số gửi lên
# -----------------------------------------------------------------------------
def clean_params(raw: dict) -> dict:
    """Ép tham số về khoảng hợp lệ, báo lỗi RÕ thay vì âm thầm sửa sai.

    Máy chủ không tin dữ liệu từ trình duyệt: một `max_new_tokens` khổng lồ hay
    `top_k` âm đều làm treo hoặc làm hỏng phân phối lấy mẫu.
    """
    prompt = str(raw.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Chưa nhập prompt.")
    if len(prompt) > 4000:
        raise ValueError("Prompt dài quá 4000 ký tự.")

    def num(name, default, lo, hi, cast):
        v = raw.get(name, default)
        try:
            v = cast(v)
        except (TypeError, ValueError):
            raise ValueError(f"Tham số {name} không phải số: {v!r}")
        if not (lo <= v <= hi):
            raise ValueError(f"Tham số {name} phải nằm trong [{lo}; {hi}], nhận được {v}.")
        return v

    return {
        "prompt": prompt,
        "max_new_tokens": num("max_new_tokens", 40, 1, MAX_NEW_TOKENS, int),
        "temperature": num("temperature", 0.9, 0.0, 5.0, float),
        "top_k": num("top_k", 40, 0, 16000, int),
        "seed": num("seed", 0, 0, 10**9, int),
        "ban_invisible": bool(raw.get("ban_invisible", True)),
    }


def run_one(b: ModelBundle, p: dict, device: str) -> dict:
    """Sinh văn bản bằng một mô hình. Trả về đúng thứ giao diện cần hiển thị."""
    with b.lock:
        out = generate_text(
            b.model, b.tok, p["prompt"],
            max_new_tokens=p["max_new_tokens"], temperature=p["temperature"],
            top_k=p["top_k"], seed=p["seed"], device=device,
            ban_invisible=p["ban_invisible"],
        )
    info = b.info
    return {
        "key": b.key, "name": b.name, "run_name": info["run_name"],
        "layer_spec": info["layer_spec"], "test_ppl": info["test_ppl"],
        "prompt": out["prompt"], "continuation": out["continuation"],
        "n_new": out["n_new"], "stop_reason": out["stop_reason"],
        "seconds": out["seconds"],
        "ms_per_token": (out["seconds"] / out["n_new"] * 1000) if out["n_new"] else None,
        "prompt_n_tokens": out["prompt_n_tokens"], "prompt_unk": out["prompt_unk"],
        "n_banned_invisible": out["n_banned_invisible"],
    }


# -----------------------------------------------------------------------------
# HTTP
# -----------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "HyenaDemo/1.0"

    def __init__(self, *a, bundles=None, device="cpu", **kw):
        self.bundles, self.device = bundles, device
        super().__init__(*a, **kw)

    # bớt ồn: không in từng dòng log cho mỗi tệp tĩnh
    def log_message(self, fmt, *args):
        if self.path.startswith("/api/"):
            print(f"  [{self.log_date_time_string()}] {fmt % args}")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            f = UI_DIR / "index.html"
            if not f.exists():
                self._json(500, {"ok": False, "error": f"thiếu tệp giao diện {f}"})
                return
            self._send(200, f.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/meta":
            self._json(200, {"ok": True, "models": [b.info for b in self.bundles],
                             "device": self.device,
                             "max_new_tokens_cap": MAX_NEW_TOKENS})
        else:
            self._json(404, {"ok": False, "error": "không có đường dẫn này"})

    def do_POST(self):  # noqa: N802
        if self.path != "/api/generate":
            self._json(404, {"ok": False, "error": "không có đường dẫn này"})
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                raise ValueError("thân yêu cầu rỗng hoặc quá lớn")
            raw = json.loads(self.rfile.read(n).decode("utf-8"))
            p = clean_params(raw)
        except ValueError as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return
        except Exception as exc:  # noqa: BLE001
            self._json(400, {"ok": False, "error": f"yêu cầu không hợp lệ: {exc}"})
            return

        try:
            results = [run_one(b, p, self.device) for b in self.bundles]
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            return
        self._json(200, {"ok": True, "params": p, "results": results})


def build_server(bundles, device: str, host: str, port: int) -> ThreadingHTTPServer:
    handler = partial(Handler, bundles=bundles, device=device)
    return ThreadingHTTPServer((host, port), handler)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Giao diện web demo sinh văn bản")
    p.add_argument("--ckpt", default="results/DEMO_vi_HHHH_s0.pt")
    p.add_argument("--compare", default="results/DEMO_vi_AAAA_s0.pt",
                   help='checkpoint thứ hai để so sánh; đặt "" để tắt')
    p.add_argument("--host", default="127.0.0.1",
                   help="CHỈ nên để 127.0.0.1; máy chủ này không có xác thực")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no_browser", action="store_true", help="không tự mở trình duyệt")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for f in [args.ckpt] + ([args.compare] if args.compare else []):
        if not Path(f).exists():
            print(f"\nKHONG THAY CHECKPOINT: {f}")
            print("Chay notebooks/kaggle_demo_generate.ipynb tren Kaggle GPU, tai")
            print("demo_checkpoints.zip ve roi giai nen vao thu muc results/.\n")
            return 1

    bundles = load_bundles(args.ckpt, args.compare or None, args.device)
    try:
        httpd = build_server(bundles, args.device, args.host, args.port)
    except OSError as exc:
        print(f"\nKhong mo duoc cong {args.port}: {exc}")
        print("Cong dang bi chiem. Chay lai voi --port 8080 chang han.\n")
        return 1

    url = f"http://{args.host}:{args.port}/"
    print(f"\nGiao dien demo: {url}")
    print("Nhan Ctrl+C de dung.\n")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nDa dung may chu.")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

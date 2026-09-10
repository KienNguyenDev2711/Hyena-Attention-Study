"""
Lưu và nạp lại trọng số mô hình, phục vụ demo sinh văn bản.

VÌ SAO MODULE NÀY TỒN TẠI:

Toàn bộ thí nghiệm E1 tới E6 chỉ cần ĐIỂM SỐ, nên `train.py` trước đây huấn
luyện xong là vứt trọng số đi. Muốn demo sinh văn bản thì phải giữ lại. Nhưng
giữ mỗi `state_dict` là chưa đủ: nạp lại cần đúng `LMConfig` đã dựng ra nó, và
cần đúng BỘ TỪ ĐIỂN đã token hoá corpus, nếu không thì không giải mã được id
thành chữ.

Từ điển âm tiết được dựng TỪ CHÍNH CORPUS lúc chạy (`SyllableTokenizer.train`),
và phụ thuộc vào `n_docs`, `data_seed`, `vocab_size`. Một checkpoint đi kèm sai
từ điển sẽ giải mã ra chữ khác hoàn toàn mà KHÔNG báo lỗi gì. Đó là loại lỗi im
lặng nguy hiểm nhất, nên ở đây từ điển được nhúng THẲNG vào tệp checkpoint chứ
không để rời.

RÀNG BUỘC ĐỊNH DẠNG: payload chỉ chứa kiểu nguyên thuỷ (dict, list, str, số,
bool, None) và tensor. Nhờ vậy `torch.load(..., weights_only=True)` nạp được,
tức nạp checkpoint không thực thi mã tuỳ ý từ tệp.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch

from .data.corpus import TOKENIZERS
from .models import HyenaFilterConfig, LMConfig, SequenceLM

# Tăng số này khi đổi bố cục payload theo cách không tương thích ngược.
CKPT_FORMAT = 1


# -----------------------------------------------------------------------------
# Bộ từ điển: tuần tự hoá qua chính save/load của nó
# -----------------------------------------------------------------------------
def _tokenizer_to_text(tok, kind: str) -> str:
    """Ép bộ token hoá về một chuỗi văn bản.

    Cố ý đi qua đúng `tok.save()` của lớp gốc thay vì tự bới nội tạng: BPE lưu
    bằng định dạng riêng của thư viện `tokenizers`, tự chép tay sẽ sai.
    """
    if kind not in TOKENIZERS:
        raise ValueError(f"tokenizer phải thuộc {list(TOKENIZERS)}, nhận được {kind!r}")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"tokenizer_{kind}.json"
        tok.save(p)
        return p.read_text(encoding="utf-8")


def _tokenizer_from_text(blob: str, kind: str):
    if kind not in TOKENIZERS:
        raise ValueError(f"tokenizer phải thuộc {list(TOKENIZERS)}, nhận được {kind!r}")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / f"tokenizer_{kind}.json"
        p.write_text(blob, encoding="utf-8")
        return TOKENIZERS[kind].load(p)


# -----------------------------------------------------------------------------
# Cấu hình mô hình: dataclass <-> dict
# -----------------------------------------------------------------------------
def lm_config_to_dict(cfg: LMConfig) -> dict:
    return asdict(cfg)


def lm_config_from_dict(d: dict) -> LMConfig:
    d = dict(d)
    filt = d.pop("hyena_filter", None)
    if filt is not None:
        d["hyena_filter"] = HyenaFilterConfig(**filt)
    return LMConfig(**d)


# -----------------------------------------------------------------------------
# Ghi
# -----------------------------------------------------------------------------
def build_payload(*, model, cfg: LMConfig, tok, tokenizer_kind: str,
                  run_name: str, metrics: dict | None = None,
                  corpus: dict | None = None,
                  train_config: dict | None = None,
                  extra: dict | None = None) -> dict:
    """Gói mọi thứ cần để dựng lại mô hình và giải mã đầu ra."""
    if cfg.vocab_size != tok.vocab_size:
        raise ValueError(
            f"vocab_size của mô hình ({cfg.vocab_size}) khác của bộ từ điển "
            f"({tok.vocab_size}). Checkpoint này sẽ giải mã sai, từ chối lưu."
        )
    return {
        "format_version": CKPT_FORMAT,
        "run_name": run_name,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "lm_config": lm_config_to_dict(cfg),
        "tokenizer_kind": tokenizer_kind,
        "tokenizer_blob": _tokenizer_to_text(tok, tokenizer_kind),
        "metrics": metrics or {},
        "corpus": corpus or {},
        "train_config": _primitives_only(train_config or {}),
        "extra": _primitives_only(extra or {}),
        # str() BẮT BUỘC: `torch.__version__` là đối tượng TorchVersion (lớp con
        # của str), và `torch.load(weights_only=True)` từ chối mọi lớp không nằm
        # trong danh sách cho phép. Để nguyên thì checkpoint ghi được nhưng KHÔNG
        # nạp lại được. Test G1 bắt đúng lỗi này.
        "torch_version": str(torch.__version__),
    }


def _primitives_only(d: dict) -> dict:
    """Bỏ mọi giá trị không phải kiểu nguyên thuỷ.

    `vars(args)` của argparse thường sạch, nhưng nếu lọt vào một đối tượng lạ thì
    `torch.load(weights_only=True)` sẽ từ chối nạp, mà lúc đó checkpoint đã ghi
    xong rồi. Lọc ngay lúc ghi để hỏng sớm và hỏng rõ.
    """
    ok = (str, int, float, bool, type(None))
    out = {}
    for k, v in d.items():
        if isinstance(v, ok):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, ok) for x in v):
            out[k] = list(v)
        elif isinstance(v, dict):
            out[k] = _primitives_only(v)
        else:
            out[k] = repr(v)
    return out


def save_checkpoint(path: str | Path, **kwargs) -> Path:
    """Ghi checkpoint. Tham số giống hệt `build_payload`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(build_payload(**kwargs), path)
    return path


# -----------------------------------------------------------------------------
# Đọc
# -----------------------------------------------------------------------------
def load_checkpoint(path: str | Path, device: torch.device | str | None = None):
    """Dựng lại (model đã eval, tokenizer, meta) từ tệp checkpoint.

    Returns:
        (model, tokenizer, meta) với `meta` là toàn bộ payload trừ trọng số.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"không thấy checkpoint: {path}")
    device = torch.device(device) if device is not None else torch.device("cpu")

    payload = torch.load(path, map_location="cpu", weights_only=True)

    fmt = payload.get("format_version")
    if fmt != CKPT_FORMAT:
        raise ValueError(
            f"checkpoint có format_version={fmt} nhưng mã hiện tại đọc "
            f"format_version={CKPT_FORMAT}. Không đoán mò, hãy sinh lại checkpoint."
        )

    cfg = lm_config_from_dict(payload["lm_config"])
    tok = _tokenizer_from_text(payload["tokenizer_blob"], payload["tokenizer_kind"])
    if cfg.vocab_size != tok.vocab_size:
        raise ValueError(
            f"checkpoint hỏng: vocab_size mô hình {cfg.vocab_size} khác từ điển "
            f"{tok.vocab_size}"
        )

    # strict=True: thà hỏng to còn hơn nạp thiếu trọng số rồi sinh ra chữ vô nghĩa
    # mà tưởng là mô hình học kém. Lưu ý alpha của bộ lọc là buffer KHÔNG bền
    # (persistent=False) nên không nằm trong state_dict; nó được dựng lại từ
    # `lm_config`, đó chính là lý do cấu hình phải đi kèm trọng số.
    model = SequenceLM(cfg)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()

    meta = {k: v for k, v in payload.items() if k not in ("model_state", "tokenizer_blob")}
    meta["tokenizer_vocab_size"] = tok.vocab_size
    return model, tok, meta

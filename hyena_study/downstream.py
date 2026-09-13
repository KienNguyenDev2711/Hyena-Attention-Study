"""
Đánh giá NGOẠI TẠI (extrinsic) trên tác vụ hạ nguồn — bổ sung cho đánh giá NỘI TẠI
(perplexity) ở E1.

VÌ SAO PHẢI CÓ PHẦN NÀY:

Perplexity là thước đo NỘI TẠI: nó chỉ nói mô hình dự đoán token kế tiếp tốt đến
đâu TRÊN CHÍNH phân phối dữ liệu huấn luyện. Nó KHÔNG trả lời được câu hỏi mà
người dùng mô hình thực sự quan tâm: biểu diễn học được có dùng được cho việc
khác không. Hai mô hình chênh nhau vài điểm PPL hoàn toàn có thể ngang nhau (hoặc
đảo chiều) khi đem đi phân loại. Vì vậy kết luận "toán tử nào tốt hơn" mà chỉ dựa
trên PPL là kết luận thiếu căn cứ.

THIẾT KẾ SO SÁNH — ba thứ bắt buộc phải có, thiếu thứ nào cũng làm mất kết luận:

1. ĐƯỜNG CƠ SỞ NGẪU NHIÊN (`--init random`). Cùng kiến trúc, KHÔNG tiền huấn
   luyện. Thiếu nó thì không biết điểm số đạt được là nhờ tiền huấn luyện hay chỉ
   nhờ kiến trúc + đầu phân loại. Đây là lỗi phổ biến nhất trong các báo cáo
   "fine-tune cho kết quả tốt".

2. ĐƯỜNG CƠ SỞ LỚP ĐA SỐ (majority). Tập dữ liệu lệch lớp nặng (nhãn `neutral`
   chỉ chiếm khoảng 4%). Chỉ báo accuracy thì một mô hình đoán bừa lớp đông nhất
   đã có vẻ "khá tốt". Vì thế macro-F1 mới là chỉ số chính.

3. GIAO THỨC GIỐNG HỆT NHAU cho Hyena và Transformer: cùng đầu phân loại, cùng
   cách gộp biểu diễn, cùng siêu tham số, cùng số seed. Đây đúng là nguyên tắc đã
   dùng ở E1 — chỉ thay đổi DUY NHẤT toán tử trộn token.

VỀ CÁCH GỘP BIỂU DIỄN (pooling) — một điểm tinh vi:

Cả Hyena lẫn Attention trong bài này đều NHÂN QUẢ: trạng thái ở vị trí t chỉ thấy
các token <= t. Do đó chỉ vị trí CUỐI CÙNG mới nhìn thấy trọn câu, nên `last` là
lựa chọn có cơ sở. `mean` (trung bình có mặt nạ) thường ổn định hơn nhưng trộn cả
những vị trí đầu câu mới thấy được vài token. Cả hai đều được cài đặt và chạy cho
CẢ HAI kiến trúc, báo cáo cả hai — KHÔNG chọn cái nào cho ra số đẹp hơn rồi mới
báo cáo.

VỀ ĐỆM (padding) — MỘT CẠM BẪY RIÊNG CỦA HYENA, ĐO ĐƯỢC BẰNG SỐ:

Trực giác thông thường là "mô hình nhân quả thì đệm bên phải vô hại, vì đầu ra tại
t không phụ thuộc đầu vào tại > t". Với Attention điều đó ĐÚNG (lệch cỡ 1e-7, mức
làm tròn). Với Hyena điều đó SAI.

Lý do: bộ lọc ngầm của Hyena lấy tham số trên trục thời gian ĐÃ CHUẨN HOÁ,
`t = linspace(0, 1, L)` (`models/hyena.py`, PositionalEncoding.forward và
HyenaFilter.window). L ở đây là độ dài của batch hiện tại. Đổi L là co giãn TOÀN
BỘ bộ lọc, nên hệ số tại mỗi độ trễ đổi theo, và đầu ra tại các vị trí THẬT cũng
đổi theo — dù không hề có rò rỉ thông tin từ tương lai.

Đo được trên mô hình tí hon (test D3): thêm ĐÚNG MỘT token đệm đã làm trạng thái
ẩn lệch 3,2e-3; thêm 20 token lệch 9,9e-3. Đệm về ĐỘ DÀI CỐ ĐỊNH thì lệch đúng 0.

Hai hệ quả, cả hai đều làm hỏng so sánh nếu bỏ qua:
  (a) Điểm số của Hyena sẽ phụ thuộc vào việc câu nào rơi chung batch với câu nào.
  (b) Bộ lọc lúc suy diễn sẽ khác bộ lọc lúc tiền huấn luyện (khi đó L luôn = 512).

Vì vậy ở đây MỌI chuỗi đều được đệm về cùng một độ dài cố định `--pad_to`, mặc
định bằng `max_seq_len` của chính mô hình (512) — đúng độ dài đã dùng khi tiền
huấn luyện. Đắt hơn về tính toán, nhưng đây là điều kiện đúng. Test D3 kiểm chứng
cả hai chiều: biến thiên khi L đổi, và bằng 0 khi L cố định.

HẠN CHẾ ĐÃ BIẾT, PHẢI GHI TRONG BÁO CÁO:
  - Câu trong VSFC RẤT NGẮN (trung vị 11 token). Tác vụ này KHÔNG kiểm tra được
    lợi thế tầm xa của Hyena. Nó kiểm tra chất lượng biểu diễn, không kiểm tra
    khả năng xử lý ngữ cảnh dài.
  - Chỉ có checkpoint seed 0 (E1 gốc không lưu trọng số). Nhiều seed ở đây là
    seed TINH CHỈNH, không phải seed TIỀN HUẤN LUYỆN. Độ lệch báo cáo vì vậy là
    độ lệch của bước tinh chỉnh, KHÔNG bao gồm độ lệch của bước tiền huấn luyện.

Ví dụ chạy:
    python -m hyena_study.downstream --smoke
    python -m hyena_study.downstream --ckpt results/DEMO_vi_HHHH_s0.pt \\
        --compare results/DEMO_vi_AAAA_s0.pt --task sentiment --mode probe
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .checkpoint import load_checkpoint, lm_config_from_dict
from .data.corpus import PAD, UNK, normalize_text
from .models import SequenceLM
from .train import set_seed

# -----------------------------------------------------------------------------
# Bộ dữ liệu
# -----------------------------------------------------------------------------
# Nguồn: UIT-VSFC (Vietnamese Students' Feedback Corpus), bản do CHÍNH nhóm
# UIT-NLP đăng. Trích dẫn gốc: Kiet Van Nguyen, Vu Duc Nguyen, Phu X. V. Nguyen,
# Tham T. H. Truong, Ngan Luu-Thuy Nguyen, "UIT-VSFC: Vietnamese Students'
# Feedback Corpus for Sentiment Analysis", KSE 2018, tr. 19-24.
#
# ⚠ Giấy phép trên HuggingFace ghi "unknown". Dùng cho mục đích nghiên cứu/học
# tập; nếu công bố thì phải xin phép nhóm tác giả. Đã ghi rõ, không lờ đi.
VSFC_REPO = "uitnlp/vietnamese_students_feedback"
VSFC_URL = ("https://huggingface.co/api/datasets/" + VSFC_REPO
            + "/parquet/default/{split}/0.parquet")
SPLITS = ("train", "validation", "test")

# Ánh xạ nhãn: xác minh bằng cách đối chiếu bản `uitnlp` (chỉ có cột số) với bản
# `tridm/UIT-VSFC` (có cả cột chuỗi lẫn cột số) — phân bố đếm khớp tuyệt đối ở
# cả ba split. KHÔNG phải suy đoán theo thứ tự bảng chữ cái.
TASKS = {
    "sentiment": {"column": "sentiment",
                  "names": ["negative", "neutral", "positive"]},
    "topic": {"column": "topic",
              "names": ["lecturer", "program", "facility", "others"]},
}


def download_vsfc(data_dir: str | Path) -> dict[str, Path]:
    """Tải parquet VSFC về `data_dir`, bỏ qua tệp đã có.

    Tải thẳng parquet thay vì dùng `datasets.load_dataset` để không phụ thuộc
    phiên bản thư viện và không cần script thực thi từ xa.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for split in SPLITS:
        f = data_dir / f"vsfc_{split}.parquet"
        if not f.exists():
            url = VSFC_URL.format(split=split)
            print(f"[tai] {split} <- {url}")
            with urllib.request.urlopen(url, timeout=120) as r:
                f.write_bytes(r.read())
        out[split] = f
    return out


def load_vsfc(data_dir: str | Path, task: str) -> dict[str, tuple[list[str], list[int]]]:
    """Trả về {split: (câu, nhãn)} cho một tác vụ."""
    if task not in TASKS:
        raise ValueError(f"task không hợp lệ: {task}; chỉ nhận {sorted(TASKS)}")
    import pyarrow.parquet as pq

    files = download_vsfc(data_dir)
    col = TASKS[task]["column"]
    n_classes = len(TASKS[task]["names"])
    out = {}
    for split, path in files.items():
        t = pq.read_table(path)
        if "sentence" not in t.column_names or col not in t.column_names:
            raise ValueError(f"{path} thiếu cột: có {t.column_names}")
        sents = t.column("sentence").to_pylist()
        labels = t.column(col).to_pylist()
        bad = [y for y in labels if not (0 <= int(y) < n_classes)]
        if bad:
            raise ValueError(f"{split}: nhãn ngoài khoảng [0;{n_classes}): {bad[:5]}")
        out[split] = (sents, [int(y) for y in labels])
    return out


def encode_dataset(sents: list[str], labels: list[int], tok, max_len: int
                   ) -> tuple[list[list[int]], list[int], dict]:
    """Token hoá + cắt bớt. Trả về thêm thống kê OOV/cắt bớt để BÁO CÁO, không giấu."""
    ids_all, n_tok, n_unk, n_trunc, n_empty = [], 0, 0, 0, 0
    for s in sents:
        ids = tok.encode(normalize_text(s))
        if not ids:
            ids = [UNK]          # câu rỗng sau chuẩn hoá: giữ 1 token để không vỡ batch
            n_empty += 1
        if len(ids) > max_len:
            ids = ids[:max_len]
            n_trunc += 1
        n_tok += len(ids)
        n_unk += sum(1 for i in ids if i == UNK)
        ids_all.append(ids)
    stats = {
        "n_examples": len(sents), "n_tokens": n_tok,
        "oov_rate": (n_unk / n_tok) if n_tok else 0.0,
        "n_truncated": n_trunc, "n_empty": n_empty,
        "len_mean": float(np.mean([len(x) for x in ids_all])) if ids_all else 0.0,
        "len_max": int(max((len(x) for x in ids_all), default=0)),
    }
    return ids_all, list(labels), stats


def make_batches(ids: list[list[int]], labels: list[int], batch_size: int,
                 shuffle: bool, rng: np.random.Generator | None = None,
                 pad_to: int | None = None):
    """Sinh batch đã đệm PHẢI, tất cả về cùng độ dài `pad_to`.

    `pad_to=None` chỉ dùng cho test: nó đệm theo câu dài nhất trong batch, và với
    Hyena cách đó làm kết quả phụ thuộc cách chia batch (xem giải thích ở đầu
    tệp). Đường chạy thật LUÔN truyền `pad_to`.
    """
    order = np.arange(len(ids))
    if shuffle:
        if rng is None:
            raise ValueError("shuffle=True thì phải truyền rng")
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        sel = order[i:i + batch_size]
        chunk = [ids[j] for j in sel]
        L = max(len(c) for c in chunk) if pad_to is None else pad_to
        if any(len(c) > L for c in chunk):
            raise ValueError(f"có câu dài hơn pad_to={L}; hãy giảm max_len")
        x = np.full((len(chunk), L), PAD, dtype=np.int64)
        lens = np.zeros(len(chunk), dtype=np.int64)
        for r, c in enumerate(chunk):
            x[r, :len(c)] = c
            lens[r] = len(c)
        yield (torch.from_numpy(x), torch.from_numpy(lens),
               torch.tensor([labels[j] for j in sel], dtype=torch.long))


# -----------------------------------------------------------------------------
# Mô hình phân loại
# -----------------------------------------------------------------------------
def pool_hidden(h: torch.Tensor, lens: torch.Tensor, how: str) -> torch.Tensor:
    """Gộp (B, L, D) -> (B, D), bỏ qua vị trí đệm.

    `last`: lấy trạng thái tại token THẬT cuối cùng (index lens-1) — không phải
    index L-1, vì L-1 có thể là token đệm.
    `mean`: trung bình trên các vị trí thật.
    """
    if how == "last":
        idx = (lens - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(1, idx.to(h.device)).squeeze(1)
    if how == "mean":
        L = h.size(1)
        mask = (torch.arange(L, device=h.device)[None, :] < lens.to(h.device)[:, None])
        mask = mask.unsqueeze(-1).to(h.dtype)
        return (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    raise ValueError(f"pooling không hợp lệ: {how}; chỉ nhận last/mean")


class Classifier(nn.Module):
    """Thân mô hình ngôn ngữ + đầu tuyến tính.

    Đầu phân loại CỐ Ý chỉ là một lớp tuyến tính: thêm MLP nhiều lớp sẽ làm mờ
    câu hỏi cần trả lời (biểu diễn của thân mô hình tốt đến đâu) vì đầu phân loại
    đủ mạnh có thể tự học bù cho biểu diễn kém.
    """

    def __init__(self, backbone: SequenceLM, n_classes: int, pooling: str,
                 freeze: bool):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        self.freeze = freeze
        self.head = nn.Linear(backbone.cfg.d_model, n_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            self.backbone.eval()           # tắt dropout của thân khi đóng băng
            with torch.no_grad():
                h = self.backbone.hidden_states(x)
        else:
            h = self.backbone.hidden_states(x)
        return self.head(pool_hidden(h, lens, self.pooling))


def build_backbone(ckpt: str | Path | None, init: str, device: str,
                   ref_ckpt: str | Path | None = None):
    """Dựng thân mô hình.

    `init="pretrained"`: nạp trọng số từ checkpoint.
    `init="random"`: dựng ĐÚNG kiến trúc đó nhưng trọng số ngẫu nhiên — đường cơ
    sở đối chứng. Vẫn phải đọc checkpoint để lấy đúng cấu hình và bộ token hoá,
    nếu không thì hai nhánh khác nhau ở nhiều biến.
    """
    src = ckpt if ckpt is not None else ref_ckpt
    if src is None:
        raise ValueError("cần checkpoint để lấy cấu hình kiến trúc")
    model, tok, meta = load_checkpoint(src, device=device)
    if init == "pretrained":
        return model, tok, meta
    if init != "random":
        raise ValueError(f"init không hợp lệ: {init}; chỉ nhận pretrained/random")
    cfg = lm_config_from_dict(meta["lm_config"])
    fresh = SequenceLM(cfg).to(device)
    return fresh, tok, meta


# -----------------------------------------------------------------------------
# Chỉ số
# -----------------------------------------------------------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 tự cài để không phụ thuộc sklearn trên Kaggle.

    Lớp không có mẫu thật lẫn mẫu dự đoán nào -> F1 = 0 cho lớp đó (quy ước giống
    sklearn với zero_division=0). Đã đối chiếu với sklearn trong test D6.
    """
    f1s = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(f1s))


def majority_baseline(train_labels: list[int], eval_labels: list[int],
                      n_classes: int) -> dict:
    """Luôn đoán lớp đông nhất của tập huấn luyện."""
    c = Counter(train_labels).most_common(1)[0][0]
    y_true = np.asarray(eval_labels)
    y_pred = np.full_like(y_true, c)
    return {"accuracy": float((y_pred == y_true).mean()),
            "macro_f1": macro_f1(y_true, y_pred, n_classes),
            "predicted_class": int(c)}


@torch.no_grad()
def evaluate(clf: Classifier, ids, labels, n_classes: int, batch_size: int,
             device: str, pad_to: int) -> dict:
    clf.eval()
    preds, trues = [], []
    for x, lens, y in make_batches(ids, labels, batch_size, False, pad_to=pad_to):
        logits = clf(x.to(device), lens.to(device))
        preds.append(logits.argmax(-1).cpu().numpy())
        trues.append(y.numpy())
    yp, yt = np.concatenate(preds), np.concatenate(trues)
    return {"accuracy": float((yp == yt).mean()),
            "macro_f1": macro_f1(yt, yp, n_classes), "n": int(len(yt))}


@torch.no_grad()
def extract_features(backbone: SequenceLM, ids, pooling: str, batch_size: int,
                     pad_to: int, device: str) -> torch.Tensor:
    """Tinh san vecto gop cho che do probe.

    Than mo hinh bi dong bang va chay o che do eval (khong dropout), nen dac
    trung la TAT DINH: tinh mot lan roi dung lai cho moi seed va moi epoch. Nho
    vay probe chay trong vai giay thay vi vai phut, va dieu do cho phep giu du
    so seed lan duong co so ngau nhien trong ngan sach GPU.
    """
    backbone.eval()
    out = []
    for x, lens, _ in make_batches(ids, [0] * len(ids), batch_size, False,
                                   pad_to=pad_to):
        h = backbone.hidden_states(x.to(device))
        out.append(pool_hidden(h, lens.to(device), pooling).float().cpu())
    return torch.cat(out, dim=0)


def _eval_head(head: nn.Module, feats: torch.Tensor, labels: list[int],
               n_classes: int, device: str) -> dict:
    head.eval()
    with torch.no_grad():
        yp = head(feats.to(device)).argmax(-1).cpu().numpy()
    yt = np.asarray(labels)
    return {"accuracy": float((yp == yt).mean()),
            "macro_f1": macro_f1(yt, yp, n_classes), "n": int(len(yt))}


# -----------------------------------------------------------------------------
# Mot lan chay tinh chinh
# -----------------------------------------------------------------------------
def run_one(*, data, tok, ckpt, init, task: str, mode: str, pooling: str,
            seed: int, epochs: int, batch_size: int, lr: float,
            weight_decay: float, grad_clip: float, max_len: int,
            device: str, pad_to: int | None = None, ref_ckpt=None,
            feature_cache: dict | None = None, log: bool = True) -> dict:
    """Huan luyen dau phan loai (va ca than neu mode=finetune), chon theo VALIDATION.

    Chon mo hinh theo macro-F1 tren tap validation roi moi bao cao tren tap test.
    KHONG duoc chon theo test - lam the la ro ri tap test va con so bao cao se
    lac quan gia.

    `feature_cache`: tu dien do ben goi cap, dung lai dac trung probe giua cac
    seed. Truyen None thi van chay dung, chi cham hon.
    """
    set_seed(seed)
    n_classes = len(TASKS[task]["names"])
    backbone, _, meta = build_backbone(ckpt, init, device, ref_ckpt=ref_ckpt)
    freeze = (mode == "probe")
    if pad_to is None:
        pad_to = backbone.cfg.max_seq_len
    if max_len > pad_to:
        raise ValueError(f"max_len={max_len} vuot pad_to={pad_to}")

    enc = {s: encode_dataset(*data[s], tok=tok, max_len=max_len) for s in SPLITS}
    tr_ids, tr_y, tr_stats = enc["train"]
    va_ids, va_y, _ = enc["validation"]
    te_ids, te_y, te_stats = enc["test"]

    clf = Classifier(backbone, n_classes, pooling, freeze).to(device)
    params = [p for p in clf.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    best = {"val_macro_f1": -1.0}
    t0 = time.time()

    if freeze:
        # ---- probe: dac trung co dinh, chi hoc dau tuyen tinh ----
        # Dac trung KHONG phu thuoc tac vu (hai tac vu dung chung cau), nen khoa
        # dem lay theo van tay cua chinh chuoi da token hoa, khong lay theo ten
        # tac vu - nho vay probe cua "topic" dung lai duoc dac trung cua
        # "sentiment" thay vi tinh lai tu dau.
        fp = (len(tr_ids), len(va_ids), len(te_ids),
              tuple(tr_ids[0][:8]), tuple(te_ids[-1][:8]))
        # init='random' dung than KHAC NHAU o moi seed, nen seed phai nam trong
        # khoa. Bo qua diem nay se lam moi seed dung chung mot dac trung va
        # do lech cua duong doi chung bi hep lai mot cach gia tao.
        ck = (str(ckpt), init, pooling, pad_to, max_len, fp,
              seed if init == 'random' else None)
        if feature_cache is not None and ck in feature_cache:
            fe = feature_cache[ck]
        else:
            fe = {sp: extract_features(backbone, enc[sp][0], pooling, batch_size,
                                       pad_to, device) for sp in SPLITS}
            if feature_cache is not None:
                feature_cache[ck] = fe
        head = clf.head
        for ep in range(epochs):
            head.train()
            order = np.arange(len(tr_y))
            rng.shuffle(order)
            tot, nb = 0.0, 0
            for i in range(0, len(order), batch_size):
                sel = order[i:i + batch_size]
                xb = fe["train"][sel].to(device)
                yb = torch.tensor([tr_y[j] for j in sel], dtype=torch.long,
                                  device=device)
                loss = F.cross_entropy(head(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
                tot += float(loss.item())
                nb += 1
            va = _eval_head(head, fe["validation"], va_y, n_classes, device)
            if log:
                print(f"    ep{ep + 1}/{epochs} loss {tot / max(nb, 1):.4f} "
                      f"| val acc {va['accuracy']:.4f} f1 {va['macro_f1']:.4f}")
            if va["macro_f1"] > best["val_macro_f1"]:
                te = _eval_head(head, fe["test"], te_y, n_classes, device)
                best = {"epoch": ep + 1, "val_accuracy": va["accuracy"],
                        "val_macro_f1": va["macro_f1"],
                        "test_accuracy": te["accuracy"],
                        "test_macro_f1": te["macro_f1"]}
    else:
        # ---- finetune: lan truyen nguoc qua toan bo than ----
        for ep in range(epochs):
            clf.train()
            tot, nb = 0.0, 0
            for x, lens, y in make_batches(tr_ids, tr_y, batch_size, True, rng,
                                           pad_to=pad_to):
                loss = F.cross_entropy(clf(x.to(device), lens.to(device)),
                                       y.to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
                tot += float(loss.item())
                nb += 1
            va = evaluate(clf, va_ids, va_y, n_classes, batch_size, device, pad_to)
            if log:
                print(f"    ep{ep + 1}/{epochs} loss {tot / max(nb, 1):.4f} "
                      f"| val acc {va['accuracy']:.4f} f1 {va['macro_f1']:.4f}")
            if va["macro_f1"] > best["val_macro_f1"]:
                te = evaluate(clf, te_ids, te_y, n_classes, batch_size, device,
                              pad_to)
                best = {"epoch": ep + 1, "val_accuracy": va["accuracy"],
                        "val_macro_f1": va["macro_f1"],
                        "test_accuracy": te["accuracy"],
                        "test_macro_f1": te["macro_f1"]}

    return {
        "task": task, "mode": mode, "pooling": pooling, "init": init,
        "seed": seed, "lr": lr, "epochs": epochs, "batch_size": batch_size,
        "pad_to": pad_to, "max_len": max_len,
        "layer_spec": meta.get("lm_config", {}).get("layer_spec"),
        "run_name": meta.get("run_name"),
        "pretrain_test_ppl": meta.get("metrics", {}).get("test_ppl"),
        "n_trainable_params": int(sum(p.numel() for p in params)),
        "n_backbone_params": int(sum(p.numel() for p in clf.backbone.parameters())),
        "train_oov_rate": tr_stats["oov_rate"], "test_oov_rate": te_stats["oov_rate"],
        "train_len_max": tr_stats["len_max"], "test_len_max": te_stats["len_max"],
        "n_truncated_train": tr_stats["n_truncated"],
        "seconds": time.time() - t0,
        **best,
    }


# -----------------------------------------------------------------------------
# Tổng hợp nhiều seed
# -----------------------------------------------------------------------------
# t hai phía, alpha=0.05. Chỉ liệt kê các bậc tự do thực sự dùng; thiếu thì báo
# lỗi thay vì lấy đại một giá trị gần đúng.
T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def summarize(runs: list[dict], key: str) -> dict:
    """Trung bình + khoảng tin cậy 95% theo phân phối t.

    n=1 thì KHÔNG có khoảng tin cậy — trả về None chứ không bịa ra số 0.
    """
    v = np.asarray([r[key] for r in runs], dtype=float)
    n = len(v)
    out = {"mean": float(v.mean()), "n_seeds": n,
           "values": [float(x) for x in v]}
    if n < 2:
        out["std"] = None; out["ci95"] = None
        return out
    sd = float(v.std(ddof=1))
    df = n - 1
    if df not in T_CRIT:
        raise ValueError(f"chưa có t tới hạn cho df={df}")
    out["std"] = sd
    out["ci95"] = float(T_CRIT[df] * sd / np.sqrt(n))
    return out


def compare_arms(a: dict, b: dict, key: str) -> dict:
    """So hai nhánh bằng khoảng tin cậy: CHỒNG LẤN thì KHÔNG kết luận được.

    Đây là cách diễn giải thận trọng, nhất quán với phần nội tại của báo cáo.
    Khoảng tin cậy tách rời là bằng chứng đủ mạnh; chồng lấn thì phải nói thẳng
    là chưa phân biệt được, KHÔNG được viết "A tốt hơn B" chỉ vì trung bình cao hơn.
    """
    sa, sb = summarize(a["runs"], key), summarize(b["runs"], key)
    if sa["ci95"] is None or sb["ci95"] is None:
        return {"metric": key, "a": sa, "b": sb, "separated": None,
                "verdict": "chỉ có 1 seed — không đánh giá được độ lệch"}
    lo_a, hi_a = sa["mean"] - sa["ci95"], sa["mean"] + sa["ci95"]
    lo_b, hi_b = sb["mean"] - sb["ci95"], sb["mean"] + sb["ci95"]
    sep = (lo_a > hi_b) or (lo_b > hi_a)
    if sep:
        win = a["name"] if sa["mean"] > sb["mean"] else b["name"]
        verdict = f"tách rời — {win} cao hơn ở cùng siêu tham số và cùng quy mô này"
    else:
        verdict = "chồng lấn — chưa phân biệt được ở số seed hiện có"
    return {"metric": key, "a": sa, "b": sb, "separated": bool(sep),
            "verdict": verdict}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Đánh giá ngoại tại trên tác vụ hạ nguồn UIT-VSFC")
    g = p.add_argument_group("mô hình")
    g.add_argument("--ckpt", default="results/DEMO_vi_HHHH_s0.pt")
    g.add_argument("--compare", default="results/DEMO_vi_AAAA_s0.pt",
                   help='checkpoint thứ hai; đặt "" để chỉ chạy một mô hình')
    g.add_argument("--no_random_control", action="store_true",
                   help="BỎ đường cơ sở ngẫu nhiên (KHÔNG khuyến khích: mất đối chứng)")

    g = p.add_argument_group("tác vụ")
    g.add_argument("--task", default="both",
                   choices=["sentiment", "topic", "both"])
    g.add_argument("--mode", default="both", choices=["probe", "finetune", "both"],
                   help="probe = đóng băng thân; finetune = huấn luyện toàn bộ")
    g.add_argument("--pooling", default="last", choices=["last", "mean", "both"],
                   help="last la lua chon co co so voi mo hinh nhan qua va duoc "
                        "CHON TRUOC khi xem ket qua")
    g.add_argument("--max_len", type=int, default=256,
                   help="cat bot cau dai hon nguong nay")
    g.add_argument("--pad_to", type=int, default=None,
                   help="dem MOI chuoi ve dung do dai nay; mac dinh = max_seq_len "
                        "cua mo hinh (512), bang dung do dai luc tien huan luyen. "
                        "BAT BUOC co dinh: bo loc Hyena phu thuoc L (xem dau tep)")

    g = p.add_argument_group("tối ưu")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    g.add_argument("--epochs", type=int, default=5)
    g.add_argument("--batch_size", type=int, default=32)
    g.add_argument("--lr_probe", type=float, default=1e-3)
    g.add_argument("--lr_finetune", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--grad_clip", type=float, default=1.0)

    g = p.add_argument_group("chạy")
    g.add_argument("--data_dir", default="data_cache/vsfc")
    g.add_argument("--out", default="results/downstream_vsfc.json")
    g.add_argument("--device", default=None)
    g.add_argument("--smoke", action="store_true",
                   help="chạy thử cực nhanh trên dữ liệu giả, không cần mạng")
    return p.parse_args(argv)


def _smoke_data(n_classes: int, n: int = 96):
    """Dữ liệu giả có tín hiệu học được, dùng để kiểm tra ĐƯỜNG DÂY chứ không phải
    để đánh giá chất lượng. Cố ý không chạm mạng."""
    rng = np.random.default_rng(0)
    out = {}
    for split, m in zip(SPLITS, (n, n // 2, n // 2)):
        y = rng.integers(0, n_classes, size=m).tolist()
        s = [" ".join(["lop" + str(c)] * 3 + ["va", "cac", "ban"]) for c in y]
        out[split] = (s, y)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = [("A", args.ckpt)] + ([("B", args.compare)] if args.compare else [])
    for _, c in ckpts:
        if not Path(c).exists():
            print(f"\nKHONG THAY CHECKPOINT: {c}")
            print("Chay notebooks/kaggle_downstream.ipynb tren Kaggle GPU truoc.\n")
            return 1

    # Do dai co dinh lay tu chinh checkpoint: dung bang L luc tien huan luyen.
    _probe_model, tok, _probe_meta = load_checkpoint(args.ckpt, device="cpu")
    pad_to = args.pad_to or _probe_model.cfg.max_seq_len
    max_len = min(args.max_len, pad_to)
    del _probe_model
    print(f"[cau hinh] pad_to={pad_to} (co dinh) | max_len={max_len} | device={device}")

    tasks = list(TASKS) if args.task == "both" else [args.task]
    modes = ["probe", "finetune"] if args.mode == "both" else [args.mode]
    poolings = ["last", "mean"] if args.pooling == "both" else [args.pooling]
    seeds = [0] if args.smoke else args.seeds
    epochs = 1 if args.smoke else args.epochs

    arms = [{"ckpt": path, "init": "pretrained"} for _, path in ckpts]
    if not args.no_random_control:
        arms += [{"ckpt": path, "init": "random"} for _, path in ckpts]

    feature_cache: dict = {}
    results, comparisons = [], []
    for task in tasks:
        n_classes = len(TASKS[task]["names"])
        data = (_smoke_data(n_classes) if args.smoke
                else load_vsfc(args.data_dir, task))
        base = majority_baseline(data["train"][1], data["test"][1], n_classes)
        print(f"\n### {task}: {n_classes} lop | majority acc {base['accuracy']:.4f} "
              f"f1 {base['macro_f1']:.4f}")

        for mode in modes:
            lr = args.lr_probe if mode == "probe" else args.lr_finetune
            for pooling in poolings:
                arm_runs = []
                for arm in arms:
                    runs = []
                    for seed in seeds:
                        r = run_one(
                            data=data, tok=tok, ckpt=arm["ckpt"], init=arm["init"],
                            task=task, mode=mode, pooling=pooling, seed=seed,
                            epochs=epochs, batch_size=args.batch_size, lr=lr,
                            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
                            max_len=max_len, device=device, pad_to=pad_to,
                            feature_cache=feature_cache, log=not args.smoke)
                        spec = r["layer_spec"] or "?"
                        arch = ("Hyena" if set(spec) == {"H"}
                                else "Transformer" if set(spec) == {"A"} else spec)
                        r["arch"] = arch
                        r["arm"] = f"{arch}-{arm['init']}"
                        r["majority_test_accuracy"] = base["accuracy"]
                        r["majority_test_macro_f1"] = base["macro_f1"]
                        runs.append(r); results.append(r)
                        print(f"  [{task}|{mode}|{pooling}] {r['arm']:22s} seed {seed} "
                              f"-> test acc {r['test_accuracy']:.4f} "
                              f"f1 {r['test_macro_f1']:.4f} ({r['seconds']:.0f}s)")
                    arm_runs.append({"name": runs[0]["arm"], "runs": runs})

                pre = [a for a in arm_runs if a["name"].endswith("-pretrained")]
                if len(pre) == 2:
                    for metric in ("test_macro_f1", "test_accuracy"):
                        c = compare_arms(pre[0], pre[1], metric)
                        c.update({"task": task, "mode": mode, "pooling": pooling,
                                  "a_name": pre[0]["name"], "b_name": pre[1]["name"]})
                        comparisons.append(c)
                        print(f"    -> {metric}: {c['verdict']}")

    payload = {
        "dataset": {"repo": VSFC_REPO, "citation":
                    "Nguyen et al., UIT-VSFC: Vietnamese Students' Feedback Corpus "
                    "for Sentiment Analysis, KSE 2018, pp. 19-24",
                    "license": "unknown (theo trang HuggingFace)"},
        "device": device, "smoke": bool(args.smoke),
        "pad_to": pad_to, "max_len_effective": max_len,
        "config": {k: v for k, v in vars(args).items()},
        "runs": results, "comparisons": comparisons,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nDa ghi {out} ({len(results)} lan chay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

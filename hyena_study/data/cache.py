"""
Cache dong token da xu ly - tiet kiem khoang 6 gio GPU cho ca do an.

VI SAO CAN:

`train.py` truoc day stream corpus tu HuggingFace roi token hoa lai TU DAU o moi
lan chay. Do tu log E0-a tren Kaggle T4: 3000 bai -> 2,0M token mat 21 giay. Mot
lan chay 50M token can khoang 90.000 bai, tuc ~10 phut chuan bi du lieu. Nhan
voi ~36 lan chay trong ke hoach la ~6 gio GPU bi dot vo ich - NHIEU HON ca thoi
gian huan luyen that (~6,2 gio).

Ma toan bo 36 lan chay do chi dung vai to hop (lang, tokenizer) khac nhau. Cung
mot dong token duoc dung lai cho moi seed, moi kien truc, moi ngan sach token.

CACH LAM:

Luu (train, val, test) dang .npy + bo token hoa + thong ke, khoa theo cac tham so
THUC SU quyet dinh du lieu. Lan chay sau nap thang tu dia.

DIEM NGUY HIEM NHAT - KHOA CACHE:

Neu khoa thieu mot tham so co anh huong toi du lieu, lan chay sau se AM THAM nap
nham du lieu cu va toan bo bang ket qua sai ma khong ai biet. Day dung la loai
loi im lang da lam hong bon lan chay Kaggle truoc do.

Nen:
  - Khoa gom DU moi tham so anh huong toi noi dung: lang, tokenizer, vocab_size,
    n_docs, data_seed, val_frac, test_frac.
  - `max_tokens` CO Y KHONG nam trong khoa: no chi cat bot tap train o cuoi, nen
    mot cache phuc vu duoc moi ngan sach token. Viec cat duoc lam LUC NAP.
  - Moi thu muc cache co `meta.json` ghi lai nguyen ven tham so, doc duoc bang mat.
  - `tests/test_cache.py` co test doi tung tham so mot va doi hoi khoa phai doi.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

from .corpus import (
    TOKENIZERS,
    CorpusStats,
    build_token_stream,
    load_wiki_texts,
)

# Cac tham so quyet dinh NOI DUNG cua dong token. Doi bat ky cai nao thi phai
# sinh cache moi. `max_tokens` KHONG nam day - xem docstring dau file.
KEY_FIELDS = ("lang", "tokenizer", "vocab_size", "n_docs", "data_seed",
              "val_frac", "test_frac")


def cache_key(**params) -> str:
    """Bam cac tham so quyet dinh du lieu thanh mot khoa ngan, on dinh."""
    missing = set(KEY_FIELDS) - set(params)
    if missing:
        raise ValueError(f"thieu tham so khoa cache: {sorted(missing)}")
    payload = json.dumps({k: params[k] for k in KEY_FIELDS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def cache_dir_for(root: Path, **params) -> Path:
    """Duong dan thu muc cache. Ten co ca phan doc duoc lan phan bam.

    Phan doc duoc giup nguoi dung liec mat la biet trong do la gi; phan bam bao
    dam hai cau hinh khac nhau khong bao gio dung chung mot thu muc.
    """
    k = cache_key(**params)
    name = (f"{params['lang']}_{params['tokenizer']}"
            f"_v{params['vocab_size']}_d{params['n_docs']}"
            f"_s{params['data_seed']}_{k}")
    return Path(root) / name


def _save(path: Path, train: np.ndarray, val: np.ndarray, test: np.ndarray,
          tok, stats: CorpusStats, params: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "train.npy", train)
    np.save(path / "val.npy", val)
    np.save(path / "test.npy", test)
    tok.save(path / f"tokenizer_{params['tokenizer']}.json")
    (path / "meta.json").write_text(json.dumps({
        "params": params,
        "stats": asdict(stats),
        "n_tokens": {"train_full": int(len(train)), "val": int(len(val)),
                     "test": int(len(test))},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _load(path: Path, params: dict):
    train = np.load(path / "train.npy")
    val = np.load(path / "val.npy")
    test = np.load(path / "test.npy")
    tok = TOKENIZERS[params["tokenizer"]].load(
        path / f"tokenizer_{params['tokenizer']}.json")
    meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    stats = CorpusStats(**meta["stats"])
    return train, val, test, tok, stats


def cached_token_stream(
    *,
    lang: str,
    tokenizer: str = "syllable",
    vocab_size: int = 16000,
    n_docs: int = 20000,
    data_seed: int = 0,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    max_tokens: int | None = None,
    cache_root: str | Path = "data_cache",
    use_cache: bool = True,
    hf_cache_dir: str | None = None,
    texts_provider: Callable[[], list[str]] | None = None,
    verbose: bool = True,
):
    """Nap dong token, dung cache neu co.

    Args:
        max_tokens: cat tap TRAIN xuong ngan sach nay. CO Y khong nam trong khoa
            cache, nen mot cache phuc vu duoc moi ngan sach.
        texts_provider: ham tra ve danh sach van ban. De trong thi lay tu
            Wikipedia. Tach ra lam tham so de test chay duoc offline.

    Returns:
        (train, val, test, tokenizer, stats) - giong het `build_token_stream`.
    """
    params = {"lang": lang, "tokenizer": tokenizer, "vocab_size": vocab_size,
              "n_docs": n_docs, "data_seed": data_seed,
              "val_frac": val_frac, "test_frac": test_frac}
    path = cache_dir_for(Path(cache_root), **params)

    hit = use_cache and (path / "meta.json").exists()
    if hit:
        train, val, test, tok, stats = _load(path, params)
        if verbose:
            print(f"[cache] DUNG LAI {path}  ({len(train):,} token train)")
    else:
        if texts_provider is None:
            def texts_provider():  # noqa: E306
                return load_wiki_texts(lang, n_docs, seed=data_seed,
                                       cache_dir=hf_cache_dir)
        if verbose:
            print(f"[cache] khong co san, dang dung {path}")
        texts = texts_provider()
        # Luu ban DAY DU (max_tokens=None) de mot cache dung duoc cho moi ngan sach
        train, val, test, tok, stats = build_token_stream(
            texts, tokenizer_kind=tokenizer, vocab_size=vocab_size,
            val_frac=val_frac, test_frac=test_frac, max_tokens=None, lang=lang,
        )
        if use_cache:
            _save(path, train, val, test, tok, stats, params)
            if verbose:
                print(f"[cache] da luu {len(train):,} token train")

    # Cat theo ngan sach SAU khi nap - day la ly do max_tokens khong o trong khoa
    if max_tokens is not None and len(train) > max_tokens:
        train = train[:max_tokens]
    stats.n_tokens_train = int(len(train))
    return train, val, test, tok, stats

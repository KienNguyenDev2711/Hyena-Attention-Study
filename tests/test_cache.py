"""
Test cache dong token - tap trung vao LOAI LOI IM LANG nguy hiem nhat.

Cache co mot che nguy hiem rieng: neu khoa cache thieu mot tham so co anh huong
toi du lieu, lan chay sau se AM THAM nap nham du lieu cu. Khong co gi bao loi,
mo hinh van train binh thuong, va ca bang ket qua sai ma khong ai biet - dung
loai loi da lam hong bon lan chay Kaggle truoc do.

Nen bo test nay khong chi kiem "cache co chay khong" ma con doi hoi:
  C1. Nap tu cache phai cho du lieu GIONG HET lan dung dau tien.
  C2. Doi BAT KY tham so nao trong khoa deu phai sinh cache khac.
  C3. max_tokens CO Y khong nam trong khoa, va phai cat dung luc nap.
  C4. Bo token hoa phai song sot qua cache (ma hoa cho cung ket qua).
  C5. Cache khong lam thay doi ket qua so voi duong khong cache.

Toan bo chay OFFLINE nho `texts_provider` - khong dung toi Wikipedia.

Chay: python tests/test_cache.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyena_study.data.cache import (  # noqa: E402
    KEY_FIELDS,
    cache_dir_for,
    cache_key,
    cached_token_stream,
)
from hyena_study.data.corpus import build_token_stream, normalize_text  # noqa: E402

TOY = [
    "trường đại học công nghệ thông tin là một trường thành viên",
    "sinh viên cao học nghiên cứu xử lý ngôn ngữ tự nhiên",
    "mô hình ngôn ngữ tích chập dài thay thế cơ chế chú ý",
    "bộ lọc tham số hoá ngầm được sinh ra bởi một mạng nhỏ",
]


def _texts(n_repeat: int = 300) -> list[str]:
    return [normalize_text(s) for s in TOY * n_repeat]


BASE = dict(lang="vi", tokenizer="syllable", vocab_size=200, n_docs=1200,
            data_seed=0, val_frac=0.05, test_frac=0.05)


# -----------------------------------------------------------------------------
def test_cache_hit_returns_identical_data():
    """Lan hai phai cho du lieu GIONG HET lan mot, tung phan tu."""
    tmp = Path(tempfile.mkdtemp())
    try:
        a = cached_token_stream(cache_root=tmp, texts_provider=_texts,
                                verbose=False, **BASE)
        b = cached_token_stream(cache_root=tmp, texts_provider=_texts,
                                verbose=False, **BASE)
        for i, name in enumerate(("train", "val", "test")):
            assert np.array_equal(a[i], b[i]), f"{name} khac nhau giua hai lan nap"
            assert a[i].dtype == b[i].dtype, f"{name} doi kieu du lieu"
        assert a[4].vocab_size == b[4].vocab_size
        assert a[4].unk_rate == b[4].unk_rate
        return int(len(a[0]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_actually_avoids_rebuilding():
    """Lan hai KHONG duoc goi lai texts_provider - do la ca muc dich cua cache."""
    tmp = Path(tempfile.mkdtemp())
    calls = {"n": 0}

    def counting_provider():
        calls["n"] += 1
        return _texts()

    try:
        cached_token_stream(cache_root=tmp, texts_provider=counting_provider,
                            verbose=False, **BASE)
        assert calls["n"] == 1, "lan dau phai dung du lieu"
        cached_token_stream(cache_root=tmp, texts_provider=counting_provider,
                            verbose=False, **BASE)
        assert calls["n"] == 1, (
            f"lan hai van goi provider {calls['n']} lan - cache KHONG co tac dung, "
            "moi lan chay se lai ton ~10 phut token hoa"
        )
        return calls["n"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_every_key_field_changes_the_cache_key():
    """Doi BAT KY truong nao trong khoa deu phai sinh khoa khac.

    Day la test quan trong nhat cua file. Neu mot truong bi bo quen khoi khoa,
    hai cau hinh khac nhau se dung chung thu muc cache va lan chay sau nap nham
    du lieu - sai am tham, khong co dau hieu gi.
    """
    base = dict(BASE)
    k0 = cache_key(**base)
    variants = {
        "lang": "en", "tokenizer": "bpe", "vocab_size": 999,
        "n_docs": 4321, "data_seed": 7, "val_frac": 0.11, "test_frac": 0.12,
    }
    assert set(variants) == set(KEY_FIELDS), (
        f"test chua phu het truong khoa: thieu {set(KEY_FIELDS) - set(variants)}"
    )
    for field, new in variants.items():
        alt = dict(base)
        alt[field] = new
        assert cache_key(**alt) != k0, (
            f"doi '{field}' ma khoa cache khong doi - hai cau hinh khac nhau se "
            "dung chung cache va nap nham du lieu"
        )
    return len(variants)


def test_max_tokens_is_not_in_the_key_but_is_applied():
    """max_tokens khong nam trong khoa (mot cache dung cho moi ngan sach),
    nhung phai duoc cat dung luc nap."""
    tmp = Path(tempfile.mkdtemp())
    try:
        full = cached_token_stream(cache_root=tmp, texts_provider=_texts,
                                   verbose=False, **BASE)
        budget = 500
        cut = cached_token_stream(cache_root=tmp, texts_provider=_texts,
                                  verbose=False, max_tokens=budget, **BASE)
        assert len(cut[0]) == budget, f"khong cat dung ngan sach: {len(cut[0])}"
        assert cut[4].n_tokens_train == budget, "thong ke khong khop sau khi cat"
        assert np.array_equal(cut[0], full[0][:budget]), "phan cat khong khop dau day"
        # val/test khong bi anh huong
        assert np.array_equal(cut[1], full[1]) and np.array_equal(cut[2], full[2])
        # chi co dung MOT thu muc cache cho ca hai lan goi
        dirs = [p for p in tmp.iterdir() if p.is_dir()]
        assert len(dirs) == 1, (
            f"max_tokens tao ra {len(dirs)} thu muc cache - no da lot vao khoa"
        )
        return budget
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_tokenizer_survives_the_cache():
    """Bo token hoa nap tu cache phai ma hoa cho ket qua y het ban goc."""
    tmp = Path(tempfile.mkdtemp())
    try:
        _, _, _, tok_a, _ = cached_token_stream(cache_root=tmp,
                                                texts_provider=_texts,
                                                verbose=False, **BASE)
        _, _, _, tok_b, _ = cached_token_stream(cache_root=tmp,
                                                texts_provider=_texts,
                                                verbose=False, **BASE)
        s = normalize_text("mô hình ngôn ngữ tiếng việt")
        assert tok_a.encode(s) == tok_b.encode(s), "bo token hoa doi sau khi qua cache"
        assert tok_a.vocab_size == tok_b.vocab_size
        assert "ườ" in tok_b.decode(tok_b.encode(normalize_text("trường học"))), (
            "dau tieng Viet bi hong sau khi qua cache"
        )
        return tok_b.vocab_size
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_matches_the_uncached_path():
    """Duong co cache va duong khong cache phai cho ket qua nhu nhau.

    Neu lech, tuc cache da lam thay doi thi nghiem - moi so lieu truoc va sau khi
    bat cache se khong so sanh duoc voi nhau.
    """
    tmp = Path(tempfile.mkdtemp())
    try:
        direct = build_token_stream(_texts(), tokenizer_kind=BASE["tokenizer"],
                                    vocab_size=BASE["vocab_size"],
                                    val_frac=BASE["val_frac"],
                                    test_frac=BASE["test_frac"],
                                    max_tokens=None, lang=BASE["lang"])
        via = cached_token_stream(cache_root=tmp, texts_provider=_texts,
                                  verbose=False, **BASE)
        for i, name in enumerate(("train", "val", "test")):
            assert np.array_equal(direct[i], via[i]), (
                f"{name} khac nhau giua duong cache va duong truc tiep"
            )
        assert direct[4].unk_rate == via[4].unk_rate
        assert direct[4].chars_per_token == via[4].chars_per_token
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_use_cache_false_skips_disk_entirely():
    """use_cache=False phai khong doc va khong ghi gi len dia."""
    tmp = Path(tempfile.mkdtemp())
    try:
        cached_token_stream(cache_root=tmp, texts_provider=_texts,
                            use_cache=False, verbose=False, **BASE)
        assert not any(tmp.iterdir()), "use_cache=False van ghi ra dia"
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_meta_json_is_human_readable():
    """meta.json phai doc duoc bang mat de nguoi dung tu kiem cache co dung khong."""
    tmp = Path(tempfile.mkdtemp())
    try:
        cached_token_stream(cache_root=tmp, texts_provider=_texts,
                            verbose=False, **BASE)
        path = cache_dir_for(tmp, **BASE)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        for f in KEY_FIELDS:
            assert f in meta["params"], f"meta.json thieu tham so '{f}'"
            assert meta["params"][f] == BASE[f], f"meta.json ghi sai '{f}'"
        assert meta["n_tokens"]["train_full"] > 0
        assert "unk_rate" in meta["stats"]
        return len(meta["params"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# -----------------------------------------------------------------------------
def main() -> int:
    tests = [
        ("C1  Nap lai cho du lieu giong het", test_cache_hit_returns_identical_data),
        ("C2  Cache thuc su tranh dung lai", test_cache_actually_avoids_rebuilding),
        ("C3  MOI truong khoa deu doi khoa", test_every_key_field_changes_the_cache_key),
        ("C4  max_tokens ngoai khoa, cat dung", test_max_tokens_is_not_in_the_key_but_is_applied),
        ("C5  Bo token hoa song sot qua cache", test_tokenizer_survives_the_cache),
        ("C6  Cache khop duong khong cache", test_cache_matches_the_uncached_path),
        ("C7  use_cache=False khong dung dia", test_use_cache_false_skips_disk_entirely),
        ("C8  meta.json doc duoc bang mat", test_meta_json_is_human_readable),
    ]
    n_fail = 0
    for name, fn in tests:
        try:
            out = fn()
            extra = f"  ({out:,})" if isinstance(out, int) and not isinstance(out, bool) else ""
            print(f"  [PASS] {name}{extra}")
        except AssertionError as exc:
            n_fail += 1
            print(f"  [FAIL] {name}\n         {exc}")
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [LOI ] {name}\n         {type(exc).__name__}: {exc}")
    print()
    print("==> " + (f"Toan bo {len(tests)} test DAT" if not n_fail
                    else f"{n_fail}/{len(tests)} test THAT BAI"))
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())

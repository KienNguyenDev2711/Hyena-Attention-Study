"""
Test duong day checkpoint + sinh van ban.

VI SAO BO TEST NAY TON TAI:

Demo sinh van ban co mot loai loi IM LANG rat de xay ra: nap sai bo tu dien hoac
nap thieu trong so thi mo hinh VAN chay, VAN nha ra chu, chi la chu vo nghia.
Nguoi xem se ket luan "mo hinh hoc kem" trong khi that ra la loi ky thuat. Nen
cac test o day khong chi hoi "co chay khong" ma doi hoi:

  G1. Trong so di qua luu/nap phai cho logits GIONG HET.
  G2. Cau hinh mo hinh phai song sot qua luu/nap (ke ca alpha cua bo loc, von la
      buffer KHONG ben nen khong nam trong state_dict).
  G3. Bo tu dien phai song sot: ma hoa va giai ma cho cung ket qua.
  G4. Cung seed phai cho cung ket qua; khac seed phai cho ket qua khac.
  G5. temperature <= 0 la argmax, tat dinh, khong phu thuoc seed.
  G6. Khong bao gio nha ra token bi cam (<pad>, <unk>).
  G7. Gap <eos> thi dung, va bao dung ly do.
  G8. Prompt dai hon max_seq_len van chay duoc (phai cat ngu canh).
  G9. Checkpoint sai dinh dang phai BAO LOI, khong duoc doan mo.
  G10. vocab_size lech giua mo hinh va tu dien phai bi tu choi ngay luc luu.
  G11. detokenize chi sua khoang trang, khong duoc them bot chu.
  G12. Prompt co chu ngoai tu dien phai duoc dem va bao ra.
  G13. Nhanh attention (AAAA) cung phai luu/nap/sinh duoc, vi demo so sanh can no.
  G14. DUONG DAY THAT: goi ham `train()` voi --save_ckpt roi sinh chu tu tep .pt.

Toan bo chay tren CPU voi mo hinh ti hon, khong dung toi GPU hay Wikipedia.
Cache dong token duoc gieo san bang `texts_provider` nen G14 cung chay offline.

Chay: python tests/test_generate.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyena_study.checkpoint import (  # noqa: E402
    CKPT_FORMAT,
    load_checkpoint,
    lm_config_from_dict,
    lm_config_to_dict,
    save_checkpoint,
)
from hyena_study.data.corpus import (  # noqa: E402
    EOS, PAD, SPECIAL_TOKENS, UNK, SyllableTokenizer,
)
from hyena_study.generate import (  # noqa: E402
    _is_invisible as _inv, detokenize, generate_ids, generate_text,
)


def _vis(x):
    return not _inv(x)
from hyena_study.models import HyenaFilterConfig, LMConfig, SequenceLM  # noqa: E402

TOY = [
    "trường đại học công nghệ thông tin là trường thành viên",
    "sinh viên cao học nghiên cứu xử lý ngôn ngữ tự nhiên",
    "mô hình ngôn ngữ tích chập dài thay thế cơ chế chú ý",
    "bộ lọc được sinh ra bởi một mạng nhỏ và cửa sổ suy giảm",
    "việt nam là một quốc gia nằm ở đông nam á .",
]
SEQ_LEN = 32


def _toy_tokenizer(vocab_size: int = 60) -> SyllableTokenizer:
    return SyllableTokenizer.train(TOY, vocab_size)


def _toy_model(tok, *, layer_spec: str = "HH", alpha_values=None) -> tuple[SequenceLM, LMConfig]:
    torch.manual_seed(0)
    cfg = LMConfig(
        vocab_size=tok.vocab_size, d_model=16, layer_spec=layer_spec,
        max_seq_len=SEQ_LEN, dropout=0.0, n_heads=2, hyena_order=2,
        hyena_filter=HyenaFilterConfig(emb_dim=5, hidden=8, n_layers=2,
                                       alpha_values=alpha_values),
    )
    return SequenceLM(cfg).eval(), cfg


def _roundtrip(tmp: Path, tok, model, cfg, **kw):
    p = save_checkpoint(tmp / "toy.pt", model=model, cfg=cfg, tok=tok,
                        tokenizer_kind="syllable", run_name="toy", **kw)
    return p, load_checkpoint(p, device="cpu")


# -----------------------------------------------------------------------------
def test_weights_survive_the_roundtrip():
    """G1. Logits truoc va sau khi luu/nap phai trung khop tuyet doi."""
    tok = _toy_tokenizer()
    model, cfg = _toy_model(tok)
    x = torch.randint(0, tok.vocab_size, (2, SEQ_LEN))
    with torch.no_grad():
        before = model(x)
    with tempfile.TemporaryDirectory() as d:
        _, (m2, _, _) = _roundtrip(Path(d), tok, model, cfg)
        with torch.no_grad():
            after = m2(x)
    gap = float((before - after).abs().max())
    assert gap == 0.0, f"logits lech {gap} sau khi nap lai, trong so khong khop"
    return True


def test_config_including_filter_alpha_survives():
    """G2. alpha cua bo loc la buffer KHONG ben, phai duoc dung lai tu cau hinh."""
    tok = _toy_tokenizer()
    alphas = [0.5 + 0.1 * i for i in range(16)]
    model, cfg = _toy_model(tok, alpha_values=alphas)
    with tempfile.TemporaryDirectory() as d:
        _, (m2, _, meta) = _roundtrip(Path(d), tok, model, cfg)
    assert meta["lm_config"]["hyena_filter"]["alpha_values"] == alphas, \
        "alpha_values khong duoc ghi vao checkpoint"
    got = m2.blocks[0].op.filter.alpha().tolist()
    assert max(abs(a - b) for a, b in zip(alphas, got)) < 1e-6, \
        f"alpha sau khi nap = {got[:3]}..., mong doi {alphas[:3]}..."
    # va cau hinh phai di duoc ca vong dict -> dataclass -> dict
    again = lm_config_to_dict(lm_config_from_dict(lm_config_to_dict(cfg)))
    assert again == lm_config_to_dict(cfg), "LMConfig khong on dinh qua vong chuyen doi"
    return True


def test_tokenizer_survives_the_roundtrip():
    """G3. Tu dien phai giai ma y het truoc va sau."""
    tok = _toy_tokenizer()
    model, cfg = _toy_model(tok)
    probe = "sinh viên nghiên cứu ngôn ngữ"
    with tempfile.TemporaryDirectory() as d:
        _, (_, t2, _) = _roundtrip(Path(d), tok, model, cfg)
    assert t2.vocab_size == tok.vocab_size, "vocab_size doi sau khi nap"
    assert t2.encode(probe) == tok.encode(probe), "ma hoa khac nhau sau khi nap"
    ids = tok.encode(probe)
    assert t2.decode(ids) == tok.decode(ids), "giai ma khac nhau sau khi nap"
    return True


def test_same_seed_same_output_different_seed_differs():
    """G4. Lay mau phai tai lap duoc theo seed, va seed khac phai doi ket qua."""
    tok = _toy_tokenizer()
    model, _ = _toy_model(tok)
    ids = tok.encode("trường đại học")
    kw = dict(max_new_tokens=12, temperature=1.0, top_k=0)
    a = generate_ids(model, ids, seed=0, **kw)["new_ids"]
    b = generate_ids(model, ids, seed=0, **kw)["new_ids"]
    assert a == b, "cung seed cho hai ket qua khac nhau, khong tai lap duoc"
    diffs = [generate_ids(model, ids, seed=s, **kw)["new_ids"] for s in (1, 2, 3)]
    assert any(d != a for d in diffs), "moi seed deu cho cung ket qua, seed khong co tac dung"
    return True


def test_greedy_is_deterministic_regardless_of_seed():
    """G5. temperature <= 0 phai la argmax, khong dinh gi toi seed."""
    tok = _toy_tokenizer()
    model, _ = _toy_model(tok)
    ids = tok.encode("mô hình ngôn ngữ")
    outs = [generate_ids(model, ids, max_new_tokens=10, temperature=0.0, seed=s)["new_ids"]
            for s in (0, 7, 123)]
    assert outs[0] == outs[1] == outs[2], f"argmax lai phu thuoc seed: {outs}"
    return True


def test_banned_tokens_never_appear():
    """G6. <pad> va <unk> khong bao gio duoc nha ra."""
    tok = _toy_tokenizer()
    model, _ = _toy_model(tok)
    ids = tok.encode("việt nam")
    bad = 0
    for s in range(6):
        out = generate_ids(model, ids, max_new_tokens=20, temperature=1.5,
                           top_k=0, seed=s)
        bad += sum(1 for i in out["new_ids"] if i in (PAD, UNK))
    assert bad == 0, f"{bad} token bi cam da lot ra ngoai"
    return True


def test_eos_stops_generation():
    """G7. Gap <eos> thi dung ngay va bao dung ly do."""
    tok = _toy_tokenizer()
    model, _ = _toy_model(tok)
    ids = tok.encode("trường đại học")
    # ep <eos> thang bang cach cam moi token khac
    others = tuple(i for i in range(tok.vocab_size) if i != EOS)
    out = generate_ids(model, ids, max_new_tokens=9, temperature=1.0,
                       top_k=0, banned=others, seed=0)
    assert out["stop_reason"] == "gap_eos", f"ly do dung = {out['stop_reason']}"
    assert out["n_new"] == 0, "dung o <eos> nhung van ghi token vao ket qua"
    assert EOS not in out["new_ids"], "<eos> bi dua vao chuoi tra ve"
    return True


def test_prompt_longer_than_context_still_works():
    """G8. Prompt dai hon max_seq_len phai duoc cat ngu canh, khong duoc no."""
    tok = _toy_tokenizer()
    model, _ = _toy_model(tok)
    long_ids = (tok.encode(" ".join(TOY)) * 4)[: SEQ_LEN * 3]
    assert len(long_ids) > SEQ_LEN, "prompt thu chua du dai de kiem tra viec cat"
    out = generate_ids(model, long_ids, max_new_tokens=4, temperature=0.0)
    assert out["n_new"] == 4, "khong sinh du token khi prompt dai hon ngu canh"
    return len(long_ids)


def test_bad_format_version_is_rejected():
    """G9. Checkpoint la dinh dang khac phai bao loi, khong duoc doan mo."""
    tok = _toy_tokenizer()
    model, cfg = _toy_model(tok)
    with tempfile.TemporaryDirectory() as d:
        p = save_checkpoint(Path(d) / "toy.pt", model=model, cfg=cfg, tok=tok,
                            tokenizer_kind="syllable", run_name="toy")
        payload = torch.load(p, map_location="cpu", weights_only=True)
        payload["format_version"] = CKPT_FORMAT + 99
        torch.save(payload, p)
        try:
            load_checkpoint(p)
        except ValueError as exc:
            assert "format_version" in str(exc), f"thong bao loi khong ro: {exc}"
            return True
    raise AssertionError("nap checkpoint sai dinh dang ma khong bao loi")


def test_vocab_mismatch_is_refused_at_save_time():
    """G10. Mo hinh va tu dien lech vocab thi phai chan NGAY luc luu."""
    tok = _toy_tokenizer()
    model, cfg = _toy_model(tok)
    cfg_bad = lm_config_from_dict({**lm_config_to_dict(cfg),
                                   "vocab_size": tok.vocab_size + 5})
    with tempfile.TemporaryDirectory() as d:
        try:
            save_checkpoint(Path(d) / "bad.pt", model=model, cfg=cfg_bad, tok=tok,
                            tokenizer_kind="syllable", run_name="bad")
        except ValueError as exc:
            assert "vocab" in str(exc).lower(), f"thong bao loi khong ro: {exc}"
            return True
    raise AssertionError("luu checkpoint lech vocab ma khong bao loi")


def test_detokenize_only_fixes_spacing():
    """Dan lai khoang trang quanh dau cau, KHONG duoc them bot chu nao."""
    tok = _toy_tokenizer()
    ids = tok.encode("việt nam là một quốc gia nằm ở đông nam á .")
    raw, pretty = tok.decode(ids), detokenize(tok, ids)
    assert pretty.endswith("á."), f"dau cham chua duoc dan lai: {pretty!r}"
    assert raw.replace(" ", "") == pretty.replace(" ", ""), \
        "detokenize da lam thay doi noi dung chu khong chi khoang trang"
    return True


def test_generate_text_reports_unk_in_prompt():
    """Prompt co chu ngoai tu dien phai duoc DEM va bao ra, khong im lang."""
    tok = _toy_tokenizer()
    model, _ = _toy_model(tok)
    out = generate_text(model, tok, "zzzqqq trường", max_new_tokens=3, temperature=0.0)
    assert out["prompt_unk"] >= 1, "khong dem duoc token ngoai tu dien trong prompt"
    assert out["prompt_n_tokens"] == 2, f"so token prompt = {out['prompt_n_tokens']}"
    assert isinstance(out["continuation"], str) and out["continuation"], "khong sinh ra chu nao"
    return out["prompt_unk"]


def test_transformer_branch_also_works():
    """Demo so sanh can ca nhanh AAAA chay duoc, khong chi HH."""
    tok = _toy_tokenizer()
    model, cfg = _toy_model(tok, layer_spec="AA")
    with tempfile.TemporaryDirectory() as d:
        _, (m2, t2, meta) = _roundtrip(Path(d), tok, model, cfg)
    assert meta["lm_config"]["layer_spec"] == "AA"
    out = generate_text(m2, t2, "sinh viên", max_new_tokens=5, temperature=0.0)
    assert out["n_new"] == 5, "nhanh attention khong sinh du token"
    return True


def test_train_cli_writes_a_usable_checkpoint():
    """G14. Duong day THAT: train.py --save_ckpt -> tep .pt -> sinh duoc chu.

    Day la test quan trong nhat trong tep nay. Cac test tren goi truc tiep
    `save_checkpoint`, con test nay chay dung ham `train()` ma Kaggle se goi.
    Bon lan chay Kaggle hong truoc day deu vi mot doan ma khong co test nao di
    qua; co `--save_ckpt` moi them cung phai duoc doi xu nhu vay.

    Chay offline hoan toan: gieo san cache dong token bang `texts_provider` nen
    khong dung toi Wikipedia.
    """
    from hyena_study.data.cache import cached_token_stream
    from hyena_study.train import parse_args, train

    n_docs, vocab = 40, 80
    texts = [TOY[i % len(TOY)] for i in range(n_docs)]

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        cache_root, out_dir = d / "cache", d / "out"
        # Gieo cache voi DUNG bo tham so khoa ma train() se dung lai.
        cached_token_stream(lang="vi", tokenizer="syllable", vocab_size=vocab,
                            n_docs=n_docs, data_seed=0, cache_root=cache_root,
                            texts_provider=lambda: texts * 60, verbose=False)

        args = parse_args([
            "--layers", "HH", "--d_model", "16", "--n_heads", "2",
            "--lang", "vi", "--tokenizer", "syllable",
            "--vocab_size", str(vocab), "--n_docs", str(n_docs),
            "--seq_len", "32", "--batch_size", "4",
            "--token_budget", "1280", "--eval_batches", "2",
            "--log_every", "1000", "--eval_every", "1000",
            "--token_cache", str(cache_root), "--out_dir", str(out_dir),
            "--run_name", "SMOKE", "--seed", "0", "--save_ckpt",
        ])
        summary = train(args)

        ckpt = out_dir / "SMOKE.pt"
        assert ckpt.exists(), "bat --save_ckpt nhung khong thay tep .pt"
        assert summary.get("checkpoint") == "SMOKE.pt",             f"summary khong ghi ten checkpoint: {summary.get('checkpoint')!r}"
        # tep JSON tren dia phai KHOP voi gia tri tra ve
        on_disk = json.loads((out_dir / "SMOKE.json").read_text(encoding="utf-8"))
        assert on_disk.get("checkpoint") == "SMOKE.pt",             "tep JSON tren dia thieu ten checkpoint, lech voi gia tri tra ve"

        model, tok, meta = load_checkpoint(ckpt, device="cpu")
        assert meta["metrics"]["test_ppl"] == summary["test_ppl"],             "test_ppl trong checkpoint khac trong summary"
        assert meta["lm_config"]["vocab_size"] == tok.vocab_size
        out = generate_text(model, tok, "sinh viên", max_new_tokens=6, temperature=0.0)
        assert out["n_new"] == 6, "checkpoint nap duoc nhung khong sinh duoc chu"
    return True


def test_invisible_tokens_are_banned_by_default():
    """G15. Token khong ve ra glyph nao phai bi chan, tru khi doi hoi nguoc lai.

    Da quan sat that tren checkpoint 16.000 tu: mo hinh nha ra U+200B (ZERO WIDTH
    SPACE) lap lai 16 lan o seed 7. Tren man hinh trong nhu treo may. Tu dien that
    chua 26 token loai nay, lan vao tu Wikipedia (chu Thai, Tang, Khmer, dau phu).
    """
    from hyena_study.generate import invisible_token_ids

    # dung mot tu dien co CHU Y cai ky tu vo hinh vao
    tok = SyllableTokenizer(list(SPECIAL_TOKENS) + ["trường", "học", "​", "­", "viên"])
    inv = invisible_token_ids(tok)
    assert set(inv) == {5, 6}, f"nhan dien sai token vo hinh: {inv}"
    assert all(not _vis(tok.vocab[i]) for i in inv)

    model, _ = _toy_model(tok)
    # ep mo hinh chi con duong sinh ra token vo hinh neu khong bi chan
    allowed_only_invisible = tuple(i for i in range(tok.vocab_size) if i not in (5, 6))
    out = generate_ids(model, tok.encode("trường học"), max_new_tokens=5,
                       temperature=1.0, top_k=0, banned=allowed_only_invisible,
                       eos=None, seed=0)
    assert set(out["new_ids"]) <= {5, 6}, "test tu no khong ep duoc dung nhanh can kiem"

    # mac dinh cua generate_text phai chan chung
    out2 = generate_text(model, tok, "trường học", max_new_tokens=8,
                         temperature=1.5, top_k=0, seed=0)
    assert out2["n_banned_invisible"] == 2, f"dem sai: {out2['n_banned_invisible']}"
    assert not any(i in (5, 6) for i in out2["new_ids"]), "token vo hinh van lot ra"

    # va phai tat duoc khi nguoi dung muon xem nguyen trang
    out3 = generate_text(model, tok, "trường học", max_new_tokens=8, temperature=1.5,
                         top_k=0, seed=0, ban_invisible=False)
    assert out3["n_banned_invisible"] == 0
    return len(inv)


# -----------------------------------------------------------------------------
def main() -> int:
    print()
    print("=" * 70)
    print("TEST CHECKPOINT + SINH VAN BAN")
    print("=" * 70)
    tests = [
        ("G1  Trong so song sot qua luu/nap", test_weights_survive_the_roundtrip),
        ("G2  Cau hinh + alpha bo loc song sot", test_config_including_filter_alpha_survives),
        ("G3  Bo tu dien song sot", test_tokenizer_survives_the_roundtrip),
        ("G4  Cung seed cung ket qua", test_same_seed_same_output_different_seed_differs),
        ("G5  argmax tat dinh, khong theo seed", test_greedy_is_deterministic_regardless_of_seed),
        ("G6  Khong nha ra <pad>/<unk>", test_banned_tokens_never_appear),
        ("G7  Gap <eos> thi dung", test_eos_stops_generation),
        ("G8  Prompt dai hon ngu canh van chay", test_prompt_longer_than_context_still_works),
        ("G9  Sai format_version thi bao loi", test_bad_format_version_is_rejected),
        ("G10 Lech vocab thi chan luc luu", test_vocab_mismatch_is_refused_at_save_time),
        ("G11 detokenize chi sua khoang trang", test_detokenize_only_fixes_spacing),
        ("G12 Dem token ngoai tu dien trong prompt", test_generate_text_reports_unk_in_prompt),
        ("G13 Nhanh attention cung chay", test_transformer_branch_also_works),
        ("G14 train.py --save_ckpt dau-cuoi", test_train_cli_writes_a_usable_checkpoint),
        ("G15 Chan token khong hien thi duoc", test_invisible_tokens_are_banned_by_default),
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

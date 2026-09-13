"""
Thi nghiem bo sung sau khi review bao cao (2026-09-13). Chay tren Kaggle GPU.

HAI CAU HOI, MOI CAU MOT THI NGHIEM:

1. HOAN DOI ALPHA (E4x). Bao cao ket luan bo loc suy tu corpus "khong dac thu
   ngon ngu" chi dua tren HINH DANG hai vector do dai hieu dung (lech 14,3%).
   Con so do phu thuoc quy uoc anh xa I(d) -> alpha: nhan them be rong o thi
   lech 36,2%. Hon nua PPL gan nhu khong nhay voi phan bo alpha (corpus lech
   logspace 75,8% ma PPL chi khac 0,05 - 0,29), nen E4 khong du do nhay de
   tra loi cau hoi dac thu ngon ngu. Phep thu truc tiep o muc huan luyen: mo
   hinh VI dung alpha do tren EN, mo hinh EN dung alpha do tren VI, roi so voi
   chinh ngon ngu do dung alpha cua minh.

2. SEED 2 CHO ABLATION (E3_*_s2). Voi n = 2, bac tu do Welch chi 1,4 - 3,0 nen
   uoc luong phuong sai rat bat on. Them mot seed la cach re nhat de ket luan
   E3 dung vung.

VI SAO CO BUOC "DAU VAN TAY CORPUS" TRUOC KHI DOT GPU:

Mau Wikipedia streaming phu thuoc phien ban `datasets` (docs/06: cung 90k bai
tieng Anh, Colab ra 35,3M token con Kaggle goc ra 42,1M). Cac lan chay VI goc
KHONG ghi phien ban. Neu Kaggle bay gio lay ra mot tap tai lieu VI khac thi:
  - seed 2 cua ablation se so voi seed 0-1 tren corpus KHAC, vo nghia;
  - E4x_vi so voi E4_corpus_vi cua corpus cu, tron hai bien lam mot.
Vi vay module do dau van tay (so tai lieu, so ky tu, so token, ty le unk) va
so voi file ket qua goc TRUOC khi huan luyen:
  - khop: chay ke hoach chinh;
  - lech: tu chuyen sang ke hoach co doi chung CUNG PHIEN (E4c = alpha cua
    chinh ngon ngu, chay lai cung seed), BO QUA ablation seed 2, ghi ro vao
    manifest. Khong bao gio im lang tron corpus.

Cache token dat trong thu muc co ten phien ban `datasets`, vi khoa cache khong
chua phien ban: doi phien ban ma dung chung thu muc se nap nham cache cu.

Chay (Kaggle, GPU T4, BAT Internet):
    python -m hyena_study.followup --lang vi --dry_run        # in ke hoach, khong nap gi
    python -m hyena_study.followup --lang vi --prepare_only   # ~10 phut, chi do dau van tay
    python -m hyena_study.followup --lang vi                  # ke hoach VI
    python -m hyena_study.followup --lang en                  # ke hoach EN
    python -m hyena_study.followup --lang vi --smoke          # CPU, kiem tra duong day
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

FINGERPRINT_INT = ("n_docs", "n_chars", "n_tokens_train", "n_tokens_val", "n_tokens_test")
FINGERPRINT_FLOAT = ("chars_per_token", "unk_rate")

# Moi co anh huong toi ket qua deu ghi TUONG MINH. Mac dinh cua train.py
# (n_docs 20000, token_budget 40M) KHAC voi cau hinh da dung cho bao cao, nen
# dua vao mac dinh la cach nhanh nhat de chay sai ma khong ai biet.
COMMON_ARGV = [
    "--layers", "HHHH", "--d_model", "256", "--n_heads", "8", "--seq_len", "512",
    "--vocab_size", "16000", "--batch_size", "16", "--lr", "0.0003",
    "--weight_decay", "0.1", "--grad_clip", "1.0", "--warmup_frac", "0.05",
    "--dropout", "0.1", "--data_seed", "0", "--token_budget", "50000000",
    "--eval_every", "500", "--eval_batches", "50",
]

LANGS = {
    "vi": {
        "reference": "results/E1_vi_HHHH_s0.json",
        "data_argv": ["--lang", "vi", "--tokenizer", "syllable", "--n_docs", "90000"],
        "own_alpha": "results/alpha_vi_bpe500.json",
        "other": "en",
    },
    "en": {
        "reference": "results/E4_corpus_en_s0.json",
        "data_argv": ["--lang", "en", "--tokenizer", "syllable", "--n_docs", "110000",
                      "--max_train_tokens", "38250964"],
        "own_alpha": "results/alpha_en_bpe500.json",
        "other": "vi",
    },
}

ABLATIONS = (
    ("no_window", ["--no_window"]),
    ("no_sine", ["--no_sine"]),
    ("order1", ["--hyena_order", "1"]),
    ("order3", ["--hyena_order", "3"]),
    ("no_posemb", ["--pos_emb", "none"]),
)

# Ngan sach smoke: du de chay qua MOI nhanh code, khong du de hoc gi.
SMOKE_OVERRIDES = [
    "--seq_len", "32", "--vocab_size", "300", "--n_docs", "400", "--batch_size", "4",
    "--token_budget", "1280", "--eval_every", "5", "--eval_batches", "2",
    "--log_every", "5", "--no_amp",
]

RUN_SPECIFIC = ("seed", "run_name", "out_dir", "token_cache")


@dataclass
class RunSpec:
    name: str
    argv: list[str]
    kind: str                      # "swap" | "control" | "ablation"
    reference: str                 # file ket qua ma cau hinh phai tai lap
    differs_in: tuple[str, ...]    # cac khoa DUOC PHEP khac file tham chieu


# -----------------------------------------------------------------------------
# Dau van tay corpus
# -----------------------------------------------------------------------------
def fingerprint_mismatches(got: dict, ref: dict, rel_tol: float = 1e-9) -> list[str]:
    """Liet ke moi truong lech giua hai thong ke corpus. Rong = cung corpus."""
    msgs = []
    for k in FINGERPRINT_INT:
        if int(got[k]) != int(ref[k]):
            msgs.append(f"{k}: {got[k]} != {ref[k]} (tham chieu)")
    for k in FINGERPRINT_FLOAT:
        if not math.isclose(float(got[k]), float(ref[k]), rel_tol=rel_tol, abs_tol=0.0):
            msgs.append(f"{k}: {got[k]!r} != {ref[k]!r} (tham chieu)")
    return msgs


# -----------------------------------------------------------------------------
# Ke hoach
# -----------------------------------------------------------------------------
def build_plan(lang: str, corpus_matches: bool, seeds: tuple[int, ...] = (0, 1, 2),
               ablation_seed: int = 2) -> list[RunSpec]:
    """Danh sach lan chay. Xem docstring dau file ve hai truong hop khop/lech."""
    if lang not in LANGS:
        raise ValueError(f"lang phai thuoc {sorted(LANGS)}, nhan {lang!r}")
    cfg = LANGS[lang]
    other = cfg["other"]
    base = COMMON_ARGV + cfg["data_argv"]
    corpus_ref = f"results/E4_corpus_{lang}_s0.json"

    plan: list[RunSpec] = []
    # Xen ke theo seed: phien Kaggle chet giua chung thi cac cap da xong van dung duoc.
    for s in seeds:
        if not corpus_matches:
            plan.append(RunSpec(
                name=f"E4c_{lang}_alpha{lang}_s{s}",
                argv=base + ["--decay_mode", "corpus", "--alpha_file", cfg["own_alpha"],
                             "--seed", str(s)],
                kind="control", reference=corpus_ref, differs_in=RUN_SPECIFIC))
        plan.append(RunSpec(
            name=f"E4x_{lang}_alpha{other}_s{s}",
            argv=base + ["--decay_mode", "corpus", "--alpha_file",
                         LANGS[other]["own_alpha"], "--seed", str(s)],
            kind="swap", reference=corpus_ref,
            differs_in=RUN_SPECIFIC + ("alpha_file",)))

    if lang == "vi" and corpus_matches:
        for tag, flags in ABLATIONS:
            plan.append(RunSpec(
                name=f"E3_{tag}_s{ablation_seed}",
                argv=base + flags + ["--seed", str(ablation_seed)],
                kind="ablation", reference=f"results/E3_{tag}_s0.json",
                differs_in=RUN_SPECIFIC))
    return plan


def full_argv(spec: RunSpec, out_dir: Path, token_cache: str, smoke: bool) -> list[str]:
    argv = list(spec.argv)
    # Duong dan alpha la tuong doi goc repo. Tren Kaggle cwd = goc repo nen giu
    # nguyen (de JSON ghi duong dan ngan, doc duoc); o noi khac thi quy ve goc repo.
    if "--alpha_file" in argv:
        i = argv.index("--alpha_file") + 1
        if not Path(argv[i]).exists():
            argv[i] = str(REPO_ROOT / argv[i])
    argv += ["--run_name", spec.name, "--out_dir", str(out_dir), "--token_cache", token_cache]
    if smoke:
        argv += SMOKE_OVERRIDES
    return argv


# -----------------------------------------------------------------------------
def _datasets_version() -> str:
    try:
        from datasets import __version__ as v
        return v
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _toy_texts() -> list[str]:
    from .data import normalize_text
    base = [
        "trường đại học công nghệ thông tin là một trường thành viên",
        "sinh viên cao học nghiên cứu xử lý ngôn ngữ tự nhiên",
        "mô hình ngôn ngữ tích chập dài thay thế cơ chế chú ý",
        "bộ lọc tham số hoá ngầm được sinh ra bởi một mạng nhỏ",
    ]
    return [normalize_text(s) for s in base * 100]


def prepare_corpus(argv: list[str], texts_provider=None) -> dict:
    """Dung (hoac nap) cache token dung nhu train.py se nap, tra ve thong ke."""
    from .data import cached_token_stream, stats_to_dict
    from .train import parse_args, resolve_max_train_tokens

    a = parse_args(argv)
    *_, stats = cached_token_stream(
        lang=a.lang, tokenizer=a.tokenizer, vocab_size=a.vocab_size,
        n_docs=a.n_docs, data_seed=a.data_seed,
        max_tokens=resolve_max_train_tokens(a), cache_root=a.token_cache,
        use_cache=True, hf_cache_dir=a.cache_dir, texts_provider=texts_provider,
    )
    return stats_to_dict(stats)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _banner(msg: str) -> None:
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)


def main(argv: list[str] | None = None, texts_provider=None) -> int:
    p = argparse.ArgumentParser(description="Thi nghiem bo sung: hoan doi alpha va seed 2 ablation")
    p.add_argument("--lang", required=True, choices=sorted(LANGS))
    p.add_argument("--out_dir", default="results_followup")
    p.add_argument("--token_cache", default=None,
                   help="mac dinh data_cache/datasets-<phien ban>")
    p.add_argument("--reference", default=None,
                   help="file JSON ket qua goc de so dau van tay (mac dinh theo ngon ngu)")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--prepare_only", action="store_true",
                   help="chi dung cache va so dau van tay; ma thoat 2 neu lech")
    p.add_argument("--dry_run", action="store_true",
                   help="in ke hoach cho ca hai truong hop, khong nap du lieu")
    p.add_argument("--smoke", action="store_true",
                   help="CPU, corpus do choi, ngan sach si hon; khong so dau van tay tru khi co --reference")
    args = p.parse_args(argv)

    lang = args.lang
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip() != "")
    if args.smoke and args.seeds == "0,1,2":
        seeds = (0,)

    if args.dry_run:
        for matched in (True, False):
            _banner(f"KE HOACH {lang.upper()} neu corpus {'KHOP' if matched else 'LECH'}")
            for spec in build_plan(lang, matched, seeds):
                print(f"  {spec.kind:<9}{spec.name}")
        return 0

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    token_cache = args.token_cache or (
        str(out / "data_cache") if args.smoke else f"data_cache/datasets-{_datasets_version()}")
    if args.smoke and texts_provider is None:
        texts_provider = _toy_texts

    import torch
    if not (args.smoke or args.prepare_only) and not torch.cuda.is_available():
        raise SystemExit("KHONG thay GPU. Bat GPU T4 trong Kaggle Settings roi chay lai. "
                         "Chay CPU chi duoc voi --smoke hoac --prepare_only.")

    ref_path = args.reference
    if ref_path is None and not args.smoke:
        ref_path = str(REPO_ROOT / LANGS[lang]["reference"])
    ref_corpus = None
    if ref_path is not None:
        if not Path(ref_path).exists():
            raise SystemExit(f"khong thay file tham chieu {ref_path}")
        ref_corpus = json.loads(Path(ref_path).read_text(encoding="utf-8"))["corpus"]

    base_argv = COMMON_ARGV + LANGS[lang]["data_argv"] + ["--token_cache", token_cache]
    if args.smoke:
        base_argv += SMOKE_OVERRIDES
    _banner(f"[{lang}] dung/nap cache token tai {token_cache} (datasets {_datasets_version()})")
    stats = prepare_corpus(base_argv, texts_provider)

    if ref_corpus is None:
        mismatches, matched = [], True
        print("  (smoke) bo qua so dau van tay")
    else:
        mismatches = fingerprint_mismatches(stats, ref_corpus)
        matched = not mismatches
    if matched:
        print(f"  DAU VAN TAY KHOP voi {ref_path}: dung corpus cua bao cao.")
    else:
        _banner("!!! CORPUS KHAC VOI BAO CAO !!!")
        for m in mismatches:
            print(f"  - {m}")
        print("  => chay DOI CHUNG CUNG PHIEN (E4c) va BO QUA ablation seed 2.")

    manifest_path = out / f"followup_{lang}_manifest.json"
    manifest = {
        "lang": lang, "smoke": args.smoke, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(), "datasets_version": _datasets_version(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "token_cache": token_cache, "reference": ref_path,
        "corpus": stats, "corpus_matches": matched, "mismatches": mismatches,
        "runs": [],
    }
    _write(manifest_path, manifest)
    if args.prepare_only:
        return 0 if matched else 2

    from .train import parse_args, train

    plan = build_plan(lang, matched, seeds)
    manifest["plan"] = [s.name for s in plan]
    if lang == "vi" and not matched:
        manifest["skipped"] = "ablation seed 2: corpus khac voi seed 0-1 nen khong so sanh duoc"
    _write(manifest_path, manifest)

    for i, spec in enumerate(plan, 1):
        jf = out / f"{spec.name}.json"
        if jf.exists() and "test_ppl" in json.loads(jf.read_text(encoding="utf-8")):
            print(f"\n[{i}/{len(plan)}] {spec.name}: da co ket qua, bo qua")
            status = "skipped_existing"
            ppl = json.loads(jf.read_text(encoding="utf-8"))["test_ppl"]
        else:
            _banner(f"[{i}/{len(plan)}] {spec.kind}: {spec.name}")
            summary = train(parse_args(full_argv(spec, out, token_cache, args.smoke)),
                            texts_provider=texts_provider)
            drift = fingerprint_mismatches(summary["corpus"], stats)
            if drift:
                raise RuntimeError(f"{spec.name} huan luyen tren corpus khac da do: {drift}")
            status, ppl = "done", summary["test_ppl"]
        manifest["runs"].append({"name": spec.name, "kind": spec.kind,
                                 "status": status, "test_ppl": ppl})
        _write(manifest_path, manifest)

    _banner("TOM TAT")
    for r in manifest["runs"]:
        print(f"  {r['kind']:<9}{r['name']:<26}PPL {r['test_ppl']:.3f}  ({r['status']})")

    from .analyze import print_followup
    dirs = [out] if args.smoke else [REPO_ROOT / "results", out]
    print_followup(dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

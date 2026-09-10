"""
Sinh văn bản từ checkpoint đã huấn luyện. Dùng cho demo trước hội đồng.

PHẢI ĐỌC TRƯỚC KHI TRÌNH DIỄN — KỲ VỌNG ĐÚNG VỀ CHẤT LƯỢNG:

Mô hình của nhóm có 7,55 triệu tham số, huấn luyện trên 50 triệu token, perplexity
khoảng 51 ở mức âm tiết. Ở quy mô đó, văn bản sinh ra sẽ ĐÚNG DÁNG tiếng Việt
(chính tả âm tiết, cụm từ ngắn, dấu câu hợp lý) nhưng KHÔNG mạch lạc về ngữ nghĩa
sau vài âm tiết. Đây là hệ quả của quy mô, không phải lỗi cài đặt. Đừng quảng cáo
quá tay khi demo; giá trị của demo này là cho thấy mô hình ĐÃ HỌC ĐƯỢC cấu trúc
cục bộ của tiếng Việt, và cho phép so sánh trực tiếp Hyena với Transformer ở cùng
ngân sách huấn luyện.

VÌ SAO SINH LẠI CHẬM HƠN BẠN TƯỞNG:

Cả hai kiến trúc ở đây đều KHÔNG có cache trạng thái khi suy diễn. Mỗi token mới
phải chạy lại toàn bộ lượt tiến trên cả ngữ cảnh. Với Hyena, bộ lọc dài được tính
qua FFT trên toàn chuỗi ở mỗi bước. Bài báo gốc có nói tới suy diễn hồi quy hiệu
quả cho tích chập dài, nhóm KHÔNG cài phần đó, nên đây là hạn chế của bản tái
hiện, không phải giới hạn của Hyena. Phải nói rõ nếu bị hỏi.

Ví dụ chạy:
    python -m hyena_study.generate --ckpt results/DEMO_vi_HHHH_s0.pt \
        --prompt "Trường Đại học Công nghệ Thông tin" --max_new_tokens 60

    python -m hyena_study.generate --ckpt results/DEMO_vi_HHHH_s0.pt \
        --compare results/DEMO_vi_AAAA_s0.pt --prompt "Việt Nam là một quốc gia"
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .data.corpus import EOS, PAD, UNK, normalize_text


# -----------------------------------------------------------------------------
# Ghép token thành chữ
# -----------------------------------------------------------------------------
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%\)\]\}»…])")
_SPACE_AFTER_OPEN = re.compile(r"([\(\[\{«])\s+")


def detokenize(tok, ids: list[int]) -> str:
    """Giải mã id thành chuỗi đọc được.

    `SyllableTokenizer.decode` nối token bằng dấu cách, nên dấu câu bị tách rời
    ("xin chào ."). Ở đây chỉ dán lại khoảng trắng quanh dấu câu, KHÔNG sửa chữ,
    KHÔNG thêm bớt token nào. Muốn xem bản thô chưa dán thì dùng `tok.decode`.
    """
    text = tok.decode(list(ids))
    if getattr(tok, "name", "") == "syllable":
        text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
        text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------------------------------------------------------
# Lấy mẫu
# -----------------------------------------------------------------------------
def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """Cắt đuôi phân phối theo top-k rồi top-p. Trả về logits đã che."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True)
        probs = F.softmax(srt, dim=-1)
        cum = probs.cumsum(dim=-1)
        # giữ lại token đầu tiên vượt ngưỡng, nếu không có thể loại sạch
        drop = cum - probs > top_p
        srt = srt.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, srt)
    return logits


@torch.no_grad()
def generate_ids(model, prompt_ids: list[int], *, max_new_tokens: int = 60,
                 temperature: float = 0.9, top_k: int = 40, top_p: float = 0.0,
                 repetition_penalty: float = 1.0,
                 banned: tuple[int, ...] = (PAD, UNK),
                 eos: int | None = EOS, seed: int | None = None,
                 device: torch.device | str = "cpu") -> dict:
    """Sinh tiếp `max_new_tokens` id sau `prompt_ids`.

    `temperature <= 0` nghĩa là lấy argmax (tất định). `banned` mặc định chặn
    <pad> và <unk> vì chúng không phải chữ thật; để mô hình nhả <unk> ra màn hình
    chỉ làm demo khó đọc chứ không nói thêm điều gì.
    """
    if not prompt_ids:
        raise ValueError("prompt rỗng sau khi token hoá, không có gì để sinh tiếp")
    device = torch.device(device)
    max_len = model.cfg.max_seq_len
    gen = None
    if seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed))

    ids = list(prompt_ids)
    new_ids: list[int] = []
    stop = "het_so_token"
    t0 = time.time()

    for _ in range(max_new_tokens):
        ctx = ids[-max_len:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :].float().cpu()

        if repetition_penalty and repetition_penalty != 1.0 and ids:
            seen = torch.tensor(sorted(set(ids)), dtype=torch.long)
            vals = logits[seen]
            logits[seen] = torch.where(vals > 0, vals / repetition_penalty,
                                       vals * repetition_penalty)
        for b in banned:
            if b is not None and 0 <= b < logits.numel():
                logits[b] = float("-inf")

        if temperature is None or temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            logits = _filter_logits(logits / float(temperature), top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum()) <= 0:
                raise RuntimeError("phân phối lấy mẫu hỏng; kiểm tra top_k/top_p")
            nxt = int(torch.multinomial(probs, num_samples=1, generator=gen))

        if eos is not None and nxt == eos:
            stop = "gap_eos"
            break
        ids.append(nxt)
        new_ids.append(nxt)

    return {
        "prompt_ids": list(prompt_ids),
        "new_ids": new_ids,
        "all_ids": ids,
        "stop_reason": stop,
        "n_new": len(new_ids),
        "seconds": round(time.time() - t0, 3),
    }


def generate_text(model, tok, prompt: str, **kw) -> dict:
    """Bọc `generate_ids` ở mức chữ: chuẩn hoá, token hoá, sinh, giải mã."""
    clean = normalize_text(prompt)
    prompt_ids = tok.encode(clean)
    if not prompt_ids:
        raise ValueError(f"prompt {prompt!r} token hoá ra rỗng")
    unk_in_prompt = sum(1 for i in prompt_ids if i == UNK)
    out = generate_ids(model, prompt_ids, **kw)
    out.update({
        "prompt": clean,
        "prompt_n_tokens": len(prompt_ids),
        "prompt_unk": unk_in_prompt,
        "continuation": detokenize(tok, out["new_ids"]),
        "full_text": detokenize(tok, out["all_ids"]),
        "raw_continuation": tok.decode(out["new_ids"]),
    })
    return out


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _load(path: str, device: str):
    model, tok, meta = load_checkpoint(path, device=device)
    ppl = meta.get("metrics", {}).get("test_ppl")
    spec = meta.get("lm_config", {}).get("layer_spec", "?")
    print(f"[nap] {Path(path).name} | layer_spec={spec} "
          f"| vocab={tok.vocab_size:,} | test PPL={ppl if ppl is None else round(ppl, 3)}")
    return model, tok, meta


def _one(model, tok, meta, args, label: str) -> dict:
    res = generate_text(
        model, tok, args.prompt,
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed, device=args.device,
    )
    res["label"] = label
    res["run_name"] = meta.get("run_name")
    res["layer_spec"] = meta.get("lm_config", {}).get("layer_spec")
    res["test_ppl"] = meta.get("metrics", {}).get("test_ppl")
    res["params_total"] = meta.get("metrics", {}).get("params_total")
    return res


def _print(res: dict) -> None:
    print()
    print("=" * 78)
    print(f"{res['label']}  (layer_spec={res['layer_spec']}, "
          f"test PPL={res['test_ppl'] if res['test_ppl'] is None else round(res['test_ppl'], 3)})")
    print("-" * 78)
    print(f"  prompt      : {res['prompt']}")
    if res["prompt_unk"]:
        print(f"  CANH BAO    : {res['prompt_unk']}/{res['prompt_n_tokens']} token trong "
              f"prompt ngoai tu dien, mo hinh khong thay chung")
    print(f"  sinh tiep   : {res['continuation']}")
    print(f"  dung vi     : {res['stop_reason']} | {res['n_new']} token | {res['seconds']}s")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sinh văn bản từ checkpoint Hyena/Transformer đã huấn luyện")
    p.add_argument("--ckpt", required=True, help="đường dẫn tệp .pt")
    p.add_argument("--compare", default=None,
                   help="checkpoint thứ hai, chạy cùng prompt để so sánh trực tiếp")
    p.add_argument("--prompt", default="Trường Đại học Công nghệ Thông tin")
    p.add_argument("--prompts_file", default=None,
                   help="tệp văn bản, mỗi dòng một prompt; ghi đè --prompt")
    p.add_argument("--max_new_tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.9,
                   help="<= 0 nghĩa là lấy argmax, tất định")
    p.add_argument("--top_k", type=int, default=40, help="0 = tắt")
    p.add_argument("--top_p", type=float, default=0.0, help="0 = tắt")
    p.add_argument("--repetition_penalty", type=float, default=1.0, help="1.0 = tắt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=1,
                   help="số mẫu cho mỗi prompt, mỗi mẫu dùng seed + i")
    p.add_argument("--device", default="cpu", help="cpu hoặc cuda")
    p.add_argument("--samples_json", default=None,
                   help="ghi toàn bộ kết quả kèm siêu dữ liệu ra tệp JSON")
    p.add_argument("--smoke", action="store_true",
                   help="chạy nhanh để kiểm đường dây: 8 token, 1 mẫu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.max_new_tokens, args.n_samples = 8, 1

    prompts = [args.prompt]
    if args.prompts_file:
        lines = [l.strip() for l in Path(args.prompts_file).read_text(encoding="utf-8").splitlines()]
        prompts = [l for l in lines if l]
        if not prompts:
            raise SystemExit(f"{args.prompts_file} không có dòng nào dùng được")

    loaded = [(_load(args.ckpt, args.device), args.ckpt)]
    if args.compare:
        loaded.append((_load(args.compare, args.device), args.compare))

    results = []
    base_seed = args.seed
    for prompt in prompts:
        args.prompt = prompt
        for i in range(args.n_samples):
            args.seed = base_seed + i
            for (model, tok, meta), path in loaded:
                label = meta.get("run_name") or Path(path).stem
                res = _one(model, tok, meta, args, label)
                res["sample_index"] = i
                res["ckpt"] = Path(path).name
                res["sampling"] = {
                    "temperature": args.temperature, "top_k": args.top_k,
                    "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
                    "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                }
                results.append(res)
                _print(res)

    if args.samples_json:
        out = Path(args.samples_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoints": [{"file": Path(p).name,
                             "run_name": m[2].get("run_name"),
                             "layer_spec": m[2].get("lm_config", {}).get("layer_spec"),
                             "metrics": m[2].get("metrics", {}),
                             "corpus": m[2].get("corpus", {})}
                            for m, p in loaded],
            "samples": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[ghi] {out}  ({len(results)} mẫu)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

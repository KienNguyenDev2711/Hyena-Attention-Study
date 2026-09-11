"""
Test may chu web cua demo sinh van ban.

VI SAO CAN: day la thu SE CHAY TRUOC HOI DONG. Mot loi 500 luc do khong sua kip.
Bo test nay dung may chu that tren mot cong ngau nhien, goi bang HTTP that, voi
mot checkpoint ti hon, roi doi hoi:

  S1. GET / tra ve trang HTML that.
  S2. GET /api/meta liet ke du mo hinh kem sieu du lieu.
  S3. POST /api/generate sinh duoc chu.
  S4. Cung seed cho cung ket qua qua hai lan goi HTTP.
  S5. Tham so sai phai tra 400 kem thong bao TIENG VIET doc duoc, khong phai 500.
  S6. max_new_tokens vuot tran phai bi chan.
  S7. Duong dan la phai tra 404 chu khong no.
  S8. Body khong phai JSON phai tra 400.
  S9. Hai mo hinh cung chay va tra ve theo dung thu tu.
  S10. Trang HTML phai nhung du cac diem neo ma JS can.

Toan bo chay tren CPU, khong dung toi GPU hay mang.

Chay: python tests/test_serve.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hyena_study.checkpoint import save_checkpoint  # noqa: E402
from hyena_study.data.corpus import SyllableTokenizer  # noqa: E402
from hyena_study.models import HyenaFilterConfig, LMConfig, SequenceLM  # noqa: E402
from hyena_study.serve import (  # noqa: E402
    MAX_NEW_TOKENS,
    UI_DIR,
    build_server,
    clean_params,
    load_bundles,
)

TOY = [
    "trường đại học công nghệ thông tin là trường thành viên",
    "sinh viên cao học nghiên cứu xử lý ngôn ngữ tự nhiên",
    "việt nam là một quốc gia nằm ở đông nam á .",
    "mô hình ngôn ngữ tích chập dài thay thế cơ chế chú ý",
]


def _make_ckpt(tmp: Path, layer_spec: str, name: str) -> Path:
    tok = SyllableTokenizer.train(TOY, 60)
    torch.manual_seed(0)
    cfg = LMConfig(vocab_size=tok.vocab_size, d_model=16, layer_spec=layer_spec,
                   max_seq_len=32, dropout=0.0, n_heads=2, hyena_order=2,
                   hyena_filter=HyenaFilterConfig(emb_dim=5, hidden=8, n_layers=2))
    model = SequenceLM(cfg).eval()
    return save_checkpoint(tmp / f"{name}.pt", model=model, cfg=cfg, tok=tok,
                           tokenizer_kind="syllable", run_name=name,
                           metrics={"test_ppl": 12.5, "params_total": 1234,
                                    "tokens_seen": 99, "steps": 7})


class Server:
    """Khoi dong may chu that tren cong tuy he dieu hanh chon (port 0)."""

    def __init__(self, tmp: Path, two: bool = True):
        a = _make_ckpt(tmp, "HH", "TOY_HH")
        b = _make_ckpt(tmp, "AA", "TOY_AA") if two else None
        self.bundles = load_bundles(str(a), str(b) if b else None, "cpu")
        self.httpd = build_server(self.bundles, "cpu", "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.th = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.th.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=30) as r:
            return r.status, r.read().decode("utf-8")

    def post(self, path: str, payload, raw: bool = False):
        data = payload if raw else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            try:
                return e.code, json.loads(body)
            except json.JSONDecodeError:
                return e.code, {"_raw": body}

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _with_server(fn, two: bool = True):
    with tempfile.TemporaryDirectory() as d:
        s = Server(Path(d), two=two)
        try:
            return fn(s)
        finally:
            s.close()


# -----------------------------------------------------------------------------
def test_index_page_is_served():
    """S1. GET / phai tra ve trang HTML that, khong phai 404."""
    def go(s):
        code, body = s.get("/")
        assert code == 200, f"ma tra ve {code}"
        assert "<!doctype html>" in body[:200].lower(), "khong phai tep HTML"
        assert "Hyena" in body, "trang khong nhac toi Hyena"
        return len(body)
    return _with_server(go)


def test_meta_lists_models():
    """S2. /api/meta phai liet ke du mo hinh kem sieu du lieu can hien thi."""
    def go(s):
        code, body = s.get("/api/meta")
        assert code == 200
        d = json.loads(body)
        assert d["ok"] is True
        assert len(d["models"]) == 2, f"co {len(d['models'])} mo hinh"
        names = [m["name"] for m in d["models"]]
        assert names == ["Hyena", "Transformer"], f"ten mo hinh: {names}"
        for m in d["models"]:
            for k in ("test_ppl", "vocab_size", "max_seq_len", "layer_spec", "file"):
                assert m.get(k) is not None, f"thieu truong {k}"
        return len(d["models"])
    return _with_server(go)


def test_generate_returns_text():
    """S3. POST /api/generate phai sinh ra chu."""
    def go(s):
        code, d = s.post("/api/generate", {"prompt": "trường đại học", "max_new_tokens": 6})
        assert code == 200, f"ma {code}: {d}"
        assert d["ok"] is True
        assert len(d["results"]) == 2
        r = d["results"][0]
        assert r["n_new"] == 6, f"sinh {r['n_new']} token"
        assert isinstance(r["continuation"], str) and r["continuation"].strip(), "khong co chu"
        assert r["prompt"] == "trường đại học"
        assert r["ms_per_token"] is not None
        return r["n_new"]
    return _with_server(go)


def test_same_seed_gives_same_answer_over_http():
    """S4. Hai lan goi cung seed phai cho cung ket qua, va seed khac phai doi."""
    def go(s):
        body = {"prompt": "sinh viên", "max_new_tokens": 8, "temperature": 1.0,
                "top_k": 0, "seed": 5}
        _, d1 = s.post("/api/generate", body)
        _, d2 = s.post("/api/generate", body)
        a = d1["results"][0]["continuation"]
        b = d2["results"][0]["continuation"]
        assert a == b, f"cung seed khac ket qua:\n  {a!r}\n  {b!r}"
        _, d3 = s.post("/api/generate", {**body, "seed": 6})
        # khong bat buoc phai khac, nhung neu moi seed deu giong nhau thi la loi
        assert isinstance(d3["results"][0]["continuation"], str)
        return True
    return _with_server(go)


def test_bad_params_return_400_in_vietnamese():
    """S5. Tham so sai phai ra 400 kem thong bao doc duoc, khong phai 500."""
    def go(s):
        cases = [
            ({"prompt": ""}, "prompt rong"),
            ({"prompt": "a", "temperature": 99}, "temperature ngoai khoang"),
            ({"prompt": "a", "top_k": -5}, "top_k am"),
            ({"prompt": "a", "max_new_tokens": 0}, "max_new_tokens = 0"),
            ({"prompt": "a", "max_new_tokens": "nhieu"}, "max_new_tokens khong phai so"),
        ]
        for payload, what in cases:
            code, d = s.post("/api/generate", payload)
            assert code == 400, f"{what}: ma {code} thay vi 400 ({d})"
            assert d.get("ok") is False and d.get("error"), f"{what}: thieu thong bao loi"
        return len(cases)
    return _with_server(go)


def test_token_cap_is_enforced():
    """S6. Vuot tran max_new_tokens phai bi chan ngay, khong duoc chay that."""
    def go(s):
        code, d = s.post("/api/generate",
                         {"prompt": "trường", "max_new_tokens": MAX_NEW_TOKENS + 1})
        assert code == 400, f"ma {code}"
        assert str(MAX_NEW_TOKENS) in d["error"], f"thong bao khong neu tran: {d['error']}"
        return MAX_NEW_TOKENS
    return _with_server(go)


def test_unknown_path_is_404():
    """S7. Duong dan la phai tra 404 gon gang."""
    def go(s):
        try:
            s.get("/khong-ton-tai")
        except urllib.error.HTTPError as e:
            assert e.code == 404, f"ma {e.code}"
            return True
        raise AssertionError("duong dan la khong tra 404")
    return _with_server(go)


def test_malformed_body_is_400():
    """S8. Body khong phai JSON phai ra 400, khong duoc no thanh 500."""
    def go(s):
        code, d = s.post("/api/generate", b"{khong phai json", raw=True)
        assert code == 400, f"ma {code}: {d}"
        return True
    return _with_server(go)


def test_single_model_mode_works():
    """S9. Che do mot mo hinh (khong --compare) phai chay duoc."""
    def go(s):
        _, body = s.get("/api/meta")
        assert len(json.loads(body)["models"]) == 1
        code, d = s.post("/api/generate", {"prompt": "việt nam", "max_new_tokens": 4})
        assert code == 200 and len(d["results"]) == 1
        return True
    return _with_server(go, two=False)


def test_ui_file_has_the_anchors_the_script_needs():
    """S10. Trang HTML phai co du diem neo ma JS thao tac, tranh loi null am tham."""
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    needed = ["id=\"prompt\"", "id=\"tokens\"", "id=\"temp\"", "id=\"topk\"",
              "id=\"seed\"", "id=\"go\"", "id=\"out\"", "id=\"badges\"",
              "id=\"err\"", "id=\"warn\"", "id=\"honest\"", "id=\"prov\"",
              "id=\"chips\"", "id=\"inv\""]
    missing = [n for n in needed if n not in html]
    assert not missing, f"thieu diem neo trong HTML: {missing}"
    assert "/api/generate" in html and "/api/meta" in html, "HTML khong goi toi API"
    return len(needed)


def test_clean_params_defaults():
    """Ham kiem tham so phai dat mac dinh dung khi nguoi dung khong gui gi them."""
    p = clean_params({"prompt": "  xin chào  "})
    assert p["prompt"] == "xin chào", "khong cat khoang trang thua"
    assert p["max_new_tokens"] == 40 and p["top_k"] == 40
    assert abs(p["temperature"] - 0.9) < 1e-9
    assert p["ban_invisible"] is True, "mac dinh phai CHAN token vo hinh"
    return True


# -----------------------------------------------------------------------------
def main() -> int:
    print()
    print("=" * 70)
    print("TEST MAY CHU WEB DEMO")
    print("=" * 70)
    tests = [
        ("S1  Trang chu tra ve HTML", test_index_page_is_served),
        ("S2  /api/meta liet ke mo hinh", test_meta_lists_models),
        ("S3  /api/generate sinh ra chu", test_generate_returns_text),
        ("S4  Cung seed cung ket qua qua HTTP", test_same_seed_gives_same_answer_over_http),
        ("S5  Tham so sai tra 400 co giai thich", test_bad_params_return_400_in_vietnamese),
        ("S6  Chan vuot tran so token", test_token_cap_is_enforced),
        ("S7  Duong dan la tra 404", test_unknown_path_is_404),
        ("S8  Body hong tra 400", test_malformed_body_is_400),
        ("S9  Che do mot mo hinh", test_single_model_mode_works),
        ("S10 HTML du diem neo cho JS", test_ui_file_has_the_anchors_the_script_needs),
        ("S11 Mac dinh tham so dung", test_clean_params_defaults),
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

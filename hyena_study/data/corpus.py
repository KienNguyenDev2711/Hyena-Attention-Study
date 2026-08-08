"""
Tầng dữ liệu: corpus tiếng Việt / tiếng Anh + các bộ token hoá.

VÌ SAO TRỤC TOKEN HOÁ LÀ MỘT TRỤC NGHIÊN CỨU, KHÔNG PHẢI CHI TIẾT KỸ THUẬT:

Chữ Quốc ngữ viết RỜI THEO ÂM TIẾT. "trường đại học" là ba token nếu tách theo
khoảng trắng, nhưng chỉ là MỘT đơn vị từ vựng. Hệ quả trực tiếp: cùng một nội
dung, token hoá theo âm tiết sinh ra chuỗi DÀI HƠN so với token hoá bằng BPE.

Chuỗi dài hơn => chi phí O(L²) của attention cắn mạnh hơn => về lý thuyết, lợi thế
dưới bậc hai của Hyena có giá trị thực tiễn với tiếng Việt hơn. Đó là giả thuyết
H2 trong đề cương, và tầng dữ liệu này tồn tại để kiểm chứng nó — chứ không phải
chỉ để "nạp dữ liệu vào mô hình".

CÔNG BẰNG XUYÊN NGÔN NGỮ: nhánh tiếng Anh phải bị cắt xuống ĐÚNG cùng số token
với nhánh tiếng Việt. So sánh hai ngôn ngữ trên hai lượng dữ liệu khác nhau thì
không kết luận được gì về ngôn ngữ cả — chỉ kết luận được về lượng dữ liệu.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# Token đặc biệt — giữ cố định ở đầu từ điển cho mọi bộ token hoá.
PAD, UNK, EOS = 0, 1, 2
SPECIAL_TOKENS = ["<pad>", "<unk>", "<eos>"]


# -----------------------------------------------------------------------------
# Làm sạch văn bản
# -----------------------------------------------------------------------------
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Chuẩn hoá tối thiểu, có chủ đích.

    Dùng **NFC** cho tiếng Việt: dấu thanh và dấu phụ được tổ hợp sẵn. Nếu để lẫn
    NFC và NFD thì cùng một chữ "ế" có thể thành hai chuỗi ký tự khác nhau, làm
    phình từ điển và tạo ra khác biệt giả giữa các mô hình.

    Cố ý KHÔNG hạ chữ thường (lowercase) và KHÔNG bỏ dấu: chữ hoa mang thông tin
    (tên riêng, đầu câu) và bỏ dấu sẽ phá huỷ tiếng Việt.
    """
    text = unicodedata.normalize("NFC", text)
    return _WS.sub(" ", text).strip()


# -----------------------------------------------------------------------------
# Bộ token hoá
# -----------------------------------------------------------------------------
class SyllableTokenizer:
    """Token hoá theo ÂM TIẾT — tách theo khoảng trắng, tách riêng dấu câu.

    Với tiếng Việt đây là đơn vị tự nhiên của chữ viết. Với tiếng Anh, cùng quy
    tắc này cho ra đơn vị TỪ — nên nhánh đối chứng tiếng Anh phải được diễn giải
    là "từ", không phải "âm tiết". Khác biệt này phải nói rõ trong báo cáo.
    """

    name = "syllable"

    def __init__(self, vocab: list[str]):
        self.vocab = vocab
        self.stoi = {s: i for i, s in enumerate(vocab)}

    @staticmethod
    def _split(text: str) -> list[str]:
        # tách dấu câu thành token riêng để không làm phình từ điển
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    @classmethod
    def train(cls, texts: list[str], vocab_size: int) -> "SyllableTokenizer":
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(cls._split(t))
        keep = [w for w, _ in counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
        return cls(SPECIAL_TOKENS + keep)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(tok, UNK) for tok in self._split(text)]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.vocab[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"kind": "syllable", "vocab": self.vocab},
                                   ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SyllableTokenizer":
        return cls(json.loads(path.read_text(encoding="utf-8"))["vocab"])


class BPETokenizer:
    """BPE huấn luyện trên chính corpus, dùng thư viện `tokenizers` của HuggingFace.

    Đối chiếu với `SyllableTokenizer`: BPE gộp các âm tiết hay đi cùng nhau thành
    một token => chuỗi NGẮN hơn. Chênh lệch độ dài giữa hai bộ này chính là đại
    lượng cần đo cho giả thuyết H2.
    """

    name = "bpe"

    def __init__(self, tk):
        self._tk = tk

    @classmethod
    def train(cls, texts: list[str], vocab_size: int) -> "BPETokenizer":
        try:
            from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Cần thư viện `tokenizers`. Trên Kaggle đã có sẵn; "
                "nếu thiếu: pip install tokenizers"
            ) from exc

        tk = Tokenizer(models.BPE(unk_token="<unk>"))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            show_progress=False,
        )
        tk.train_from_iterator(texts, trainer=trainer)
        return cls(tk)

    def encode(self, text: str) -> list[int]:
        return self._tk.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tk.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    def save(self, path: Path) -> None:
        self._tk.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        from tokenizers import Tokenizer
        return cls(Tokenizer.from_file(str(path)))


TOKENIZERS = {"syllable": SyllableTokenizer, "bpe": BPETokenizer}


# -----------------------------------------------------------------------------
# Nạp corpus
# -----------------------------------------------------------------------------
WIKI_CONFIG = {"vi": "20231101.vi", "en": "20231101.en"}


@dataclass
class CorpusStats:
    """Số liệu phải được ghi vào kết quả — đây là bằng chứng so sánh công bằng."""

    lang: str
    tokenizer: str
    vocab_size: int
    n_docs: int
    n_chars: int
    n_tokens_train: int
    n_tokens_val: int
    n_tokens_test: int
    chars_per_token: float
    unk_rate: float


def load_wiki_texts(lang: str, n_docs: int, seed: int = 0,
                    cache_dir: str | None = None, streaming: bool = True,
                    shuffle_buffer: int = 10_000) -> list[str]:
    """Lấy `n_docs` bài Wikipedia (đã xáo trộn) cho ngôn ngữ `lang`.

    Nguồn: `wikimedia/wikipedia`, config `20231101.vi` (665.622 bài, ~1,13 GB) và
    `20231101.en`. Đã kiểm chứng tồn tại ngày 2026-08-08.

    MẶC ĐỊNH DÙNG STREAMING — lý do quan trọng:
    Wikipedia tiếng Anh có ~6,4 triệu bài. Tải trọn bộ về chỉ để lấy vài chục
    nghìn bài sẽ ăn hết dung lượng đĩa của Kaggle và tốn rất nhiều thời gian,
    trong khi nhánh đối chứng tiếng Anh chỉ cần đúng bằng lượng của tiếng Việt.
    Streaming chỉ kéo về phần thực sự dùng.

    Đánh đổi phải biết: ở chế độ streaming, `shuffle` chỉ xáo trong một bộ đệm
    `shuffle_buffer` phần tử chứ không xáo toàn cục. Mẫu thu được vì thế lệch về
    phía đầu tập dữ liệu. Điều này **chấp nhận được** vì ta áp dụng **đúng cùng
    một quy trình lấy mẫu cho cả hai ngôn ngữ** — thiên lệch triệt tiêu khi so
    sánh. Nhưng phải ghi rõ trong phần hạn chế của báo cáo, không được giấu.

    Kaggle cần BẬT Internet trong Settings của notebook.
    """
    if lang not in WIKI_CONFIG:
        raise ValueError(f"lang phải thuộc {list(WIKI_CONFIG)}, nhận được {lang!r}")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Cần thư viện `datasets`: pip install datasets") from exc

    if streaming:
        ds = load_dataset("wikimedia/wikipedia", WIKI_CONFIG[lang],
                          split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer).take(n_docs)
        texts = [normalize_text(row["text"]) for row in ds
                 if row.get("text") and row["text"].strip()]
    else:
        ds = load_dataset("wikimedia/wikipedia", WIKI_CONFIG[lang],
                          split="train", cache_dir=cache_dir)
        ds = ds.shuffle(seed=seed).select(range(min(n_docs, len(ds))))
        texts = [normalize_text(t) for t in ds["text"] if t and t.strip()]

    if not texts:
        raise RuntimeError(
            f"Không lấy được bài nào cho lang={lang!r}. Kiểm tra đã BẬT Internet "
            "trong Kaggle Settings chưa."
        )
    return texts


def build_token_stream(
    texts: list[str],
    tokenizer_kind: str = "syllable",
    vocab_size: int = 16000,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    max_tokens: int | None = None,
    lang: str = "vi",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, CorpusStats]:
    """Văn bản thô -> ba mảng token id liền mạch (train/val/test) + bộ token hoá.

    `max_tokens` cắt tập TRAIN xuống đúng một ngân sách token — đây là cơ chế bảo
    đảm công bằng giữa hai ngôn ngữ và giữa hai bộ token hoá (đề cương §5.0 điểm 3).
    """
    if tokenizer_kind not in TOKENIZERS:
        raise ValueError(f"tokenizer phải thuộc {list(TOKENIZERS)}")

    tok = TOKENIZERS[tokenizer_kind].train(texts, vocab_size)

    ids: list[int] = []
    for t in texts:
        ids.extend(tok.encode(t))
        ids.append(EOS)
    arr = np.asarray(ids, dtype=np.int32)

    n = len(arr)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test, val, train = arr[-n_test:], arr[-(n_test + n_val):-n_test], arr[: n - n_test - n_val]

    if max_tokens is not None and len(train) > max_tokens:
        train = train[:max_tokens]

    stats = CorpusStats(
        lang=lang,
        tokenizer=tokenizer_kind,
        vocab_size=tok.vocab_size,
        n_docs=len(texts),
        n_chars=sum(len(t) for t in texts),
        n_tokens_train=len(train),
        n_tokens_val=len(val),
        n_tokens_test=len(test),
        chars_per_token=sum(len(t) for t in texts) / max(n, 1),
        unk_rate=float((arr == UNK).mean()),
    )
    return train, val, test, tok, stats


# -----------------------------------------------------------------------------
# Dataset cho PyTorch
# -----------------------------------------------------------------------------
class LMWindowDataset:
    """Cắt dòng token liền mạch thành các cửa sổ dài L+1 KHÔNG chồng lấn.

    Cố ý không chồng lấn: cửa sổ chồng nhau khiến cùng một token được học nhiều
    lần, làm "số token huấn luyện" ghi trong báo cáo không còn phản ánh đúng
    lượng tín hiệu thực sự — phá vỡ tính so sánh cùng ngân sách.
    """

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.seq_len = seq_len
        n_win = (len(tokens) - 1) // seq_len
        if n_win < 1:
            raise ValueError(
                f"chỉ có {len(tokens)} token, không đủ cho một cửa sổ dài {seq_len + 1}"
            )
        usable = n_win * seq_len + 1
        self.data = np.ascontiguousarray(tokens[:usable])
        self.n_win = n_win

    def __len__(self) -> int:
        return self.n_win

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        s = i * self.seq_len
        chunk = self.data[s : s + self.seq_len + 1]
        return chunk[:-1].astype(np.int64), chunk[1:].astype(np.int64)


def stats_to_dict(stats: CorpusStats) -> dict:
    return asdict(stats)

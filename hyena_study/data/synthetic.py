"""
Tac vu tong hop ASSOCIATIVE RECALL - cong cu chan doan tri nho tam xa.

TAI SAO PHAI CO TAC VU NAY (khong phai lam cho vui):

Phep do E0-b cho ra bo loc voi do dai hieu dung trung vi chi 2,9 token, va
142/256 kenh nam trong 4 token. Cau hinh do bien Hyena thanh gan nhu mot tich
chap cuc bo. No CO THE giam perplexity - vi thong tin trong van ban that su tap
trung o khoang cach ngan - trong khi da vut bo chinh kha nang tam xa lam nen gia
tri cua Hyena.

Perplexity KHONG lo ra dieu do: mot mo hinh chi nhin 4 token van dat PPL kha tot
tren van ban tu nhien. Can mot tac vu ma cau tra loi nam CHINH XAC o xa, va mo
hinh khong the doan mo tu ngu canh cuc bo. Do la associative recall.

DINH NGHIA (Poli et al., Table 1):

    prompt: a, 1, b, e, 3, f, b   ->  target: e

Chuoi gom cac cap (khoa, gia tri) noi tiep nhau, roi mot khoa truy van o cuoi.
Mo hinh phai tra ve dung gia tri da di kem khoa do truoc day. Khoang cach tu truy
van nguoc ve cap goc co the dai tuy y, nen tac vu do THANG kha nang nho xa.

Paper cung dung ho tac vu nay lam thuoc do chinh khi so sanh cac toan tu
(Table 5: Hyena 97,2% vs H3 0,6% o do dai 131072, vocab 30).

QUYET DINH CAI DAT (paper khong dac ta chi tiet, day la lua chon cua nhom):

  (S1) Moi mau co mot anh xa khoa -> gia tri rieng, sinh bang mot HOAN VI ngau
       nhien cua bang chu cai. Nho vay anh xa luon don tri: mot khoa chi ung voi
       dung mot gia tri trong cung mot chuoi. Neu lay ngau nhien khong rang buoc
       thi cung mot khoa co the di kem hai gia tri khac nhau, tac vu tro nen
       KHONG GIAI DUOC va ket qua 0% se vo nghia.
  (S2) Khoa duoc lay CO HOAN LAI, nen cap co the lap lai trong chuoi - dung nhu
       paper ghi nhan o cac chuoi dai ("multiple copies of key-value tuples
       appear in the prompt").
  (S3) Ham mat mat chi tinh tren VI TRI CUOI CUNG. Tinh tren moi vi tri se lam
       loang tin hieu bang phan du doan cac token dem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RecallConfig:
    """Sieu tham so cua tac vu associative recall."""

    vocab_size: int = 20        # so ky hieu; paper quet 10, 20, 30, 40
    seq_len: int = 128          # do dai chuoi (ke ca token truy van)
    n_train: int = 20000
    n_val: int = 2000
    n_test: int = 2000
    seed: int = 0

    @property
    def n_pairs(self) -> int:
        """So cap (khoa, gia tri) nhet vua vao chuoi."""
        return (self.seq_len - 1) // 2

    def __post_init__(self) -> None:
        if self.vocab_size < 4:
            raise ValueError("vocab_size phai >= 4")
        if self.n_pairs < 1:
            raise ValueError(
                f"seq_len={self.seq_len} qua ngan, khong chua noi mot cap khoa-gia tri"
            )


def make_recall_split(cfg: RecallConfig, n_samples: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Sinh mot tap du lieu associative recall.

    Returns:
        x: (n_samples, L) chuoi dau vao, L = 2*n_pairs + 1
        y: (n_samples,) gia tri dung ung voi khoa truy van

    Cach dung: mo hinh doc x, du doan tai VI TRI CUOI, so voi y.
    """
    V, P = cfg.vocab_size, cfg.n_pairs
    L = 2 * P + 1

    x = np.empty((n_samples, L), dtype=np.int64)
    y = np.empty(n_samples, dtype=np.int64)

    for i in range(n_samples):
        # (S1) hoan vi ngau nhien => anh xa khoa -> gia tri luon don tri
        mapping = rng.permutation(V)
        # (S2) lay khoa co hoan lai
        keys = rng.integers(0, V, size=P)
        vals = mapping[keys]

        x[i, 0:2 * P:2] = keys
        x[i, 1:2 * P:2] = vals

        q = int(rng.choice(keys))       # truy van phai la khoa DA xuat hien
        x[i, -1] = q
        y[i] = mapping[q]

    return x, y


def build_recall_dataset(cfg: RecallConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Sinh ca ba tap train/val/test voi cung mot seed goc."""
    rng = np.random.default_rng(cfg.seed)
    return {
        "train": make_recall_split(cfg, cfg.n_train, rng),
        "val": make_recall_split(cfg, cfg.n_val, rng),
        "test": make_recall_split(cfg, cfg.n_test, rng),
    }


def query_distance(x: np.ndarray) -> np.ndarray:
    """Khoang cach tu token truy van nguoc ve LAN XUAT HIEN CUOI cua cung khoa.

    Dung de phan tich: gom do chinh xac theo khoang cach cho biet mo hinh nho xa
    duoc bao nhieu. Mot mo hinh chi nhin 4 token se dat gan 100% khi khoang cach
    ngan va roi ve muc doan mo khi khoang cach dai - chinh la dau hieu can tim.

    Tra ve mang (n_samples,) so token tinh tu vi tri truy van nguoc ve khoa do.
    """
    n, L = x.shape
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        q = x[i, -1]
        # duyet nguoc, bo qua chinh token truy van; khoa nam o cac vi tri chan
        for pos in range(L - 3, -1, -2):
            if x[i, pos] == q:
                out[i] = (L - 1) - pos
                break
    return out


def chance_accuracy(cfg: RecallConfig) -> float:
    """Do chinh xac cua phep doan mo. Moc so sanh BAT BUOC trong moi bang ket qua.

    Anh xa la mot hoan vi ngau nhien nen gia tri dung phan bo deu tren V ky hieu:
    doan mo cho 1/V. Bao cao do chinh xac ma khong kem moc nay thi nguoi doc khong
    biet 20% la gioi hay la vo dung.
    """
    return 1.0 / cfg.vocab_size

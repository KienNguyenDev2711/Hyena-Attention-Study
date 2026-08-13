# Task sửa báo cáo cuối kỳ (Nhóm 8)

Nguồn: rà soát `Report_Nhom8.pdf` (9 trang, biên dịch 2026-08-11) đối chiếu trực tiếp
với `results/`, `hyena_study/`, `tests/` và toàn văn bài báo gốc
(`../Hyena_paper_fulltext_extracted.txt`).

**Đã kiểm và ĐÚNG (không cần sửa):** toàn bộ Bảng 4, 5, 6 (giá trị PPL), 7, 8; 22/22 ô
tăng tốc E5; 11/11 ô Bảng 2; số bộ nhớ; số tham số 7.553.280 / 7.383.552 / 2,30%; mọi
khoảng tin cậy (phân phối t, df = n-1); mọi trích dẫn bài báo gốc (2x vs FlashAttention
@8192, footnote "FlashAttention already 2-4x faster", Table 5 AR 97,2% @131072 vocab 30,
355M ở Table 3, Định nghĩa 3.1); 18/18 test trong `test_models.py` + `test_morphology.py`
chạy PASS; bố cục và dấu tiếng Việt trên bản in.

Tổng thời gian phần không cần GPU: khoảng 3 giờ.

---

## Thứ tự ưu tiên

| Nhóm | Task | Ghi chú |
|---|---|---|
| Bắt buộc | T1, T2, T3, T4, T5 | Sai sự thật hoặc mâu thuẫn nội bộ, kiểm được trong vài giây |
| Nên có | T6, T7, T8, T10 | Sai nguồn hoặc thống kê yếu |
| Nếu còn giờ | T9, T11, T12 | Trình bày và minh bạch |
| Tuỳ chọn | T13 | Cần GPU |

---

## T1. Sửa câu sai về dữ liệu, mục 4.2 — NẶNG

- [ ] Chưa làm

**Hiện có:** "Nhánh tiếng Anh được cắt xuống đúng cùng số token với nhánh tiếng Việt."

**Bằng chứng phủ định:**

| File | `corpus.n_tokens_train` |
|---|---|
| `results/E1_vi_HHHH_s0.json` | 38.250.964 |
| `results/E1_en_HHHH_s0.json` | 42.147.057 |

Cơ chế cắt là `max_tokens = token_budget = 50.000.000` (`hyena_study/train.py:125`).
Cả hai corpus đều nhỏ hơn 50M nên lệnh cắt **không bao giờ chạy**. Ý định có ghi trong
`hyena_study/data/corpus.py:15` nhưng không được thực thi.

**Sửa thành:** "cùng 90.000 tài liệu và cùng ngân sách 50 triệu token huấn luyện".

**Thêm vào mục Hạn chế:** hai nhánh không cùng số token duy nhất, nên VI chạy khoảng
1,31 epoch còn EN khoảng 1,19 epoch trên tập token của mình.

Thời gian: 10 phút.

---

## T2. Sửa hai dòng corpus của Bảng 3 — NẶNG

- [ ] Chưa làm

**Vấn đề:** Bảng 3 in số của `alpha_vi.json` / `alpha_en.json` (đo bằng âm tiết,
top_k = 1000), nhưng E4 huấn luyện bằng `alpha_vi_bpe500.json` / `alpha_en_bpe500.json`
(xem `config.alpha_file` trong `results/E4_corpus_*.json`). Kết quả là Bảng 3 in trung vị
3,6 / 4,8 trong khi mục 5.3 và 5.9 trích 2,95 / 2,85 / 2,9 cho **cùng một cấu hình mang
cùng một tên**, và bộ lọc trong Bảng 3 chưa từng được huấn luyện.

**Sửa thành:**

| Cấu hình | <= 4 | <= 34 | trung vị |
|---|---|---|---|
| corpus, tiếng Việt | 142 | 227 | 2,95 |
| corpus, tiếng Anh | 141 | 214 | 2,85 |

Hai dòng `uniform` (0 / 0 / 169,3) và `logspace` (57 / 145 / 22,6) giữ nguyên, đã kiểm đúng.

Thời gian: 5 phút.

---

## T3. Sửa "90% thông tin nằm trong 34 token" — NẶNG

- [ ] Chưa làm

**Số thật**, tính theo đúng quy ước của nhóm:

| Phép đo | tích luỹ tại d <= 34 | đạt 90% tại |
|---|---|---|
| VI âm tiết | 87,5% | d = 46 |
| VI BPE | 88,7% | d = 46 |
| EN âm tiết | 79,9% | d = 103 |
| EN BPE | 83,5% | d = 75 |

Docstring của chính nhóm ghi "80-90%" (`hyena_study/morphology.py:213`,
`tests/test_morphology.py:178`).

**Sửa thành:** "80-90% (tuỳ ngôn ngữ và bộ token hoá)".

**Sửa ở ba chỗ:** mục 3.6, chú thích Bảng 3, chú giải trong Hình 2.

Lưu ý: lập luận "uniform là hình nộm" vẫn đứng vững, vì `uniform` đặt 100% số kênh ra xa
hơn 64 token. Chỉ con số phải sửa.

Thời gian: 10 phút.

---

## T4. Sửa "1,66 tới 2,27 lần ở mọi khoảng cách" — NẶNG

- [ ] Chưa làm

**Vấn đề:** câu này mâu thuẫn với Bảng 2 nằm ngay cạnh. Cột BPE: d = 34 cho 1,55;
d = 88 cho 1,27; d = 167 cho 1,03. Dải thật trong vùng tin cậy (d <= 167) là **1,03 đến
2,27**, không phải 1,66 đến 2,27.

**Vấn đề thứ hai:** câu tiếp theo viết "Kết quả này bền qua cả hai cách token hoá", trong
khi ở cột âm tiết tiếng Việt **thấp hơn** tiếng Anh tại d = 88, 103, 121 (0,94 / 0,94 /
0,90), và đoạn ngay sau đó lại thừa nhận đúng điều này.

**Sửa thành:** "gấp 1,7 đến 2,3 lần ở khoảng cách ngắn (d <= 13), giảm dần về khoảng 1,0
khi tiến tới d = 167; với token hoá âm tiết, ưu thế đảo chiều từ d = 88 do nhiễu loạn
token ngoài từ điển."

**Bỏ hẳn** cụm "bền qua cả hai cách token hoá".

**Sửa ở ba chỗ:** phần Tóm tắt, mục 5.1, mục 6 (Thảo luận).

Thời gian: 15 phút.

---

## T5. Thêm mục Hạn chế về lưới lấy mẫu I(d) — NẶNG

- [ ] Chưa làm

**Vấn đề kỹ thuật:** `hyena_study/morphology.py:132` lấy mẫu `lags` theo thang log
(`np.logspace(0, log10(512), 40)`), hoàn toàn hợp lý cho việc đo.
Nhưng `hyena_study/morphology.py:176-187` chuẩn hoá `w = mi / mi.sum()` **trên chính lưới
log đó**, coi mỗi lag là một khối lượng xác suất, **không nhân bề rộng ô**. Lưới log lấy
dày ở d nhỏ (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, ...) và thưa ở d lớn (317, 372, 436, 512),
nên phân bố bị kéo về phía 0 một cách cơ học.

**Bảng độ nhạy cần đưa vào báo cáo:**

| Phép đo | trung vị (token) | <= 4 kênh | <= 34 kênh |
|---|---|---|---|
| VI-BPE, coi lưới log là khối lượng xác suất (cách nhóm làm) | 2,95 | 142 | 227 |
| VI-BPE, có trọng số bề rộng ô | 90,54 | 39 | 87 |
| EN-BPE, coi lưới log là khối lượng xác suất | 2,85 | 141 | 214 |
| EN-BPE, có trọng số bề rộng ô | 160,45 | 27 | 57 |

Trung vị lệch khoảng 30 lần chỉ vì một quy ước chưa được nêu.

**Nội dung cần viết (một đoạn):**
1. Nêu rõ quy ước đã dùng.
2. In bảng độ nhạy trên.
3. Nói rõ điều này **không** làm sai kết quả PPL ở Bảng 7 (chạy sao thì ra vậy), nhưng nó
   đổi cách diễn giải mệnh đề "thông tin dự đoán của corpus nằm trong khoảng 3 token".

**Câu hỏi phản biện phải trả lời được:** "Vì sao thông tin lại chỉ nằm trong 3 token trong
khi Hình 1 cho thấy I(d) còn trên nền nhiễu tới d = 167?"

Thời gian: 30 phút.

---

## T6. Sửa số throughput ở mục 5.8 — VỪA

- [ ] Chưa làm

**Hiện có:** "Transformer đạt 150,0 nghìn token mỗi giây còn Hyena đạt 92,9 nghìn, tức
Transformer nhanh hơn 1,61 lần."

**Nguồn thật của hai số đó:** `results/E0a_transformer_vi.json` (150.000 tok/s) và
`results/E0a_hyena_vi.json` (92.937 tok/s) — đây là lần **chạy thử E0a**, n_docs = 3000,
chỉ 3 triệu token.

**Số của thí nghiệm chính (E1, cùng L = 512, cùng T4, 50 triệu token):**

| Mô hình | tok/s | Nguồn |
|---|---|---|
| Transformer | 138.100 - 138.600 | `E1_*_AAAA_s*.json` |
| Hyena | 91.500 - 92.500 | `E1_*_HHHH_s*.json` |

Tỉ lệ đúng là **1,50**, không phải 1,61.

**Sửa:** thay bằng số E1, hoặc giữ nguyên nhưng ghi rõ "đo trên lần chạy thử E0a".

Thời gian: 10 phút.

---

## T7. Sửa "61 phép kiểm tự động" thành 62 — VỪA

- [ ] Chưa làm

`python -m pytest tests/ --collect-only -q` đếm được **62**:

| File | Số test |
|---|---|
| `test_analyze.py` | 10 |
| `test_cache.py` | 8 |
| `test_e6_script.py` | 7 |
| `test_models.py` | 11 |
| `test_morphology.py` | 7 |
| `test_pipeline.py` | 11 |
| `test_recall.py` | 8 |
| **Tổng** | **62** |

Đây là con số thầy kiểm được bằng một dòng lệnh.

Thời gian: 2 phút.

---

## T8. Hạ giọng dòng "Bậc N = 1" ở Bảng 6 — VỪA

- [x] Đã làm (Quang, 2026-08-13) — sửa cả ba chỗ: dòng Bảng 6 + chú thích, mục 4.1, mục 5.6

**Vấn đề 1:** với n = 2, t(df = 1) = 12,706. KTC của `order1` là [51,724; 54,488]; KTC của
cấu hình gốc là [51,052; 51,708]. Khoảng hở chỉ **0,016 PPL**. Một seed thứ ba gần như
chắc chắn lật kết luận này.

**Sửa:** giữ chữ "có" nhưng thêm chú thích "tách rời nhưng biên rất sát (0,016 PPL)".

**Vấn đề 2:** mục 4.1 khẳng định "Mọi cấu hình chỉ khác nhau ở toán tử trộn token" — không
đúng với nhánh ablation:

| Nhánh | Tổng tham số | Chênh so với gốc |
|---|---|---|
| Gốc (bậc 2) | 7.553.280 | - |
| Bậc N = 1 | 7.221.504 | -4,4% |
| Bậc N = 3 | 7.885.056 | +4,4% |
| Bỏ positional emb | 7.422.208 | -1,7% |

Thiệt hại của N = 1 bị lẫn với việc mất 331.776 tham số. Phải ghi rõ trong mục 5.6.

Thời gian: 20 phút.

---

## T9. Thêm cột khoảng tin cậy vào Bảng 6 — NHẸ

- [x] Đã làm (Quang, 2026-08-13) — cột KTC 95% tính từ `test_ppl` trong `results/`, caption ghi rõ n và t

Hiện chỉ có cột "có / không", người đọc không kiểm được. In khoảng cụ thể cho từng nhánh
(n = 2, t = 12,706).

Thời gian: 15 phút.

---

## T10. Sửa cách trình bày 14,3% và tương quan 0,9974 — VỪA

- [ ] Chưa làm

**Điểm 1 — quy ước của 14,3%.** Công thức đã tái lập được là
`mean(|ell_vi - ell_en| / ell_en)` = 14,35%. Nhưng đổi mẫu số sang `ell_vi` cho **22,29%**,
bản đối xứng cho 17,27%. Số nhảy 1,55 lần tuỳ hướng.
Kết luận vẫn vững (14-22% so với 76% trở lên), chỉ cần **nêu rõ quy ước**.

**Điểm 2 — tương quan log 0,9974 không phải bằng chứng.** Hai vector đều là đường phân vị
đơn điệu đã sắp xếp, nên tương quan cao là điều không thể không xảy ra. Bằng chứng: cặp
`corpus` với `logspace`, mà báo cáo gọi là rất khác nhau, cũng đạt tương quan log
**0,9377** (bản âm tiết: 0,9491).
**Sửa:** bỏ con số 0,9974, hoặc nêu kèm 0,9377 để người đọc thấy thang đo.

**Điểm 3 — không có script tái sinh.** Hai con số 14,3% và 75,8% chỉ xuất hiện dưới dạng
văn xuôi trong `docs/00_de_cuong_nghien_cuu.md`, `docs/02_ke_hoach_tong_the.md`,
`docs/03_qa_phan_bien.md`. Không script nào trong repo sinh ra chúng.
**Việc cần làm:** thêm một hàm trong `hyena_study/analyze.py` để tái sinh, kèm test.

Thời gian: 20 phút (chưa tính phần viết hàm).

---

## T11. Ghi chú lệch đơn vị BPE và âm tiết — NHẸ

- [ ] Chưa làm

`E4_corpus_vi` dùng `alpha_vi_bpe500.json`, đo trên corpus BPE (3,851 ký tự mỗi token),
nhưng huấn luyện với `tokenizer: syllable` (4,118 ký tự mỗi token). Độ dài hiệu dụng tính
bằng **token BPE** được áp thẳng lên **vị trí token âm tiết**.

| Ngôn ngữ | ký tự/token khi đo (BPE) | ký tự/token khi huấn luyện (âm tiết) | lệch thang |
|---|---|---|---|
| VI | 3,851 | 4,118 | khoảng 6,5% |
| EN | 4,155 | 5,115 | khoảng 19% |

Nhỏ so với T5, nhưng với một đóng góp mà toàn bộ ý tưởng là khớp độ dài bộ lọc với khoảng
cách thông tin, việc không nêu phép quy đổi đơn vị là chỗ phản biện sẽ nhắm vào.

**Sửa:** thêm một câu vào mục 3.5 hoặc mục Hạn chế.

Thời gian: 10 phút.

---

## T12. Các lỗi vụn — NHẸ

- [ ] Chưa làm

1. **Mục 5.5, số ký tự trên token.** 4,101 và 5,121 là của corpus đo MI 30.000 tài liệu
   (`alpha_vi.json`, `alpha_en.json`), không phải corpus huấn luyện 90.000 tài liệu
   (4,118 và 5,115, tỉ lệ 1,242 chứ không phải 1,249). Ghi rõ nguồn.

2. **Bảng 7.** Nói rõ ba dòng `uniform` chính là các lần chạy Hyena ở Bảng 4
   (`E1_vi_HHHH_*`, `E1_en_HHHH_*`), không phải sáu lần chạy thêm.

3. **Mục 6 và mục 5.9 khung hoá khác nhau.** Mục 6 viết "hai trong ba lần chạy đạt tới
   0,98", mục 5.9 viết "3 trên 5". Cả hai đúng với hai tập con khác nhau. Chọn một cách nói.

4. **Hình 3, panel phải (bộ nhớ).** Chỉ là lượt tiến (`E5_benchmark_fp16_fwd.csv`), trong
   khi panel trái gồm cả tiến và lùi. Phải ghi nhãn.

5. **Mục 4.3, hai con số kiểm chứng nhân quả.** `_future_grad_ratio`
   (`tests/test_models.py:119`) gọi `torch.randn` không đặt seed, nên 2,6e-16 và 9,1e-1 chỉ
   là một lần bốc ngẫu nhiên. Chạy lại cho: Hyena 3,78e-16; bộ lọc ngắn 1,67e-16; toàn mô
   hình 5,39e-16; phép kiểm ngược 1,98.
   **Sửa:** đổi thành "dưới 1e-15 trên mọi biến thể" và nêu `FFT_LEAK_TOL`, hoặc seed lại test.

6. **Mục 5.9, hai con số không có artifact.** "attention giảm còn 0,196 ở độ dài 129" và
   "có warmup 1,000 còn không warmup 0,187" không nằm trong file nào dưới `results/` (hai
   file CSV E6 chỉ có L = 65). Ghi rõ "đo một lần, không lưu artifact".
   Liên quan: `results/NGUON_GOC_E6.txt` do chính nhóm viết đã ghi
   `E6_recall_corpus_k37440.csv` là **chép tay từ log Kaggle**, bản gốc nằm trong
   `E6_A.zip` — file này không có trong repo.

7. **Typography.** Trên bản in thấy `"có"mới` (chú thích Bảng 6) và `"ngây thơ"là` (chú
   thích Bảng 8) bị mất dấu cách, do dùng dấu nháy kép thẳng trong LaTeX. Thay bằng cặp
   nháy LaTeX (hai dấu huyền mở, hai dấu nháy đơn đóng).

8. **Bộ biên dịch.** PDF hiện tại biên dịch bằng pdfTeX với vntex T5 (metadata:
   `pdfTeX-1.40.26`; font VNR / VNTI / VNBX), không phải LuaLaTeX như chính phần đầu file
   `.tex` khuyến nghị. Hệ quả: chữ ra Computer Modern thay vì Times theo chuẩn ACL, và lớp
   text mất dấu cách sau ký tự có dấu móc khi copy-paste (ảnh hưởng công cụ đối chiếu trùng
   lặp). Bản in vẫn đúng, đã kiểm bằng cách render trang 1, 2, 4, 5, 6, 7.

Thời gian: 30 phút.

---

## T13. (Tuỳ chọn, cần GPU) Chạy lại nhánh tiếng Anh cùng số token

- [ ] Hạ tầng xong (Quang, 2026-08-13): cờ `--max_train_tokens` trong `train.py` + notebook
  `notebooks/colab_E1_en_matched.ipynb`. Còn chờ: push lên GitHub → chạy Colab (~1h GPU) →
  đưa `E1_en_matched_results.zip` về tích hợp vào `results/`

Chỉ làm nếu muốn **giữ nguyên** khẳng định "cùng số token" ở T1 thay vì viết lại.

Cắt corpus EN xuống 38.250.964 token, chạy lại 6 lần: `E1_en` x 2 mô hình x 3 seed.

Ngân sách: khoảng 1 giờ GPU trên T4.

---

## Phản biện học thuật (không phải lỗi, nhưng hội đồng sẽ hỏi)

Không cần sửa file, nhưng phải chuẩn bị câu trả lời.

1. **"Hyena vượt Transformer 19,5%".** Báo cáo đã tự hãm rất đúng mực. Nhưng để ý hướng của
   lời hãm: Hyena chậm hơn 1,5 lần trên mỗi token, nên **ở cùng thời gian thực, Transformer
   thấy nhiều hơn khoảng 1,5 lần số token**. Một dòng compute-matched (Transformer chạy 75
   triệu token) tốn một lần chạy khoảng 9 phút và chặn đứng câu hỏi khó nhất.

2. **Nhan đề vẫn dẫn bằng "khởi tạo bộ lọc từ cấu trúc thông tin của corpus"**, trong khi
   đó chính là phần **không** kết luận được (+0,285 và +0,167 PPL, KTC chồng lấn). Cân nhắc
   để nhan đề dẫn bằng thứ đã đứng vững.

3. **Kết quả mạnh nhất đang bị chôn ở mục 5.7:** `logspace` vượt `uniform` khoảng 1 PPL với
   KTC tách rời trên **cả hai** ngôn ngữ. Đây là phát hiện sạch, tái lập được, về đúng một
   siêu tham số mà bài báo gốc **để ngỏ** (mục 2.3 của chính báo cáo). Nên đưa lên đầu.

4. **KTC chỉ bắt nhiễu khởi tạo.** Ba seed trên một tập test duy nhất, không bootstrap trên
   tập test. Chênh lệch 0,285 PPL nằm dưới cả sai số dữ liệu chưa được ước lượng.

5. **"Vùng tin cậy" là tên gọi quá mạnh.** Nó được định nghĩa là `signal_to_bias > 1`, tức
   tín hiệu **bằng** độ chệch. Và `n_shuffle = 1` (mặc định, `hyena_study/morphology.py:114`)
   nên đường nền nhiễu không có thanh sai số. Nên gọi là "vùng tín hiệu trên nền" và tăng
   số lần xáo.

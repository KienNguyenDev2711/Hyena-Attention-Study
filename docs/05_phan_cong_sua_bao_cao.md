# Bảng phân công sửa báo cáo — Nhóm 8

**Môn:** Chuyên đề nghiên cứu và Xử lý ngôn ngữ tự nhiên
**Tên đề tài:** Hyena cho tiếng Việt: Mô hình ngôn ngữ tích chập dài dưới bậc hai, phân tích
ablation và khởi tạo bộ lọc từ cấu trúc thông tin của corpus
**Bài báo nền:** Poli et al., *Hyena Hierarchy: Towards Larger Convolutional Language Models*,
ICML 2023 (Oral)

**Nguồn:** danh sách lỗi chi tiết ở [`04_task_sua_bao_cao.md`](04_task_sua_bao_cao.md).
File này chỉ chia việc; mọi bằng chứng và câu sửa cụ thể nằm ở file 04.

**Phạm vi:** đây là phần **bổ sung** cho bảng phân công gốc (`Task_Nhóm 8.pdf`), điền vào
mục "Project cần những điểm sau để cải thiện" đang để trống ở trang 4 của file đó.

---

## 1. Thành viên

| Họ và tên | MSHV | Vai trò (giữ theo giai đoạn seminar) |
|---|---|---|
| Nguyễn Cao Trung Kiên | 250201069 | Báo cáo, biên dịch, soát cuối |
| Tô Huỳnh Minh Tiến | 250201095 | Phương pháp, công thức |
| Trần Tú Quang | 250201084 | Mã nguồn, kết quả, thống kê |

> **Cần xác nhận:** `Task_Nhóm 8.pdf` trang 1 ghi "Tô Huỳnh Minh **Tiền**", còn
> `report/acl_latex.tex` và bản PDF báo cáo ghi "Tô Huỳnh Minh **Tiến**". Hai tài liệu nộp
> cho thầy đang ghi tên khác nhau. Phải thống nhất trước khi nộp.

---

## 2. Bảng tổng hợp phân công

| Task | Nội dung | Mức | Người | Thời gian |
|---|---|---|---|---|
| T1 | Sửa câu sai về dữ liệu, mục 4.2 | NẶNG | Kiên | 10 phút |
| T2 | Sửa hai dòng corpus của Bảng 3 | NẶNG | Kiên | 5 phút |
| T3 | Sửa "90% thông tin nằm trong 34 token" | NẶNG | Kiên | 10 phút |
| T4 | Sửa "1,66 tới 2,27 lần ở mọi khoảng cách" | NẶNG | Kiên | 15 phút |
| T5 | Thêm mục Hạn chế về lưới lấy mẫu I(d) | NẶNG | Tiến | 30 phút |
| T6 | Sửa số throughput ở mục 5.8 | VỪA | Kiên | 10 phút |
| T7 | Sửa "61 phép kiểm" thành 62 | VỪA | Kiên | 2 phút |
| T8 | Hạ giọng dòng "Bậc N = 1" ở Bảng 6 | VỪA | Quang | 20 phút |
| T9 | Thêm cột khoảng tin cậy vào Bảng 6 | NHẸ | Quang | 15 phút |
| T10 | Sửa cách trình bày 14,3% và tương quan 0,9974 | VỪA | Tiến | 20 phút |
| T11 | Ghi chú lệch đơn vị BPE và âm tiết | NHẸ | Tiến | 10 phút |
| T12 | Các lỗi vụn (8 mục) | NHẸ | Kiên | 30 phút |
| T13 | Chạy lại nhánh tiếng Anh cùng số token | TUỲ CHỌN | Quang | ~1 giờ GPU |

**Tổng không cần GPU:** khoảng 3 giờ. Kiên 1 giờ 22 phút · Tiến 1 giờ · Quang 35 phút.

---

## 3. Chi tiết theo từng người

### 3.1. Nguyễn Cao Trung Kiên — sửa văn bản báo cáo và biên dịch

Toàn bộ phần này thao tác trên `report/acl_latex.tex`.

| Task | Việc phải làm | Sản phẩm cần có |
|---|---|---|
| T1 | Bỏ câu "Nhánh tiếng Anh được cắt xuống đúng cùng số token", thay bằng "cùng 90.000 tài liệu và cùng ngân sách 50 triệu token huấn luyện". Thêm một câu vào Hạn chế về chênh lệch epoch (VI ~1,31 · EN ~1,19) | Mục 4.2 và mục Hạn chế đã sửa |
| T2 | Thay hai dòng corpus của Bảng 3: VI thành `142 / 227 / 2,95`, EN thành `141 / 214 / 2,85` | Bảng 3 khớp với file alpha thật sự dùng để huấn luyện |
| T3 | Đổi "90%" thành "80–90% (tuỳ ngôn ngữ và bộ token hoá)" ở **ba chỗ**: mục 3.6, chú thích Bảng 3, chú giải trong Hình 2 | Ba chỗ đã sửa. Hình 2 phải sinh lại vì chữ nằm trong hình |
| T4 | Sửa dải tỉ lệ thông tin tương hỗ, bỏ cụm "bền qua cả hai cách token hoá". Sửa ở **ba chỗ**: Tóm tắt, mục 5.1, mục 6 | Ba chỗ đã sửa, không còn mâu thuẫn với Bảng 2 |
| T6 | Thay 150,0k / 92,9k / 1,61 bằng số của E1 (138,1–138,6k / 91,5–92,5k / **1,50**), hoặc ghi rõ là số của lần chạy thử E0a | Mục 5.8 đã sửa |
| T7 | Đổi "61 phép kiểm tự động" thành **62** ở mục 4.3 | Đã sửa |
| T12 | Tám mục vụn: nguồn số ký tự/token; chú thích dòng `uniform` của Bảng 7; thống nhất "3 trên 5" với "hai trong ba"; ghi nhãn Hình 3 panel phải là lượt tiến; sửa hai số kiểm chứng nhân quả; ghi chú hai số E6 không có artifact; sửa hai dấu nháy kép bị dính chữ; **biên dịch lại bằng LuaLaTeX** | Bản PDF mới |

**Cảnh báo về biên dịch (T12 mục 8).** `Task_Nhóm 8.pdf` mục 3.1 ghi *"Bắt buộc dùng
LuaLaTeX, không dùng pdfLaTeX"*. Nhưng `Report_Nhom8.pdf` hiện tại có metadata
`pdfTeX-1.40.26` và nhúng font VNR / VNTI / VNBX (vntex T5), tức **đã biên dịch bằng
pdfLaTeX**. Bản in vẫn đúng dấu tiếng Việt (tôi đã render kiểm trang 1, 2, 4, 5, 6, 7), nên
không phải lỗi chặn nộp, nhưng:

- chữ ra Computer Modern thay vì Times theo chuẩn ACL;
- lớp text mất dấu cách sau ký tự có dấu móc khi copy-paste, ảnh hưởng công cụ đối chiếu
  trùng lặp.

Hạng mục "biên dịch bằng LuaLaTeX" trong bảng phân công gốc **chưa hoàn thành đúng đặc tả**.

**Phụ thuộc:** T1 phụ thuộc T13. Nếu Quang chạy lại được nhánh EN thì T1 chỉ cần giữ nguyên
câu cũ; nếu không thì bắt buộc sửa câu như trên.

---

### 3.2. Tô Huỳnh Minh Tiến — phương pháp và chỉ số so sánh

| Task | Việc phải làm | Sản phẩm cần có |
|---|---|---|
| T5 | Viết một đoạn Hạn chế về việc `alphas_from_mi` coi lưới log là khối lượng xác suất mà không nhân bề rộng ô. Kèm bảng độ nhạy (trung vị 2,95 so với 90,5 với VI; 2,85 so với 160,5 với EN) | Một đoạn trong mục Hạn chế + một bảng nhỏ |
| T10 | Nêu rõ quy ước của con số 14,3% (mẫu số là tiếng Anh; đổi mẫu số ra 22,3%). Bỏ hoặc bổ nghĩa con số tương quan log 0,9974, vì cặp `corpus` với `logspace` cũng đạt 0,9377. Viết hàm trong `hyena_study/analyze.py` để tái sinh 14,3% và 75,8% | Mục 5.3 đã sửa + hàm mới + test |
| T11 | Thêm một câu vào mục 3.5 hoặc Hạn chế về việc alpha đo bằng BPE nhưng huấn luyện bằng âm tiết (lệch thang ~6,5% với VI, ~19% với EN) | Một câu |

**Vì sao giao cho Tiến:** cả ba đều nằm ở tầng phương pháp (`hyena_study/morphology.py`), là
phần Tiến đã phụ trách từ giai đoạn seminar, và cả ba đều là chỗ hội đồng sẽ hỏi sâu về công
thức.

**Câu hỏi phản biện Tiến phải trả lời được sau T5:**
> "Vì sao thông tin dự đoán của corpus lại chỉ nằm trong khoảng 3 token, trong khi Hình 1 cho
> thấy I(d) còn nằm trên nền nhiễu tới d = 167?"

---

### 3.3. Trần Tú Quang — thống kê và chạy lại thí nghiệm

| Task | Việc phải làm | Sản phẩm cần có |
|---|---|---|
| T8 | Thêm chú thích cho dòng "Bậc N = 1" ở Bảng 6: tách rời nhưng chỉ hở 0,016 PPL. Sửa mục 4.1 và 5.6: nhánh ablation **không** cùng số tham số (order1 −4,4%, order3 +4,4%, no_posemb −1,7%) | Bảng 6 + hai mục đã sửa |
| T9 | Thêm cột khoảng tin cậy 95% vào Bảng 6 (n = 2, t = 12,706) thay vì chỉ in "có / không" | Bảng 6 có cột khoảng |
| T13 | *Tuỳ chọn, chỉ làm nếu muốn giữ nguyên khẳng định "cùng số token".* Cắt corpus EN xuống 38.250.964 token, chạy lại 6 lần `E1_en` (2 mô hình × 3 seed) | 6 file JSON mới trong `results/` |

**Ràng buộc hạ tầng:** T13 chạy trên Kaggle GPU, **không** train trên máy cá nhân. Máy cá nhân
chỉ chạy unit test CPU.

**Quy tắc bắt buộc khi chạy T13:** logic thí nghiệm **không** được viết trong ô notebook.
Kaggle chỉ clone repo rồi gọi module đã có test đường dây.

---

### 3.4. Cả nhóm

| Việc | Sản phẩm cần có | Ghi chú |
|---|---|---|
| Soát chéo: mỗi người kiểm phần của hai người kia | Danh sách lỗi còn sót | Không ai tự soát phần của mình |
| Đối chiếu lại toàn bộ con số trong slide với bản báo cáo đã sửa | Slide đã cập nhật | Nhiều số trong slide lấy từ bản báo cáo cũ |
| Cập nhật `docs/03_qa_phan_bien.md` | Phần Q&A đã cập nhật | Ba con số 14,3% / 75,8% / 1,66–2,27 đang nằm trong file này ở dạng cũ |
| Thống nhất cách trả lời năm câu ở mục 5 dưới đây | Câu trả lời chung | |

---

## 4. Đính chính cho chính bảng phân công gốc (`Task_Nhóm 8.pdf`)

Bảng phân công gốc cũng chứa vài số cần sửa trước khi nộp.

### 4.1. Đã kiểm và ĐÚNG, giữ nguyên

Phụ lục trang 5 của file gốc, các dòng sau tôi đã tính lại từ `results/` và đều khớp:

| Đại lượng | Giá trị trong file gốc | Kiểm |
|---|---|---|
| PPL Hyena so với Transformer, tiếng Việt | 51,38 so với 63,86 | đúng |
| PPL Hyena so với Transformer, tiếng Anh | 61,51 so với 72,82 | đúng |
| Số tham số | 7.553.280 so với 7.383.552 | đúng |
| Tăng tốc ở độ dài 8192, lượt tiến và lùi | 1,88 lần | đúng |
| Ablation gây thiệt hại lớn nhất | bỏ cửa sổ suy giảm, +1,92 PPL | đúng |
| corpus so với logspace, tiếng Việt | 0,285 PPL, KTC chồng lấn | đúng |
| logspace so với uniform, tiếng Việt | 0,902 PPL, KTC tách rời | đúng |

### 4.2. Phải sửa

| Vị trí | Hiện có | Sửa thành |
|---|---|---|
| Mục 2, dòng "Bộ kiểm thử tự động" | 61 phép kiểm | **62** phép kiểm |
| Phụ lục, "Số phép kiểm tự động" | 61 | **62** |
| Phụ lục, "Thông tin tương hỗ VI so với EN" | 1,66 tới 2,27 lần | **1,03 tới 2,27 lần** trong vùng tin cậy (d ≤ 167); mức 1,7–2,3 chỉ đúng ở d ≤ 13 |
| Mục 3.2, ghi chú phép kiểm ngược | "bản đúng rò rỉ 1e-16, bản cố tình sai 9e-1" | "bản đúng rò rỉ dưới 1e-15, bản cố tình sai vượt ngưỡng 1e-6". Hai con số cũ lấy từ test **không đặt seed**; tôi chạy lại ra 3,78e-16 và 1,98 |
| Mục 3.1, hạng mục biên dịch | "Bắt buộc dùng LuaLaTeX" | Ghi rõ hạng mục này **chưa hoàn thành**: PDF hiện tại biên dịch bằng pdfTeX |

### 4.3. Con số "44 lần chạy" — đã xác minh (Quang, 2026-08-13)

Kiểm kê đầy đủ từ `results/` (kể cả `NGUON_GOC_E6.txt`):

| Nhóm | Đếm được | Ghi chú |
|---|---|---|
| E1 | 12 | 2 ngôn ngữ × 2 mô hình × 3 seed |
| E2 | 6 | BPE × 2 mô hình × 3 seed |
| E3 | 10 | 5 nhánh ablation × 2 seed |
| E4 | 12 | {corpus, logspace} × 2 ngôn ngữ × 3 seed |
| **Huấn luyện LM chính thức** | **40** | |
| E0a | 2 | chạy hiệu chỉnh (3M token) |
| E0b | 4 | lần đo MI: {vi, en} × {âm tiết, BPE500} |
| E5 | 1 phiên (2 CSV) | benchmark fwd + bwd, 22 ô đo |
| E6 | 11 giữ lại | 8 (lần 2) + 3 (lần 3); phiên đầu thêm 8 lần nhưng bị loại vì bão hoà |

Con số 44 ra được theo **ba** quy ước khác nhau: (a) 40 + 4 E0b; (b) 40 + 2 E0a + 2 lượt
benchmark E5; (c) 40 + 2 E0a + 2 phiên E6 giữ lại. Vì không quy ước nào trội hơn, con số
44 trơ trọi là không bảo vệ được — người đếm kiểu khác sẽ ra 40, 42, 58, thậm chí 66.

**Sửa dòng trong phụ lục `Task_Nhóm 8.pdf` thành bản phân rã tường minh:**

> Tổng số lần chạy: **40 lần huấn luyện LM chính thức** (E1: 12, E2: 6, E3: 10, E4: 12),
> cộng 2 lần hiệu chỉnh E0-a, 4 lần đo thông tin tương hỗ E0-b, 1 phiên benchmark E5
> (22 ô đo) và 11 lần chạy recall E6 (8 + 3; phiên đầu 8 lần bị loại vì bão hoà, ghi
> trong `NGUON_GOC_E6.txt`).

Nếu bắt buộc giữ một con số duy nhất: dùng **42** ("lần huấn luyện mô hình ngôn ngữ, kể cả
2 lần hiệu chỉnh") — quy ước ít tranh cãi nhất.

---

## 5. "Project cần những điểm sau để cải thiện"

Phần này điền vào trang 4 đang để trống của `Task_Nhóm 8.pdf`. Đây **không phải lỗi**, mà là
những câu hội đồng nhiều khả năng sẽ hỏi.

1. **Chưa có so sánh cùng thời gian thực.** Hyena chậm hơn 1,5 lần trên mỗi token, nên ở cùng
   thời gian thực Transformer thấy nhiều hơn khoảng 1,5 lần số token. Một lần chạy Transformer
   ở 75 triệu token (khoảng 9 phút) sẽ chặn đứng câu hỏi khó nhất về kết quả 19,5%.

2. **Nhan đề dẫn bằng phần không kết luận được.** Tên đề tài nhấn "khởi tạo bộ lọc từ cấu trúc
   thông tin của corpus", nhưng đó chính là phần cho kết quả âm (+0,285 và +0,167 PPL, khoảng
   tin cậy chồng lấn). Cân nhắc để nhan đề dẫn bằng thứ đã đứng vững.

3. **Kết quả mạnh nhất đang bị chôn ở mục 5.7.** `logspace` vượt `uniform` khoảng 1 PPL với
   khoảng tin cậy tách rời trên **cả hai** ngôn ngữ. Đây là phát hiện sạch, tái lập được, về
   đúng một siêu tham số mà bài báo gốc **để ngỏ**. Nên đưa lên đầu.

4. **Khoảng tin cậy chỉ bắt nhiễu khởi tạo.** Ba seed trên một tập test duy nhất, không
   bootstrap trên tập test. Chênh lệch 0,285 PPL nằm dưới cả sai số dữ liệu chưa được ước lượng.

5. **"Vùng tin cậy" là tên gọi quá mạnh.** Nó được định nghĩa là tín hiệu **bằng** độ chệch
   (`signal_to_bias > 1`), và đường nền nhiễu chỉ dựng từ **một** lần xáo trộn
   (`n_shuffle = 1`, mặc định trong `hyena_study/morphology.py`), nên không có thanh sai số.

---

## 6. Thứ tự thực hiện

| Đợt | Task | Ai | Điều kiện chuyển đợt |
|---|---|---|---|
| 1 | T1, T2, T3, T4 | Kiên | Bốn lỗi nặng đã hết |
| 1 | T5 | Tiến | Đoạn Hạn chế đã viết |
| 2 | T6, T7 | Kiên | |
| 2 | T8, T10 | Quang, Tiến | |
| 3 | T9, T11, T12 | Quang, Tiến, Kiên | Biên dịch lại bằng LuaLaTeX |
| 4 | Soát chéo + cập nhật slide + `03_qa_phan_bien.md` | Cả nhóm | |
| Song song | T13 | Quang | Chỉ nếu còn ngân sách Kaggle |

**Mốc chặn:** đợt 1 phải xong trước khi ai đó bắt đầu sửa slide, vì bốn lỗi nặng đều nằm ở
những con số đang được trích sang slide.

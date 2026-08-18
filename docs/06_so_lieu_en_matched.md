# Số liệu nhánh EN sau khi khớp token (T13) — bàn giao cho Kiên & Tiến

**Nguồn:** T13 hoàn tất 2026-08-18. `E1_en` và `E4_*_en` chạy lại trên corpus cắt đúng
38.250.964 token (bằng nhánh VI), budget giữ 50M → cùng 6.103 bước, cùng ~1,31 epoch.
Chạy trên Colab T4, `datasets 4.0.0`, `torch 2.11.0+cu128` (đều ghi trong JSON).
Bằng chứng gốc: `notebooks/E1_en_matched_results.zip`, `notebooks/E4_en_matched_results.zip`
(commit vào repo có chủ đích, ngoại lệ của `*.zip` trong gitignore) + hai notebook đã chạy
kèm output ngay cạnh.

**Vì sao corpus đổi:** mẫu Wikipedia streaming phụ thuộc phiên bản `datasets` — 90k bài
trên Colab chỉ ra 35,3M token (bản Kaggle gốc ra 42,1M). Phải lấy 110.000 bài rồi cắt
xuống đúng đích. Từ giờ `train.py` ghi `datasets_version` vào mọi JSON.

## 1. Số mới (3 seed, KTC 95% phân phối t, t = 4,303)

| Nhánh | PPL | ± sd | KTC 95% |
|---|---|---|---|
| E1 EN Hyena (uniform) | **55,82** | 0,35 | [54,96; 56,68] |
| E1 EN Transformer | 65,43 | 0,18 | [64,99; 65,87] |
| E4 EN logspace | 54,97 | 0,16 | [54,56; 55,37] |
| E4 EN corpus | **54,91** | 0,15 | [54,53; 55,29] |

Tỉ lệ Transformer/Hyena EN = 1,172 → Hyena thấp hơn **14,7%** (cũ: 15,5%).
VI giữ nguyên: 19,5%, tỉ lệ 1,243. Khoảng cách VI–EN **giãn ra** sau khi khớp token —
đúng chiều giả thuyết.

Corpus mới: 110.000 tài liệu, 245.466.748 ký tự, val/test 2.406.613 mỗi tập,
chars/token 5,100 (cũ 5,115), unk 10,46% (cũ 10,55%).

## 2. Thay số trong `report/acl_latex.tex` (old → new, kèm số dòng hiện tại)

| Dòng | Hiện có | Sửa thành |
|---|---|---|
| 107, 514, 855 | "15,5\% trên tiếng Anh" | "**14,7\%** trên tiếng Anh" |
| 505 | `\textbf{61,51} $\pm$ 0,41 & [60,49; 62,54]` | `\textbf{55,82} $\pm$ 0,35 & [54,96; 56,68]` |
| 506 | `72,82 $\pm$ 0,52 & [71,52; 74,12]` | `65,43 $\pm$ 0,18 & [64,99; 65,87]` |
| 621 | `uniform 61,51 & [60,49; 62,54]` | `55,82 & [54,96; 56,68]` |
| 622 | `logspace 60,34 & [60,20; 60,49]` | `54,97 & [54,56; 55,37]` |
| 623 | `corpus \textbf{60,18} & [59,89; 60,47]` | `\textbf{54,91} & [54,53; 55,29]` |
| 333–334 | câu "cắt xuống đúng cùng số token" | GIỮ (giờ đã đúng), thêm chi tiết: "nhánh tiếng Anh lấy dư 110.000 tài liệu rồi cắt xuống đúng 38.250.964 token của nhánh tiếng Việt; cả hai nhánh huấn luyện cùng 49.995.776 token (~1,31 epoch)" |
| 650–651 (T6) | `150,0 / 92,9 / 1,61` (số E0a) | dùng số E1 **VI** (cùng Kaggle T4): Transformer 138,5–138,6k, Hyena 91,7–92,5k, tỉ lệ **1,50**. KHÔNG trộn số EN mới vào đây (EN matched đo trên Colab T4, host khác): nếu muốn nêu, ghi riêng "trên Colab T4, nhánh EN cho 122,4k so với 85,7k, tỉ lệ 1,43 — cùng chiều" |

## 3. ⚠️ Mục 5.7 phải viết lại (kết luận đổi)

Đoạn "Phần thứ nhất là khẳng định được..." (dòng 632–637): với EN matched,
**logspace và corpus KHÔNG còn tách rời khỏi uniform trên tiếng Anh** —
uniform [54,96; 56,68] chồng lấn cả hai (nguyên nhân: 3 seed uniform mới phân tán
55,50–56,18 nên KTC phình). Chiều vẫn đúng: logspace −0,85 PPL, corpus −0,91 PPL.

- Cũ: "vượt uniform với KTC tách rời, trên cả hai ngôn ngữ. Khoảng cách 0,90 PPL (VI)
  và 1,17 PPL (EN)."
- Mới (đề xuất): "vượt \textsf{uniform} với khoảng tin cậy tách rời trên tiếng Việt
  (0,90 PPL); trên tiếng Anh cùng chiều (0,85–0,91 PPL) nhưng khoảng tin cậy chồng lấn
  do phương sai giữa các seed của \textsf{uniform} lớn."
- Đoạn "corpus chỉ hơn logspace 0,285 PPL (VI) và 0,167 PPL (EN)": EN mới là **0,055 PPL**,
  KTC vẫn chồng lấn → kết luận "không khẳng định được" giữ nguyên.
- Dòng 863–864 (kết luận, "chênh lệch nhỏ còn lại xuất hiện đồng đều trên cả hai ngôn
  ngữ"): sửa "đồng đều" → "cùng chiều" (0,285 vs 0,055 không còn gọi là đồng đều).
- Lan toả: mục 5 của `docs/05` từng gọi "logspace vượt uniform tách rời trên CẢ HAI ngôn
  ngữ" là kết quả mạnh nhất — giờ chỉ còn đứng vững trên tiếng Việt. Slide và Q&A nói theo.

## 4. Việc của Tiến (số phụ trợ đổi nhẹ)

- **T11** (dòng 289): EN đo BPE 4,155 vs huấn luyện âm tiết giờ là **5,100** (cũ 5,115)
  → tính lại phần trăm lệch thang theo quy ước của Tiến.
- **T12.1** (dòng 532–533, Kiên sửa nhưng số của Tiến): nếu chuyển sang corpus huấn luyện
  thì VI 4,118 vs EN **5,100**, tỉ lệ **1,239** (cũ ghi 4,101/5,121/1,249 của corpus đo MI).
- 14,3%/75,8% (T10) dựa trên file alpha — **không đổi**.

## 5. Không đổi

Bảng 3 (alpha từ file đo riêng), E2/E3 (chỉ VI), E5 (synthetic), E6, Bảng 2/Hình 1-2
(MI đo trên corpus 30k bài riêng). Số phép kiểm: **67** (T7).

## 6. Thêm vào Hạn chế (một đoạn ngắn)

Mẫu tài liệu lấy qua streaming shuffle của `wikimedia/wikipedia` không tái lập được giữa
các phiên bản `datasets` (cùng seed, khác phiên bản → khác tài liệu). Nhánh EN vì vậy được
lấy lại mẫu (110k bài, `datasets 4.0.0`, ghi trong JSON) và cắt xuống đúng ngân sách token
của nhánh VI; các nhánh VI giữ nguyên mẫu gốc. Phép so sánh chính (cùng số token, cùng số
bước) nhờ đó chặt hơn trước, đổi lại hai ngôn ngữ không còn chung một lần lấy mẫu.

---
*Trạng thái: đã đối chiếu từng chữ số với JSON trong `E4_en_matched_results.zip` và chép
12 file `E4_*_en_*` vào `results/` (2026-08-18). Mọi số trong file này là số chốt.*

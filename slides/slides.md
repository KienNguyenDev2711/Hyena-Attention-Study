---
marp: true
title: "Hyena cho tiếng Việt (báo cáo cuối kỳ Nhóm 08)"
author: "Trần Tú Quang · Tô Huỳnh Minh Tiến · Nguyễn Cao Trung Kiên (Nhóm 08)"
paginate: true
html: true
math: katex
backgroundColor: "#ffffff"
color: "#1d2b36"
style: |
  @import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');
  :root {
    --navy:#1F3A68; --navy-deep:#16294d; --ink:#1d2b36;
    --accent:#2a6df4; --soft:#eef3fb; --line:#d4deee; --muted:#6b7a90;
    --card-bg:#f7faff; --card-border:#cdd9ec; --card-radius:8px;
  }
  section {
    font-family: "Be Vietnam Pro", "Segoe UI", system-ui, sans-serif;
    font-size: 24px;
    padding: 96px 70px 60px 70px;
    background:
      radial-gradient(1200px 380px at 88% -8%, #eef3fb 0%, rgba(238,243,251,0) 60%),
      #ffffff;
    color: var(--ink);
    display: flex; flex-direction: column;
    justify-content: flex-start !important; align-content: flex-start;
    letter-spacing:.1px;
  }
  h2 {
    position: absolute; top: 0; left: 0; right: 0; margin: 0;
    background: linear-gradient(100deg, var(--navy-deep) 0%, var(--navy) 58%, #28508f 100%);
    color: #ffffff !important; font-size: 29px; font-weight: 600;
    padding: 18px 70px 16px 70px;
    box-shadow: 0 3px 14px rgba(15,29,56,.18);
  }
  h3 {
    color: var(--navy); font-size: 23px; font-weight: 700; margin: 2px 0 14px 0;
    padding-bottom: 7px; border-bottom: 2px solid var(--line); display: inline-block;
  }
  p { margin: 9px 0; }
  strong { color: var(--navy); font-weight: 700; }
  em { color: var(--accent); font-style: normal; font-weight: 600; }
  a { color: var(--navy); }
  code {
    background: var(--soft); color: var(--navy); padding: 1px 7px;
    border-radius: 5px; font-size: .94em;
  }
  ul { list-style: none; padding-left: 4px; margin: 8px 0; }
  ul li { position: relative; padding-left: 26px; margin: 12px 0; line-height: 1.45; }
  ul li::before {
    content: ""; position: absolute; left: 3px; top: .55em;
    width: 8px; height: 8px; border-radius: 3px;
    background: linear-gradient(135deg, var(--navy), var(--accent));
  }
  ol { padding-left: 22px; } ol li { margin: 11px 0; line-height: 1.45; }
  table { font-size: 21px; border-collapse: separate; border-spacing: 0; margin: 10px 0; width: 100%;
    border-radius: var(--card-radius); overflow: visible; }
  thead th { background: var(--navy); color: #fff; font-weight: 600; }
  thead th:first-child { border-top-left-radius: var(--card-radius); }
  thead th:last-child { border-top-right-radius: var(--card-radius); }
  tbody tr:last-child td:first-child { border-bottom-left-radius: var(--card-radius); }
  tbody tr:last-child td:last-child { border-bottom-right-radius: var(--card-radius); }
  tbody tr:nth-child(even) td { background: #f6f9fe; }
  td, th { border: 1px solid var(--line); padding: 8px 14px; }
  blockquote {
    border: 1px solid var(--card-border); border-left: 5px solid var(--accent);
    background: var(--card-bg); color: #20324f; padding: 12px 22px;
    border-radius: var(--card-radius); margin: 12px 0;
  }
  /* ── Display math as an elegant card ── */
  .katex-display {
    background: var(--card-bg);
    border: 1px solid var(--card-border); border-left: 5px solid var(--navy);
    border-radius: var(--card-radius); padding: 16px 22px; margin: 14px 0;
  }
  .katex { font-size: 1.18em; }
  /* ── Code as a clean light card ── */
  pre {
    background: var(--card-bg); border: 1px solid var(--card-border);
    border-radius: var(--card-radius); padding: 14px 18px;
  }
  pre code { background: none; color: #20324f; font-size: 19px; line-height: 1.6; }
  footer {
    left:0; bottom:0; width:100%; box-sizing:border-box; display:flex; padding:0;
    height:26px; font-size:13px; color:#ffffff;
    background: linear-gradient(90deg,#0e1d38 0%,#16294d 30%,#1f3a68 62%,#2a4d86 100%);
  }
  footer span { flex:1; display:flex; align-items:center; justify-content:center;
    border-right:1px solid rgba(255,255,255,.28); }
  footer span:last-child { border-right:none; }
  section::after {
    position:absolute; right:20px; bottom:5px; z-index:10; color:#ffffff;
    font-weight:600; font-size:13px;
    content: attr(data-marpit-pagination) " / " attr(data-marpit-pagination-total);
  }
  /* ── UIT logo (top-right, above banner) ── */
  header {
    position:absolute; top:9px; right:16px; left:auto; margin:0; padding:0;
    background:none; box-shadow:none; z-index:40;
  }
  header img {
    height:50px; width:50px; object-fit:contain; display:block; background:#ffffff;
    border-radius:50%; padding:5px; box-sizing:border-box; box-shadow:0 1px 5px rgba(0,0,0,.22);
  }
  /* ── Lead / title ── */
  section.lead { text-align:center; justify-content:center; }
  section.lead::before {
    content:""; position:absolute; top:0; left:0; right:0; height:8px;
    background: linear-gradient(90deg, var(--navy) 0%, var(--accent) 100%);
  }
  .titlebox {
    width:100%; box-sizing:border-box;
    background: linear-gradient(120deg, #16294d 0%, #1F3A68 60%, #2a558f 100%);
    border-radius:16px; padding:30px 44px; margin:10px 0 26px 0;
    box-shadow:0 10px 30px rgba(15,29,56,.22); text-align:center;
  }
  .titlebox h1 { background:none; border:none; color:#fff !important;
    font-size:42px; margin:0; padding:0; letter-spacing:.3px; }
  .titlebox h3 { color:#cfe0ff !important; font-weight:400; border:none;
    margin:10px 0 0 0; display:block; }
  section.lead h1 { color:var(--navy); font-size:42px; }
  section.lead h3 { color:var(--ink); font-weight:400; border:none; display:block; }
  /* ── Components ── */
  .small { font-size:18px; color:var(--muted); }
  .box {
    background:var(--card-bg); border:1px solid var(--card-border); border-left:5px solid var(--accent);
    border-radius:var(--card-radius); padding:13px 22px;
  }
  .warn {
    background:#fff8ec; border:1px solid #f3dca6; border-left:5px solid #e0a51e;
    border-radius:var(--card-radius); padding:13px 22px;
  }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start; }
  .center { text-align:center; }
  .yes { color:#15803d; font-weight:700; }
  .no  { color:#b04a4a; font-weight:700; }
  /* vertical flow of steps */
  .flow { display:flex; flex-direction:column; align-items:center; gap:5px; margin:14px 0; }
  .flow .step {
    background:#fff; border:1.5px solid var(--card-border); border-radius:var(--card-radius);
    padding:9px 22px; font-weight:600; color:var(--navy); font-size:21px;
  }
  .flow .step.fill { background:var(--navy); color:#fff; border-color:var(--navy); }
  .flow .ar { color:var(--accent); font-size:17px; line-height:1; }
  /* horizontal variant — dùng khi slide có nhiều khối dọc (tránh tràn) */
  .flow.row { flex-direction:row; flex-wrap:wrap; justify-content:center; gap:8px; margin:8px 0 4px; }
  .flow.row .step { padding:7px 15px; font-size:19px; }
  .flow.row .ar { font-size:14px; }
  /* horizontal pill timeline */
  .chips { display:flex; flex-wrap:wrap; align-items:center; gap:7px; justify-content:center; margin:8px 0 4px; }
  .chip { background:var(--navy); color:#fff; border-radius:999px; padding:6px 15px; font-size:18px; font-weight:600; }
  .chip.alt { background:#eef3fb; color:var(--navy); border:1px solid #cdd9ec; }
  .chip.hot { background:linear-gradient(135deg,var(--navy),var(--accent)); }
  .sep { color:#9bb0cf; font-weight:700; }
  /* mono illustration (matrix) */
  .mono {
    background:#0f1f3d; color:#dbe7ff; border-radius:12px; padding:16px 22px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size:19px; line-height:1.55;
    display:inline-block;
  }
  .mono .dim { color:#7f93bd; }
  /* monospace pipeline / step card (Tiến) */
  .pipeline {
    background:var(--card-bg); border:1px solid var(--card-border); border-left:5px solid var(--accent);
    border-radius:var(--card-radius);
    padding:12px 16px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size:20px; line-height:1.65; color:#20324f;
  }
  .pill {
    display:inline-block; border:1px solid #cdd9ec; background:#fff;
    color:var(--navy); border-radius:999px; padding:3px 12px;
    margin:3px 4px 3px 0; font-size:18px;
  }
  .diagram { text-align:center; margin-top:12px; }
  .diagram img { max-height:280px; width:auto; }
  /* khoảng cách dọc giữa các block component (tránh dính nhau) */
  .box, .warn, .pipeline, .grid2, pre, table { margin-top:16px; margin-bottom:16px; }
  .mono { margin:8px 0; }
  .tight table { font-size:17.5px; }
  .tight li { font-size:22px; }
  section.compact h3 { margin-bottom:8px; }
  section.compact .small { font-size:16px; }
  section.compact .katex-display { padding:10px 18px; margin:8px 0 10px; }
  section.compact table { font-size:16px; margin-top:8px; margin-bottom:10px; }
  section.compact td, section.compact th { padding:5px 10px; }
  section.compact .pipeline { font-size:18px; line-height:1.45; padding:9px 14px; margin-top:8px; margin-bottom:10px; }
  section.compact .box { padding:10px 18px; margin-top:12px; margin-bottom:12px; }
  /* ── Section divider (navy + ghost number) ── */
  section.divider {
    background-color:#16294d !important;
    background-image: linear-gradient(135deg,#0e1d38 0%,#16294d 45%,#1F3A68 100%) !important;
    color:#eaf1fc; justify-content:center !important; align-content:center;
    padding:96px 80px; overflow:hidden;
  }
  section.divider .dnum {
    position:absolute; top:14px; right:50px;
    font-size:260px; font-weight:800; line-height:1;
    color:rgba(255,255,255,.06); letter-spacing:-6px; z-index:0; pointer-events:none;
  }
  section.divider .dbar {
    width:64px; height:6px; border-radius:3px; position:relative; z-index:1;
    background:linear-gradient(90deg,var(--accent),#86b4ff); margin:0 0 20px 0;
  }
  section.divider h1 {
    color:#ffffff !important; background:none; border:none; box-shadow:none;
    font-size:48px; line-height:1.12; margin:0 0 16px 0; padding:0;
    position:relative; z-index:1; max-width:80%;
  }
  section.divider .dsub {
    color:#cfe0ff; font-size:23px; line-height:1.5; max-width:80%;
    position:relative; z-index:1;
  }
  section.divider .dmeta {
    color:#9db4d8; font-size:18px; margin-top:30px; position:relative; z-index:1;
  }
footer: '<span>Nhóm 08</span><span>Hyena cho tiếng Việt</span><span>2026</span>'
header: '<img src="assets/UIT_logo.svg" alt="UIT">'
---

<!-- _class: lead -->
<!-- _paginate: false -->

<div class="titlebox">

# Hyena cho tiếng Việt
### Mô hình ngôn ngữ tích chập dài dưới bậc hai, phân tích ablation và khởi tạo bộ lọc từ cấu trúc thông tin của corpus

</div>

<span class="small">Bài báo nền: Poli et al., *Hyena Hierarchy*, ICML 2023 (Oral) · Báo cáo cuối kỳ, môn Chuyên đề nghiên cứu và Xử lý ngôn ngữ tự nhiên</span>

<br>

**Nhóm 08**
Nguyễn Cao Trung Kiên · Tô Huỳnh Minh Tiến · Trần Tú Quang

<span class="small">University of Information Technology, VNU-HCM (UIT) · 2026</span>

<!--
Notes:
XÁC NHẬN tên GVHD trước khi trình chiếu (deck seminar ghi TS. Nguyễn Văn Kiệt).
Mở đầu: hội đồng đã nghe seminar về bài báo gốc; buổi này là THÍ NGHIỆM CỦA NHÓM.
-->

---

## Nội dung trình bày

1. **Bài toán & ba giả thuyết**: vì sao thử Hyena trên tiếng Việt
2. **Phương pháp**: đo cấu trúc thông tin I(d), ánh xạ sang bộ lọc, thiết lập công bằng
3. **Kết quả**: E1 chất lượng · E2 token hoá · E3 ablation · E4 khởi tạo · E5 hiệu năng · E6 recall
4. **Hạn chế & kết luận**

<div class="box">

Toàn bộ số liệu truy ngược được về `results/` trong repo; **67 phép kiểm tự động**; hai nhánh ngôn ngữ huấn luyện **cùng 38.250.964 token, cùng 6.103 bước**.

</div>

<!--
Notes:
30 giây. Nhấn câu trong box: đây là điểm phân biệt của nhóm: mọi con số tái lập được.
-->

---

<!-- _class: divider -->
<!-- footer: '<span>Nhóm 08</span><span>1 · Bài toán & giả thuyết</span><span>2026</span>' -->

<div class="dnum">1</div>

<div class="dbar"></div>

# Bài toán & ba giả thuyết

<div class="dsub">Chữ Quốc ngữ viết rời theo âm tiết · chuỗi dài hơn · O(L²) cắn mạnh hơn</div>

<div class="dmeta">Phần 1</div>

---

<!-- _class: compact -->

## Vì sao thử Hyena trên tiếng Việt?

- Bài báo gốc chỉ thực nghiệm trên **tiếng Anh**.
- Chữ Quốc ngữ viết **rời theo âm tiết**: cùng một nội dung, chuỗi tiếng Việt **dài hơn**.
- Chuỗi dài hơn thì chi phí $O(L^2)$ của attention nặng hơn, nên lợi thế dưới bậc hai của Hyena đáng giá hơn với tiếng Việt.

<div class="grid2">
<div class="box">

**H1.** Cùng ngân sách token, Hyena có giữ được chất lượng so với Transformer trên tiếng Việt?

</div>
<div class="box">

**H2.** Lợi thế đó có **lớn hơn** khi token hoá theo âm tiết so với BPE?

</div>
</div>

<div class="box">

**H3.** Khoảng suy giảm $\alpha$ của bộ lọc, siêu tham số mà bài báo gốc **để ngỏ**: có nên **đo từ corpus** thay vì chọn tay?

</div>

<!--
Notes:
H3 là đóng góp riêng của nhóm. Nói trước: kết quả H3 là kết quả ÂM có kiểm soát; nhóm trình bày trung thực.
-->

---

## Nhắc nhanh cơ chế Hyena

$$z^{n+1}_t = x^n_t \cdot (h^n * z^n)_t, \quad n = 1 \dots N$$

- Thay attention bằng **tích chập dài** (kernel dài bằng chuỗi, tính qua FFT, $O(L \log L)$) xen kẽ **cổng nhân theo phần tử**.
- Bộ lọc $h$ **tham số hoá ngầm**: FFN nhỏ + kích hoạt sine + **cửa sổ suy giảm mũ** $\exp(-\alpha t)$; mỗi kênh một $\alpha$, tức một *độ dài hiệu dụng*.
- Phân bố $\alpha$ quyết định bộ nhớ của mô hình nằm ở khoảng cách nào → chính là chỗ H3 can thiệp.

<div class="chips">
<span class="chip alt">bậc N = 1 ≈ GSS</span><span class="sep">·</span>
<span class="chip">bậc N = 2 ≈ H3 của Dao et al. (cấu hình chính)</span><span class="sep">·</span>
<span class="chip alt">bậc N = 3</span>
</div>

<!--
Notes:
Đi nhanh vì hội đồng đã nghe seminar. Chỉ cần đọng lại: alpha = độ dài hiệu dụng của từng kênh bộ nhớ.
-->

---

<!-- _class: divider -->
<!-- footer: '<span>Nhóm 08</span><span>2 · Phương pháp</span><span>2026</span>' -->

<div class="dnum">2</div>

<div class="dbar"></div>

# Phương pháp

<div class="dsub">Đo I(d) trên corpus · ánh xạ sang α · thiết lập so sánh công bằng</div>

<div class="dmeta">Phần 2</div>

---

<!-- _class: compact -->

## Đo cấu trúc thông tin: I(d) → phân bố α

<div class="pipeline">
corpus → I(d): thông tin tương hỗ giữa token cách nhau d
       → chuẩn hoá thành khối lượng w(d) → độ dài hiệu dụng ℓ → α = L / ℓ
</div>

- Ước lượng plug-in có **trừ độ chệch** bằng xáo trộn; công cụ được **hiệu chuẩn** trên chuỗi tổng hợp biết trước đáp án.
- Tiếng Việt giữ tín hiệu trên nền nhiễu tới $d = 167$; gấp **1,7–2,3 lần** tiếng Anh ở khoảng cách ngắn ($d \le 13$), giảm dần về $\approx 1$ ở $d = 167$.
- **80–90%** khối lượng thông tin (tuỳ ngôn ngữ và bộ token hoá) nằm trong 34 token đầu, trong khi khởi tạo `uniform` mặc định đặt *toàn bộ* 256 kênh xa hơn 64 token.

<div class="warn">

**Quy ước phải khai:** chuẩn hoá trên **lưới log** không nhân bề rộng ô. Có trọng số bề rộng, trung vị độ dài hiệu dụng đổi từ **2,95 → 90,5** token (VI-BPE). Kết quả PPL không đổi, nhưng cách diễn giải "thông tin nằm trong ~3 token" phụ thuộc quy ước này.

</div>

<!--
Notes:
Slide phòng thủ cho câu hỏi chắc chắn bị hỏi (T5). Chủ động khai quy ước trước khi bị hỏi.
-->

---

<!-- _class: compact -->

## Thiết lập so sánh công bằng

| Ràng buộc | Giá trị |
|---|---|
| Kiến trúc chung | 4 lớp · $d_{\text{model}}=256$ · $L=512$ · từ điển 16k · chỉ khác toán tử trộn token |
| Tham số | Hyena 7.553.280 · Transformer 7.383.552 · lệch 2,30% |
| Dữ liệu | Wikipedia VI/EN, nhánh EN **cắt xuống đúng 38.250.964 token của VI** |
| Ngân sách | cùng 49.995.776 token huấn luyện = 6.103 bước ≈ 1,31 epoch mỗi nhánh |
| Seed | 3 seed cho E1/E2/E4 · 2 seed cho E3 · KTC 95% phân phối $t$ |

- **67 phép kiểm tự động**, gồm kiểm chứng nhân quả: rò rỉ tương lai $< 10^{-15}$ trên mọi biến thể; bản cố tình sai vọt lên 1,98, tức bộ test bắt được lỗi thật.
- Kết luận khoa học so với đối chứng công bằng `logspace`, không so với hình nộm `uniform`.

<!--
Notes:
"Cắt xuống đúng cùng số token" giờ đúng theo nghĩa đen: nhánh EN lấy dư 110k tài liệu rồi cắt (T13). Nếu bị hỏi vì sao 110k: mẫu streaming phụ thuộc phiên bản datasets, đã ghi version vào artifact.
-->

---

<!-- _class: divider -->
<!-- footer: '<span>Nhóm 08</span><span>3 · Kết quả</span><span>2026</span>' -->

<div class="dnum">3</div>

<div class="dbar"></div>

# Kết quả

<div class="dsub">E1 chất lượng · E2 token hoá · E3 ablation · E4 khởi tạo bộ lọc · E5 hiệu năng · E6 recall</div>

<div class="dmeta">Phần 3</div>

---

## E1 · Hyena vượt Transformer trên cả hai ngôn ngữ

| Ngôn ngữ | Hyena (PPL) | Transformer (PPL) | Hyena thấp hơn |
|---|---|---|---|
| Tiếng Việt | **51,38** [51,05; 51,71] | 63,86 [63,14; 64,58] | **19,5%** |
| Tiếng Anh (cùng số token) | **55,82** [54,96; 56,68] | 65,43 [64,99; 65,87] | **14,7%** |

- Khoảng tin cậy hai mô hình **tách rời rõ** trên cả hai ngôn ngữ → **H1 đứng vững**.
- Lợi thế trên tiếng Việt (19,5%) **lớn hơn** tiếng Anh (14,7%), đúng chiều dự đoán từ cấu trúc thông tin.

<div class="box">

Hai nhánh nhìn thấy đúng **cùng một lượng tín hiệu**: 38.250.964 token duy nhất, 49.995.776 token huấn luyện, 6.103 bước, nên chênh lệch chỉ còn quy được cho ngôn ngữ.

</div>

<!--
Notes:
Số EN là bản chạy lại token-matched (T13). Nếu hỏi "sao EN đổi số so với bản nộp trước": corpus lấy mẫu lại + cắt đúng ngân sách, ghi rõ trong Hạn chế.
-->

---

## E2 · Token hoá âm tiết so với BPE

| Token hoá (tiếng Việt) | Tỉ lệ PPL Transformer/Hyena |
|---|---|
| Âm tiết (chuỗi dài hơn) | **1,243** |
| BPE (chuỗi ngắn hơn) | 1,202 |

- PPL **không so sánh được** giữa hai từ điển khác nhau, chỉ so được *tỉ lệ giữa hai mô hình trong cùng từ điển*.
- Lợi thế Hyena lớn hơn ở âm tiết, đúng chiều H2, nhưng chênh lệch nhỏ và chưa có KTC cho tỉ lệ: **dấu hiệu, chưa phải bằng chứng**.

<!--
Notes:
Chủ động nêu cạm bẫy diễn giải trước khi hội đồng nêu. Đừng overclaim H2.
-->

---

<!-- _class: compact -->

## E3 · Ablation bộ lọc

| Cấu hình | PPL | Δ | KTC 95% | Tách rời? |
|---|---|---|---|---|
| Gốc (bậc 2, đủ) | 51,38 | | [51,05; 51,71] | |
| Bỏ cửa sổ suy giảm | 53,30 | +1,92 | [52,24; 54,35] | **có** |
| Bậc N = 1 | 53,11 | +1,73 | [51,72; 54,49] | có* |
| Bỏ kích hoạt sine | 51,80 | +0,42 | [51,49; 52,10] | không |
| Bậc N = 3 | 51,63 | +0,25 | [49,51; 53,76] | không |
| Bỏ positional emb | 50,79 | −0,59 | [50,14; 51,45] | không |

- **Cửa sổ suy giảm là thành phần sống còn**, đúng vai trò của $\mathrm{Window}(t)$; đây cũng là nơi giả thuyết H3 tác động.
- *N = 1: tách rời nhưng biên chỉ hở **0,016 PPL**, và nhánh này mất 331.776 tham số (−4,4%), thiệt hại trộn hai nguyên nhân nên kết luận mong manh.
- Bỏ positional embedding **không gây hại**: bộ lọc tích chập đã mang sẵn thông tin vị trí.

<!--
Notes:
Trung thực về dòng N=1 (T8). 2 seed mỗi nhánh, KTC rộng, chỉ 2 dòng đầu được kết luận.
-->

---

<!-- _class: compact -->

## E4 · Khởi tạo bộ lọc từ corpus

| Khởi tạo | VI (PPL, KTC) | EN cùng số token (PPL, KTC) |
|---|---|---|
| `uniform` | 51,38 [51,05; 51,71] | 55,82 [54,96; 56,68] |
| `logspace` | 50,48 [50,33; 50,62] | 54,97 [54,56; 55,37] |
| corpus (H3) | **50,19** [49,93; 50,45] | **54,91** [54,53; 55,29] |

- **Tiếng Việt:** `logspace` và corpus đều **tách rời** khỏi `uniform` (−0,90 / −1,19 PPL) → *khoảng* $\alpha$ thực sự quan trọng, chọn tuỳ tiện gây thiệt hại đo được.
- **Tiếng Anh:** cùng chiều (−0,85 / −0,91 PPL) nhưng KTC **chồng lấn** do phương sai seed của `uniform` lớn.
- **corpus vs `logspace`:** chỉ hơn 0,285 (VI) / 0,055 (EN) PPL, chồng lấn cả hai → **không kết luận được H3**.

<div class="warn">

Kết quả **âm có kiểm soát**: đo $\alpha$ từ corpus không tốt hơn một lựa chọn cách đều theo log. Đây vẫn là câu trả lời có giá trị cho siêu tham số mà bài báo gốc để ngỏ.

</div>

<!--
Notes:
Slide nhạy cảm nhất: tên đề tài nhấn H3 mà H3 không kết luận được. Chủ động nói trước, đừng để hội đồng "bắt" ra.
-->

---

## E5 · Hiệu năng theo độ dài chuỗi

- Ở $L = 8192$, Hyena nhanh hơn kernel attention tiết kiệm bộ nhớ **1,88 lần** (lượt tiến và lùi); attention dày đặc hết bộ nhớ từ trước đó.
- Bộ nhớ Hyena tăng xấp xỉ **gấp đôi khi L gấp đôi** (gần tuyến tính); attention tăng gấp bốn (bậc hai).
- Đổi lại, ở $L = 512$ mỗi token của Hyena **chậm hơn 1,50 lần** Transformer (91,7–92,5k so với 138,5–138,6k token/giây, Kaggle T4); chi phí cố định của FFT chỉ được bù khi $L \ge 8$K.

<div class="box">

Kết luận hiệu năng: lợi thế dưới bậc hai là **lợi thế tiệm cận**: có thật, đo được, nhưng chỉ xuất hiện ở ngữ cảnh dài.

</div>

<!--
Notes:
Nếu hỏi "vậy sao so sánh chất lượng ở L=512": vì ngân sách GPU; so cùng thời gian thực Transformer sẽ thấy ~1,5× token, đã ghi ở Hạn chế, một lần chạy compute-matched (~9 phút) là việc tiếp theo.
-->

---

## E6 · Recall liên kết

- Tác vụ truy hồi key–value tổng hợp ($L = 65$, vocab 10): attention giải **hoàn hảo** (1,000 ở mọi seed).
- Hyena: phân bố **lưỡng cực**. Gộp 5 seed: 0,092 · 0,141 · 0,922 · 0,981 · 0,987. **3/5 lần thành công (≥ 0,92), 2/5 lần sập (≤ 0,14)**, không có giá trị trung gian.
- Gấp đôi ngân sách bước làm lần thành công tốt lên (0,92 → 0,98) nhưng **không cứu được kiểu hỏng** → thất bại tối ưu hoá lượng cực, không phải hội tụ chậm.

<div class="warn">

Với phân bố lưỡng cực, báo mean ± sd là **sai lệch** vì không lần chạy nào nằm gần trung bình. Nhóm báo **tỉ lệ thành công + độ chính xác khi thành công**.

</div>

<!--
Notes:
Đây là điểm yếu thật của Hyena mà chính bài báo gốc cũng nhận (recall/induction). Nói thẳng; hội đồng đánh giá cao sự trung thực thống kê.
-->

---

<!-- _class: divider -->
<!-- footer: '<span>Nhóm 08</span><span>4 · Hạn chế & kết luận</span><span>2026</span>' -->

<div class="dnum">4</div>

<div class="dbar"></div>

# Hạn chế & kết luận

<div class="dsub">Những gì nhóm biết là chưa chắc · và những gì đứng vững</div>

<div class="dmeta">Phần 4</div>

---

<!-- _class: compact -->

## Hạn chế

1. **Quy ước lưới I(d):** chuẩn hoá trên lưới log không nhân bề rộng ô; trung vị độ dài hiệu dụng nhạy với quy ước (2,95 ↔ 90,5 token).
2. **KTC chỉ bắt nhiễu khởi tạo:** 3 seed (2 với ablation) trên một tập test duy nhất, không bootstrap; nền nhiễu I(d) dựng từ **một** lần xáo trộn.
3. **Lệch đơn vị:** $\alpha$ đo trên corpus BPE nhưng huấn luyện bằng âm tiết (lệch thang ~6,5% VI, ~18,5% EN).
4. **Chưa so cùng thời gian thực:** Hyena chậm hơn 1,50×/token → cùng giờ GPU, Transformer thấy nhiều token hơn.
5. **Mẫu dữ liệu phụ thuộc phiên bản `datasets`:** nhánh EN phải lấy mẫu lại (110k tài liệu, cắt về đúng ngân sách VI); phiên bản thư viện nay được ghi vào mọi artifact.

<!--
Notes:
Mỗi mục ứng với một câu phản biện dự kiến. Khai trước = giữ thế chủ động.
-->

---

## Kết luận

<div class="box">

**Đứng vững:** Hyena đạt PPL thấp hơn Transformer **19,5%** (VI) và **14,7%** (EN) ở cùng ngân sách token; cửa sổ suy giảm là thành phần sống còn; *khoảng* $\alpha$ quan trọng (`logspace` vượt `uniform` tách rời trên tiếng Việt); tăng tốc 1,88× ở $L = 8192$.

</div>

<div class="box">

**Không kết luận:** đo $\alpha$ từ thống kê corpus (H3) không hơn `logspace`; ưu thế token hoá âm tiết (H2) mới là dấu hiệu; recall tầm xa vẫn là điểm yếu lưỡng cực của Hyena.

</div>

- Đóng góp: so sánh có kiểm soát Hyena–Transformer **đầu tiên trên tiếng Việt** ở quy mô này + trả lời có bằng chứng cho một siêu tham số bài báo gốc để ngỏ + hạ tầng tái lập đầy đủ (67 test, cache token, artifact truy ngược được).

<!--
Notes:
Kết bằng một câu: "lợi thế của Hyena với tiếng Việt là có thật và đo được, nhưng nằm ở khoảng suy giảm của bộ lọc, không nằm ở việc đo nó từ corpus."
-->

---

<!-- _class: lead -->

# Cảm ơn hội đồng!

**Nhóm 08** · Nguyễn Cao Trung Kiên · Tô Huỳnh Minh Tiến · Trần Tú Quang

<span class="small">Mã nguồn, 67 phép kiểm và toàn bộ artifact: github.com/KienNguyenDev2711/Hyena-Attention-Study</span>

<!--
Notes:
Chuẩn bị sẵn 5 câu trả lời phản biện trong docs/05 mục 5 + docs/03 Q&A.
-->

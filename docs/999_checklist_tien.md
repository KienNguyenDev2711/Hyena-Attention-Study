# Checklist Tô Huỳnh Minh Tiến

## Tick nhanh

- [x] T5 - limitation lưới `I(d)` trong report đã có bảng `tab:alpha_sensitivity`.
- [x] T10 - wording `14,3%`, `75,8%`, `0,9974` trong source report đã làm rõ denominator/metric/correlation.
- [x] T11 - mismatch BPE/âm tiết đã cập nhật theo EN matched: `5,100`, `18,5%`.
- [x] Slide source/HTML/PDF đã cập nhật `~18,5% EN`.
- [x] Checklist chính `docs/04_task_sua_bao_cao.md` đã tick T5/T10/T11.
- [x] Mapping với team và câu trả lời phản biện T5 đã ghi trong file này.
- [ ] `Report_Nhom8.pdf` chưa regenerate từ `report/acl_latex.tex`.

## Vị trí cần kiểm khi có PDF report mới

- [x] Trang 1 source report: abstract đã hạ giọng `14,3%` và bỏ cách dùng `0,9974` như bằng chứng mạnh.
- [x] Trang 5 source report: mục 5.3 đã nêu metric, mẫu số, 256 kênh, correlation và `75,8%`.
- [x] Mục Hạn chế source report: đã có T5 bảng sensitivity và T11 `5,100`/`18,5%`.
- [ ] PDF report mới: sau khi Kiên/nhóm compile, kiểm lại trang 1, trang 5, và mục Hạn chế có đúng các điểm trên không.

| Task | Nội dung cần làm | Status | Evidence | Note |
|---|---|---|---|---|
| T5.1 | Kiểm tra `alphas_from_mi` có đang chuẩn hoá `I(d)` trực tiếp trên lưới log mà không nhân bề rộng ô hay không | DONE | `hyena_study/morphology.py`, `hyena_study/analyze.py::alpha_mapping_sensitivity` | Code hiện tại dùng `w = w / w.sum()` trên các lag đã lấy mẫu log; `analyze.py` đã có docstring note `T5/Tien`. |
| T5.2 | Bổ sung sensitivity analysis cho lưới log: so sánh cách hiện tại với cách nhân bề rộng ô | DONE | `python3 -m hyena_study.analyze`; `tests/test_analyze.py::test_lag_width_sensitivity_reproduces_report_numbers` | Reproduce: VI 2,95/142/227 vs 90,54/39/87; EN 2,85/141/214 vs 160,45/27/57. |
| T5.3 | Viết limitation trong report, nêu rõ điều này đổi diễn giải chứ không làm sai PPL đã chạy | DONE | `report/acl_latex.tex`, Bảng `tab:alpha_sensitivity` | Bảng width-weighted là sensitivity analysis, không phải kết quả huấn luyện mới. |
| T10.1 | Tái sinh 14,3%, 22,3%, 75,8%, correlation 0,9974 và correlation corpus-logspace từ code | DONE | `hyena_study/analyze.py::alpha_comparison_metrics`; `tests/test_analyze.py::test_alpha_comparison_metrics_reproduce_report_numbers` | Reproduce: 14,35% denominator EN; 22,29% denominator VI; symmetric 17,27%; corr log VI-EN 0,9974; VI-logspace 75,81% và corr 0,9377; code/test đã có docstring note `T10/Tien`. |
| T10.2 | Sửa wording report: nêu denominator, metric, sample size, biến lấy correlation | DONE | `report/acl_latex.tex`, mục 5.3 và abstract | Correlation được hạ giọng thành mô tả hình dạng của 256 effective-length quantiles, không dùng như bằng chứng độc lập mạnh. |
| T11.1 | Kiểm tra scale mismatch: alpha đo bằng BPE token nhưng E4 train/evaluate bằng syllable token | DONE | `results/E4_corpus_*.json`, `results/alpha_*_bpe500.json`, `report/acl_latex.tex`, `docs/06_so_lieu_en_matched.md` | VI: 3,851 char/BPE-token vs 4,118 char/syllable-token; EN matched: 4,155 vs 5,100. |
| T11.2 | Bổ sung limitation trong report về scale mismatch | DONE | `report/acl_latex.tex`, mục 3.5 | Ghi rõ 6,5%/18,5% là lệch thang đơn vị, chưa phải sai số trực tiếp của PPL vì chưa có experiment cô lập. |
| T14.1 | Kiểm lại slide/Q&A nếu có nhắc `14,3%`, `75,8%`, `0,9974`, hoặc claim corpus/logspace "đồng đều" | DONE | `rg -n "14,3|14\\.3|75,8|75\\.8|0,9974|0\\.9974|đồng đều|dong deu" slides docs`; `slides/slides.md`; `slides/slides.html` | Không thấy các số/claim nhạy cảm trong slide/Q&A; chỉ thấy slide limitation còn `~19% EN`, đã cập nhật thành `~18,5% EN`. Repo hiện không có `docs/03_qa_phan_bien.md`, nên chưa kiểm được Q&A ngoài repo. |
| T14.2 | Chuẩn bị trả lời phản biện T5: vì sao median alpha khoảng 3 token nhưng MI còn tín hiệu tới `d=167` | DONE | `report/acl_latex.tex`, Bảng `tab:alpha_sensitivity`; `hyena_study/analyze.py::alpha_mapping_sensitivity` | Câu trả lời: median khoảng 3 token là kết quả của quy ước ánh xạ trong `alphas_from_mi`, nơi `I(d)` được chuẩn hoá trên lưới log như các điểm rời rạc, chưa nhân bề rộng ô. Vì lưới log dày ở lag ngắn, phân vị bị kéo về nhỏ; nếu nhân bề rộng ô, median sensitivity tăng lên 90,54 (VI) và 160,45 (EN). Do đó median alpha không phủ định việc MI còn trên nền tới `d=167`; nó chỉ là một quy ước rời rạc để sinh alpha, không phải kết luận duy nhất về toàn bộ phân bố thông tin theo khoảng cách liên tục. |

## Bản đồ mapping với team

| Mảng | Vị trí chính | Liên quan team | Điểm Tiến cần giữ khi sync |
|---|---|---|---|
| T5 - limitation lưới `I(d)` | `report/acl_latex.tex` dòng 849-878; Bảng `tab:alpha_sensitivity`; `hyena_study/analyze.py::alpha_mapping_sensitivity`; `tests/test_analyze.py::test_lag_width_sensitivity_reproduces_report_numbers` | Kiên dùng đoạn này khi chốt PDF và khi bị hỏi vì sao median corpus filter chỉ khoảng 3 token. Quang có thể dẫn code/test để chứng minh số tái sinh được. | Không nói "thông tin chỉ nằm trong 3 token" như kết luận tuyệt đối. Nói "3 token là median theo quy ước ánh xạ rời rạc hiện tại"; sensitivity có nhân bề rộng ô cho median 90,54 VI và 160,45 EN. |
| T10 - `14,3%`, `75,8%`, `0,9974` | `report/acl_latex.tex` dòng 481-491; `hyena_study/analyze.py::alpha_comparison_metrics`; `tests/test_analyze.py::test_alpha_comparison_metrics_reproduce_report_numbers` | Không bị T13 ảnh hưởng vì dựa trên file alpha, không dựa trên PPL EN matched. Kiên/slide chỉ cần tránh dùng correlation như bằng chứng mạnh. | Khi nói `14,3%`, phải nói rõ mẫu số là `ell_en`: `mean(|ell_vi - ell_en| / ell_en)`. Nếu đổi mẫu số sang VI là 22,3%, symmetric là 17,3%. Correlation log 0,9974 chỉ mô tả hình dạng 256 phân vị đơn điệu; VI-logspace cũng 0,9377. |
| T11 - mismatch BPE/âm tiết | `report/acl_latex.tex` dòng 286-293; `slides/slides.md` dòng 563-567; `docs/06_so_lieu_en_matched.md` mục 4 | Phụ thuộc T13 vì số EN huấn luyện đổi từ 5,115 sang 5,100 chars/token. Kiên cần dùng số mới trong report/PDF; cả nhóm cần dùng số mới trong slide. | Chỉ gọi là lệch thang đơn vị, không gọi là sai số PPL. Số hiện tại: VI 3,851 BPE vs 4,118 syllable, khoảng 6,5%; EN 4,155 BPE vs 5,100 syllable, khoảng 18,5%. |
| Slide/Q&A | `slides/slides.md` dòng 563-567; `slides/slides.html` dòng 3905-3908; `docs/03_qa_phan_bien.md` nếu nhóm bổ sung lại | Cả nhóm dùng khi cập nhật slide/Q&A. Quang/Kiên cần biết không còn claim "đồng đều"; kết luận mới là "cùng chiều nhưng không cùng độ lớn". | Đã cập nhật slide từ `~19% EN` sang `~18,5% EN`. Repo hiện chưa có `docs/03_qa_phan_bien.md`, nên nếu ai giữ file Q&A ngoài repo thì cần tìm các số cũ `14,3%`, `75,8%`, `0,9974`, `1,66-2,27`, và claim "đồng đều". |

## Câu trả lời phản biện Tiến nên nói

**Hỏi:** Vì sao bộ lọc corpus có median khoảng 3 token, trong khi Hình 1 cho thấy MI còn tín hiệu tới `d = 167`?

**Trả lời ngắn:** Hai con số đang nói hai việc khác nhau. Hình 1 đo đường cong MI còn nằm trên nền tới khoảng cách xa. Còn median khoảng 3 token là kết quả của bước ánh xạ `I(d) -> alpha`: code hiện chuẩn hoá `I(d)` trên lưới log như các điểm rời rạc, chưa nhân bề rộng ô. Vì lưới log lấy rất dày ở khoảng cách ngắn, phân vị bị kéo về nhỏ. Khi làm sensitivity có nhân bề rộng ô, median đổi mạnh lên 90,54 với VI và 160,45 với EN. Vì vậy median 3 token là quy ước sinh alpha hiện tại, không phải kết luận rằng corpus không còn thông tin sau 3 token.

**Câu cần tránh:** "Thông tin của corpus chỉ nằm trong 3 token." Câu đúng là: "Theo quy ước rời rạc hiện dùng để sinh alpha, nhiều kênh được cấp cho vùng lag ngắn; diễn giải này nhạy với cách gán khối lượng trên lưới lag."

## Việc còn cần phối hợp

| Việc | Ai giữ | Trạng thái | Ghi chú cho Tiến |
|---|---|---|---|
| Chốt PDF cuối | Kiên | TODO theo bảng team | Sau khi Kiên biên dịch, kiểm nhanh T5/T10/T11 còn hiện đúng trong PDF: Bảng sensitivity, số 18,5%, và wording correlation. |
| Q&A `docs/03` | Chưa có trong repo | BLOCKED | Nếu team đưa file này lại vào repo, Tiến cần quét các số cũ và thêm câu trả lời T5 ở trên. |
| Slide cuối | Cả nhóm | IN_PROGRESS | Source `slides/slides.md` đã đúng; nếu render lại slide, kiểm `slides/slides.html`/PDF không quay về `~19% EN`. |
| Soát chéo | Quang kiểm Kiên + Tiến theo bảng team | PENDING | Phần Tiến có evidence code/test đủ; điểm còn dễ bị hỏi nhất là T5, không phải T10/T11. |

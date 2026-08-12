# Checklist Tô Huỳnh Minh Tiến

| Task | Nội dung cần làm | Status | Evidence | Note |
|---|---|---|---|---|
| T5.1 | Kiểm tra `alphas_from_mi` có đang chuẩn hoá `I(d)` trực tiếp trên lưới log mà không nhân bề rộng ô hay không | DONE | `hyena_study/morphology.py`, `hyena_study/analyze.py::alpha_mapping_sensitivity` | Code hiện tại dùng `w = w / w.sum()` trên các lag đã lấy mẫu log; `analyze.py` đã có docstring note `T5/Tien`. |
| T5.2 | Bổ sung sensitivity analysis cho lưới log: so sánh cách hiện tại với cách nhân bề rộng ô | DONE | `python3 -m hyena_study.analyze`; `tests/test_analyze.py::test_lag_width_sensitivity_reproduces_report_numbers` | Reproduce: VI 2,95/142/227 vs 90,54/39/87; EN 2,85/141/214 vs 160,45/27/57. |
| T5.3 | Viết limitation trong report, nêu rõ điều này đổi diễn giải chứ không làm sai PPL đã chạy | DONE | `report/acl_latex.tex`, Bảng `tab:alpha_sensitivity` | Bảng width-weighted là sensitivity analysis, không phải kết quả huấn luyện mới. |
| T10.1 | Tái sinh 14,3%, 22,3%, 75,8%, correlation 0,9974 và correlation corpus-logspace từ code | DONE | `hyena_study/analyze.py::alpha_comparison_metrics`; `tests/test_analyze.py::test_alpha_comparison_metrics_reproduce_report_numbers` | Reproduce: 14,35% denominator EN; 22,29% denominator VI; symmetric 17,27%; corr log VI-EN 0,9974; VI-logspace 75,81% và corr 0,9377; code/test đã có docstring note `T10/Tien`. |
| T10.2 | Sửa wording report: nêu denominator, metric, sample size, biến lấy correlation | DONE | `report/acl_latex.tex`, mục 5.3 và abstract | Correlation được hạ giọng thành mô tả hình dạng của 256 effective-length quantiles, không dùng như bằng chứng độc lập mạnh. |
| T11.1 | Kiểm tra scale mismatch: alpha đo bằng BPE token nhưng E4 train/evaluate bằng syllable token | DONE | `results/E4_corpus_*.json`, `results/alpha_*_bpe500.json`, `report/acl_latex.tex` | VI: 3,851 char/BPE-token vs 4,118 char/syllable-token; EN: 4,155 vs 5,115. |
| T11.2 | Bổ sung limitation trong report về scale mismatch | DONE | `report/acl_latex.tex`, mục 3.5 | Ghi rõ 6,5%/19% là lệch thang đơn vị, chưa phải sai số trực tiếp của PPL vì chưa có experiment cô lập. |

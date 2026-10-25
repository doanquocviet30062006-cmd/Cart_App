# Báo cáo xây dựng và kiểm thử CART Student Risk Lab

Ngày chạy: 25/10/2026. Ứng dụng là bản local mới trong workspace, được tạo vì source repository nêu trong báo cáo gốc không truy cập được. Không có tuyên bố sửa website Vercel công khai.

## Dataset, nhãn và pipeline

- Nguồn dữ liệu chính thức: Open University Learning Analytics Dataset (OULAD), tải từ UCI, DOI 10.24432/C5KK69. Giấy phép CC BY 4.0. Bài mô tả dữ liệu: Kuzilek, Hlosta, Zdrahal, *Open University Learning Analytics dataset*, Scientific Data 4, 170171 (2017), DOI 10.1038/sdata.2017.171.
- Tệp archive UCI `data/raw/oulad_valid.zip` có 44,793,856 byte và vượt qua kiểm tra ZIP (`testzip = None`); 8 tệp CSV/tài liệu được giải nén trong `data/raw/oulad/`.
- Dataset gốc có 32,593 lượt đăng ký. Cohort tại cuối ngày 28 còn 27,515 lượt: loại 45 bản ghi thiếu ngày đăng ký, 16 đăng ký sau ngày 28, và 5,055 lượt đã rút vào hoặc trước ngày 28. Cohort có 5,105 `Withdrawn` và 22,410 nhãn còn lại.
- Target dương: `final_result == Withdrawn`; target âm: `Pass`, `Fail`, hoặc `Distinction`. Không gộp Fail với Withdrawn.
- Đủ điều kiện ở cuối ngày 28 nếu `date_registration <= 28` và `date_unregistration` trống hoặc `> 28`. Các ngày đăng ký/rút chỉ dùng xác định cohort, không đi vào feature. Thiếu ngày đăng ký bị loại vì không thể xác minh đủ điều kiện.
- VLE `sum_click` và số ngày hoạt động được tổng hợp cho ngày 0–28 inclusive. Sáu feature: `code_module`, `code_presentation`, `num_of_prev_attempts`, `studied_credits`, `clicks_day_0_28`, `active_days_day_0_28`. ID sinh viên, giới tính, tuổi, học vấn, `final_result`, ngày đăng ký và ngày rút không được dùng làm predictor.
- Train/test: `GroupShuffleSplit(test_size=0.20, random_state=42)` theo `id_student`, để lượt đăng ký lặp lại của cùng sinh viên không xuất hiện ở cả hai tập. Train có 22,029 lượt / 19,860 sinh viên; test có 5,486 lượt / 4,966 sinh viên; giao nhau bằng 0.
- Model: `DecisionTreeClassifier(criterion="gini", class_weight="balanced", max_depth=8, random_state=42)`. Alpha pruning được chọn trong 5-fold `StratifiedGroupKFold` trên train, nhóm theo sinh viên, dùng F1 lớp Withdrawn. Chọn cây đơn giản nhất (ít lá nhất, sau đó độ sâu thấp nhất) nằm trong một độ lệch chuẩn so với F1 CV cao nhất. Không dùng test để chọn alpha.

## Kết quả test từ lần chạy thực tế

Confusion matrix theo thứ tự nhãn `[Not withdrawn, Withdrawn]`, hàng là nhãn thật, cột là dự báo:

```text
[[2608, 1861],
 [ 371,  646]]
```

| Cấu hình | Accuracy | Precision Withdrawn | Recall Withdrawn | F1 Withdrawn | Độ sâu | Lá | Nút |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority-class baseline | 81.46% | 0.00% | 0.00% | 0.000 | — | — | — |
| CART chưa pruning (depth cap 8, class_weight balanced) | 61.41% | 26.35% | 60.28% | 0.367 | 8 | 207 | 413 |
| CART sau pruning (alpha 0.0005895681) | 59.31% | 25.77% | 63.52% | 0.367 | 6 | 15 | 29 |

Các số cho thấy accuracy baseline cao do dự báo lớp đa số nhưng không phát hiện trường hợp Withdrawn nào. CART recall cao hơn nhưng false positive lớn (1,861/4,469 lượt không rút bị cảnh báo) và còn bỏ sót 371/1,017 lượt Withdrawn. Pruning giảm mạnh kích thước cây và tăng recall khoảng 3.2 điểm phần trăm, trong khi F1 gần như không đổi và accuracy giảm. Đây là kết quả hạn chế, không phải bằng chứng hệ thống cải thiện kết quả học tập.

Leaf fraction hiển thị trong app là tỷ lệ lớp ở nút lá, không được gọi là xác suất đã hiệu chuẩn.

## Kiểm thử và build

| Kiểm tra | Lệnh hoặc thao tác | Kết quả |
|---|---|---|
| Pipeline dữ liệu và huấn luyện | `python ml/train.py` | Thành công; tạo `artifacts/cart_withdrawal.joblib`, `model_metadata.json`, `metrics.json` và `train_features.csv`. |
| Bộ kiểm thử backend | `python -m unittest discover -s tests -v` | 14/14 passed; gồm inference/artifact, path khớp dự báo, schema thiếu/thừa, sai kiểu, ngoài miền, category không hợp lệ, endpoint HTTP. |
| Kiểm tra feature leakage | Bài test `test_model_features_exclude_outcome_and_postcutoff_dates` | Passed: target, ID, ngày đăng ký/rút không có trong feature schema. |
| Kiểm tra JavaScript | `node --check app/static/app.js` | Thành công, không có lỗi cú pháp. |
| Smoke test giao diện | `node tests/browser_smoke.js` (Chrome headless, local app) | Passed: 4 khu vực điều hướng; prediction thật; Decision Path; input sai miền; trang metric; dataset/limitations; viewport 390×844; không có `pageerror`. |
| Frontend build | — | Không có bundler/compiler vì frontend là HTML/CSS/JS tĩnh. Đã kiểm tra cú pháp JS và chạy smoke test trên Chrome thực. |
| Production deployment | — | Chưa triển khai. Chỉ xác minh local tại `http://127.0.0.1:8765`; không có credential/deployment proof và không coi localhost là production. |

## Ảnh giao diện đã chụp từ app local

- `screenshots/overview.png`
- `screenshots/prediction.png`
- `screenshots/evaluation.png`
- `screenshots/dataset-mobile.png`

## Giới hạn còn lại

- OULAD là dữ liệu ẩn danh lịch sử từ các năm 2013–2014; kết quả không đảm bảo chuyển sang trường, học phần hoặc thời kỳ khác.
- Precision 25.77% cho thấy nhiều cảnh báo giả; cách đặt ngưỡng/quy trình hỗ trợ cần được xác định cùng người dùng nghiệp vụ trước khi có bất kỳ thử nghiệm thực tế.
- Chưa có hiệu chuẩn xác suất, kiểm toán fairness theo nhóm, đánh giá prospective, kiểm định ngoài mẫu theo kỳ mở lớp hoặc đo lợi ích của can thiệp.
- Đây là nguyên mẫu local phục vụ học thuật, không phải hệ thống quản lý học tập hoặc quyết định tự động về sinh viên.

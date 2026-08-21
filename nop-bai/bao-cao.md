# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | Nguyễn Thành Công |
| MSSV | 2A202601396 |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/___/___ |
| Ngày nộp | 21/8/2026 |

---

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---|---|---|---|---|
| 1 | 50 | 0.05 | 2 | 0.6051 | 0.846 |
| 2 | 100 | 0.1 | 3 | 0.7109 | 0.878 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.874 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Lần chạy 3 mang lại `f1_score` cao nhất (0.7149), vượt xa ngưỡng chặn `F1 >= 0.65` của pipeline. Đáng chú ý, lần chạy 2 đạt `accuracy` cao nhất (0.878) nhưng `f1_score` lại thấp hơn lần 3 (0.7109 < 0.7149). Điều này chứng minh Accuracy cao dễ bị chi phối bởi việc dự đoán đúng lớp đa số, trong khi F1 phản ánh chính xác năng lực nhận diện lớp thu nhập cao. Giữa các lần chạy, ta thấy sự đánh đổi: cấu hình sâu và nhiều cây hơn giúp mô hình bù đắp tốt cho `learning_rate` nhỏ, tối ưu hoá dự đoán ở lớp thiểu số.

---

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Tập dữ liệu Adult mang tính mất cân bằng lớp rõ rệt khi lớp dương (thu nhập > 50K) chỉ chiếm 24.8% tổng số mẫu. Nếu một mô hình ngớ ngẩn luôn dự đoán mọi mẫu là thu nhập thấp, Accuracy vẫn đạt mức 75.2% (0.752) — một con số dễ gây hiểu nhầm rằng mô hình hoạt động tốt dù thực tế hoàn toàn vô dụng.

$F_1\text{-score}$ của lớp dương là trung bình điều hòa giữa Precision và Recall, đo lường trực tiếp khả năng phát hiện đúng người có thu nhập cao và kiểm soát lượng dự đoán nhầm. Việc không sử dụng `average="weighted"` hay `average="macro"` là bắt buộc, vì các tùy chọn này sẽ đưa độ đo của lớp đa số vào tính toán, làm thổi phồng điểm số và vô hiệu hóa cơ chế kiểm soát chất lượng (Quality Gate).

---

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| Lệnh thiết lập biến môi trường MLflow báo lỗi `CommandNotFoundException`. | Sử dụng lệnh `export` của Linux/Bash trong môi trường PowerShell trên Windows. | Chuyển sang cú pháp PowerShell `$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"`. |
| Giao diện MLflow UI mặc định không hiển thị các cột tham số và chỉ số. | Chưa cấu hình tùy chọn hiển thị cột trong bảng Experiments. | Nhấn vào nút `Columns` trên UI và tích chọn các trường `f1_score`, `accuracy`, `learning_rate`, `max_depth`, `n_estimators`. |

---

## 4. So Sánh Bước 2 và Bước 3 (bắt buộc, 2 - 3 câu)

<!-- Lấy số liệu từ bảng ở mục 3.6 của tasks/buoc-3.md. -->

| | f1_score | accuracy |
|---|---|---|
| Bước 2 (chỉ `train_batch1`) | ___ | ___ |
| Bước 3 (thêm `train_batch2`) | ___ | ___ |

**Nhận xét:** ___

<!--
Một câu trả lời trung thực kiểu "f1 giảm 0,01 vì dữ liệu mới cùng phân phối, không mang
thêm thông tin mới" được đánh giá cao hơn kết luận sai rằng thêm dữ liệu luôn tốt hơn.
-->

---

## 5. Phần Bonus Đã Thực Hiện (nếu có)

<!-- Xóa cả mục 5 nếu không làm bonus. Mỗi bonus tối đa 1 dòng. -->

- [ ] Bonus 1 - Tracking MLflow từ xa với DagsHub: ___
- [ ] Bonus 2 - Điều chỉnh ngưỡng quyết định: ___
- [ ] Bonus 3 - Báo cáo precision / recall tự động: ___
- [ ] Bonus 4 - Hoàn trả về phiên bản trước: ___
- [ ] Bonus 5 - Cảnh báo lệch lạc dữ liệu: ___

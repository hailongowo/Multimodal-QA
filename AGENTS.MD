# Quy tắc làm việc dành cho AI agent

## Bắt buộc trước khi code

1. Đọc toàn bộ `README.md`, `AGENTS.md`, `CHECKPOINT.md` và `GUIDE.md`.
2. Chỉ code khi người dùng yêu cầu cho phép code.

## Quy tắc dữ liệu

- Corpus gốc kèm ảnh nằm trong data\law_db
- Data dùng để train nằm ở data\train data
- Data dùng để đánh giá nằm ở data\public_test data
- Với tất cả các hướng giải quyết bài toán ngoài Baseline, cần sử dụng structured_corpus, ở data\structured_corpus\structured_corpus.jsonl

## Quy trình thực hiện

- Mọi hướng giải quyết phải được chạy trên GPU nếu có thể.
- Chỉ thực hiện **một hướng giải quyết bài toán tại một thời điểm**.
- Chỉ sửa những file cần thiết cho hướng hiện tại.
- Output duy nhất của mỗi hướng là 1 file JSON lưu tại `outputs/predictions/{hướng}_predictions.json` chứa prediction của hướng đó trên tập public_test, và 1 file JSON lưu tại `outputs/metrics/{hướng}_metrics.json` chứa đánh giá của hướng đó với metric F2, Recall@1, 5, 10.
- Đối với mỗi hướng, hãy viết 1 file `README.md` trong folder code của hướng đó, mô tả toàn bộ về pipeline từ chuẩn bị dữ liệu cho đến đánh giá, kèm config và giải thích chúng, không cần hướng dẫn cách chạy.
- Sau mỗi hướng, cập nhật `GUIDE.md` bằng lệnh PowerShell khớp code thực tế, `GUIDE.md` chỉ bao gồm các lệnh chạy full pipeline các hướng kèm với mô tả các param, các setting có thể chạy, và lệnh để track progress.
- Sau khi viết code, dừng và liệt kê ngắn gọn các file đã thay đổi.
- Để người dùng tự chạy và tự kiểm thử; agent **không chạy code, test, benchmark hoặc thí nghiệm**.
- Không đánh dấu hướng giải quyết hoàn thành trước khi người dùng xác nhận.
- Chỉ chuyển hướng giải quyết khi người dùng yêu cầu rõ ràng.

## Khi người dùng báo lỗi

- Chỉ chẩn đoán và sửa lỗi thuộc hướng hiện tại.
- Sau khi sửa, dừng để người dùng kiểm thử lại.
- Không triển khai thêm hướng khác trong lúc sửa lỗi.

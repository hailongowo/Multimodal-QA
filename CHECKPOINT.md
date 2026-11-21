# CHECKPOINT — Trạng thái hiện tại

Cập nhật: 18/11/2026.

## Bài toán đang thực hiện

Project hiện đang thực hiện **MLQA-TSR — Subtask 1**: truy xuất các điều luật liên quan từ corpus pháp luật cho câu hỏi bằng văn bản và ảnh đi kèm. Đầu ra là danh sách các cặp `law_id`, `article_id` trong `relevant_articles`.

## Tiêu chí hoàn thành một hướng

Một hướng được tính là **đã hoàn thành khi đã được evaluate và có kết quả F2**. Việc đã viết code, chuẩn bị dữ liệu hoặc tạo prediction chưa đủ để đánh dấu hoàn thành.

Trạng thái hoàn thành dưới đây áp dụng cho từng hướng đã đánh giá, không có nghĩa toàn bộ Subtask 1 đã hoàn thành.

## Các hướng đã hoàn thành

Các kết quả dưới đây được lấy từ metric đã lưu trên cùng 100 câu public test tại `data/public_test data/vlsp_2025_public_test.json`. Metric là F2 tính cho từng câu hỏi rồi lấy trung bình.

| Hướng | Biến thể / stage đã evaluate | F2 | Ngày đánh giá ghi trong metric |
| --- | --- | ---: | --- |
| Official baseline — Visualized-BGE | Final | 0.13671849 | 04/11/2026 |
| Image-to-text — BGE-M3 | Dense | 0.21825867 | 06/11/2026 |
| Image-to-text — BGE-M3 | BGE reranker | 0.14670008 | 06/11/2026 |
| Structured corpus + reranker | Dense | 0.15939168 | 06/11/2026 |
| Structured corpus + reranker | Reranked | 0.19352547 | 06/11/2026 |

Có **3 hướng đã hoàn thành** theo tiêu chí trên; dense và reranked là các stage của cùng một hướng.

Nguồn kết quả:

- Official baseline: `outputs/evaluation/official_baseline/metrics.json`.
- Image-to-text: `outputs/evaluation/img_to_txt/metrics.json`.
- Structured corpus + reranker: `outputs/evaluation/structured_reranker/metrics.json`.
- Bảng tổng hợp đã lưu: `outputs/evaluation/summary.md`.

Trong các kết quả đã lưu ở trên, image-to-text dùng BGE-M3 dense có F2 cao nhất: **0.21825867**.

## Hướng chưa đủ điều kiện đánh dấu hoàn thành

- **Fine-tune Visualized-BGE**: đã có tài liệu và code trong `src/legalqa/finetune_visual_bge/`, nhưng chưa tìm thấy kết quả evaluate F2 đã lưu trong workspace. Chưa đánh dấu hoàn thành.

Checkpoint này tổng hợp các kết quả có sẵn; không chạy lại code, đánh giá, benchmark hoặc thí nghiệm.

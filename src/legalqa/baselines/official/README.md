# Official baseline — Visualized-BGE

Hướng baseline dùng Visualized-BGE để truy xuất điều luật từ câu hỏi và ảnh, không huấn luyện trên train data và không dùng structured corpus. Lệnh chạy và track progress nằm trong [GUIDE.md](../../../../GUIDE.md).

## Dữ liệu và chuẩn bị

- Corpus gốc: `data/law_db/vlsp2025_law.json`; ảnh corpus: `data/law_db/images.fld/`.
- Câu hỏi: `data/public_test data/vlsp_2025_public_test.json`; ảnh: `data/public_test data/public_test_images/{image_id}.jpg`.
- Trọng số: `artifacts/models/Visualized_m3.pth`; backbone `BAAI/bge-m3`. Môi trường cần dependencies project và module `visual_bge`.
- Loader ưu tiên package `visual_bge` đã cài; nếu không tìm thấy module, dùng mã nguồn tại `.runtime/FlagEmbedding/research/visual_bge` (phải chứa `visual_bge/modeling.py`). Đường dẫn được xác định từ vị trí loader, không phụ thuộc thư mục chạy lệnh. Nếu cả hai nguồn đều thiếu, loader báo đường dẫn cần có; lỗi thiếu dependency bên trong được giữ nguyên.

## Pipeline

1. `scripts/build_official_baseline_data.py` chuyển bảng HTML trong marker TABLE thành Markdown, giữ cấu trúc documents/articles và ghi `data/baselines/official/vlsp2025_law.json`.
2. `scripts/run_official_baseline.py` encode từng điều luật. Điều không có marker ảnh dùng text; điều có marker ảnh được bỏ marker khỏi text và tạo một vector text + ảnh cho mỗi ảnh nhận diện được.
3. Lưu ID và embedding vào `artifacts/official_baseline/law_id.npy`, `law_embedding.npy`.
4. Encode câu hỏi cùng ảnh, tính tích vô hướng với vector corpus, giữ chính sách baseline: chọn 5 vector cao nhất rồi loại trùng cặp ID, xuất theo thứ tự corpus (có thể ít hơn 5 điều luật). Riêng Recall@1/5/10 dùng danh sách tối đa 10 điều luật phân biệt theo score giảm dần từ cùng lượt encode.
5. Ghi hai file kết quả public test: `outputs/predictions/official_baseline_predictions.json` và `outputs/metrics/official_baseline_metrics.json`. Metrics gồm `f2`, `recall_at_1`, `recall_at_5`, `recall_at_10`. F2 dùng prediction từ top 5 vector theo chính sách baseline; recall dùng thứ hạng điều luật phân biệt trước khi cắt top 5, trung bình theo câu hỏi.

## Cấu hình

| Setting | Giá trị / tác dụng |
| --- | --- |
| Backbone | `BAAI/bge-m3`, cố định trong loader |
| Trọng số, dữ liệu và output | Hằng số trong `scripts/run_official_baseline.py` |
| Top-k điều luật | Prediction 5; thứ hạng đánh giá tối đa 10 |
| Tiền xử lý | Script chuẩn bị nhận `--input`, `--output`, mặc định theo đường dẫn trên |
| Device | Do implementation Visualized-BGE quyết định |
| Cache | Full pipeline luôn tạo lại corpus baseline và embeddings |
| Giới hạn token inference | Loader bật truncation cho corpus và query, gồm special tokens. Text-only bị giới hạn bởi tokenizer và bảng position embedding, trừ offset padding của XLM-R. Text + ảnh dành chỗ cho toàn bộ image patch tokens trong context trước khi tokenize text. Văn bản vượt giới hạn bị cắt theo truncation side của tokenizer (BGE-M3 mặc định cắt cuối), có thể mất nội dung điều luật. |

Loader triển khai giới hạn trong subclass riêng của baseline, áp dụng cho cả package đã cài và source fallback, không sửa mã nguồn runtime dùng chung. Inference `encode` tắt gradient. Giới hạn context lấy giá trị nhỏ hơn giữa `tokenizer.model_max_length` và số hàng position embedding; số image tokens lấy từ `model_visual.visual.patch_embed.num_patches`, tương ứng nhánh EVA của Visualized-BGE M3.

Wrapper `run_pipeline.py` nhận `--device`, `--reuse-index`, `--rebuild-corpus`, `--batch-size`, `--accumulation-steps`, `--epochs`, `--max-train-samples` nhưng không dùng các giá trị này để thay đổi pipeline baseline.

## Đánh giá và trạng thái

Nhãn `relevant_articles` chỉ dùng đánh giá. F2 tính trên tập cặp ID từng câu rồi lấy trung bình: `5PR / (4P + R + 1e-9)`. Log nằm tại `runs/official_baseline/pipeline.log`. Metric lịch sử tại `outputs/evaluation/official_baseline/` được ghi nhận riêng trong CHECKPOINT.md; là kết quả lịch sử, không được ghi đè bởi pipeline hiện tại. Tài liệu này không xác nhận lần chạy mới.



Thư mục gốc kết quả chứa hai thư mục con: `predictions/` và `metrics/`; `--output-dir` (ở script có tham số này) chọn thư mục gốc, mặc định public test là `outputs/`.


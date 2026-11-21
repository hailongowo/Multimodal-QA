# Structured corpus + Visualized-BGE + reranker

Hướng này truy xuất nhiều chunk cho mỗi điều luật từ structured corpus. Visualized-BGE nhận câu hỏi và ảnh; reranker nhận văn bản câu hỏi và evidence. Không fine-tune, không dùng mô tả ảnh sinh sẵn. Lệnh chạy và track progress nằm trong [GUIDE.md](../../../GUIDE.md).

## Chuẩn bị dữ liệu

Corpus bắt buộc là `data/structured_corpus/structured_corpus.jsonl`. Wrapper tạo corpus từ `data/law_db/vlsp2025_law.json` nếu thiếu hoặc bật `--rebuild-corpus`. Câu hỏi nằm tại `data/public_test data/vlsp_2025_public_test.json`, ảnh tại `data/public_test data/public_test_images/{image_id}.jpg`. Không dùng train data để học trọng số.

Cần module `visual_bge`, backbone `BAAI/bge-m3`, trọng số `artifacts/models/Visualized_m3.pth` và reranker `BAAI/bge-reranker-v2-m3`. Script có fallback tìm Visualized-BGE tại `.runtime/FlagEmbedding/research/visual_bge` khi thiếu module.

## Pipeline

1. Đọc structured records, chuyển phần tử cấu trúc thành text và chia chunks, giữ header văn bản/điều cùng block_ids để truy nguồn. Không đưa description_vlm vào retrieval.
2. Encode chunks bằng text encoder Visualized-BGE, lưu candidates và embeddings vào index.
3. Encode câu hỏi cùng ảnh, tính tích vô hướng với toàn bộ chunks.
4. Lấy top chunks, nhóm theo `(law_id, label_id)`, giữ shortlist điều luật theo score chunk lớn nhất và một số evidence mỗi điều.
5. Cross-encoder chấm văn bản câu hỏi với từng evidence; score điều luật là score evidence lớn nhất. Ảnh chỉ tham gia dense stage.
6. Chỉ xuất prediction reranked cuối, tối đa top-k cặp ID khác nhau/câu. Giữ thứ hạng đến max(10, top-k) để tính recall. Tập đánh giá phải có nhãn và không rỗng; nhãn không tham gia retrieval/rerank.

## Cấu hình

Các tham số chi tiết thuộc `scripts/run_structured_reranker.py`.

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--chunk-tokens` | `512` | Ngân sách chunk mềm và giới hạn tokenize candidates |
| `--chunk-overlap` | `1` | Số phần tử overlap, không phải token |
| `--chunk-shortlist-size` | `200` | Chunks dense xét để nhóm điều luật |
| `--article-shortlist-size` | `50` | Điều luật tối đa trước rerank, phải ≥ max(10, top-k) |
| `--chunks-per-article` | `3` | Evidence tối đa mỗi điều |
| `--dense-batch-size` | `8` | Batch encode corpus |
| `--top-k` | `5` | Số cặp ID đầu ra tối đa |
| `--reranker-model` | `BAAI/bge-reranker-v2-m3` | Cross-encoder |
| `--reranker-batch-size` | `4` | Batch câu hỏi/evidence |
| `--reranker-max-length` | `1024` | Giới hạn token một cặp |
| `--device` | `cuda` | Chỉ điều khiển reranker, không đổi device loader Visualized-BGE |
| `--rebuild-index` | Tắt ở script stage | Buộc xây lại cache |
| `--corpus`, `--questions`, `--images-dir`, `--model-weight` | Đường dẫn ở trên | Chọn đầu vào |
| `--index-dir`, `--output-dir` | `runs/structured_reranker`, `outputs` | Thư mục artifact |


Phần tử quá dài có thể vượt ngân sách chunk và bị tokenizer cắt khi encode. Wrapper mặc định xây lại index; `--reuse-index` cho phép dùng index cũ nhưng cache không kiểm tra fingerprint corpus/model/config. `--rebuild-corpus` tạo lại structured corpus. Các tham số training mà wrapper nhận không tác động hướng này.

## Artifact và đánh giá

- `runs/structured_reranker/structured_index.npz`: candidates và embeddings tương ứng.
- `outputs/predictions/structured_reranker_predictions.json`: kết quả reranked cuối theo schema chính thức.
- `outputs/metrics/structured_reranker_metrics.json`: chỉ gồm `f2`, `recall_at_1`, `recall_at_5`, `recall_at_10`.
- `runs/structured_reranker/pipeline.log`: log full pipeline.

Metric dùng eval_retrieval: so sánh tập cặp ID trên từng câu, lấy trung bình F2. Kết quả lịch sử nằm trong CHECKPOINT.md; tài liệu này không xác nhận lần chạy mới.

F2 dùng prediction cuối (mặc định top 5). Recall@1/5/10 dùng thứ hạng reranked trước khi cắt prediction, loại trùng cặp ID và lấy trung bình recall theo câu; câu có nhãn rỗng được tính 0. Cache và log là artifact nội bộ, không phải kết quả cuối. Các file kết quả cũ không được tự xóa.


Thư mục gốc kết quả chứa hai thư mục con: `predictions/` và `metrics/`; `--output-dir` (ở script có tham số này) chọn thư mục gốc, mặc định public test là `outputs/`.


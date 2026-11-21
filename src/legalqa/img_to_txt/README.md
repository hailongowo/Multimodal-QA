# Image-to-text — BGE-M3 dense + reranker

Hướng này dùng mô tả ảnh đã chuẩn bị sẵn, truy xuất bằng `BAAI/bge-m3`, rồi rerank bằng `BAAI/bge-reranker-v2-m3`. Pipeline không đọc pixels, không sinh mô tả ảnh và không fine-tune. Lệnh chạy và track progress nằm trong [GUIDE.md](../../../GUIDE.md).

## Chuẩn bị dữ liệu

| Đầu vào | Đường dẫn mặc định |
| --- | --- |
| Structured corpus | `data/structured_corpus/structured_corpus.jsonl` |
| Mô tả ảnh corpus | `data/img_to_txt/image_descriptions.json`, schema `2.0` |
| Mô tả ảnh test | `data/img_to_txt/img_to_txt_test_descriptions.json`, schema `1.0` |
| Câu hỏi | `data/public_test data/vlsp_2025_public_test.json` |

Wrapper tạo structured corpus từ `data/law_db/vlsp2025_law.json` nếu thiếu hoặc bật `--rebuild-corpus`. Hai file mô tả phải có sẵn. Không dùng dữ liệu train để học trọng số.

Mô tả corpus có `records`, liên kết bằng `record_id`, `article_key`, `document_id`, `article_id`; chỉ record có `status=success` đóng góp mô tả. ID phải khớp corpus. Mô tả test có dictionary `descriptions` với khóa `image_id`. Description gồm đúng `image_category`, `summary`, `items`; mỗi item gồm `identifier`, `visual_or_geometric_features`, `meaning_or_specification`. Category hợp lệ: `SIGN_ILLUSTRATION`, `TECHNICAL_DIAGRAM`, `MARKING_AND_SIGNAL`.

## Pipeline

1. Ghép tên văn bản, tên điều, paragraph và text trong ô bảng với mô tả ảnh. Gộp các record cùng `(law_id, article_id)`.
2. Chia theo dòng thành chunks với ngân sách token mềm, overlap theo số phần tử. Dòng quá dài có thể vượt ngân sách và bị tokenizer cắt khi encode.
3. Encode bằng CLS của BGE-M3, chuẩn hóa L2 và lưu index. Query là câu hỏi cộng mô tả ảnh tương ứng; thiếu mô tả cho image_id thì dùng câu hỏi đơn thuần.
4. Tính tích vô hướng query/chunks; lấy top chunks, nhóm theo cặp ID, chọn shortlist điều luật bằng score chunk lớn nhất và giữ evidence mỗi điều.
5. Cross-encoder chấm query/evidence; score điều luật là score evidence lớn nhất. Giữ thứ hạng reranked tối đa max(10, top-k) cặp ID khác nhau; prediction cuối chỉ lấy top-k. Dense là stage nội bộ, không xuất file kết quả riêng.
6. Yêu cầu tập câu hỏi có nhãn và không rỗng để xuất đủ hai file kết quả. Nhãn chỉ dùng đánh giá, không đưa vào query.

## Cấu hình

Tham số chi tiết thuộc `evaluate.py`; wrapper chỉ cung cấp một tập con.

| Tham số | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--dense-model` | `BAAI/bge-m3` | Encoder text |
| `--reranker-model` | `BAAI/bge-reranker-v2-m3` | Cross-encoder |
| `--chunk-tokens` | `512` | Ngân sách chunk và giới hạn encode dense, cả query |
| `--chunk-overlap` | `1` | Số dòng overlap |
| `--chunk-shortlist-size` | `200` | Số chunks dense giữ lại |
| `--article-shortlist-size` | `50` | Số điều luật trước rerank, phải ≥ max(10, top-k) |
| `--chunks-per-article` | `3` | Evidence tối đa mỗi điều |
| `--dense-batch-size` | `8` | Batch encode dense |
| `--reranker-batch-size` | `4` | Batch cross-encoder |
| `--reranker-max-length` | `1024` | Giới hạn token cặp query/evidence |
| `--top-k` | `5` | Số cặp ID đầu ra tối đa |
| `--device` | `cuda` | Device cả hai model; hỗ trợ cpu |
| `--rebuild-index` | Tắt ở evaluate | Buộc xây lại cache |
| `--corpus`, `--descriptions`, `--test-descriptions`, `--questions` | Đường dẫn ở trên | Chọn dữ liệu |
| `--index-dir`, `--output-dir` | `runs/img_to_txt`, `outputs` | Thư mục artifact |

Wrapper mặc định buộc xây lại index; `--reuse-index` cho phép cache nếu fingerprint khớp tên model, chunk config, đường dẫn và nội dung corpus/mô tả corpus. `--rebuild-corpus` tạo lại structured corpus. Các tham số training mà wrapper nhận không tác động hướng này.

## Artifact và đánh giá

- `runs/img_to_txt/img_to_txt_index.npz`: embeddings, candidates và fingerprint.
- `outputs/predictions/img_to_txt_predictions.json`: kết quả reranked cuối theo schema chính thức.
- `outputs/metrics/img_to_txt_metrics.json`: chỉ gồm `f2`, `recall_at_1`, `recall_at_5`, `recall_at_10`.
- `runs/img_to_txt/pipeline.log`: log các stage và heartbeat.

F2 so sánh tập cặp ID từng câu rồi lấy trung bình, ưu tiên recall. Metric lịch sử được tổng hợp trong CHECKPOINT.md; cập nhật README không tạo thêm kết quả đánh giá.

F2 dùng prediction cuối (mặc định top 5). Recall@1/5/10 dùng thứ hạng reranked trước khi cắt prediction, loại trùng cặp ID và lấy trung bình recall theo câu; câu có nhãn rỗng được tính 0. Cache và log là artifact nội bộ, không phải kết quả cuối. Các file kết quả cũ không được tự xóa.


Thư mục gốc kết quả chứa hai thư mục con: `predictions/` và `metrics/`; `--output-dir` (ở script có tham số này) chọn thư mục gốc, mặc định public test là `outputs/`.


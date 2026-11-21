# Fine-tune Visualized-BGE

Hướng này fine-tune Visualized-BGE bằng câu hỏi + ảnh train và candidates văn bản từ structured corpus. Pipeline gồm chuẩn bị dữ liệu, mining, training, chọn checkpoint trên validation và đánh giá public test. Lệnh chạy và track progress nằm trong [GUIDE.md](../../../GUIDE.md).

## Đầu vào

| Dữ liệu | Đường dẫn mặc định |
| --- | --- |
| Structured corpus | `data/structured_corpus/structured_corpus.jsonl` |
| Train và ảnh | `data/train data/vlsp_2025_train.json`, `data/train data/train_images/` |
| Public test và ảnh | `data/public_test data/vlsp_2025_public_test.json`, `data/public_test data/public_test_images/` |
| Trọng số khởi tạo | `artifacts/models/Visualized_m3.pth` |
| Alias nhãn | `data/finetune_visual_bge/label_aliases.json` |

Cần dependencies project và implementation Visualized-BGE tương thích, backbone mặc định BAAI/bge-m3. Ảnh query được tra theo `{image_id}.jpg`; corpus được serialize thành text, không encode ảnh corpus trực tiếp.

## Pipeline

1. Wrapper tạo structured corpus từ `data/law_db/vlsp2025_law.json` nếu thiếu hoặc yêu cầu rebuild.
2. `prepare_visual_bge_prerequisites.py` kiểm tra đầu vào, ảnh và nhãn train; giữ alias đã có hoặc tạo mặc định nếu thiếu. Không dùng nhãn public test để tạo alias.
3. `prepare_visual_bge_finetune.py` tạo candidates cấp article và clause từ paragraph, caption, bảng và description_vlm nếu có trong corpus. Units cùng cặp ID được gộp, giữ provenance. Alias ánh xạ nhãn sang candidates và giữ ID đầu ra phục vụ metric chính thức.
4. Loại cả câu hỏi không hợp lệ hoặc không giải được đầy đủ positives, ghi báo cáo rejection. Chia train/validation theo nhóm image_id, mặc định 80/20, seed 42, tránh cùng ảnh ở cả hai split.
5. Mining dùng checkpoint gốc để encode query train và toàn bộ candidates, loại positives rồi giữ tối đa 20 hard negatives/query. Lưu rank, score và fingerprint manifest.
6. Training tạo batch query ảnh + text, positives và số negatives được cấu hình. Loss multi-positive InfoNCE chuẩn hóa L2 embeddings, dùng temperature và mask positives/valid documents. Tối ưu AdamW, tích lũy gradient và clip gradient.
7. Sau mỗi epoch, đánh giá validation Recall@5, lưu best.pt khi cải thiện và last.pt để resume. Early stopping mặc định sau 2 epoch liên tiếp không cải thiện.
8. Evaluation nạp best.pt, encode corpus/query và lấy top-k cặp ID. Wrapper đánh giá validation trước, public test sau; không dùng public test chọn checkpoint.

## Cấu hình full pipeline

| Tham số wrapper | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `--device` | `cuda` | Device mining/training/evaluation; CPU training dùng FP32 |
| `--batch-size` | `1` | Query mỗi batch; documents phụ thuộc positives/negatives |
| `--accumulation-steps` | `16` | Batch tích lũy gradient; không chia nhỏ forward documents |
| `--hard-negative-count` | `1` | Negatives tối đa/query khi train, cho phép 0; không đổi pool mining |
| `--query-max-length` | `256` | Token text query training, gồm special tokens, không gồm image tokens |
| `--document-max-length` | `512` | Token document training; cắt cuối text dài |
| `--precision` | `fp16` | CUDA chọn fp16/bf16/fp32; bf16 cần GPU hỗ trợ; loss FP32, fp16 dùng GradScaler |
| `--gradient-checkpointing` | Bật | Checkpoint text encoder để giảm activation; tắt bằng --no-gradient-checkpointing |
| `--epochs` | `5` | Số epoch tối đa |
| `--max-train-samples` | `0` | 0 dùng toàn bộ split train; số dương giới hạn query train theo seed |
| `--aliases` | Đường dẫn alias ở trên | Chọn file ánh xạ, giữ file đã tồn tại |
| `--rebuild-corpus` | Tắt | Tạo lại structured corpus |

Token limits, precision và checkpointing trên chỉ áp dụng training; mining/evaluation và validation trong training dùng inference config riêng. Không checkpoint vision encoder. Wrapper train toàn bộ encoder, không có resume; mỗi lần chạy chuẩn bị lại manifest, mining và train từ đầu.

## Cấu hình các stage

Các setting sau thuộc script stage hoặc TrainingConfig, không phải tất cả đều được wrapper cung cấp.

| Setting | Mặc định | Vai trò |
| --- | --- | --- |
| Preparation --validation-fraction, --seed | 0.2, 42 | Chia split theo ảnh |
| Mining --top-n | 20 | Kích thước pool negatives |
| Training --learning-rate, --temperature | 2e-5, 0.02 | Learning rate và độ sắc logits contrastive |
| TrainingConfig weight_decay, max_gradient_norm | 0.01, 1.0 | AdamW và gradient clipping |
| TrainingConfig patience | 2 | Early stopping theo validation Recall@5 |
| Training --freeze-policy | none | Script hỗ trợ none, vision, all_but_projection |
| Training --max-train-samples | 100 ở script riêng | Khác wrapper mặc định 0; 0 dùng toàn bộ split train |
| Training --seed, --log-every-steps | 42, 10 | Chọn subset/lặp lại và tần suất log tiến độ |
| Training --resume | Không có | Script riêng hỗ trợ tiếp tục checkpoint |
| Evaluation --top-k | 5 | Số cặp ID đầu ra |
| Evaluation --checkpoint | runs/finetune_visual_bge/best.pt | Checkpoint chọn trên validation |
| Evaluation --questions | Không có | Mặc định validation từ manifest; file ngoài dùng để evaluate/predict |

Các script stage còn nhận đường dẫn manifest, ảnh, trọng số, output và negatives tương ứng. Giới hạn train queries không giới hạn validation/public test; wrapper vẫn mine toàn bộ split train trước khi chọn subset training.

## Artifact và đánh giá

- `data/finetune_visual_bge/label_aliases.json`, `input_rejections.json`: alias và báo cáo kiểm tra đầu vào.
- `data/finetune_visual_bge/manifest.json`, `manifest_rejected.json`: candidates, positives đã resolve, split, provenance/fingerprint và câu bị loại.
- `data/finetune_visual_bge/hard_negatives.json`: negatives từ model gốc.
- `runs/finetune_visual_bge/best.pt`, `last.pt`, `history.json`: checkpoints và lịch sử training.
- `runs/finetune_visual_bge/train.log`: config, batch/token shapes, loss, validation, checkpoint và bộ nhớ CUDA; ghi mới khi train bắt đầu.
- `runs/finetune_visual_bge/pipeline.log`: stage, heartbeat và lỗi; ghi nối tiếp.
- `runs/finetune_visual_bge/validation/predictions/finetune_visual_bge_predictions.json` và `runs/finetune_visual_bge/validation/metrics/finetune_visual_bge_metrics.json`: artifact validation nội bộ, tách khỏi kết quả public test.
- `outputs/predictions/finetune_visual_bge_predictions.json` và `outputs/metrics/finetune_visual_bge_metrics.json`: hai file kết quả public test duy nhất từ wrapper.

Evaluation yêu cầu tập câu hỏi có nhãn và không rỗng. Metrics chỉ gồm `f2`, `recall_at_1`, `recall_at_5`, `recall_at_10`. F2 dùng prediction top-k (mặc định 5); recall dùng thứ hạng tối đa max(10, top-k) cặp ID khác nhau trước khi cắt prediction, trung bình theo câu, nhãn rỗng tính 0. Evaluation nhận `--output-dir` thay cho `--output`; tên hai file cố định theo hướng. Không tự xóa file kết quả cũ. Cấu hình giảm VRAM vẫn là full fine-tuning, chưa bảo đảm chạy trên GPU 8 GB. CHECKPOINT.md hiện chưa ghi nhận F2 đã lưu cho hướng này; cập nhật README không thay đổi trạng thái hoàn thành.


Thư mục gốc kết quả chứa hai thư mục con: `predictions/` và `metrics/`; `--output-dir` (ở script có tham số này) chọn thư mục gốc, mặc định public test là `outputs/`.


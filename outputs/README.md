# Subtask 1 outputs

Fine-tuned Visualized-BGE ghi prediction vào `raw/finetuned_visual_bge_predictions.json` và metric
vào file cùng tên với hậu tố `_metrics.json`. Checkpoint nằm trong `runs/finetune_visual_bge/`.

Kết quả được chia thành hai nhóm:

- `raw/`: prediction nguyên bản của từng pipeline.
- `evaluation/`: metric, so sánh và nhận xét trên cùng 100 câu public test.

## Raw predictions

- `raw/official_baseline_predictions.json`: output của official Visualized-BGE baseline.
- `raw/img_to_txt_precomputed_predictions.json`: output BM25 dùng mô tả ảnh precomputed.

## Evaluation

- `evaluation/metrics.json`: dữ liệu metric có cấu trúc để xử lý bằng chương trình.
- `evaluation/summary.md`: bản tóm tắt kết quả và kết luận.


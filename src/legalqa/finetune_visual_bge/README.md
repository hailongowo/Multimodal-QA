# Fine-tune Visualized-BGE

Hướng này giữ nguyên official baseline và fine-tune Visualized-BGE bằng train data cùng `structured_corpus`.
Candidate retrieval gồm cả article và clause. Nhãn lỗi định dạng đã review nằm trong
`data/finetune_visual_bge/label_aliases.json`; alias giữ nguyên ID đầu ra cho metric chính thức.

## Quy trình

```powershell
python scripts/prepare_visual_bge_finetune.py
python scripts/mine_visual_bge_hard_negatives.py --device cuda
python scripts/train_visual_bge.py --device cuda --batch-size 4 --accumulation-steps 4
python scripts/evaluate_finetuned_visual_bge.py --device cuda
```

Preparation chia 80/20 theo `image_id`. Mining dùng checkpoint gốc, loại toàn bộ positive rồi giữ
top-20 hard negatives. Training lấy 4 negatives mỗi query, dùng multi-positive InfoNCE và chọn
`best.pt` theo validation Recall@5.

Mặc định lệnh train chỉ lấy tối đa 100 query từ split train để các vòng thử nghiệm chạy nhanh hơn.
Tập con được chọn xác định theo `--seed`, còn toàn bộ split validation vẫn được dùng để đánh giá.
Có thể đổi kích thước tập con hoặc dùng lại toàn bộ train data:

```powershell
# Thử nhanh với 50 mẫu, log sau mỗi 5 batch
python scripts/train_visual_bge.py --max-train-samples 50 --log-every-steps 5

# Train toàn bộ split train
python scripts/train_visual_bge.py --max-train-samples 0
```

Tiến độ được in ra console và ghi đồng thời vào
`runs/finetune_visual_bge/train.log`. Log gồm cấu hình hiệu lực, số mẫu train/validation, loss và
learning rate đang chạy, elapsed/ETA, validation Recall@5, checkpoint và trạng thái early stopping.

Artifact mặc định:

- `data/finetune_visual_bge/manifest.json`: candidate corpus, label resolution và split.
- `data/finetune_visual_bge/hard_negatives.json`: negative IDs, rank, score và fingerprint.
- `runs/finetune_visual_bge/{best.pt,last.pt,history.json,train.log}`: checkpoint, metric có cấu trúc
  và log tiến độ.
- `outputs/raw/finetuned_visual_bge_predictions.json`: prediction và file metric đi kèm.

Resume bằng `--resume runs/finetune_visual_bge/last.pt`. Nếu thiếu VRAM, dùng
`--freeze-policy vision`, giảm batch size và tăng gradient accumulation.

Prediction-only cho public test:

```powershell
python scripts/evaluate_finetuned_visual_bge.py `
  --questions "data/public_test data/vlsp_2025_public_test.json" `
  --images-dir "data/public_test data/public_test_images" `
  --output outputs/raw/finetuned_visual_bge_public_test.json
```

Nếu input không có `relevant_articles`, script chỉ ghi prediction và không tính metric.

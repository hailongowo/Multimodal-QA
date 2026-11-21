# Full pipeline, tham số và theo dõi tiến độ

Các lệnh chạy tại thư mục gốc project, dùng môi trường `.venv` đã chuẩn bị. Chọn một setting của một hướng mỗi lần. Mỗi full pipeline ghi hai file `outputs/predictions/{hướng}_predictions.json` và `outputs/metrics/{hướng}_metrics.json`, với tên hướng tương ứng `official_baseline`, `img_to_txt`, `structured_reranker`, `finetune_visual_bge`. Metrics gồm F2 và Recall@1/5/10; prediction lấy top 5, recall tính từ thứ hạng trước khi cắt top 5. Validation fine-tune lưu nội bộ tại `runs/finetune_visual_bge/validation/`. Lệnh track chạy trong PowerShell thứ hai sau khi pipeline tạo log.

```powershell
Set-Location "C:\Users\long3\Documents\Multimodal QA"
```

GPU là setting mặc định cho các lệnh full pipeline bên dưới: dùng `--device cuda` theo yêu cầu trong `AGENTS.md`. Chỉ chọn CPU khi không thể dùng GPU. Với baseline và dense Visualized-BGE của structured reranker, model tự chọn CUDA khi `torch.cuda.is_available()` trả về `True`, ngược lại tự dùng CPU; tham số wrapper không ép CUDA cho hai phần này. Các bước chuẩn bị dữ liệu và tính metric vẫn dùng CPU theo implementation hiện tại.

## Official baseline — Visualized-BGE

### Chạy full pipeline — GPU mặc định

```powershell
.\.venv\Scripts\python.exe src/legalqa/baselines/official/run_pipeline.py --device cuda
```

Setting hiện tại: chuẩn bị corpus baseline, tạo embedding, retrieval và evaluate public test. Wrapper nhận `--device`, `--reuse-index`, `--rebuild-corpus`, `--batch-size`, `--accumulation-steps`, `--epochs`, `--max-train-samples` nhưng không sử dụng các giá trị này để thay đổi pipeline baseline. `--device cuda` trong lệnh thể hiện setting GPU mặc định; Visualized-BGE thực tế tự chọn CUDA nếu khả dụng, nên tham số này không bảo đảm model đã chạy GPU.

Loader dùng package `visual_bge` đã cài hoặc tự tìm mã nguồn tại `.runtime/FlagEmbedding/research/visual_bge/visual_bge/modeling.py` nếu thiếu module; không cần đặt `PYTHONPATH` trong lệnh chạy.

Setting token inference tự động trong loader, không có tham số CLI: truncate văn bản corpus/query theo giới hạn tokenizer và position embedding, dành chỗ cho image patch tokens khi có ảnh. Văn bản dài bị cắt cuối theo mặc định tokenizer BGE-M3. Sau lỗi CUDA device-side assert, chạy lại lệnh full pipeline ở tiến trình Python mới; pipeline tạo lại embeddings từ đầu, không resume ở điều luật bị lỗi.

### Track progress

```powershell
Get-Content -Encoding UTF8 "runs/official_baseline/pipeline.log" -Tail 50 -Wait
```

## Image-to-text — BGE-M3 dense + reranker

### Chạy full pipeline — CUDA, xây lại index

```powershell
.\.venv\Scripts\python.exe src/legalqa/img_to_txt/run_pipeline.py --device cuda
```

Pipeline tạo structured corpus nếu thiếu, sau đó xây index, dense retrieval, rerank và evaluate public test. Cần có sẵn mô tả ảnh corpus và public test; pipeline không sinh mô tả ảnh.

| Tham số | Mặc định | Tác dụng |
| --- | --- | --- |
| `--device` | `cuda` | Device cho dense model và reranker; có thể dùng `cpu`. |
| `--reuse-index` | Không bật | Cho phép dùng lại index theo cơ chế kiểm tra cache của image-to-text. Nếu không bật, buộc xây lại index. |
| `--rebuild-corpus` | Không bật | Tạo lại `data/structured_corpus/structured_corpus.jsonl` dù đã tồn tại. |

Các tham số `--batch-size`, `--accumulation-steps`, `--epochs`, `--max-train-samples` được wrapper nhận nhưng không dùng cho hướng này.

### Setting CPU — chỉ khi không thể dùng GPU

```powershell
.\.venv\Scripts\python.exe src/legalqa/img_to_txt/run_pipeline.py --device cpu
```

### Setting dùng lại index

```powershell
.\.venv\Scripts\python.exe src/legalqa/img_to_txt/run_pipeline.py --device cuda --reuse-index
```

### Setting tạo lại corpus và index

```powershell
.\.venv\Scripts\python.exe src/legalqa/img_to_txt/run_pipeline.py --device cuda --rebuild-corpus
```

### Track progress

```powershell
Get-Content -Encoding UTF8 "runs/img_to_txt/pipeline.log" -Tail 50 -Wait
```

## Structured corpus + Visualized-BGE + reranker

### Chạy full pipeline — CUDA, xây lại index

```powershell
.\.venv\Scripts\python.exe src/legalqa/structured_reranker/run_pipeline.py --device cuda
```

Pipeline tạo structured corpus nếu thiếu, sau đó xây index, dense retrieval, rerank và evaluate public test.

| Tham số | Mặc định | Tác dụng |
| --- | --- | --- |
| `--device` | `cuda` | Device của reranker; không điều khiển device tải Visualized-BGE. `cpu` chỉ chuyển reranker sang CPU. |
| `--reuse-index` | Không bật | Cho phép dùng lại index hiện có; cache hướng này không kiểm tra fingerprint. Chỉ dùng khi corpus, trọng số và cấu hình index vẫn phù hợp. |
| `--rebuild-corpus` | Không bật | Tạo lại `data/structured_corpus/structured_corpus.jsonl` dù đã tồn tại. |

Các tham số `--batch-size`, `--accumulation-steps`, `--epochs`, `--max-train-samples` được wrapper nhận nhưng không dùng cho hướng này.

### Setting reranker dùng CPU — chỉ khi không thể dùng GPU

```powershell
.\.venv\Scripts\python.exe src/legalqa/structured_reranker/run_pipeline.py --device cpu
```

### Setting dùng lại index

```powershell
.\.venv\Scripts\python.exe src/legalqa/structured_reranker/run_pipeline.py --device cuda --reuse-index
```

### Setting tạo lại corpus và index

```powershell
.\.venv\Scripts\python.exe src/legalqa/structured_reranker/run_pipeline.py --device cuda --rebuild-corpus
```

### Track progress

```powershell
Get-Content -Encoding UTF8 "runs/structured_reranker/pipeline.log" -Tail 50 -Wait
```

## Fine-tune Visualized-BGE

### Chạy full pipeline — CUDA, cấu hình giảm VRAM, toàn bộ split train

```powershell
.\.venv\Scripts\python.exe src/legalqa/finetune_visual_bge/run_pipeline.py --device cuda --batch-size 1 --accumulation-steps 16 --hard-negative-count 1 --query-max-length 256 --document-max-length 512 --precision fp16 --gradient-checkpointing --epochs 5 --max-train-samples 0
```

Pipeline dùng structured corpus, kiểm tra đầu vào và chuẩn bị alias, tạo manifest/split, mine hard negatives, train, evaluate validation và evaluate public test. Mỗi lần chạy tạo lại manifest, negatives và train từ đầu; wrapper không có resume. Cấu hình giảm VRAM chưa được kiểm thử trên GPU 8 GB; full fine-tuning vẫn có thể OOM ở backward hoặc AdamW.

| Tham số | Mặc định | Tác dụng |
| --- | --- | --- |
| `--device` | `cuda` | Device cho mining, training và evaluation; CPU training dùng FP32. |
| `--batch-size` | `1` | Số query mỗi batch training; documents gồm các positive và hard negatives. |
| `--accumulation-steps` | `16` | Số batch tích lũy gradient trước cập nhật; không chia nhỏ forward documents. |
| `--hard-negative-count` | `1` | Số hard negative tối đa/query khi training; không thay pool mining. Cho phép 0. |
| `--query-max-length` | `256` | Giới hạn token văn bản query trong training, gồm special tokens; không gồm image tokens. |
| `--document-max-length` | `512` | Giới hạn token/document trong training, gồm special tokens; cắt phần cuối văn bản dài. |
| `--precision` | `fp16` | CUDA training chọn `fp16`, `bf16` (GPU hỗ trợ) hoặc `fp32`; loss tính FP32, FP16 dùng GradScaler. |
| `--gradient-checkpointing` / `--no-gradient-checkpointing` | Bật | Tính lại text encoder khi backward để giảm activation; tăng thời gian tính toán. Không bật checkpointing vision encoder. |
| `--epochs` | `5` | Số epoch training. |
| `--max-train-samples` | `0` | 0 dùng toàn bộ split train; số dương giới hạn query train, không giới hạn validation/public test. |
| `--aliases` | `data/finetune_visual_bge/label_aliases.json` | Giữ file alias hiện có hoặc tạo mặc định nếu thiếu. |
| `--rebuild-corpus` | Không bật | Tạo lại `data/structured_corpus/structured_corpus.jsonl`. |

Các tham số token, precision và checkpointing trên chỉ áp dụng training. Mining và evaluation giữ cấu hình inference riêng; validation trong training cũng không dùng giới hạn token training. Model vẫn train toàn bộ encoder, không đóng băng vision.

### Setting giới hạn 100 query train, 1 epoch

```powershell
.\.venv\Scripts\python.exe src/legalqa/finetune_visual_bge/run_pipeline.py --device cuda --batch-size 1 --accumulation-steps 16 --hard-negative-count 1 --query-max-length 256 --document-max-length 512 --precision fp16 --gradient-checkpointing --epochs 1 --max-train-samples 100
```

Lệnh vẫn chạy full pipeline, gồm mining và evaluate validation/public test.

### Setting giảm token document xuống 256

```powershell
.\.venv\Scripts\python.exe src/legalqa/finetune_visual_bge/run_pipeline.py --device cuda --document-max-length 256 --epochs 1 --max-train-samples 100
```

Các tham số không ghi dùng mặc định trong bảng. Giảm token có thể mất nội dung điều luật.

### Setting tạo lại corpus hoặc chỉ định alias

```powershell
.\.venv\Scripts\python.exe src/legalqa/finetune_visual_bge/run_pipeline.py --device cuda --rebuild-corpus --aliases "data/finetune_visual_bge/label_aliases.json"
```

### Track progress

```powershell
Get-Content -Encoding UTF8 "runs/finetune_visual_bge/pipeline.log" -Tail 50 -Wait
```

```powershell
Get-Content -Encoding UTF8 "runs/finetune_visual_bge/train.log" -Tail 50 -Wait
```

Training ghi `Batch start` kèm query IDs và số documents; `Tokenized before encoder` ghi shape `(batch, padded_tokens)` trước từng encoder. `Batch CUDA MiB` ghi allocated, reserved và peak sau mỗi batch; `Failure CUDA MiB` ghi bộ nhớ khi lỗi. Peak được reset đầu mỗi batch. `train.log` được ghi mới khi bắt đầu training; `pipeline.log` ghi nối tiếp.

```powershell
Get-Content -Encoding UTF8 "data/finetune_visual_bge/input_rejections.json"
Get-Content -Encoding UTF8 "data/finetune_visual_bge/manifest_rejected.json"
```

## Tham số theo dõi log

| Tham số / dấu mốc | Ý nghĩa |
| --- | --- |
| `-Encoding UTF8` | Đọc log tiếng Việt theo UTF-8. |
| `-Tail 50` | Hiển thị 50 dòng cuối khi bắt đầu theo dõi. |
| `-Wait` | Tiếp tục hiển thị dòng được ghi thêm; `Ctrl+C` tại cửa sổ track dừng theo dõi. |
| `START stage i/n`, `DONE` | Stage đang bắt đầu hoặc đã kết thúc. |
| `RUNNING` | Heartbeat khoảng 30 giây; không phải phần trăm tiến độ. |
| `FAILED`, `INTERRUPTED` | Pipeline lỗi hoặc bị ngắt. |
| `PIPELINE SUCCEEDED` | Các lệnh đã thoát thành công; cần đọc kết quả đánh giá trước khi xác nhận hướng hoàn thành. |

Log pipeline được ghi nối tiếp qua các lần chạy; đọc từ dòng `BEGIN` gần nhất để theo dõi lần chạy hiện tại.



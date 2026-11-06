# MLQA-TSR — Subtask 1

Code gồm phần dùng chung cho các hướng giải quyết Subtask 1 và baseline chính thức đặt riêng trong `baselines/official/`.

## Cấu trúc

```text
src/legalqa/
├── data/loaders.py          # Dùng chung: đọc JSON
├── evaluation/metrics.py    # Dùng chung: F2 chính thức
├── experiment.py            # Dùng chung: chạy truy xuất, xuất JSON, tính F2
└── baselines/
    └── official/           # Riêng baseline chính thức
        ├── preprocess.py   # Chuyển bảng HTML thành Markdown cho baseline
        ├── model.py        # Tải Visualized-BGE
        ├── embedding.py    # Xử lý bảng/marker ảnh và tạo embedding corpus
        ├── retrieval.py    # Dot product, top-5, ánh xạ ID, loại trùng
        └── pipeline.py     # Nạp model, lưu/đọc .npy, gọi phần chạy chung
scripts/run_official_baseline.py
scripts/build_official_baseline_data.py
tests/test_shared.py
tests/test_official_baseline.py
requirements.txt            # Toàn bộ dependency của project
```

Phần dùng chung không import model hoặc module của baseline. Các hướng mới có thể dùng lại việc đọc dữ liệu, xuất kết quả và đánh giá mà không cần cài Visualized-BGE.

## Thêm hướng giải quyết khác

Hướng fine-tune Visualized-BGE bằng train data, hard negatives và candidate article/clause từ
`structured_corpus` được mô tả tại `src/legalqa/finetune_visual_bge/README.md`. Hướng này độc lập với
official baseline và có các script prepare, mine, train, evaluate riêng.

Mỗi hướng tự chuẩn bị corpus, xử lý dữ liệu, ảnh, embedding và xếp hạng. Phần dùng chung chỉ yêu cầu một hàm:

```python
retrieve(query) -> list[dict]
```

- `query` gồm `id`, `image_id`, `question`; không có nhãn, đáp án hoặc lựa chọn.
- Kết quả là danh sách `{"law_id": ..., "article_id": ...}` theo thứ tự do phương pháp quyết định.
- Mỗi phương pháp tự quyết định số kết quả, cách chọn top-k và xử lý trùng. Phần chung không sắp xếp, loại trùng hoặc lấy bù kết quả.

Trong script riêng của phương pháp, thêm `src` vào đường dẫn import như `scripts/run_official_baseline.py`, đọc câu hỏi bằng `read_json()`, rồi gọi:

```python
from pathlib import Path

from legalqa.data.loaders import read_json
from legalqa.experiment import run_experiment

questions = read_json(Path("data/public_test data/vlsp_2025_public_test.json"))
metrics = run_experiment(questions, retrieve, Path("my_predictions.json"))
```

`retrieve` trong ví dụ là hàm do hướng xử lý mới cung cấp. `run_experiment()` ghi JSON có `id`, `image_id`, `question`, `relevant_articles`; tính và in F2; trả về `{"f2": ...}`. Câu hỏi đầu vào cần có nhãn `relevant_articles` để đánh giá. Hàm ghi đè file kết quả nếu đã tồn tại; thư mục cha cần tồn tại.

`read_json()` giữ nguyên cấu trúc và nội dung dữ liệu, không làm sạch văn bản hay gộp ID. Tiền xử lý đặc thù của từng phương pháp đặt trong code của phương pháp đó.

## Tiền xử lý riêng của official baseline

Tạo artifact văn bản riêng của official baseline trước khi chạy baseline:

```powershell
python scripts/build_official_baseline_data.py
```

Script đọc `data/law_db/vlsp2025_law.json`, chuyển HTML trong các marker bảng sang
Markdown và ghi `data/baselines/official/vlsp2025_law.json`. File đầu ra giữ nguyên schema
của file gốc và giữ nguyên marker ảnh; bước này không đọc hoặc biến đổi file ảnh.
Chỉ official baseline dùng artifact này. Pipeline image-to-text đọc trực tiếp corpus gốc.

## Chạy baseline chính thức

Baseline bám sát [retrieval.py chính thức](https://github.com/sonlam1102/VLSP2025-MLQA-TSR/blob/main/retrieval.py) cho Subtask 1. Model là Visualized-BGE với `BAAI/bge-m3` và `Visualized_m3.pth`.

Cài PyTorch phù hợp với máy theo [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/), rồi cài toàn bộ dependency của project:

```powershell
python -m pip install -r requirements.txt
```

`visual_bge` được cài từ thư mục `research/visual_bge` của FlagEmbedding theo [hướng dẫn Visualized-BGE](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/visual_bge). Bước tiền xử lý của official baseline sử dụng `pandas<3`, `lxml` và `tabulate` để chuyển bảng HTML sang Markdown theo cùng cách của baseline gốc.

Tải [Visualized_m3.pth](https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_m3.pth) và đặt tại `artifacts/models/Visualized_m3.pth`. Script đã trỏ đến dữ liệu hiện tại:

```text
data/law_db/vlsp2025_law.json
data/baselines/official/vlsp2025_law.json
data/law_db/images.fld/
data/public_test data/vlsp_2025_public_test.json
data/public_test data/public_test_images/
artifacts/models/Visualized_m3.pth
```

Chạy từ project root:

```powershell
python scripts/build_official_baseline_data.py
python scripts/run_official_baseline.py
```

Luồng baseline: đọc corpus đã được tiền xử lý; encode nội dung điều luật và từng ảnh; ghi `law_id.npy`, `law_embedding.npy`; nạp lại embedding; encode ảnh + câu hỏi; chọn 5 vector bằng tích vô hướng; loại cặp ID trùng. Phần dùng chung ghi `public_test_task_1.json` và in F2 với công thức có `1e-9`. Ba file đầu ra nằm trong `artifacts/official_baseline/` và được ghi đè khi chạy lại.

Giữ cách xử lý marker ảnh của bản gốc, không ghép tiêu đề vào nội dung và không gộp ID corpus. Sau khi chọn top-5 vector, baseline duyệt theo thứ tự corpus, loại trùng mà không lấy bù, nên có thể trả dưới 5 điều luật. Các quy tắc này chỉ thuộc baseline.

Điều chỉnh đường dẫn so với repo gốc: dùng file public test và ảnh tương ứng trong workspace. Script gốc đọc tên file private-test khi retrieval nhưng dùng ảnh và nhãn public-test; bản này dùng cùng một file public-test cho truy xuất và đánh giá.

Notebook `notebooks/01_law_db_eda.ipynb` dùng để khảo sát dữ liệu, không tham gia luồng baseline.

## Structured corpus + reranker (không dùng mô tả ảnh)

Pipeline thử nghiệm này giữ cách encode query bằng ảnh + câu hỏi của official Visualized-BGE,
nhưng tạo multi-vector index từ các block có cấu trúc thay vì dùng cả article làm một chunk.
Chunk mặc định có ngân sách 512 token, overlap một structural element; table được chia theo hàng.
Mọi trường `description_vlm` đều bị loại trước khi embedding. Dense retrieval lấy top-200 chunk,
gom max-score về top-50 article, rồi `BAAI/bge-reranker-v2-m3` chấm tối đa 3 chunk tốt nhất
của mỗi article trước khi trả top-5:

```powershell
python scripts/run_structured_reranker.py
```

Artifact được ghi riêng, không đè baseline:

- `runs/structured_reranker/structured_index.npz` (chunk metadata + embedding được publish atomically)
- `outputs/structured_reranker/dense_predictions.json`
- `outputs/structured_reranker/reranked_predictions.json`
- `outputs/structured_reranker/metrics.json`

Dùng `--rebuild-index` khi structured corpus hoặc cấu hình chunk thay đổi. Các ablation chính:
`--chunk-tokens`, `--chunk-overlap`, `--chunk-shortlist-size`, `--article-shortlist-size`,
`--chunks-per-article`, `--dense-batch-size`, `--reranker-batch-size`, và `--top-k`.

## Kiểm tra

```powershell
python -m unittest discover -s tests -v
```

Tất cả dependency được quản lý trong `requirements.txt`. Test baseline dùng model giả để kiểm tra embedding, top-5 và luồng file, không tải trọng số hoặc chạy model thật.

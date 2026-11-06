# Pipeline `img_to_txt`

Pipeline dùng `BAAI/bge-m3` để dense retrieval trên structured corpus có image
descriptions, sau đó rerank article shortlist bằng `BAAI/bge-reranker-v2-m3`.
Query là câu hỏi cộng test image description; pipeline không đọc ảnh và không còn BM25.

```powershell
python src/legalqa/img_to_txt/evaluate.py
```

Cache mặc định: `runs/img_to_txt/img_to_txt_index.npz`. Cache chỉ được dùng lại khi
fingerprint của model, cấu hình chunk, corpus và descriptions khớp. Dùng
`--rebuild-index` để buộc xây lại. Output nằm trong `outputs/img_to_txt/` gồm
`dense_predictions.json`, `reranked_predictions.json`, và `metrics.json`.

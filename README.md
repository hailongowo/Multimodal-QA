# MLQA-TSR — Subtask 1

## Mô tả bài toán

Subtask 1 là bài toán truy xuất điều luật liên quan đến câu hỏi đa phương thức về luật giao thông đường bộ Việt Nam. Với một câu hỏi bằng văn bản và ảnh đi kèm, hệ thống cần tìm các điều luật trong corpus cung cấp căn cứ pháp lý để trả lời câu hỏi.

Mỗi kết quả được xác định bằng cặp `law_id` và `article_id`, tương ứng với văn bản pháp luật và điều luật trong văn bản đó. Một câu hỏi có thể liên quan đến nhiều điều luật.

## Đầu vào

- `id`: mã định danh câu hỏi.
- `question`: nội dung câu hỏi bằng văn bản.
- `image_id`: định danh ảnh đi kèm câu hỏi.
- Corpus pháp luật: tập văn bản và các điều luật, có thể chứa văn bản, bảng và hình ảnh.

Nhãn `relevant_articles` trong dữ liệu có gán nhãn được dùng để huấn luyện hoặc đánh giá, không phải đầu vào khi truy xuất.

## Đầu ra

Với mỗi câu hỏi, hệ thống trả về danh sách `relevant_articles` chứa các cặp `law_id`, `article_id` được dự đoán là liên quan. Subtask 1 yêu cầu truy xuất căn cứ pháp lý, không yêu cầu sinh câu trả lời hoặc chọn đáp án trắc nghiệm.

Ví dụ cấu trúc kết quả dưới đây dùng các ID minh họa:

```json
[
  {
    "id": "question_example",
    "image_id": "image_example",
    "question": "Nội dung câu hỏi về tình huống giao thông trong ảnh",
    "relevant_articles": [
      {
        "law_id": "law_example",
        "article_id": "article_example"
      }
    ]
  }
]
```

Các ID điều luật trong kết quả thực tế phải khớp với corpus.

## Dữ liệu trong workspace

| Dữ liệu | Đường dẫn | Vai trò |
| --- | --- | --- |
| Corpus gốc và ảnh | `data/law_db/` | Nguồn văn bản pháp luật và hình ảnh kèm theo |
| Tập huấn luyện | `data/train data/` | Dữ liệu dùng để huấn luyện |
| Tập đánh giá | `data/public_test data/` | Câu hỏi, ảnh và nhãn dùng để đánh giá trên public test |
| Corpus có cấu trúc | `data/structured_corpus/structured_corpus.jsonl` | Biểu diễn corpus dùng cho các hướng giải quyết ngoài baseline theo quy định của project |

## Đánh giá

Kết quả được đánh giá bằng F2, tính riêng cho từng câu hỏi rồi lấy trung bình trên tập đánh giá. Với mỗi câu hỏi, tập dự đoán và tập nhãn được so sánh theo cặp `(law_id, article_id)`; các cặp trùng lặp chỉ được tính một lần.

- **Precision (P)**: số điều luật dự đoán đúng chia cho tổng số điều luật được dự đoán.
- **Recall (R)**: số điều luật dự đoán đúng chia cho tổng số điều luật trong nhãn.
- **F2**: `5 × P × R / (4 × P + R + 1e-9)`.

Nếu không có điều luật được dự đoán, precision và F2 của câu hỏi bằng 0. F2 đặt trọng số cho recall cao hơn precision, khuyến khích hệ thống tìm đủ các căn cứ pháp lý liên quan.

"""Text-only BGE-M3 embedding adapter."""
from __future__ import annotations
import numpy as np

class BGEM3TextEncoder:
    def __init__(self, tokenizer, model, device: str, batch_size: int = 8, max_length: int = 512):
        import torch
        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length

    @classmethod
    def load(cls, model_name="BAAI/bge-m3", device="cuda", batch_size=8, max_length=512):
        from transformers import AutoModel, AutoTokenizer
        return cls(AutoTokenizer.from_pretrained(model_name), AutoModel.from_pretrained(model_name), device, batch_size, max_length)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        rows = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                inputs = self.tokenizer(texts[start:start + self.batch_size], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                norms = pooled.norm(dim=1, keepdim=True)
                if self.torch.any(norms == 0):
                    raise ValueError("encoder produced a zero-norm embedding")
                rows.append((pooled / norms).float().cpu().numpy())
        return np.concatenate(rows).astype(np.float32, copy=False)

    def score_query(self, query: str, candidate_embeddings: np.ndarray) -> np.ndarray:
        vector = self.encode_texts([query])[0]
        matrix = np.asarray(candidate_embeddings)
        if matrix.ndim != 2 or matrix.shape[1] != vector.shape[0]:
            raise ValueError("candidate embedding dimensions do not match query")
        return matrix @ vector

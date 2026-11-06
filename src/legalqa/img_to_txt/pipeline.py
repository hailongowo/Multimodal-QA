"""Cached dense retrieval and cross-encoder reranking for image descriptions."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np

from .evaluation import build_dense_query
from ..structured_reranker.pipeline import build_predictions, rerank_articles, select_article_evidence

INDEX_FILE = "img_to_txt_index.npz"

def build_index_fingerprint(model_name: str, chunk_tokens: int, chunk_overlap: int, inputs: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps([model_name, chunk_tokens, chunk_overlap], separators=(",", ":")).encode())
    for path in inputs:
        digest.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()

def save_description_index(output_dir: Path, candidates: list[dict], embeddings: np.ndarray, fingerprint: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / INDEX_FILE
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, embeddings=np.asarray(embeddings), candidates_json=np.asarray(json.dumps(candidates, ensure_ascii=False)), fingerprint=np.asarray(fingerprint))
    temporary.replace(target)

def load_description_index(output_dir: Path, expected_fingerprint: str):
    path = output_dir / INDEX_FILE
    if not path.is_file(): return None
    try:
        with np.load(path, allow_pickle=False) as archive:
            if str(archive["fingerprint"].item()) != expected_fingerprint: return None
            candidates = json.loads(str(archive["candidates_json"].item()))
            embeddings = archive["embeddings"].copy()
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    if len(candidates) != len(embeddings):
        raise ValueError("candidate and embedding counts differ")
    return candidates, embeddings

def run_pipeline(questions, descriptions, candidates, candidate_embeddings, encoder, reranker, chunk_shortlist_size=200, article_shortlist_size=50, chunks_per_article=3, top_k=5):
    dense_rankings, reranked_rankings = [], []
    for question in questions:
        query = build_dense_query(question, descriptions)
        scores = encoder.score_query(query, candidate_embeddings)
        evidence = select_article_evidence(scores, candidates, chunk_shortlist_size, article_shortlist_size, chunks_per_article)
        dense_rankings.append(np.asarray([item["indices"][0] for item in evidence]))
        reranked_rankings.append(rerank_articles(query, candidates, evidence, reranker))
    return build_predictions(questions, dense_rankings, candidates, top_k), build_predictions(questions, reranked_rankings, candidates, top_k)

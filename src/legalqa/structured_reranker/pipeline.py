"""Core operations for description-free structured retrieval and reranking."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..finetune_visual_bge.corpus import build_candidate_units, read_structured_corpus
from .chunks import build_structured_chunks


INDEX_FILE = "structured_index.npz"


def load_structured_candidates(path: Path) -> list[dict]:
    """Load article/clause candidates without generated image descriptions."""
    units = build_candidate_units(
        read_structured_corpus(path),
        include_image_descriptions=False,
        include_clause_candidates=False,
    )
    return [
        {
            "law_id": unit.law_id,
            "label_id": unit.label_id,
            "unit_type": unit.unit_type,
            "text": unit.text,
        }
        for unit in units
    ]


def load_structured_chunks(
    path: Path,
    count_tokens,
    max_tokens: int = 512,
    overlap_elements: int = 1,
) -> list[dict]:
    """Build JSON-serializable multi-vector chunks from structured records."""
    return [
        asdict(chunk)
        for chunk in build_structured_chunks(
            read_structured_corpus(path),
            count_tokens=count_tokens,
            max_tokens=max_tokens,
            overlap_elements=overlap_elements,
        )
    ]


def shortlist_indices(scores: np.ndarray, top_n: int) -> np.ndarray:
    """Return stable descending indices, bounded by the corpus size."""
    values = np.asarray(scores)
    if values.ndim != 1:
        raise ValueError("dense scores must be one-dimensional")
    if top_n < 1:
        raise ValueError("top_n must be positive")
    return np.argsort(-values, kind="stable")[:top_n]


def rerank_shortlist(
    question: str,
    candidates: list[dict],
    dense_scores: np.ndarray,
    reranker,
    top_n: int = 50,
) -> np.ndarray:
    """Rerank the dense shortlist using question/candidate text pairs."""
    shortlist = shortlist_indices(dense_scores, top_n)
    pairs = [(question, candidates[int(index)]["text"]) for index in shortlist]
    rerank_scores = np.asarray(reranker.score_pairs(pairs))
    if rerank_scores.shape != (len(shortlist),):
        raise ValueError("reranker returned a score count different from the shortlist")
    order = np.argsort(-rerank_scores, kind="stable")
    return shortlist[order]


def select_article_evidence(
    dense_scores: np.ndarray,
    chunks: list[dict],
    chunk_top_n: int = 200,
    article_top_n: int = 50,
    chunks_per_article: int = 3,
) -> list[dict]:
    """Group top dense chunks into an article shortlist using max aggregation."""
    if len(dense_scores) != len(chunks):
        raise ValueError("dense score and chunk counts differ")
    if article_top_n < 1 or chunks_per_article < 1:
        raise ValueError("article_top_n and chunks_per_article must be positive")
    grouped: dict[tuple[str, str], dict] = {}
    for raw_index in shortlist_indices(dense_scores, chunk_top_n):
        index = int(raw_index)
        chunk = chunks[index]
        key = (chunk["law_id"], chunk["label_id"])
        if key not in grouped:
            grouped[key] = {
                "key": key,
                "dense_score": float(dense_scores[index]),
                "indices": [],
            }
        if len(grouped[key]["indices"]) < chunks_per_article:
            grouped[key]["indices"].append(index)
    evidence = list(grouped.values())[:article_top_n]
    for item in evidence:
        item["indices"] = tuple(item["indices"])
    return evidence


def rerank_articles(
    question: str,
    chunks: list[dict],
    evidence: list[dict],
    reranker,
) -> np.ndarray:
    """Cross-encode local evidence and rank articles by their maximum score."""
    pairs = []
    locations = []
    for article_position, item in enumerate(evidence):
        for index in item["indices"]:
            pairs.append((question, chunks[index]["text"]))
            locations.append((article_position, index))
    scores = np.asarray(reranker.score_pairs(pairs))
    if scores.shape != (len(pairs),):
        raise ValueError("reranker returned a score count different from article evidence")

    best = [None] * len(evidence)
    for score, (article_position, index) in zip(scores, locations):
        if best[article_position] is None or score > best[article_position][0]:
            best[article_position] = (float(score), index)
    order = sorted(range(len(best)), key=lambda position: -best[position][0])
    return np.asarray([best[position][1] for position in order], dtype=int)


def build_predictions(
    questions: list[dict],
    rankings: list[np.ndarray],
    candidates: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Convert candidate rankings to the official prediction schema."""
    if len(questions) != len(rankings):
        raise ValueError("question and ranking counts differ")
    predictions = []
    for question, ranking in zip(questions, rankings):
        references = []
        seen = set()
        for raw_index in ranking:
            candidate = candidates[int(raw_index)]
            key = (candidate["law_id"], candidate["label_id"])
            if key in seen:
                continue
            seen.add(key)
            references.append({"law_id": key[0], "article_id": key[1]})
            if len(references) == top_k:
                break
        predictions.append(
            {
                "id": question["id"],
                "image_id": question["image_id"],
                "question": question["question"],
                "relevant_articles": references,
            }
        )
    return predictions


def run_pipeline(
    questions: list[dict],
    images_dir: Path,
    candidates: list[dict],
    candidate_embeddings: np.ndarray,
    retriever,
    reranker,
    chunk_shortlist_size: int = 200,
    article_shortlist_size: int = 50,
    chunks_per_article: int = 3,
    top_k: int = 5,
) -> tuple[list[dict], list[dict]]:
    """Run dense retrieval and reranking without exposing labels to either model."""
    dense_rankings = []
    reranked_rankings = []
    for question in questions:
        scores = retriever.score_query(
            images_dir / f"{question['image_id']}.jpg",
            question["question"],
            candidate_embeddings,
        )
        evidence = select_article_evidence(
            scores,
            candidates,
            chunk_top_n=chunk_shortlist_size,
            article_top_n=article_shortlist_size,
            chunks_per_article=chunks_per_article,
        )
        dense_rankings.append(np.asarray([item["indices"][0] for item in evidence]))
        reranked_rankings.append(rerank_articles(question["question"], candidates, evidence, reranker))
    return (
        build_predictions(questions, dense_rankings, candidates, top_k),
        build_predictions(questions, reranked_rankings, candidates, top_k),
    )


def save_index(output_dir: Path, candidates: list[dict], embeddings: np.ndarray) -> None:
    """Atomically publish aligned structured candidates and dense embeddings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / INDEX_FILE
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    candidates_json = json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            embeddings=np.asarray(embeddings),
            candidates_json=np.asarray(candidates_json),
        )
    temporary.replace(index_path)


def load_index(output_dir: Path) -> tuple[list[dict], np.ndarray]:
    """Load the aligned candidate metadata and dense matrix."""
    with np.load(output_dir / INDEX_FILE, allow_pickle=False) as archive:
        candidates = json.loads(str(archive["candidates_json"].item()))
        embeddings = archive["embeddings"].copy()
    if len(candidates) != len(embeddings):
        raise ValueError("candidate and embedding counts differ")
    return candidates, embeddings


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


class VisualizedBGERetriever:
    """Thin inference adapter retaining the official baseline encode calls."""

    def __init__(self, model, max_length: int = 512, batch_size: int = 8):
        self.model = model
        self.max_length = max_length
        self.batch_size = batch_size
        self.model.eval()

    @classmethod
    def load(cls, model_weight: Path, max_length: int = 512, batch_size: int = 8):
        from ..baselines.official.model import load_model

        return cls(load_model(model_weight), max_length=max_length, batch_size=batch_size)

    def count_tokens(self, text: str) -> int:
        encoded = self.model.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
        )
        return len(encoded["input_ids"])

    def encode_candidates(self, candidates: list[dict]) -> np.ndarray:
        rows = []
        for start in range(0, len(candidates), self.batch_size):
            texts = [candidate["text"] for candidate in candidates[start:start + self.batch_size]]
            tokens = self.model.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.model.device)
            rows.append(_to_numpy(self.model.encode_text(tokens)))
        return np.concatenate(rows, axis=0)

    def score_query(
        self,
        image_path: Path,
        question: str,
        candidate_embeddings: np.ndarray,
    ) -> np.ndarray:
        query = _to_numpy(self.model.encode(image=str(image_path), text=question))
        return (query @ candidate_embeddings.T).reshape(-1)


class CrossEncoderReranker:
    """Lazy Hugging Face sequence-classification reranker."""

    def __init__(self, tokenizer, model, device: str, batch_size: int = 4, max_length: int = 1024):
        import torch

        self.torch = torch
        self.tokenizer = tokenizer
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.max_length = max_length

    @classmethod
    def load(
        cls,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        batch_size: int = 4,
        max_length: int = 1024,
    ):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return cls(tokenizer, model, device, batch_size, max_length)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = []
        with self.torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start:start + self.batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**inputs, return_dict=True).logits.reshape(-1)
                scores.extend(logits.float().cpu().tolist())
        return np.asarray(scores, dtype=np.float32)

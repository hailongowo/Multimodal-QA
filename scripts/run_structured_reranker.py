"""Run structured-corpus Visualized-BGE retrieval with cross-encoder reranking."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.data.loaders import read_json
from legalqa.evaluation.metrics import eval_retrieval
from legalqa.structured_reranker.pipeline import (
    CrossEncoderReranker,
    INDEX_FILE,
    VisualizedBGERetriever,
    build_predictions,
    load_index,
    load_structured_chunks,
    rerank_articles,
    select_article_evidence,
    save_index,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/structured_corpus/structured_corpus.jsonl")
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data/public_test data/vlsp_2025_public_test.json",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=ROOT / "data/public_test data/public_test_images",
    )
    parser.add_argument("--model-weight", type=Path, default=ROOT / "artifacts/models/Visualized_m3.pth")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "runs/structured_reranker")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/structured_reranker")
    parser.add_argument(
        "--official-baseline",
        type=Path,
        default=ROOT / "outputs/raw/official_baseline_predictions.json",
    )
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=1)
    parser.add_argument("--chunk-shortlist-size", type=int, default=200)
    parser.add_argument("--article-shortlist-size", type=int, default=50)
    parser.add_argument("--chunks-per-article", type=int, default=3)
    parser.add_argument("--dense-batch-size", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--reranker-batch-size", type=int, default=4)
    parser.add_argument("--reranker-max-length", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _make_visualized_retriever(
    model_weight: Path,
    max_length: int,
    batch_size: int,
) -> VisualizedBGERetriever:
    try:
        return VisualizedBGERetriever.load(model_weight, max_length, batch_size)
    except ModuleNotFoundError as error:
        if error.name != "visual_bge":
            raise
        vendored = ROOT / ".runtime/FlagEmbedding/research/visual_bge"
        if not vendored.is_dir():
            raise
        sys.path.insert(0, str(vendored))
        return VisualizedBGERetriever.load(model_weight, max_length, batch_size)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.article_shortlist_size < args.top_k or args.top_k < 1:
        raise ValueError("article-shortlist-size must be at least top-k and top-k must be positive")

    questions = read_json(args.questions)
    retriever = _make_visualized_retriever(
        args.model_weight,
        max_length=args.chunk_tokens,
        batch_size=args.dense_batch_size,
    )
    index_exists = (args.index_dir / INDEX_FILE).is_file()
    if args.rebuild_index or not index_exists:
        candidates = load_structured_chunks(
            args.corpus,
            retriever.count_tokens,
            max_tokens=args.chunk_tokens,
            overlap_elements=args.chunk_overlap,
        )
        print(f"Encoding {len(candidates)} description-free structured chunks...")
        embeddings = retriever.encode_candidates(candidates)
        save_index(args.index_dir, candidates, embeddings)
    else:
        candidates, embeddings = load_index(args.index_dir)
        print(f"Loaded {len(candidates)} cached structured chunks.")

    dense_scores = []
    for index, question in enumerate(questions, 1):
        dense_scores.append(
            retriever.score_query(
                args.images_dir / f"{question['image_id']}.jpg",
                question["question"],
                embeddings,
            )
        )
        print(f"Dense query {index}/{len(questions)}", flush=True)

    del retriever
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    reranker = CrossEncoderReranker.load(
        args.reranker_model,
        device=args.device,
        batch_size=args.reranker_batch_size,
        max_length=args.reranker_max_length,
    )
    dense_rankings = []
    reranked_rankings = []

    for index, (question, scores) in enumerate(zip(questions, dense_scores), 1):
        evidence = select_article_evidence(
            scores,
            candidates,
            chunk_top_n=args.chunk_shortlist_size,
            article_top_n=args.article_shortlist_size,
            chunks_per_article=args.chunks_per_article,
        )
        dense_rankings.append(np.asarray([item["indices"][0] for item in evidence]))
        reranked_rankings.append(
            rerank_articles(question["question"], candidates, evidence, reranker)
        )
        print(f"Reranked query {index}/{len(questions)}", flush=True)

    dense_predictions = build_predictions(questions, dense_rankings, candidates, args.top_k)
    reranked_predictions = build_predictions(questions, reranked_rankings, candidates, args.top_k)
    _write_json(args.output_dir / "dense_predictions.json", dense_predictions)
    _write_json(args.output_dir / "reranked_predictions.json", reranked_predictions)

    metrics = {
        "chunk_count": len(candidates),
        "article_count": len({(item["law_id"], item["label_id"]) for item in candidates}),
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap": args.chunk_overlap,
        "chunk_shortlist_size": args.chunk_shortlist_size,
        "article_shortlist_size": args.article_shortlist_size,
        "chunks_per_article": args.chunks_per_article,
        "top_k": args.top_k,
        "image_descriptions_used": False,
    }
    if all("relevant_articles" in question for question in questions):
        metrics["structured_dense"] = {
            key: float(value) for key, value in eval_retrieval(dense_predictions, questions).items()
        }
        metrics["structured_reranked"] = {
            key: float(value) for key, value in eval_retrieval(reranked_predictions, questions).items()
        }
        if args.official_baseline.is_file():
            official = read_json(args.official_baseline)
            metrics["official_baseline"] = {
                key: float(value) for key, value in eval_retrieval(official, questions).items()
            }
    _write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

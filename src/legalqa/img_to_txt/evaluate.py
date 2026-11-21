"""Run BGE-M3 retrieval and cross-encoder reranking over image descriptions."""
from __future__ import annotations
import argparse, gc, json, sys
from pathlib import Path
if __package__ in (None, ""): sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from legalqa.data.loaders import read_json
from legalqa.evaluation.results import save_results
from legalqa.img_to_txt.chunks import load_description_chunks
from legalqa.img_to_txt.encoder import BGEM3TextEncoder
from legalqa.img_to_txt.evaluation import load_test_descriptions
from legalqa.img_to_txt.pipeline import build_index_fingerprint, load_description_index, prepare_dense_stage, rerank_dense_stage, save_description_index
from legalqa.structured_reranker.pipeline import CrossEncoderReranker
ROOT = Path(__file__).resolve().parents[3]

def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=ROOT / "data/structured_corpus/structured_corpus.jsonl")
    p.add_argument("--descriptions", type=Path, default=ROOT / "data/img_to_txt/image_descriptions.json")
    p.add_argument("--test-descriptions", type=Path, default=ROOT / "data/img_to_txt/img_to_txt_test_descriptions.json")
    p.add_argument("--questions", type=Path, default=ROOT / "data/public_test data/vlsp_2025_public_test.json")
    p.add_argument("--dense-model", default="BAAI/bge-m3"); p.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--index-dir", type=Path, default=ROOT / "runs/img_to_txt"); p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--chunk-tokens", type=int, default=512); p.add_argument("--chunk-overlap", type=int, default=1)
    p.add_argument("--chunk-shortlist-size", type=int, default=200); p.add_argument("--article-shortlist-size", type=int, default=50); p.add_argument("--chunks-per-article", type=int, default=3)
    p.add_argument("--dense-batch-size", type=int, default=8); p.add_argument("--reranker-batch-size", type=int, default=4); p.add_argument("--reranker-max-length", type=int, default=1024)
    p.add_argument("--top-k", type=int, default=5); p.add_argument("--device", default="cuda"); p.add_argument("--rebuild-index", action="store_true")
    return p

def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.top_k < 1 or a.article_shortlist_size < max(10, a.top_k): raise ValueError("article-shortlist-size must be at least max(10, top-k) and top-k must be positive")
    fingerprint = build_index_fingerprint(a.dense_model, a.chunk_tokens, a.chunk_overlap, [a.corpus, a.descriptions])
    encoder = BGEM3TextEncoder.load(a.dense_model, a.device, a.dense_batch_size, a.chunk_tokens)
    cached = None if a.rebuild_index else load_description_index(a.index_dir, fingerprint)
    if cached is None:
        candidates = load_description_chunks(a.corpus, a.descriptions, encoder.count_tokens, a.chunk_tokens, a.chunk_overlap)
        embeddings = encoder.encode_texts([x["text"] for x in candidates]); save_description_index(a.index_dir, candidates, embeddings, fingerprint)
    else: candidates, embeddings = cached
    questions = read_json(a.questions); descriptions = load_test_descriptions(a.test_descriptions)
    queries, dense_scores = prepare_dense_stage(questions, descriptions, embeddings, encoder)
    del encoder
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass
    reranker = CrossEncoderReranker.load(a.reranker_model, a.device, a.reranker_batch_size, a.reranker_max_length)
    dense, reranked = rerank_dense_stage(questions, queries, dense_scores, candidates, reranker, a.chunk_shortlist_size, a.article_shortlist_size, a.chunks_per_article, max(10, a.top_k))
    metrics = save_results(reranked, questions, reranked, a.output_dir, "img_to_txt", a.top_k)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0
if __name__ == "__main__": raise SystemExit(main())




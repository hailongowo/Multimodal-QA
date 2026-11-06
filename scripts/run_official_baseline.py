"""Run the official Subtask 1 baseline with this workspace's data paths."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalqa.baselines.official.pipeline import make_embedding, retrieval

LAW_PATH = ROOT / "data/baselines/official/vlsp2025_law.json"
LAW_IMAGE_DIR = ROOT / "data/law_db/images.fld"
QUESTION_PATH = ROOT / "data/public_test data/vlsp_2025_public_test.json"
QUESTION_IMAGE_DIR = ROOT / "data/public_test data/public_test_images"
MODEL_WEIGHT = ROOT / "artifacts/models/Visualized_m3.pth"
OUTPUT_DIR = ROOT / "artifacts/official_baseline"


def main():
    make_embedding(LAW_PATH, LAW_IMAGE_DIR, MODEL_WEIGHT, OUTPUT_DIR)
    retrieval(QUESTION_PATH, QUESTION_IMAGE_DIR, MODEL_WEIGHT, OUTPUT_DIR)


if __name__ == "__main__":
    main()

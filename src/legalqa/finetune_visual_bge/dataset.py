"""Fine-tuning examples and false-negative-safe collation."""

from pathlib import Path
import random


class FineTuneDataset:
    def __init__(self, manifest: dict, negatives: dict, images_dir: Path, split="train",
                 negative_count=4, seed=42, max_samples=None):
        if max_samples is not None and max_samples < 1:
            raise ValueError("max_samples must be at least 1 or None")
        self.candidates = manifest["candidates"]
        self.images_dir = images_dir
        self.negative_count = negative_count
        self.seed = seed
        self.epoch = 0
        allowed = set(manifest["splits"][split])
        self.questions = [q for q in manifest["questions"] if q["id"] in allowed]
        if split == "train" and max_samples is not None and len(self.questions) > max_samples:
            indexes = list(range(len(self.questions)))
            random.Random(seed).shuffle(indexes)
            selected = set(indexes[:max_samples])
            self.questions = [
                question for index, question in enumerate(self.questions) if index in selected
            ]
        candidate_index = {(c["law_id"], c["label_id"]): index for index, c in enumerate(self.candidates)}
        pools = {row["id"]: row["negatives"] for row in negatives["queries"]}
        self.negative_pools = {}
        for question in self.questions:
            positive_indexes = {p["candidate_index"] for p in question["positives"]}
            indexes = []
            for item in pools.get(question["id"], []):
                key = (item["law_id"], item["label_id"])
                if key not in candidate_index:
                    raise ValueError(f"unknown negative candidate {key} for {question['id']}")
                if candidate_index[key] not in positive_indexes:
                    indexes.append(candidate_index[key])
            self.negative_pools[question["id"]] = indexes

    def __len__(self):
        return len(self.questions)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __getitem__(self, index):
        question = self.questions[index]
        seed = f"{self.seed}:{self.epoch}:{question['id']}"
        pool = list(self.negative_pools[question["id"]])
        random.Random(seed).shuffle(pool)
        return {
            "id": question["id"], "question": question["question"],
            "image_path": self.images_dir / f"{question['image_id']}.jpg",
            "positive_indexes": [p["candidate_index"] for p in question["positives"]],
            "negative_indexes": pool[:self.negative_count],
        }


def collate_training_batch(examples: list[dict], candidates: list[dict]) -> dict:
    import torch
    document_indexes = []
    for example in examples:
        for index in example["positive_indexes"] + example["negative_indexes"]:
            if index not in document_indexes:
                document_indexes.append(index)
    positive_mask = torch.zeros((len(examples), len(document_indexes)), dtype=torch.bool)
    valid_mask = torch.ones_like(positive_mask)
    for row, example in enumerate(examples):
        positive_keys = {(candidates[i]["law_id"], candidates[i]["label_id"]) for i in example["positive_indexes"]}
        for column, index in enumerate(document_indexes):
            key = (candidates[index]["law_id"], candidates[index]["label_id"])
            positive_mask[row, column] = key in positive_keys
    return {
        "ids": [e["id"] for e in examples],
        "image_paths": [e["image_path"] for e in examples],
        "questions": [e["question"] for e in examples],
        "document_indexes": document_indexes,
        "document_texts": [candidates[index]["text"] for index in document_indexes],
        "positive_mask": positive_mask,
        "valid_mask": valid_mask,
    }

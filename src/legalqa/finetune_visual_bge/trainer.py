"""Training loop for multi-positive Visualized-BGE retrieval."""

from dataclasses import asdict, dataclass
import json
import logging
import math
from pathlib import Path
import random
import time

from .artifacts import fingerprint_json, write_json_atomic
from .checkpoints import load_checkpoint, save_checkpoint
from .dataset import FineTuneDataset, collate_training_batch
from .loss import multi_positive_info_nce
from .model import load_visualized_bge


@dataclass
class TrainingConfig:
    manifest: Path
    negatives: Path
    images_dir: Path
    weights: Path
    output_dir: Path
    model_name: str = "BAAI/bge-m3"
    device: str = "cuda"
    seed: int = 42
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    temperature: float = 0.02
    hard_negative_count: int = 4
    accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    freeze_policy: str = "none"
    patience: int = 2
    max_train_samples: int | None = 100
    log_every_steps: int = 10
    resume: Path | None = None


def _validate_config(config: TrainingConfig) -> None:
    if config.max_train_samples is not None and config.max_train_samples < 1:
        raise ValueError("max_train_samples must be at least 1 or None")
    if config.log_every_steps < 1:
        raise ValueError("log_every_steps must be at least 1")


def _create_run_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"legalqa.finetune_visual_bge.train.{id(output_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in logger.handlers[:]:
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def _format_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def train(config: TrainingConfig, adapter=None, validation_fn=None) -> dict:
    import torch
    _validate_config(config)
    logger = _create_run_logger(config.output_dir)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    try:
        effective_config = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        }
        logger.info("Training run started config=%s", json.dumps(effective_config, ensure_ascii=False))
        manifest = json.loads(config.manifest.read_text(encoding="utf-8"))
        negatives = json.loads(config.negatives.read_text(encoding="utf-8"))
        fingerprints = {"manifest": fingerprint_json(manifest), "negatives": fingerprint_json(negatives)}
        adapter = adapter or load_visualized_bge(config.model_name, config.weights, config.device, config.freeze_policy)
        dataset = FineTuneDataset(
            manifest,
            negatives,
            config.images_dir,
            negative_count=config.hard_negative_count,
            seed=config.seed,
            max_samples=config.max_train_samples,
        )
        logger.info(
            "Data ready train_samples=%d validation_samples=%d candidates=%d",
            len(dataset),
            len(manifest["splits"].get("validation", [])),
            len(manifest["candidates"]),
        )
        optimizer = torch.optim.AdamW((p for p in adapter.backend.parameters() if p.requires_grad),
                                      lr=config.learning_rate, weight_decay=config.weight_decay)
        start_epoch, global_step, best_metric, history = 0, 0, float("-inf"), []
        if config.resume:
            state = load_checkpoint(config.resume, adapter, optimizer, expected_fingerprints=fingerprints)
            start_epoch, global_step = state["epoch"] + 1, state["global_step"]
            best_metric, history = state["best_metric"], state["history"]
            logger.info("Resumed checkpoint=%s start_epoch=%d global_step=%d",
                        config.resume, start_epoch + 1, global_step)
        stale_epochs = 0
        run_started = time.monotonic()
        total_batches = math.ceil(len(dataset) / config.batch_size)
        for epoch in range(start_epoch, config.epochs):
            epoch_started = time.monotonic()
            dataset.set_epoch(epoch)
            adapter.backend.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            logger.info("Epoch started epoch=%d/%d batches=%d", epoch + 1, config.epochs, total_batches)
            for start in range(0, len(dataset), config.batch_size):
                batch_number = (start // config.batch_size) + 1
                examples = [dataset[index] for index in range(start, min(start + config.batch_size, len(dataset)))]
                batch = collate_training_batch(examples, manifest["candidates"])
                queries = adapter.forward_queries(batch["image_paths"], batch["questions"], training=True)
                documents = adapter.forward_documents(batch["document_texts"])
                loss = multi_positive_info_nce(queries, documents, batch["positive_mask"].to(adapter.device),
                                               batch["valid_mask"].to(adapter.device), config.temperature)
                (loss / config.accumulation_steps).backward()
                losses.append(float(loss.detach().cpu()))
                if batch_number % config.accumulation_steps == 0 or start + config.batch_size >= len(dataset):
                    torch.nn.utils.clip_grad_norm_(adapter.backend.parameters(), config.max_gradient_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                if batch_number % config.log_every_steps == 0 or batch_number == total_batches:
                    elapsed = time.monotonic() - epoch_started
                    eta = (elapsed / batch_number) * (total_batches - batch_number)
                    logger.info(
                        "Progress epoch=%d/%d batch=%d/%d loss=%.6f lr=%.8g elapsed=%s eta=%s",
                        epoch + 1,
                        config.epochs,
                        batch_number,
                        total_batches,
                        sum(losses) / len(losses),
                        optimizer.param_groups[0]["lr"],
                        _format_duration(elapsed),
                        _format_duration(eta),
                    )
            if validation_fn is None:
                from .evaluation import validation_recall_at_5
                metric = float(validation_recall_at_5(adapter, manifest, config.images_dir))
            else:
                metric = float(validation_fn(adapter, manifest))
            train_loss = sum(losses) / max(len(losses), 1)
            row = {"epoch": epoch, "train_loss": train_loss, "validation_recall_at_5": metric}
            history.append(row)
            logger.info(
                "Epoch completed epoch=%d/%d train_loss=%.6f validation_recall_at_5=%.6f elapsed=%s",
                epoch + 1,
                config.epochs,
                train_loss,
                metric,
                _format_duration(time.monotonic() - epoch_started),
            )
            save_checkpoint(config.output_dir / "last.pt", adapter, optimizer, None, None, epoch=epoch,
                            global_step=global_step, best_metric=max(best_metric, metric), history=history,
                            config=effective_config, fingerprints=fingerprints)
            logger.info("Saved last checkpoint path=%s", config.output_dir / "last.pt")
            if metric > best_metric:
                best_metric, stale_epochs = metric, 0
                save_checkpoint(config.output_dir / "best.pt", adapter, optimizer, None, None, epoch=epoch,
                                global_step=global_step, best_metric=best_metric, history=history,
                                config=effective_config, fingerprints=fingerprints)
                logger.info("Saved best checkpoint path=%s metric=%.6f",
                            config.output_dir / "best.pt", best_metric)
            else:
                stale_epochs += 1
                logger.info("No validation improvement stale_epochs=%d/%d", stale_epochs, config.patience)
                if stale_epochs >= config.patience:
                    logger.info("Early stopping triggered at epoch=%d", epoch + 1)
                    break
        write_json_atomic(config.output_dir / "history.json", history)
        logger.info("Training run completed best_metric=%.6f global_step=%d elapsed=%s",
                    best_metric, global_step, _format_duration(time.monotonic() - run_started))
        return {"best_metric": best_metric, "global_step": global_step, "history": history}
    except Exception:
        logger.exception("Training run failed")
        raise
    finally:
        _close_logger(logger)

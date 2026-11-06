"""Atomic resumable training checkpoints."""

from pathlib import Path
import random


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, *, epoch: int, global_step: int,
                    best_metric: float, history: list, config: dict, fingerprints: dict) -> None:
    import numpy as np
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None, "epoch": epoch, "global_step": global_step,
        "best_metric": best_metric, "history": history, "config": config, "fingerprints": fingerprints,
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None,
                    expected_fingerprints: dict | None = None) -> dict:
    import numpy as np
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (expected_fingerprints or {}).items():
        if state.get("fingerprints", {}).get(key) != expected:
            raise ValueError(f"incompatible checkpoint {key}")
    model.load_state_dict(state["model"])
    if optimizer is not None and state["optimizer"] is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state

"""Narrow differentiable adapter around the upstream Visualized-BGE model."""

from pathlib import Path
import importlib.util
import sys


INFERENCE_BATCH_SIZE = 4


def ensure_visual_bge_available():
    """Support the local upstream checkout when pip installed metadata only."""
    if importlib.util.find_spec("visual_bge") is not None:
        return
    root = Path(__file__).resolve().parents[3]
    source = root / ".runtime/FlagEmbedding/research/visual_bge"
    if (source / "visual_bge/modeling.py").is_file():
        sys.path.insert(0, str(source))
        return
    raise ModuleNotFoundError(
        "visual_bge has no importable module. Expected installed package or local source at "
        f"{source}. A metadata-only pip installation is insufficient.",
        name="visual_bge",
    )


class VisualizedBGEAdapter:
    def __init__(self, backend, tokenize, load_images, device="cpu", freeze_policy="none"):
        import torch
        self.torch = torch
        self.backend = backend.to(device)
        self.tokenize = tokenize
        self.load_images = load_images
        self.device = torch.device(device)
        self.query_max_length = None
        self.document_max_length = None
        self.token_stats = {}
        if freeze_policy not in {"none", "vision", "all_but_projection"}:
            raise ValueError(f"unknown freeze policy: {freeze_policy}")
        if freeze_policy == "vision":
            for parameter in self.backend.model_visual.parameters():
                parameter.requires_grad = False
        elif freeze_policy == "all_but_projection":
            for parameter in self.backend.parameters():
                parameter.requires_grad = False
            for parameter in self.backend.visual_proj.parameters():
                parameter.requires_grad = True
        if not any(parameter.requires_grad for parameter in self.backend.parameters()):
            raise ValueError("freeze policy left no trainable parameters")

    def configure_training(self, query_max_length, document_max_length, gradient_checkpointing):
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        if gradient_checkpointing and not getattr(self, "_text_checkpointing", False):
            from torch.utils.checkpoint import checkpoint
            original_forward = self.backend.bge_encoder.forward

            def checkpointed_forward(*args, **kwargs):
                if self.backend.training and self.torch.is_grad_enabled():
                    return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
                return original_forward(*args, **kwargs)

            self.backend.bge_encoder.forward = checkpointed_forward
            self._text_checkpointing = True

    def _tokenize(self, texts, kind, max_length):
        if self.backend.training and max_length is not None:
            tokens = self.tokenize(texts, self.device, max_length=max_length)
        else:
            tokens = self.tokenize(texts, self.device)
        self.token_stats[kind] = tuple(tokens["input_ids"].shape)
        observer = getattr(self, "token_observer", None)
        if observer is not None and self.backend.training:
            observer(kind, self.token_stats[kind])
        return tokens

    def forward_documents(self, texts):
        values = self.backend.encode_text(self._tokenize(texts, "documents", self.document_max_length))
        return self.torch.nn.functional.normalize(values, dim=-1)

    def forward_queries(self, image_paths, texts, training=True):
        images = self.load_images(image_paths, training, self.device)
        values = self.backend.encode_mm(images, self._tokenize(texts, "queries", self.query_max_length))
        return self.torch.nn.functional.normalize(values, dim=-1)

    def encode_documents(self, texts):
        self.backend.eval()
        with self.torch.no_grad():
            batches = [
                self.forward_documents(texts[start:start + INFERENCE_BATCH_SIZE]).cpu()
                for start in range(0, len(texts), INFERENCE_BATCH_SIZE)
            ]
            return self.torch.cat(batches).numpy()

    def encode_queries(self, image_paths, texts):
        if len(image_paths) != len(texts):
            raise ValueError("image path and query text counts differ")
        self.backend.eval()
        with self.torch.no_grad():
            batches = [
                self.forward_queries(
                    image_paths[start:start + INFERENCE_BATCH_SIZE],
                    texts[start:start + INFERENCE_BATCH_SIZE],
                    training=False,
                ).cpu()
                for start in range(0, len(texts), INFERENCE_BATCH_SIZE)
            ]
            return self.torch.cat(batches).numpy()

    def state_dict(self):
        return self.backend.state_dict()

    def load_state_dict(self, state):
        return self.backend.load_state_dict(state)


def load_visualized_bge(model_name: str, weight_path: Path, device: str = "cuda", freeze_policy: str = "none"):
    ensure_visual_bge_available()
    import torch
    from PIL import Image
    from visual_bge.modeling import Visualized_BGE

    backend = Visualized_BGE(model_name_bge=model_name, model_weight=str(weight_path))

    def tokenize(texts, target_device, max_length=None):
        options = {"max_length": max_length} if max_length is not None else {}
        return backend.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, **options).to(target_device)

    def load_images(paths, training, target_device):
        preprocess = backend.preprocess_train if training else backend.preprocess_val
        tensors = [preprocess(Image.open(path).convert("RGB")) for path in paths]
        return torch.stack(tensors).to(target_device)

    return VisualizedBGEAdapter(backend, tokenize, load_images, device, freeze_policy)

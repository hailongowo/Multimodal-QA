"""Load the Visualized-BGE model used only by the official baseline."""

from pathlib import Path
import sys


def load_model(model_weight):
    try:
        from visual_bge.modeling import Visualized_BGE
    except ModuleNotFoundError as error:
        if error.name != "visual_bge":
            raise
        root = Path(__file__).resolve().parents[4]
        source = root / ".runtime/FlagEmbedding/research/visual_bge"
        if not (source / "visual_bge/modeling.py").is_file():
            raise ModuleNotFoundError(
                "Cannot import visual_bge. Install the package or provide local source at "
                f"{source}. A metadata-only pip installation is insufficient.",
                name="visual_bge",
            ) from error
        sys.path.insert(0, str(source))
        from visual_bge.modeling import Visualized_BGE

    import torch
    from PIL import Image

    class LengthLimitedVisualizedBGE(Visualized_BGE):
        @torch.no_grad()
        def encode(self, image=None, text=None):
            if text is None:
                return super().encode(image=image, text=text)

            positions = self.bge_embeddings.position_embeddings
            context_limit = min(self.tokenizer.model_max_length, positions.num_embeddings)
            if image is None:
                # XLM-R generates positions starting after its padding index.
                position_offset = (positions.padding_idx + 1) if positions.padding_idx is not None else 0
                text_limit = min(context_limit, positions.num_embeddings - position_offset)
                image_tensor = None
            else:
                # encode_mm inserts all image patches after CLS and shifts text positions.
                image_tokens = self.model_visual.visual.patch_embed.num_patches
                text_limit = context_limit - image_tokens
                with Image.open(image) as source:
                    image_tensor = self.preprocess_val(source).unsqueeze(0).to(self.device)

            if text_limit < self.tokenizer.num_special_tokens_to_add(pair=False):
                raise ValueError("Visualized-BGE context has no room for text after image tokens.")
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=text_limit,
            ).to(self.device)
            if image_tensor is None:
                return self.encode_text(tokens)
            return self.encode_mm(image_tensor, tokens)

    model = LengthLimitedVisualizedBGE(model_name_bge="BAAI/bge-m3", model_weight=str(model_weight))
    model.eval()
    return model

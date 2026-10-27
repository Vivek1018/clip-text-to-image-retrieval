from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torchvision.io import read_image
from torchvision.transforms.functional import rotate as tv_rotate
from transformers import CLIPModel, CLIPProcessor

from clip_retrieval.config import Settings


def resolve_device(device_setting: str) -> torch.device:
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def _to_feature_tensor(model_output: Tensor | object) -> Tensor:
    if isinstance(model_output, Tensor):
        return model_output
    if hasattr(model_output, "pooler_output"):
        pooled = model_output.pooler_output
        if pooled is not None:
            return pooled
    raise TypeError(f"Unexpected model output type: {type(model_output)}")


class ClipEncoder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.device = resolve_device(settings.device)
        token = settings.hf_token
        self.model = CLIPModel.from_pretrained(
            settings.clip_model_id, token=token
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            settings.clip_model_id, token=token
        )
        self.model.eval()

    @torch.inference_mode()
    def encode_text(self, text: str) -> Tensor:
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_text_features(**inputs)
        return F.normalize(_to_feature_tensor(features), dim=-1)

    def _load_tensor_image(self, path: Path) -> Tensor:
        image = read_image(str(path))
        if self.settings.image_rotate_degrees:
            image = tv_rotate(image, self.settings.image_rotate_degrees)
        return image

    @torch.inference_mode()
    def encode_image_paths(self, paths: Sequence[Path]) -> Tensor:
        if not paths:
            return torch.empty(0, self.model.config.projection_dim)

        tensors = [self._load_tensor_image(p) for p in paths]
        inputs = self.processor(images=tensors, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_image_features(**inputs)
        return F.normalize(_to_feature_tensor(features), dim=-1)

    @torch.inference_mode()
    def encode_image_url(self, url: str, user_agent: str | None = None) -> Tensor:
        headers = {}
        if user_agent:
            headers["User-Agent"] = user_agent
        import requests

        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        features = self.model.get_image_features(**inputs)
        return F.normalize(_to_feature_tensor(features), dim=-1)

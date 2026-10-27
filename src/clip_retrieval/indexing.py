from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import torch
from tqdm import tqdm

from clip_retrieval.config import Settings
from clip_retrieval.encoder import ClipEncoder


@dataclass
class IndexManifest:
    clip_model_id: str
    image_glob: str
    image_rotate_degrees: int
    image_paths: list[str]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> IndexManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def discover_images(images_dir: Path, pattern: str) -> list[Path]:
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {images_dir}. "
            "Create it and add .jpg files (see data/README.md)."
        )
    paths = sorted(images_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No images matching {pattern!r} in {images_dir}"
        )
    return paths


def build_index(
    settings: Settings,
    encoder: ClipEncoder | None = None,
    image_paths: Sequence[Path] | None = None,
) -> tuple[torch.Tensor, IndexManifest]:
    encoder = encoder or ClipEncoder(settings)
    paths = list(image_paths) if image_paths is not None else discover_images(
        settings.images_dir, settings.image_glob
    )

    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)
    batch_files: list[Path] = []
    batch_size = settings.batch_size

    total_batches = (len(paths) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(total_batches), desc="Indexing images"):
        start = batch_idx * batch_size
        batch_paths = paths[start : start + batch_size]
        embeddings = encoder.encode_image_paths(batch_paths).cpu()
        batch_file = settings.embeddings_dir / f"batch_{batch_idx + 1}.pt"
        torch.save(embeddings, batch_file)
        batch_files.append(batch_file)

    stacked = torch.cat([torch.load(f, weights_only=True) for f in batch_files], dim=0)
    torch.save(stacked, settings.embeddings_dir / "embeddings.pt")

    manifest = IndexManifest(
        clip_model_id=settings.clip_model_id,
        image_glob=settings.image_glob,
        image_rotate_degrees=settings.image_rotate_degrees,
        image_paths=[str(p) for p in paths],
    )
    manifest.save(settings.embeddings_manifest)
    return stacked, manifest


def load_index(settings: Settings) -> tuple[torch.Tensor, IndexManifest]:
    manifest_path = settings.embeddings_manifest
    embeddings_path = settings.embeddings_dir / "embeddings.pt"
    if not manifest_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(
            "Index not found. Run: clip-retrieval index --images-dir data/images"
        )
    manifest = IndexManifest.load(manifest_path)
    embeddings = torch.load(embeddings_path, weights_only=True)
    return embeddings, manifest

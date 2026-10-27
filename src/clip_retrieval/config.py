from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    clip_model_id: str
    images_dir: Path
    embeddings_dir: Path
    embeddings_manifest: Path
    batch_size: int
    image_glob: str
    top_k: int
    image_rotate_degrees: int
    device: str
    hf_token: str | None

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> Settings:
        root = _repo_root()
        if env_file is None:
            env_file = root / ".env"
        load_dotenv(env_file, override=False)

        images_dir = _resolve_path(
            os.getenv("IMAGES_DIR", "data/images"), root
        )
        embeddings_dir = _resolve_path(
            os.getenv("EMBEDDINGS_DIR", "data/embeddings"), root
        )
        manifest = _resolve_path(
            os.getenv("EMBEDDINGS_MANIFEST", "data/embeddings/manifest.json"),
            root,
        )

        return cls(
            repo_root=root,
            clip_model_id=os.getenv(
                "CLIP_MODEL_ID", "openai/clip-vit-base-patch32"
            ),
            images_dir=images_dir,
            embeddings_dir=embeddings_dir,
            embeddings_manifest=manifest,
            batch_size=int(os.getenv("BATCH_SIZE", "20")),
            image_glob=os.getenv("IMAGE_GLOB", "*.jpg"),
            top_k=int(os.getenv("TOP_K", "6")),
            image_rotate_degrees=int(os.getenv("IMAGE_ROTATE_DEGREES", "-90")),
            device=os.getenv("DEVICE", "auto"),
            hf_token=os.getenv("HF_TOKEN") or None,
        )

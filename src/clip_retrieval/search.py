from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from clip_retrieval.config import Settings
from clip_retrieval.encoder import ClipEncoder
from clip_retrieval.indexing import IndexManifest, load_index


@dataclass(frozen=True)
class SearchResult:
    path: Path
    score: float


def rank_images(
    query: str,
    settings: Settings,
    encoder: ClipEncoder | None = None,
    top_k: int | None = None,
) -> list[SearchResult]:
    top_k = top_k if top_k is not None else settings.top_k
    encoder = encoder or ClipEncoder(settings)
    image_embeddings, manifest = load_index(settings)
    text_embedding = encoder.encode_text(query).cpu()

    scores = F.cosine_similarity(
        text_embedding, image_embeddings, dim=-1
    ).detach().numpy()
    order = scores.argsort()[::-1][:top_k]

    results: list[SearchResult] = []
    for idx in order:
        results.append(
            SearchResult(
                path=Path(manifest.image_paths[int(idx)]),
                score=float(scores[int(idx)]),
            )
        )
    return results

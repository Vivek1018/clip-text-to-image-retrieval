from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import rotate as tv_rotate
from torchvision.utils import make_grid

from clip_retrieval.config import Settings
from clip_retrieval.search import SearchResult


def save_result_grid(
    results: list[SearchResult],
    output_path: Path,
    settings: Settings,
    nrow: int = 2,
    cell_size: tuple[int, int] = (240, 480),
) -> Path:
    if not results:
        raise ValueError("No search results to visualize.")

    from torchvision import transforms

    resize = transforms.Resize(cell_size)
    tensors = []
    for result in results:
        image = read_image(str(result.path))
        if settings.image_rotate_degrees:
            image = tv_rotate(image, settings.image_rotate_degrees)
        tensors.append(resize(image))

    grid = make_grid(torch.stack(tensors), nrow=nrow)
    pil_image: Image.Image = ToPILImage()(grid)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_image.save(output_path)
    return output_path

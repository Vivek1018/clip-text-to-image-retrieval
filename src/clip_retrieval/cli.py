from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from clip_retrieval.config import Settings
from clip_retrieval.encoder import ClipEncoder
from clip_retrieval.indexing import build_index, discover_images
from clip_retrieval.search import rank_images
from clip_retrieval.visualize import save_result_grid


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env (default: repository .env)",
    )


def _cmd_index(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    if args.images_dir:
        settings = replace(
            settings, images_dir=Path(args.images_dir).resolve()
        )
    if args.batch_size:
        settings = replace(settings, batch_size=args.batch_size)

    encoder = ClipEncoder(settings)
    paths = discover_images(settings.images_dir, settings.image_glob)
    _, manifest = build_index(settings, encoder=encoder, image_paths=paths)
    print(f"Indexed {len(manifest.image_paths)} images.")
    print(f"Manifest: {settings.embeddings_manifest}")
    print(f"Embeddings: {settings.embeddings_dir / 'embeddings.pt'}")
    return 0


def _cmd_search(args: argparse.Namespace) -> int:
    settings = Settings.from_env(args.env_file)
    top_k = args.top_k or settings.top_k
    results = rank_images(args.query, settings, top_k=top_k)

    print(f'Query: "{args.query}"')
    for rank, hit in enumerate(results, start=1):
        print(f"{rank:2d}. score={hit.score:.4f}  {hit.path}")

    if args.output_grid:
        out = save_result_grid(
            results, Path(args.output_grid), settings, nrow=args.nrow
        )
        print(f"Saved grid: {out}")
    return 0


def _cmd_demo_url(args: argparse.Namespace) -> int:
    """Single URL vs text query (notebook-style smoke test)."""
    settings = Settings.from_env(args.env_file)
    encoder = ClipEncoder(settings)
    text_emb = encoder.encode_text(args.query).cpu()
    image_emb = encoder.encode_image_url(args.url).cpu()
    score = float(
        __import__("torch").nn.functional.cosine_similarity(
            text_emb, image_emb, dim=-1
        )
    )
    print(f'URL: {args.url}')
    print(f'Query: "{args.query}"')
    print(f"Cosine similarity: {score:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Local CLIP text-to-image retrieval.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    index_p = sub.add_parser("index", help="Embed all images and build an index.")
    _add_common_args(index_p)
    index_p.add_argument("--images-dir", type=Path, default=None)
    index_p.add_argument("--batch-size", type=int, default=None)
    index_p.set_defaults(func=_cmd_index)

    search_p = sub.add_parser("search", help="Retrieve top images for a text query.")
    _add_common_args(search_p)
    search_p.add_argument("--query", "-q", required=True, help="Text query")
    search_p.add_argument("--top-k", type=int, default=None)
    search_p.add_argument(
        "--output-grid",
        type=Path,
        default=None,
        help="Optional path to save a preview grid (PNG)",
    )
    search_p.add_argument("--nrow", type=int, default=2)
    search_p.set_defaults(func=_cmd_search)

    demo_p = sub.add_parser(
        "demo-url",
        help="Compare one image URL against a text query (no local index).",
    )
    _add_common_args(demo_p)
    demo_p.add_argument("--url", required=True)
    demo_p.add_argument("--query", "-q", required=True)
    demo_p.set_defaults(func=_cmd_demo_url)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

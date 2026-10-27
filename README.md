# CLIP Text-to-Image Retrieval

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-CLIP-ee4c2c.svg)](https://pytorch.org/)

Local text-to-image search over your own photo library using OpenAI CLIP — no paid vision APIs. Embeddings are cached on disk so repeat queries stay fast. Default retrieval returns the **top 6** images per natural-language query.

Built as a personal project while studying multimodal embeddings for an M.Sc. in Data Analytics at Stiftung Universität Hildesheim.

## Tech stack

- **PyTorch** · **Hugging Face Transformers** (`openai/clip-vit-base-patch32`)
- **torchvision** (batch image I/O, optional rotation for mobile photos)
- **NumPy** · cosine similarity ranking
- Config via **`.env`** / **`python-dotenv`**
- Install & lock env with **[uv](https://github.com/astral-sh/uv)** (`pyproject.toml`)

## Project layout

```text
src/clip_retrieval/   # encoder, indexing, search, CLI
notebooks/            # exploratory notebook (outputs cleared)
data/images/          # your images (not committed)
data/embeddings/      # index artifacts (not committed)
```

## Quickstart

```bash
# Clone and enter the repo
git clone https://github.com/Vivek1018/clip-text-to-image-retrieval.git
cd clip-text-to-image-retrieval

# Python 3.10+ with uv
uv venv
uv pip install -e .

# Configuration
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux

# Add .jpg files to data/images/, then build the index
clip-retrieval index --images-dir data/images

# Search (prints paths + scores; optional preview grid)
clip-retrieval search -q "water bottle" --top-k 6 --output-grid outputs/query.png
```

### One-image URL demo (from the original notebook)

```bash
clip-retrieval demo-url \
  --url "https://cdn.overstockart.com/blog/wp-content/uploads/2018/04/GS2840-1020x500.jpg" \
  -q "a picnic spot by the side of the river"
```

### Without installing the entry point

```bash
uv pip install -e .
python -m clip_retrieval search -q "price tag"
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CLIP_MODEL_ID` | `openai/clip-vit-base-patch32` | Hugging Face model |
| `IMAGES_DIR` | `data/images` | Image folder |
| `EMBEDDINGS_DIR` | `data/embeddings` | Cached `.pt` batches + manifest |
| `BATCH_SIZE` | `20` | Indexing batch size |
| `TOP_K` | `6` | Default results per query |
| `IMAGE_ROTATE_DEGREES` | `-90` | Match phone orientation (set `0` to disable) |
| `HF_TOKEN` | *(empty)* | Optional Hugging Face token for downloads |

## Notebook

The original walkthrough lives in [`notebooks/Image_Text_Clip_Embeddings.ipynb`](notebooks/Image_Text_Clip_Embeddings.ipynb). Production flow is the CLI above.

## Related coursework

Older multimodal tutorial notebooks (`part_1` … `part_4`) remain under `notebooks/src/` for reference; this README focuses on the portfolio **text-to-image retrieval** pipeline.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Vivek Kumar Verma** — [LinkedIn](https://www.linkedin.com/in/vivek-kumar-verma-0206765b/) · Vivek.verma1018@gmail.com

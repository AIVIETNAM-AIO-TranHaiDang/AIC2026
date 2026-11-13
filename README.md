# AIC 2026 — Multimodal Video Event Retrieval

An interactive, human-in-the-loop video search system built for the
**AI Challenge Ho Chi Minh City 2026** (*Event Retrieval from a Large Video
Collection*). An operator types a Vietnamese description of a moment (or a
question about it) and the system returns the matching `(video_id, timestamp)`
from a corpus of Vietnamese news broadcasts.

## Highlights

| | |
| --- | --- |
| Corpus | **415 videos · 81.9 hours** |
| Shots detected (OmniShotCut) | 56,250 |
| Keyframes indexed | **171,059** (75,257 near-duplicates removed with pHash) |
| Visual retrieval | SigLIP2 (`so400m-patch16-384`) + FAISS |
| Text retrieval | BGE-M3 semantic index + literal (keyword) index over OCR and ASR |
| Ranking | Reciprocal rank fusion across channels + temporal window fusion |
| Query latency | ~2 s end to end (ranking stage ~0.5 s) |

### Retrieval quality

Measured on a 16-query Vietnamese KIS-T seed fixture (6 visual, 5 OCR, 5 ASR
queries). The fixture is small and labelled by the team, so treat it as a
pipeline baseline, not a leaderboard number. See
[`benchmarks/`](benchmarks/README.md) for the fixture and raw reports.

| Configuration | R@1 | R@5 | R@10 | R@100 | Median rank |
| --- | ---: | ---: | ---: | ---: | ---: |
| Visual only (SigLIP2) | 0.063 | 0.250 | 0.313 | 0.750 | 14 |
| + OCR/ASR text channels, rank fusion | 0.313 | 0.625 | 0.688 | 0.938 | 3.5 |

Most of the gain comes from speech (ASR) queries, which a visual-only index
cannot answer.

## Task formats

| Format | What the operator must do |
| --- | --- |
| **KIS-V** | Find an exact segment given a short example clip. |
| **KIS-T** | Find an exact segment given a Vietnamese text description. |
| **KIS-C** | Same as KIS-T, but details are revealed progressively. |
| **VQA** | Answer a question about a video (retrieve, then read/reason). |
| **TRAKE** | Return an ordered sequence of moments for a multi-step event. |

## Architecture

```
Raw video
   │  ingest        OmniShotCut / TransNetV2 shots → keyframes → pHash de-dup
   ▼
Keyframes ──┬── embed       SigLIP2 vectors (safetensors shards) → FAISS index
            ├── chronicle   ASR (Faster-Whisper) · OCR (EasyOCR) · optional VLM captions
            └── textstack   BGE-M3 semantic index · literal index · overlay prior
                                   │
Online query ──▶ Query Cortex (Vietnamese parsing / translation)
             ──▶ multi-channel search (visual · semantic · literal)
             ──▶ reciprocal rank fusion + temporal window fusion
             ──▶ optional VLM re-ranking
             ──▶ operator console (FastAPI + web UI) → submit (video_id, timestamp)
```

Offline stages are resumable and can fan out across GPUs
(`--num-gpus N`) or run serverlessly on [Modal](docs/modal-keyframes.md).

## Repository layout

```
.
├── src/aic/            Pipeline package
│   ├── ingest/           shot detection, keyframe selection, de-duplication
│   ├── embed/            SigLIP2 / PE-Core encoders and shard storage
│   ├── chronicle/        ASR, OCR, captions, per-shot evidence assembly
│   ├── textstack/        BGE-M3 dense/sparse text embeddings
│   ├── index/            FAISS and sparse index wrappers
│   ├── retrieval/        channels, rank fusion, temporal windows
│   ├── cortex/           Vietnamese query parsing and dispatch
│   ├── escalate/         VLM verify / re-rank, query-performance predictor
│   ├── vqa/              question answering and counting
│   ├── eval/             fixtures, metrics, evaluation harness
│   └── service/          FastAPI app, sessions, static operator UI
├── scripts/            Entry points (local stages and Modal runners)
├── configs/            Profiles (t0.yaml, Kaggle, GTX 1650, VQA, Modal)
├── benchmarks/         Labelled fixture and retrieval reports
├── tests/              Pytest suite (synthetic-video fixtures)
├── notebooks/          Kaggle pilot notebook
├── artifacts/          Small tracked manifests used by Modal jobs
├── docs/               Guides and design notes (start with the list below)
└── data/               Corpus, keyframes, embeddings, manifests (gitignored)
```

## Getting started

Python 3.10+ and a virtual environment are required; do not install into the
system Python.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e ".[dev]"            # package + pytest + ruff

ruff check .
pytest
```

GPU-only backends (OmniShotCut, PE-Core, vLLM) have conflicting dependency
trees and are installed separately; the Modal keyframe image uses
[`requirements-modal-ingest.txt`](requirements-modal-ingest.txt).

## Quick start

Put videos in `data/videos/`, then run the stages in order. Each one is
resumable; re-running only processes what is missing.

```bash
python scripts/ingest_corpus.py      --config configs/t0.yaml   # shots + keyframes
python scripts/embed_corpus.py       --config configs/t0.yaml   # SigLIP2 embeddings
python scripts/build_chronicle.py    --config configs/t0.yaml   # ASR + OCR (+ captions)
python scripts/build_text_indexes.py --config configs/t0.yaml   # BGE-M3 + literal indexes
python scripts/serve.py              --config configs/t0.yaml   # UI at http://127.0.0.1:8000/
```

Add `--num-gpus N` to the first three to spread work across GPUs. To evaluate,
create a fixture from [`configs/fixture.example.yaml`](configs/fixture.example.yaml)
and run `python scripts/evaluate_fixture.py --config configs/t0.yaml --fixture data/fixture.yaml`.

For a small reproducible run on free GPUs, use the
[Kaggle pilot](docs/kaggle-pilot.md) (`notebooks/kaggle_pilot.ipynb`).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/pipeline-guide.md](docs/pipeline-guide.md) | Full runbook: every stage, multi-GPU/endpoint fan-out, resume and repair, accuracy knobs, model backends |
| [docs/modal-keyframes.md](docs/modal-keyframes.md) | Keyframe extraction on Modal |
| [docs/modal-siglip2-benchmark.md](docs/modal-siglip2-benchmark.md) | SigLIP2 GPU benchmark on Modal |
| [docs/retrieval-artifacts.md](docs/retrieval-artifacts.md) | Where keyframes, manifests and indexes are stored |
| [docs/huggingface-keyframes.md](docs/huggingface-keyframes.md) | Keyframe archive on Hugging Face |
| [docs/kaggle-pilot.md](docs/kaggle-pilot.md) | Kaggle pilot setup |
| [docs/guides/single-video-ingest-walkthrough.md](docs/guides/single-video-ingest-walkthrough.md) | Step-by-step ingest of one video (Vietnamese) |
| [docs/note/](docs/note/16-pipeline-current-state.md) | Design notes (pipeline state, VQA design) |
| [docs/overview.vi.md](docs/overview.vi.md) | Project overview in Vietnamese |
| [benchmarks/README.md](benchmarks/README.md) | Benchmark fixture and results |

## Data

Large data is not stored in Git. Keyframes are on Hugging Face
(`Neezidow/AIC2026-keyframes`); embeddings and indexes live in Modal Volumes.
See [docs/retrieval-artifacts.md](docs/retrieval-artifacts.md) for the layout and
what is needed to run retrieval or the full UI.

## Contributing

Run `ruff check .` and `pytest` before committing. Working conventions for
contributors and AI agents are in [CLAUDE.md](CLAUDE.md).

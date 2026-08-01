# AIC2026_prepare

Preparation codebase for the **AI Challenge Ho Chi Minh City (AIC / HCMAI) 2026** —
the *Event Retrieval from a Large Video Collection* task.

The goal is an **interactive, human-in-the-loop video retrieval system**: during a
live timed round, an operator uses the tool to find the exact video segment (or
answer a question about it) that matches a query from the moderators, over a corpus
of hundreds of hours of news broadcast video. The competition follows the
[Video Browser Showdown (VBS)](https://videobrowsershowdown.org/about-vbs/) and
Lifelog Search Challenge (LSC) format.

## Query/task formats

| Format | What the operator must do |
| --- | --- |
| **KIS-V** | Find an exact segment given a short example clip. |
| **KIS-T** | Find an exact segment given a Vietnamese text description. |
| **KIS-C** | Same as KIS-T, but details are revealed progressively during the round. |
| **VQA / Q&A** | Answer a question about a video (retrieve + read/reason). |
| **TRAKE** | Return an ordered sequence of moments for a multi-step event. |

See [`docs/note/01-task-types.md`](docs/note/01-task-types.md) for full definitions.

## Reference pipeline

Offline indexing: shot boundary + keyframe extraction -> keyframe filtering ->
vision-language embeddings (CLIP family / BEiT-3) + optional object detection / OCR /
ASR -> metadata store -> vector index (FAISS or a vector DB).

Online retrieval: query understanding (Vietnamese->English translation for text,
clip embedding for visual) -> ANN vector search -> re-ranking + temporal fusion ->
interactive UI that submits `(video_id, timestamp)` answers.

## Repository layout

```
AIC2026_prepare/
├── CLAUDE.md                # Working guide and rules for AI agents and developers.
├── README.md                # This file.
├── requirements.txt         # Runtime dependencies (pinned).
├── requirements-dev.txt     # Test/quality tooling (pytest, ruff).
├── requirements-gpu.txt     # GPU-server-only backends (conflicting deps).
├── implementation-notes.md  # Running log of decisions and trade-offs.
├── pyproject.toml           # Package metadata, pytest and ruff settings.
├── configs/                 # Profile configs (t0.yaml) and fixture example.
├── src/aic/                 # The pipeline package (config, eval, ingest, retrieval).
├── scripts/                 # Runnable entry points (ingest_corpus.py).
├── tests/                   # Pytest suite with synthetic-video fixtures.
├── data/                    # Corpus, keyframes, manifests (gitignored).
└── docs/
    ├── note/                # Research notes; start at INDEX.md.
    └── reference/           # Source training slides.
```

## Getting started

The project uses a **virtual environment**; do not install packages globally.

```bash
# From the repository root
python -m venv .venv

# Activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (Git Bash):
source .venv/Scripts/activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

Run the checks:

```bash
pytest
ruff check .
```

## Running the pipeline

For a reproducible 3–10 video Kaggle GPU pilot, use
[`notebooks/kaggle_pilot.ipynb`](notebooks/kaggle_pilot.ipynb) with
[`configs/t0-kaggle.yaml`](configs/t0-kaggle.yaml); setup, persistence, tunnel
limits, and troubleshooting are documented in
[`docs/kaggle-pilot.md`](docs/kaggle-pilot.md).

Drop input videos (mp4/mkv/avi/mov/webm) into `data/videos/`, then run the
stages in order (each is resumable — a killed Colab session just re-runs the
same command; use `--limit N` for pilots):

```bash
python scripts/ingest_corpus.py     --config configs/t0.yaml
python scripts/embed_corpus.py      --config configs/t0.yaml
python scripts/build_chronicle.py   --config configs/t0.yaml
python scripts/build_text_indexes.py --config configs/t0.yaml
python scripts/evaluate_fixture.py  --config configs/t0.yaml --fixture data/fixture.yaml
```

**Multi-GPU:** the local torch stages (shots, embed, ASR, easyocr OCR) accept
`--num-gpus N` to spread the corpus across N GPUs — one worker process per card,
pinned via `CUDA_VISIBLE_DEVICES`, with a single aggregate progress bar. The
workers pull videos from a shared queue (work-stealing): each builds its model
once, then grabs the next video whenever it finishes one, so a fast card keeps
working instead of idling while a slow card grinds through a fixed shard — the
videos are unequal in length, so this keeps every GPU busy. It defaults to every
visible GPU (`torch.cuda.device_count()`) and runs in-process on a single GPU or
CPU, so the command is identical everywhere. The run is interruption-safe: each
worker appends to its own rank manifest, and the next run folds any leftover
rank manifests from a killed session back into the canonical one before
planning, so finished videos are never re-done:

```bash
python scripts/ingest_corpus.py   --config configs/t0.yaml --num-gpus 4
python scripts/embed_corpus.py    --config configs/t0.yaml --num-gpus 4
python scripts/build_chronicle.py --config configs/t0.yaml --num-gpus 4
```

**Multi-endpoint fan-out:** the OpenAI-compatible stages (captioning, VLM OCR,
the Query Cortex, VLM verify) are unaffected by `--num-gpus`; they scale by
pointing at more endpoints, not by owning a GPU. Give any of them a **list** of
`base_url` and the stage spreads its work across all of them:

```yaml
chronicle:
  caption:
    provider: local
    base_url: ["http://gpu0:8000/v1", "http://gpu1:8000/v1"]  # two servers
    model: qwen3-vl            # one model -> applied to every endpoint
    num_parallel: 4            # concurrent requests PER endpoint (batch stages)
    host_max_retries: 3        # evict an endpoint after N transport errors
```

Deploy different models on different endpoints by passing a `model` list of the
same length (`model: [qwen3-vl, gemma-vl]` pairs position-by-position); a single
`model` is broadcast to every endpoint. The batch stages (caption, VLM OCR) keep
every endpoint busy with one shared work queue and fail over across endpoints
(so a mixed pool degrades gracefully if one server dies); the online single-call
stages (Cortex, VLM verify) load-balance and fail over but ignore `num_parallel`
(one query is one request). A plain string `base_url` with the defaults behaves
exactly as before — a single endpoint, one request at a time.

To stand those endpoints up on a multi-GPU box, `scripts/serve_local_llms.py`
auto-detects the visible GPUs and launches one `llama-server` per card (each
pinned via `CUDA_VISIBLE_DEVICES`, its own port), then prints the comma-joined
`LOCAL_LLM_BASE_URL` that the config splits into the endpoint list — so a config
whose `base_url: ${LOCAL_LLM_BASE_URL}` fans across every GPU with no YAML edit:

```bash
python scripts/serve_local_llms.py --server-bin .../llama-server \
    --model model.gguf --mmproj mmproj.gguf     # background; --foreground streams logs
```

**Storage knobs:** model weights default to the in-project `data/models/`; set
`paths.models_dir` to relocate the cache to a shared or persistent location
(reused across runs and services). Keyframe embeddings are stored as
`safetensors` by default (`embed.shard_format`), with `embed.store_dtype`
(e.g. `float16`) to trade disk for a small precision loss.

**Resume & repair:** every offline stage is resumable — it records one manifest
row per finished artifact and, on a re-run, skips whatever is already done, so
adding new videos and re-running processes only the new ones (a killed Colab
session just re-runs the same command). If a run was interrupted hard enough to
leave a truncated keyframe image or embedding shard on disk, set `verify.on_resume:
true` in the config: before resuming, the ingest / embed / chronicle scripts then
decode every referenced artifact, delete any that are missing or corrupt along
with their manifest rows — cascading the drop to downstream stages whose input
vanished — so the resume redoes exactly the broken items. It is off by default
because the check reads every file once; enable it after a known-bad interruption.
Verify also requeues videos whose ingest *failed* (the failure may have been
transient), and failed captions are retried on every resume. Artifacts that are
valid but wrong — e.g. transcripts produced before a settings change — are
re-done explicitly: `build_chronicle.py --redo-asr` (or
`--redo-asr-videos id1,id2`) drops their ASR rows so the resumable stage
re-transcribes them.

**Accuracy knobs (2026-07 audit):** `chronicle.asr.vad_filter` and
`chronicle.asr.condition_on_previous_text` gate the whisper hallucination loop
on music-heavy broadcast audio (enabled in the shipped profiles; defaults keep
the library behaviour). `retrieval.keyframe_oversample` keeps the dense visual
channel competitive in rank fusion, and `ingest.dedup.keep_one_per_shot`
guarantees every non-blank shot stays visually retrievable. Endpoint stages
retry HTTP 429/503 on the same endpoint with backoff
(`rate_limit_max_retries`) instead of counting throttling toward eviction.

**Accuracy program (note 15, 2026-07):** seven further levers, all
config-gated with off = the previous behaviour:

- `chronicle.asr.min_chars_per_second` — speaking-rate gate that catches the
  *high-confidence* music hallucinations the probability gates cannot; also
  applied at assembly, so re-running `build_chronicle.py` heals an existing
  corpus without re-transcribing.
- `chronicle.overlay_min_recurrence` — persistent-overlay OCR split:
  watermarks/tickers leave the per-shot evidence and become one per-video
  overlay document, indexed by `build_text_indexes.py` for the video prior.
- Bilingual chronicle — `chronicle.translate.enabled` (offline NLLB
  translation of `caption_en`) and/or `chronicle.caption.request_caption_vi`
  (native Vietnamese from the captioner; wins per shot, translation fills
  gaps), plus `cortex.dispatch.raw_query_weight` sending the RAW Vietnamese
  query to the multilingual text channels — no VI->EN hop.
- `retrieval.temporal` — window fusion: candidates of one video within
  `window_ms` score as a window (per-sub-query best contribution, soft
  coverage), `agreement_bonus` rewards cross-channel triangulation,
  `video_prior_weight` adds the overlay prior. `window_ms: 0` is bit-exact
  per-shot RRF.
- `retrieval.visual_chunking` — sentence chunking for the dense visual
  channel: each sentence of a multi-sentence query becomes its own SigLIP
  sub-query (the encoder reads 64 tokens and mostly the first sentence),
  so every sentence probes for its own moment and window fusion rewards
  the window covering them all; `retrieval.temporal.order_bonus`
  additionally rewards windows whose chunks match in the query's sentence
  order (vitrivr-style temporal scoring).
- `retrieval.auto_rerank` — the vlm_verify listwise rerank runs
  automatically on every search (degrade-to-unchanged on any failure or
  timeout); requires `escalations.vlm_verify.enabled` and adds one VLM call
  of latency per search.
- `scripts/generate_fixture.py` — synthetic fixture: a VLM writes AIC-style
  Vietnamese queries for sampled windows (exact labels by construction; the
  model sees only keyframes, so no caption-vocabulary leakage). The
  endpoint follows the profile's `fixture_gen` section; keep the
  hand-labelled fixture as a holdout.

The chronicle caption stage now reports one aggregate progress bar with
ok/failed counts (per-request `httpx` logging is capped at WARNING by
`setup_logging`).

Then start the online service and operator console:

```bash
python scripts/serve.py --config configs/t0.yaml
# UI at http://127.0.0.1:8000/  (host/port under service: in the config)
```

Set `service.streaming: true` to also register additive NDJSON streaming twins
of the two online routes — `/api/search/stream` and `/api/qa/stream` — so the
immediately-usable payload paints first (the fused ranking; VQA Track A cards)
and each slower stage (KIS auto-rerank; VQA reverse/count/Track B) streams in as
it finishes. Same answer, incremental delivery: the streamed result is
byte-identical to the one-shot route, the console auto-uses the `/stream` routes
and falls back to the one-shot routes when the flag is off.


Escalations (Phase 7) appear as buttons and hotkeys in the console for
every tool enabled under `escalations:` in the config; the QPP advisor hint
appears once `qpp.trained_artifact` points at a trained artifact:

```bash
python scripts/train_qpp.py --config configs/t0.yaml \
    --fixture data/fixture.yaml --out data/qpp.json
```

Operational scripts:

```bash
python scripts/warmup.py          --config configs/t0.yaml   # cold-start drill
python scripts/measure_latency.py --config configs/t0.yaml --fixture data/fixture.yaml
python scripts/faiss_gpu_parity.py --config configs/t0.yaml  # GPU server only
```


### Selectable model backends

Every stage keeps a proven default and offers the mid-2026 SOTA challenger
as a pure config switch (the ablation harness decides which wins on the
labelled fixture). Backends whose dependency trees conflict with the pinned
dev venv install on the GPU server per
[`requirements-gpu.txt`](requirements-gpu.txt).

| Stage | Default | SOTA option (config switch) |
| --- | --- | --- |
| Shot detection | `transnetv2` (CPU-capable) | `omnishotcut` (2026 SBD SOTA, GPU-only) |
| Keyframe embedding | `siglip2` | `pe_core` (Meta Perception Encoder) |
| ASR | `faster_whisper` large-v3 | PhoWhisper via `model_size` (CT2 convert), or `qwen3_asr` (+ForcedAligner timestamps) |
| OCR | `easyocr` | `vlm` — PaddleOCR-VL-1.6 served by vLLM (no PaddlePaddle deps here) |
| Captioning | Gemini API | any OpenAI-compatible endpoint (Qwen3-VL-8B local, OpenAI, ...) |
| Text embedding | `bge_m3` (dense+sparse) | `textstack.dense` override -> Qwen3-Embedding (semantic channel) |

`configs/t0.yaml` carries a commented example for every switch.

## Building the evaluation fixture (Phase 1)

Copy [`configs/fixture.example.yaml`](configs/fixture.example.yaml) to
`data/fixture.yaml` and label at least 30 KIS queries against the dev corpus. Every retrieval
component is evaluated against this fixture via `aic.eval.evaluate`.

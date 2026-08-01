# The pipeline as implemented (July 2026)

The current end-to-end system, stage by stage, as the code actually runs it
— the implementation counterpart of the reference design in
[02-system-architecture.md](02-system-architecture.md). Everything below is
config-driven from one profile YAML (`configs/t0.yaml` is the annotated
reference; the T4 profiles are its GPU variants); every accuracy lever notes
its config key so a reader can trace behaviour to a switch. History and
rationale live in note [09](09-kis-implementation-plan.md) (phases), note
[10](10-model-sota-audit-2026-07.md) (model choices), and note
[15](15-kis-accuracy-research-2026-07.md) (accuracy program).

## Offline: from raw video to indexes

Run in order; every stage writes one manifest row per finished artifact and
skips done rows on re-run (resume-by-default). An opt-in integrity pass
(`verify.on_resume`) heals a corpus corrupted by a hard interruption.

```
videos/  --1--> shots + keyframes --2--> keyframe embeddings + FAISS index
                    |                        (dense_visual channel)
                    +--3--> ASR ---+
                    +--3--> OCR ---+--> chronicle.jsonl --4--> semantic index
                    +--3--> captions                          literal index
                                                              overlay index
```

1. **Ingest** (`scripts/ingest_corpus.py` -> `src/aic/ingest/`). Shot
   boundaries (TransNetV2 default, OmniShotCut challenger), keyframes per
   shot (positional + motion sampling), perceptual-hash dedup with an
   entropy blank gate; `dedup.keep_one_per_shot` guarantees a non-blank
   shot never loses all its frames. Multi-GPU via `--num-gpus`
   (work-stealing queue, per-rank shard manifests).
2. **Embed** (`scripts/embed_corpus.py` -> `src/aic/embed/`). Every
   keyframe through the image tower (SigLIP2 default, PE-Core challenger)
   into safetensors shards; `scripts/build_text_indexes.py` or the first
   retrieval run builds the FAISS index (CPU default, `index.device: cuda`
   after the parity check).
3. **Chronicle extraction** (`scripts/build_chronicle.py` ->
   `src/aic/chronicle/`). Three independent, resumable extractors:
   - **ASR**: faster-whisper (Qwen3-ASR challenger) with the
     anti-hallucination stack — VAD, no previous-text conditioning,
     probability gates, and the speaking-rate gate
     (`asr.min_chars_per_second`) that catches high-confidence music
     hallucinations.
   - **OCR**: EasyOCR (VLM/PaddleOCR challenger), per keyframe.
   - **Captions**: any OpenAI-compatible endpoint (Gemini or local
     llama.cpp/vLLM), fanned out over an endpoint pool with health
     eviction, 429/503 backoff, cross-endpoint failover; one aggregate
     tqdm bar with ok/failed counts. Optionally bilingual
     (`caption.request_caption_vi`).
   - **Assembly** joins everything into `chronicle.jsonl`, one validated
     record per shot. Assembly is cheap and rebuilds every run, and it
     re-applies the ASR rate gate and performs the persistent-overlay OCR
     split (`chronicle.overlay_min_recurrence`): lines recurring across a
     video's shots (watermarks, tickers) move from shot evidence into a
     per-video overlay list. `translate.enabled` fills `caption_vi` by
     offline NLLB translation for shots without a native one.
4. **Text indexes** (`scripts/build_text_indexes.py` ->
   `src/aic/textstack/`). Three documents per the chronicle:
   - *semantic* per shot (captions in both languages + speech) -> dense
     vector index (BGE-M3 default, Qwen3-Embedding challenger),
   - *literal* per shot (scene OCR + speech, overlay excluded) -> learned
     sparse index (BGE-M3 lexical weights),
   - *overlay* per video (the split-out watermark/ticker text) -> sparse
     index for the retrieval video prior.

## Online: from a Vietnamese query to (video_id, timestamp)

`scripts/serve.py` (operator console + FastAPI) and
`scripts/evaluate_fixture.py --use-cortex` (the harness) run the same path.
The diagram shows every lever ON (the T4 accuracy profiles):

```
raw VI query
  |-- Cortex compile (LLM -> typed spec; raw-query fallback on any failure:
  |     the whole query becomes one visual phrase)
  |-- dispatch: spec fields -> sub-queries per channel
  |     visual_phrases/paraphrases -> dense_visual + semantic_dense
  |     entity_terms               -> semantic_dense + literal_sparse
  |     ocr_literals               -> literal_sparse
  |     asr_phrases                -> literal_sparse + semantic_dense
  |     RAW query (untranslated)   -> semantic_dense + literal_sparse
  |                                   (dispatch.raw_query_weight)
  |-- visual chunking (retrieval.visual_chunking): any multi-sentence
  |     dense_visual sub-query (fallback spec, long paraphrase; also the
  |     whole query in the no-cortex baseline) splits into one sub-query
  |     per sentence — SigLIP reads 64 tokens and mostly the first
  |     sentence, so each sentence probes for its own moment
  |-- channels rank shots (one batched encoder call per channel)
  |-- fusion:
  |     window_ms == 0 -> weighted per-shot RRF (the old engine, bit-exact)
  |     window_ms > 0  -> temporal window fusion (retrieval.temporal):
  |                       per-video two-pointer windows; score = sum of each
  |                       sub-query's best contribution in the window
  |                       * (1 + agreement_bonus * (distinct channels - 1))
  |                       * (1 + order_bonus, when the sentence chunks hit
  |                          shots in the query's sentence order)
  |                       + video_prior_weight * overlay-index contribution
  |-- negation filter (spec negations vs evidence text)
  |-- auto VLM rerank (retrieval.auto_rerank; reuses vlm_verify; listwise
  |     over top_n with keyframes; ANY failure/timeout degrades to the
  |     fused order)
  '-- grounding: winning shot -> its best dense-visual keyframe timestamp,
      else the shot midpoint -> SearchResult(video_id, timestamp_ms)
```

KIS-V (example-clip) queries bypass the Cortex into the fusion engine with
a nearest-frame-per-target clip encoding. KIS-C sessions accumulate
revealed constraints with monotone narrowing and stale-submit protection.

**Operator escalations** (buttons/hotkeys, hard timeouts,
degrade-to-unchanged): BGE-M3 ColBERT re-rank, Imagine->Match (SDXL-Turbo
renders -> image search), VLM verify (the same listwise rerank, manual).
The QPP advisor (when `qpp.trained_artifact` is set) hints whether the current
ranking looks trustworthy.

## Measurement

- `scripts/evaluate_fixture.py` — recall@k / median rank / time-to-target
  over a labelled fixture; `--debug-top N` prints per-case hits;
  `--only-channels` and the temporal/auto-rerank switches are the ablation
  levers.
- `scripts/generate_fixture.py` — synthetic fixture: a VLM (per-profile
  `fixture_gen` endpoint) writes AIC-style Vietnamese queries for randomly
  sampled windows; labels exact by construction; the model sees only
  keyframes so queries cannot leak caption vocabulary. Keep the
  hand-labelled fixture as holdout.
- `scripts/train_qpp.py` — trains the QPP artifact on fixture outcomes.
- `scripts/warmup.py`, `scripts/measure_latency.py`,
  `scripts/faiss_gpu_parity.py` — the Phase 9/10 operational drills.

## Data layout (all under `paths.data_root`, gitignored)

```
data/
  corpus/       raw videos
  keyframes/    <video_id>/*.jpg
  embeddings/   <model-slug>/shard-*.safetensors (+ manifest)
  manifests/    videos|shots|keyframes|asr|ocr|captions|chronicle .jsonl
  indexes/      keyframes-<model>/ chronicle-semantic/ chronicle-literal/
                chronicle-overlay/
  models/       every downloaded weight (paths.models_dir; never ~/.cache)
  reports/      evaluation JSON
```

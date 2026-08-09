# Retrieval benchmark

This directory holds reproducible, labelled checks for the retrieval service.
The seed fixture is intentionally small: it is a pipeline baseline, not an
official competition test set. A team member should review and extend its
ground-truth windows before using the numbers to choose a final model.

## Visual-only baseline

- Deployment: full-corpus SigLIP2 UI on Modal
- Active retrieval channel: `dense_visual`
- Inactive channels: `semantic_dense`, `literal_sparse` (text indexes absent)
- Fixture: 16 Vietnamese KIS-T queries (6 visual, 5 OCR, 5 ASR)
- Ground truth: `chronicle.jsonl` windows, with visual cases checked against
  their keyframes

Quality result:

| slice | cases | R@1 | R@5 | R@10 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 16 | 0.062 | 0.250 | 0.312 | 0.750 |
| visual | 6 | 0.167 | 0.333 | 0.333 | 0.833 |
| OCR | 5 | 0.000 | 0.400 | 0.600 | 0.800 |
| ASR | 5 | 0.000 | 0.000 | 0.000 | 0.600 |

The ASR slice has no hit in the top 10. This is the expected failure mode of
a visual-only index and is the main reason to build the text indexes next.

## Fusion result after OCR/ASR indexing

- Deployment: full-corpus SigLIP2 + BGE-M3 UI on Modal
- Active channels: `dense_visual`, `semantic_dense`, `literal_sparse`
- Report: `reports/siglip2-fusion-service.json`

| slice | cases | R@1 | R@5 | R@10 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| all | 16 | 0.312 | 0.625 | 0.688 | 0.938 |
| visual | 6 | 0.167 | 0.333 | 0.333 | 0.833 |
| OCR | 5 | 0.200 | 0.600 | 0.800 | 1.000 |
| ASR | 5 | 0.600 | 1.000 | 1.000 | 1.000 |

Compared with visual-only, misses fall from 4 to 1, median rank from 14.0 to
3.5, and mean time-to-target from 99.2 s to 34.5 s. ASR is the clearest gain:
all five labelled speech queries land in the top three. The remaining full
miss is `visual_006`; `ocr_004` is found but only at rank 25.

The generic audit query `NGUYÊN LIỆU` is not a single-target test: Chronicle
contains 517 shots with that exact OCR phrase. The literal channel returns an
exact OCR hit at rank 1 and 85 exact hits in its top 100. `L26_V078:18` sits at
rank 80 because many ingredient screens contain the same heading plus richer
text; this is ambiguity, not an indexing failure. Reproduce the channel audit
with:

```bash
modal run scripts/modal_audit_text_retrieval.py \
  --query "NGUYÊN LIỆU" \
  --expected-shots "L26_V078:18,L26_V078:94,L26_V078:95"
```

Visual-only warm service latency (16 queries x 2 passes):

| stage | p50 | p95 |
| --- | ---: | ---: |
| wall | 1879 ms | 3141 ms |
| Cortex/Gemini compile | 1248 ms | 2515 ms |
| retrieval rank | 211 ms | 232 ms |
| response construction | 1 ms | 1 ms |

FAISS/ranking is not the main latency bottleneck in this profile. Query
compilation through Gemini dominates warm response time.

Fusion warm service latency (the same 16 queries x 2 passes):

| stage | p50 | p95 |
| --- | ---: | ---: |
| wall | 2030 ms | 3931 ms |
| Cortex/Gemini compile | 1036 ms | 3146 ms |
| retrieval rank | 499 ms | 612 ms |
| response construction | 1 ms | 1 ms |

Adding both BGE-M3 channels raises median ranking time by about 288 ms while
the median wall time rises by about 151 ms in this small run (Cortex latency
varies between passes). Query compilation remains the largest median stage;
the 500 ms end-to-end target is not met.

## Run against a deployed service

```bash
python scripts/evaluate_fixture.py \
  --config configs/t0.yaml \
  --fixture benchmarks/fixtures/siglip2-visual-seed.yaml \
  --url https://neezidow--aic-ui-siglip2-fastapi-app.modal.run \
  --report benchmarks/reports/siglip2-fusion-service.json \
  --debug-top 10
```

```bash
python scripts/measure_latency.py \
  --config configs/t0.yaml \
  --fixture benchmarks/fixtures/siglip2-visual-seed.yaml \
  --url https://neezidow--aic-ui-siglip2-fastapi-app.modal.run \
  --warmup 0 \
  --repeats 2 \
  --report benchmarks/reports/siglip2-fusion-service-latency.json
```

`--url` evaluates the exact service configuration that the UI uses. It cannot
be combined with local channel ablations: deploy the desired channel profile,
then run the same fixture against it. Gemini compilation can introduce some
run-to-run rank variation, so model comparisons should use repeated passes or
a cached/deterministic compiled query set.

## Next comparison

Keep the visual-only report unchanged as the A side. Extend the hand-reviewed
fixture, then run channel-weight ablations against the same frozen queries.
Start with `visual_006` (full miss) and `ocr_004` (rank 25), rather than tuning
against the highly ambiguous `NGUYÊN LIỆU` heading.

# Retrieval benchmark

This directory holds reproducible, labelled checks for the retrieval service.
The seed fixture is intentionally small: it is a pipeline baseline, not an
official competition test set. A team member should review and extend its
ground-truth windows before using the numbers to choose a final model.

## Current baseline

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

Warm service latency (16 queries x 2 passes):

| stage | p50 | p95 |
| --- | ---: | ---: |
| wall | 1879 ms | 3141 ms |
| Cortex/Gemini compile | 1248 ms | 2515 ms |
| retrieval rank | 211 ms | 232 ms |
| response construction | 1 ms | 1 ms |

FAISS/ranking is not the main latency bottleneck in this profile. Query
compilation through Gemini dominates warm response time.

## Run against a deployed service

```bash
python scripts/evaluate_fixture.py \
  --config configs/t0.yaml \
  --fixture benchmarks/fixtures/siglip2-visual-seed.yaml \
  --url https://neezidow--aic-ui-siglip2-fastapi-app.modal.run \
  --report benchmarks/reports/siglip2-visual-service.json \
  --debug-top 10
```

```bash
python scripts/measure_latency.py \
  --config configs/t0.yaml \
  --fixture benchmarks/fixtures/siglip2-visual-seed.yaml \
  --url https://neezidow--aic-ui-siglip2-fastapi-app.modal.run \
  --warmup 0 \
  --repeats 2 \
  --report benchmarks/reports/siglip2-visual-service-latency.json
```

`--url` evaluates the exact service configuration that the UI uses. It cannot
be combined with local channel ablations: deploy the desired channel profile,
then run the same fixture against it. Gemini compilation can introduce some
run-to-run rank variation, so model comparisons should use repeated passes or
a cached/deterministic compiled query set.

## Next comparison

Build the BGE-M3 semantic and literal indexes from the existing OCR/ASR
Chronicle, enable those two channels, then rerun the same commands. Keep this
visual-only report unchanged as the A side of that comparison.

# Ngữ cảnh dự án AIC2026 (cập nhật 2026-08-09)

## Dự án

Hệ thống truy xuất video tương tác cho cuộc thi **AI Challenge HCM 2026** — tìm đoạn video khớp truy vấn từ kho hàng trăm giờ tin tức truyền hình. Theo format Video Browser Showdown (VBS).

## Pipeline tổng quan

```
Offline: Video → Shot Detection → Keyframes → Embedding → ASR/OCR/Caption → Text Indexes
Online:  Query → Cortex (VI→EN) → Multi-channel Search → Temporal Fusion → VLM Verify → UI
```

## Hạ tầng

- **Local**: Ubuntu, GTX 1650 4GB — đã test pilot 1 video end-to-end (test.md)
- **Modal.com**: Serverless GPU cloud (L4/A10/L40S), pay-per-second, profile `new-workspace` đang active
- **Kaggle**: Notebook pilot 3-10 video (docs/kaggle-pilot.md)
- **HuggingFace**: Dataset private `Neezidow/AIC2026-keyframes` — 415 cặp TAR/report đã upload

## Modal Volumes

| Volume | Nội dung |
|---|---|
| `aic-keyframe-results` | 415 TAR + 415 report.json (7 ZIPs) |
| `aic-frames` | 415 thư mục keyframes đã giải nén |
| `aic-embeddings` | Embedding shards + FAISS index + 3 manifests |
| `aic-model-cache` | Cache model (OmniShotCut, SigLIP2...) |
| `aic-embedding-benchmarks` | (trống) |

## Trạng thái pipeline — Corpus đầy đủ (415 video)

| Bước | Trạng thái | Chi tiết |
|---|---|---|
| ① Trích keyframe (OmniShotCut) | ✅ Xong | 415 video → 56,250 shots → 171,059 keyframes |
| ② Embedding (SigLIP2) | ✅ Xong | 11,043 shards, FAISS index 171,059 vectors dim=1152 |
| ③ ASR (Faster-Whisper) | ❌ Chưa | — |
| ④ OCR (EasyOCR / PaddleOCR-VL) | ❌ Chưa | — |
| ⑤ Captioning (VLM) | ❌ Chưa | — |
| ⑥ Text Indexes (BGE-M3) | ❌ Chưa | — |
| ⑦ Online service + UI | ❌ Chưa | — |

## Số liệu corpus

- **7 archives**: Videos_L21_a, L22_a, L23_a, L24_a, L25_a, L26_a, L26_b
- **415 video** — tất cả `status=done`
- **56,250 shots** — trung bình ~136 shots/video
- **171,059 keyframes** — trung bình ~412 kf/video
- **FAISS index**: 171,059 vectors × 1152 dim, ~751 MiB (vectors.npy + index.faiss)
- **ID range**: `L21_V001_s0_f2` → `L26_V199_s116_f7648`

## Model backends (chọn qua config)

| Stage | Mặc định | SOTA option |
|---|---|---|
| Shot detection | `transnetv2` | `omnishotcut` (đang dùng) |
| Keyframe embed | `siglip2` (đang dùng) | `pe_core` (Meta) |
| ASR | `faster_whisper` large-v3 | PhoWhisper, `qwen3_asr` |
| OCR | `easyocr` | `vlm` (PaddleOCR-VL) |
| Caption | Gemini API | Qwen3-VL-8B local |
| Text embed | `bge_m3` | Qwen3-Embedding |

## Scripts chính

```bash
# Offline
python scripts/ingest_corpus.py      --config configs/t0.yaml   # Shots + keyframes
python scripts/embed_corpus.py       --config configs/t0.yaml   # SigLIP2 embedding
python scripts/build_chronicle.py    --config configs/t0.yaml   # ASR + OCR + Caption
python scripts/build_text_indexes.py --config configs/t0.yaml   # BGE-M3 indexes

# Modal
modal run scripts/modal_keyframes.py::stage  --archive <ZIP>    # Stage ZIP
modal run scripts/modal_keyframes.py::run    --archive <ZIP>    # Trích keyframe

# Online
python scripts/serve.py --config configs/t0.yaml               # FastAPI + UI :8000
```

## Bước tiếp theo cần làm

1. Chạy ASR/OCR/Captioning cho 415 video (build_chronicle)
2. Xây text indexes (build_text_indexes)
3. Serve online + test retrieval end-to-end

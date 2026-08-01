# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Lệnh (Commands)

### Thiết lập và Phát triển (Setup and Development)
- Tạo và kích hoạt môi trường ảo: `python -m venv .venv && source .venv/bin/activate`
- Cài đặt thư viện phụ thuộc: `pip install -r requirements.txt -r requirements-dev.txt`
- Cài đặt gói ở chế độ có thể chỉnh sửa (editable mode): `pip install -e .`
- Kiểm tra lỗi cú pháp và định dạng mã nguồn (lint): `ruff check .`
- Chạy toàn bộ kiểm thử (test): `pytest`
- Chạy một tệp kiểm thử duy nhất: `pytest tests/test_ingest.py`
- Chạy một trường hợp kiểm thử cụ thể: `pytest tests/test_ingest.py -k test_shot_boundary`

### Quy trình Lập chỉ mục Ngoại tuyến (Offline Indexing Pipeline)
- Trích xuất shots và keyframes từ video thô: `python scripts/ingest_corpus.py --config configs/t0.yaml --num-gpus 4`
- Trích xuất vision embeddings: `python scripts/embed_corpus.py --config configs/t0.yaml --num-gpus 4`
- Chạy ASR, OCR, captions và tổng hợp chronicle: `python scripts/build_chronicle.py --config configs/t0.yaml --num-gpus 4`
- Xây dựng chỉ mục tìm kiếm semantic, literal và overlay: `python scripts/build_text_indexes.py --config configs/t0.yaml`

### Đánh giá và Tiện ích (Evaluation and Utilities)
- Đánh giá trên fixture queries: `python scripts/evaluate_fixture.py --config configs/t0.yaml --fixture data/fixture.yaml`
- Tạo các câu truy vấn giả lập (synthetic queries): `python scripts/generate_fixture.py --config configs/t0.yaml`
- Huấn luyện mô hình dự đoán hiệu suất truy vấn (Query Performance Predictor - QPP): `python scripts/train_qpp.py --config configs/t0.yaml --fixture data/fixture.yaml --out data/qpp.json`
- Xác minh tính đồng bộ (parity) của FAISS trên GPU: `python scripts/faiss_gpu_parity.py --config configs/t0.yaml`
- Đo lường độ trễ (latency) của pipeline: `python scripts/measure_latency.py --config configs/t0.yaml --fixture data/fixture.yaml`

### Khởi chạy Backend và Giao diện Console UI
- Khởi chạy FastAPI backend: `python scripts/serve.py --config configs/t0.yaml`

---

## Kiến trúc Hệ thống (High-Level Architecture)

Hệ thống được thiết kế để tìm kiếm sự kiện video tương tác đa phương thức (KIS/VQA pipeline) phục vụ cho cuộc thi AI Challenge Ho Chi Minh City 2026.

```
                  ┌──────────────────────┐
                  │      Raw Video       │
                  └──────────┬───────────┘
                             │
                             ▼ Ingest
                  ┌──────────────────────┐
                  │ Shots and Keyframes  │
                  └──────────┬───────────┘
            ┌────────────────┼────────────────┐
            ▼ Embed          ▼ Chronicle      ▼ Audio/OCR
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │ SigLIP/Vision │ │ ASR / OCR /   │ │   Overlay     │
    │  Embeddings   │ │ VLM Captions  │ │ Text Index    │
    └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
            │                 │                 │
            ▼ FAISS Index     ▼ Text Indexes    ▼ Overlay Index
    ┌───────────────────────────────────────────────────┐
    │              Online Retrieval Pipeline            │
    │                                                   │
    │  1. Vietnamese Query -> Cortex parser             │
    │  2. Visual Chunking & Multi-channel search        │
    │  3. Temporal Window Fusion                        │
    │  4. VLM Verification / Reranking                  │
    │  5. Operator UI / VQA Grounding                   │
    └───────────────────────────────────────────────────┘
```

### Các Mô-đun Chính (Key Modules dưới thư mục `src/aic/`)
- `ingest/`: Phát hiện ranh giới shot video (TransNetV2/OmniShotCut) và loại bỏ keyframe trùng lặp bằng p-hash.
- `embed/`: Tạo vector embeddings cho keyframes bằng SigLIP-2, lưu trữ dưới dạng safetensors.
- `chronicle/`: Tích hợp ASR (faster_whisper), scene OCR (EasyOCR/PaddleOCR-VL), VLM captioning và trích xuất thông tin overlay lặp đi lặp lại (tickers/logos) vào tệp `chronicle.jsonl`.
- `textstack/`: Các tầng chỉ mục tìm kiếm sparse và dense (BGE-M3 lexical/dense, Qwen3) và bộ lọc token overlay.
- `retrieval/`: Thực hiện visual chunking (cho các truy vấn nhiều câu), tìm kiếm đa kênh (multi-channel search) và thuật toán **temporal window fusion** (xếp hạng lại các keyframes ứng viên dựa trên mức độ trùng lặp cửa sổ thời gian, thứ tự xuất hiện và mật độ).
- `cortex/`: Dịch và phân tích cú pháp truy vấn tiếng Việt thành các thông số truy vấn cấu trúc (retrieval specs).
- `vqa/`: Trả lời câu hỏi và định vị phân đoạn khớp trong video ứng viên.
- `service/`: Các API endpoint của FastAPI để kết nối luồng xử lý truy vấn với giao diện điều khiển của người vận hành.

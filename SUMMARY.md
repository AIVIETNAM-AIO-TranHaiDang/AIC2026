# Tổng hợp dự án AIC2026_prepare

## 1. Tổng quan bài toán

Đây là codebase chuẩn bị cho cuộc thi **AI Challenge Hồ Chí Minh (AIC/HCMAI) 2026** — bài toán **Truy xuất sự kiện từ bộ sưu tập video lớn** (*Event Retrieval from a Large Video Collection*).

**Mục tiêu:** Xây dựng một hệ thống truy xuất video tương tác có sự tham gia của con người (human-in-the-loop). Trong mỗi vòng thi có giới hạn thời gian, người vận hành (operator) sử dụng công cụ để tìm đúng đoạn video (hoặc trả lời câu hỏi liên quan) khớp với truy vấn từ ban tổ chức, trên một kho dữ liệu hàng trăm giờ video tin tức truyền hình.

Cuộc thi theo định dạng **Video Browser Showdown (VBS)** và **Lifelog Search Challenge (LSC)**.

### Các định dạng truy vấn/nhiệm vụ

| Định dạng | Mô tả |
| --- | --- |
| **KIS-V** (Known-Item Search - Visual) | Tìm đúng đoạn video dựa trên một clip ví dụ ngắn. |
| **KIS-T** (Known-Item Search - Text) | Tìm đúng đoạn video dựa trên mô tả văn bản tiếng Việt. |
| **KIS-C** (Known-Item Search - Contextual) | Giống KIS-T, nhưng thông tin được tiết lộ dần trong vòng thi. |
| **VQA / Q&A** (Visual Question Answering) | Trả lời câu hỏi về video (truy xuất + đọc/suy luận). |
| **TRAKE** | Trả về chuỗi các khoảnh khắc theo thứ tự cho sự kiện nhiều bước. |

---

## 2. Pipeline tham khảo

### Giai đoạn Offline (Lập chỉ mục)

1. **Phát hiện ranh giới cảnh (Shot Boundary Detection)** + trích xuất khung hình chính (keyframe)
2. **Lọc keyframe** (loại bỏ khung hình trùng/trống)
3. **Tạo embedding vision-language** (CLIP family / SigLIP2 / BEiT-3) + tùy chọn nhận diện đối tượng / OCR / ASR
4. **Lưu metadata** vào manifest
5. **Xây dựng chỉ mục vector** (FAISS hoặc vector DB)

### Giai đoạn Online (Truy xuất)

1. **Hiểu truy vấn** — dịch Việt→Anh cho văn bản, embedding cho visual
2. **Tìm kiếm ANN** (Approximate Nearest Neighbor) trên vector index
3. **Re-ranking + temporal fusion** — xếp hạng lại kết quả
4. **Giao diện tương tác** gửi câu trả lời `(video_id, timestamp)`

---

## 3. Cấu trúc thư mục

```
AIC2026_prepare/
├── CLAUDE.md                # Hướng dẫn cho AI agent và lập trình viên
├── README.md                # Tệp giới thiệu dự án
├── requirements.txt         # Các dependency runtime
├── requirements-dev.txt     # Công cụ test/chất lượng (pytest, ruff)
├── requirements-gpu.txt     # Backend dành cho GPU server
├── implementation-notes.md  # Nhật ký quyết định và đánh đổi
├── pyproject.toml  u         # Metadata package, cấ hình pytest và ruff
├── configs/                 # Cấu hình profile (t0.yaml) và ví dụ fixture
├── src/aic/                 # Package pipeline chính
├── scripts/                 # Các script thực thi
├── tests/                   # Bộ test Pytest với fixture video tổng hợp
├── data/                    # Corpus, keyframes, manifests (gitignored)
└── docs/                    # Ghi chú nghiên cứu và slide tham khảo
```

---

## 4. Chức năng các module chính (`src/aic/`)

| Module | Chức năng |
| --- | --- |
| **`config.py`** | Quản lý cấu hình toàn bộ pipeline, đọc và xác thực file YAML config. |
| **`ingest/`** | Giai đoạn nhập liệu: phát hiện ranh giới cảnh (shot detection), trích xuất keyframe, lọc keyframe trùng lặp. |
| **`embed/`** | Tạo vector embedding cho keyframe sử dụng mô hình vision-language (SigLIP2, Perception Encoder). |
| **`chronicle/`** | Xây dựng "chronicle" (biên niên sử) cho video: chạy ASR (nhận dạng giọng nói), OCR (nhận dạng chữ), và captioning (mô tả hình ảnh). |
| **`textstack/`** | Tạo text embedding (BGE-M3 dense+sparse) và xây dựng chỉ mục văn bản cho tìm kiếm. |
| **`index/`** | Xây dựng và quản lý chỉ mục FAISS cho tìm kiếm vector. |
| **`retrieval/`** | Xử lý truy xuất online: tìm kiếm ANN, rank fusion, temporal window fusion, visual chunking. |
| **`cortex/`** | "Query Cortex" — hiểu và xử lý truy vấn (dịch Việt→Anh, phân tích truy vấn). |
| **`service/`** | Web server và giao diện operator console, API endpoints (`/api/search`, `/api/qa`). |
| **`eval/`** | Đánh giá hiệu năng truy xuất trên fixture được gán nhãn. |
| **`escalate/`** | Các công cụ leo thang (escalation): VLM verify (xác minh bằng vision-language model), QPP (Query Performance Prediction). |
| **`vqa/`** | Module trả lời câu hỏi video (Visual Question Answering). |
| **`parallel.py`** | Hỗ trợ xử lý song song multi-GPU (work-stealing queue). |
| **`oaicompat.py`** | Client tương thích OpenAI API cho các endpoint LLM cục bộ. |
| **`manifest.py`** | Quản lý manifest — theo dõi tiến trình xử lý, hỗ trợ resume khi bị gián đoạn. |
| **`verify.py`** | Kiểm tra và sửa chữa các artifact bị hỏng khi resume sau gián đoạn. |

---

## 5. Chức năng các script chính (`scripts/`)

| Script | Chức năng |
| --- | --- |
| **`ingest_corpus.py`** | Nhập video, phát hiện cảnh, trích xuất keyframe. |
| **`embed_corpus.py`** | Tạo embedding vector cho keyframe. |
| **`build_chronicle.py`** | Chạy ASR, OCR, captioning cho toàn bộ corpus. |
| **`build_text_indexes.py`** | Xây dựng chỉ mục văn bản (BM25, dense embedding). |
| **`evaluate_fixture.py`** | Đánh giá hiệu năng trên fixture được gán nhãn. |
| **`serve.py`** | Khởi chạy web server và giao diện operator. |
| **`serve_local_llms.py`** | Khởi chạy LLM server cục bộ (llama-server) trên nhiều GPU. |
| **`train_qpp.py`** | Huấn luyện mô hình QPP (dự đoán hiệu năng truy vấn). |
| **`generate_fixture.py`** | Tạo fixture tổng hợp bằng VLM. |
| **`warmup.py`** | Khởi động nóng (cold-start drill). |
| **`measure_latency.py`** | Đo độ trễ pipeline. |
| **`faiss_gpu_parity.py`** | Kiểm tra tính nhất quán FAISS CPU/GPU. |

---

## 6. Các model backend có thể chọn

| Giai đoạn | Mặc định | Tùy chọn SOTA (chuyển qua config) |
| --- | --- | --- |
| Phát hiện cảnh | `transnetv2` (chạy được CPU) | `omnishotcut` (SOTA 2026, chỉ GPU) |
| Embedding keyframe | `siglip2` | `pe_core` (Meta Perception Encoder) |
| ASR | `faster_whisper` large-v3 | PhoWhisper hoặc `qwen3_asr` |
| OCR | `easyocr` | `vlm` — PaddleOCR-VL-1.6 qua vLLM |
| Captioning | Gemini API | Endpoint tương thích OpenAI (Qwen3-VL-8B local, ...) |
| Text embedding | `bge_m3` (dense+sparse) | Qwen3-Embedding |

---

## 7. Hướng dẫn cài đặt và thực thi

### 7.1. Cài đặt môi trường

```bash
# Tạo virtual environment
python -m venv .venv

# Kích hoạt (Linux / macOS)
source .venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

### 7.2. Kiểm tra code

```bash
pytest                # Chạy test suite
ruff check .          # Kiểm tra style/quality
```

### 7.3. Chạy pipeline offline (lập chỉ mục)

Đặt video đầu vào (mp4/mkv/avi/mov/webm) vào thư mục `data/videos/`, sau đó chạy lần lượt:

```bash
# Bước 1: Nhập video, trích xuất keyframe
python scripts/ingest_corpus.py     --config configs/t0.yaml

# Bước 2: Tạo embedding cho keyframe
python scripts/embed_corpus.py      --config configs/t0.yaml

# Bước 3: Xây dựng chronicle (ASR, OCR, captioning)
python scripts/build_chronicle.py   --config configs/t0.yaml

# Bước 4: Xây dựng chỉ mục văn bản
python scripts/build_text_indexes.py --config configs/t0.yaml

# (Tùy chọn) Đánh giá trên fixture
python scripts/evaluate_fixture.py  --config configs/t0.yaml --fixture data/fixture.yaml
```

> **Lưu ý:** Mỗi bước đều hỗ trợ **resume** — nếu bị gián đoạn (ví dụ: Colab bị ngắt), chỉ cần chạy lại cùng lệnh, các video đã xử lý sẽ được bỏ qua.

### 7.4. Chạy với nhiều GPU

```bash
python scripts/ingest_corpus.py   --config configs/t0.yaml --num-gpus 4
python scripts/embed_corpus.py    --config configs/t0.yaml --num-gpus 4
python scripts/build_chronicle.py --config configs/t0.yaml --num-gpus 4
```

### 7.5. Khởi chạy dịch vụ online

```bash
# Khởi chạy web server + giao diện operator
python scripts/serve.py --config configs/t0.yaml
# Truy cập UI tại http://127.0.0.1:8000/
```

### 7.6. Các script vận hành

```bash
# Khởi động nóng
python scripts/warmup.py          --config configs/t0.yaml

# Đo độ trễ
python scripts/measure_latency.py --config configs/t0.yaml --fixture data/fixture.yaml

# Kiểm tra FAISS GPU parity (chỉ GPU server)
python scripts/faiss_gpu_parity.py --config configs/t0.yaml
```

### 7.7. Xây dựng evaluation fixture

```bash
# Copy file ví dụ và gán nhãn ít nhất 30 truy vấn KIS
cp configs/fixture.example.yaml data/fixture.yaml
# Chỉnh sửa data/fixture.yaml với các truy vấn đã gán nhãn

# Huấn luyện QPP
python scripts/train_qpp.py --config configs/t0.yaml \
    --fixture data/fixture.yaml --out data/qpp.json
```

---

## 8. Các tính năng nổi bật

- **Resumable:** Mọi giai đoạn offline đều có thể tiếp tục từ điểm dừng.
- **Multi-GPU:** Hỗ trợ phân tải trên nhiều GPU bằng work-stealing queue.
- **Multi-endpoint fan-out:** Các stage dùng API (captioning, VLM OCR) có thể phân tải qua nhiều endpoint LLM.
- **Verify & Repair:** Tự động phát hiện và sửa artifact bị hỏng khi resume (`verify.on_resume: true`).
- **Streaming API:** Hỗ trợ NDJSON streaming cho kết quả tìm kiếm tăng dần.
- **Temporal Fusion:** Xếp hạng theo cửa sổ thời gian, thưởng cho sự nhất quán đa kênh (cross-channel triangulation).
- **Bilingual Chronicle:** Hỗ trợ song ngữ Việt-Anh cho caption và truy vấn.

---

## 9. Bảng tổng hợp luồng hệ thống

### Nguyên tắc chia để trị (3 thư mục chính)

```
configs/   ─── Chứa tham số, flags (chọn model, đường dẫn, ngưỡng, ...)
                        │
                        ▼
scripts/   ─── File thực thi chính: đọc config → gọi hàm trong src → chạy
                        │
                        ▼
src/aic/   ─── Định nghĩa hàm, class (logic xử lý thực sự)
```

### 9.1. Giai đoạn OFFLINE (Lập chỉ mục — chạy một lần trước khi thi)

> Mục tiêu: Biến kho video thô → cơ sở dữ liệu có thể tìm kiếm được.

| Bước | Thư mục / File liên quan | Cách thực thi | Models áp dụng |
| --- | --- | --- | --- |
| **① Chuẩn bị dữ liệu** | `data/videos/` — đặt video vào đây | Thủ công: copy file mp4/mkv/avi/mov/webm vào thư mục | — |
| **② Phát hiện ranh giới cảnh** (Shot Boundary Detection) | **config:** `configs/t0.yaml` → mục `ingest.shots` | `python scripts/ingest_corpus.py --config configs/t0.yaml` | **Mặc định:** `transnetv2` (chạy CPU) · **SOTA:** `omnishotcut` (GPU-only, 2026) |
|  | **src:** `src/aic/ingest/shots.py` — logic cắt video thành từng cảnh |  |  |
|  | **src:** `src/aic/ingest/video.py` — đọc/decode video |  |  |
| **③ Trích xuất Keyframe** | **config:** `configs/t0.yaml` → mục `ingest.keyframes` | Chạy chung lệnh bước ② (cùng `ingest_corpus.py`) | Thuật toán lấy mẫu khung hình (không dùng model ML) |
|  | **src:** `src/aic/ingest/keyframes.py` — chọn khung hình đại diện |  |  |
|  | **src:** `src/aic/ingest/dedup.py` — loại bỏ keyframe trùng/trống |  |  |
| **④ Tạo Embedding hình ảnh** (Feature Extraction) | **config:** `configs/t0.yaml` → mục `embed` | `python scripts/embed_corpus.py --config configs/t0.yaml` | **Mặc định:** `siglip2` (Google SigLIP v2) · **SOTA:** `pe_core` (Meta Perception Encoder) |
|  | **src:** `src/aic/embed/encoders.py` — load model, encode ảnh → vector |  |  |
|  | **src:** `src/aic/embed/job.py` — quản lý job embedding |  |  |
|  | **src:** `src/aic/embed/shard_io.py` — lưu/đọc vector (safetensors) |  |  |
| **⑤ Xây dựng Chronicle** (ASR + OCR + Captioning) | **config:** `configs/t0.yaml` → mục `chronicle` | `python scripts/build_chronicle.py --config configs/t0.yaml` | *(xem chi tiết bên dưới)* |
|  | **src:** `src/aic/chronicle/asr.py` — nhận dạng giọng nói | *ASR:* | **Mặc định:** `faster_whisper` large-v3 · **Thay thế:** PhoWhisper, `qwen3_asr` |
|  | **src:** `src/aic/chronicle/ocr.py` — nhận dạng chữ trong ảnh | *OCR:* | **Mặc định:** `easyocr` · **Thay thế:** `vlm` (PaddleOCR-VL-1.6 qua vLLM) |
|  | **src:** `src/aic/chronicle/caption.py` — mô tả hình ảnh bằng văn bản | *Captioning:* | **Mặc định:** Gemini API · **Thay thế:** Qwen3-VL-8B local, OpenAI API |
|  | **src:** `src/aic/chronicle/translate.py` — dịch caption EN↔VI | *Dịch:* | NLLB (offline translation) |
|  | **src:** `src/aic/chronicle/entities.py` — trích xuất thực thể |  |  |
| **⑥ Xây dựng chỉ mục văn bản** (Text Indexing) | **config:** `configs/t0.yaml` → mục `textstack` | `python scripts/build_text_indexes.py --config configs/t0.yaml` | **Mặc định:** `bge_m3` (dense + sparse) · **Thay thế:** Qwen3-Embedding |
|  | **src:** `src/aic/textstack/embedder.py` — tạo text embedding |  |  |
|  | **src:** `src/aic/textstack/job.py` — quản lý job indexing |  |  |
|  | **src:** `src/aic/index/vector.py` — xây dựng FAISS index |  |  |
|  | **src:** `src/aic/index/sparse.py` — xây dựng BM25/sparse index |  |  |
| **⑦ Đánh giá (tùy chọn)** | **config:** `configs/fixture.example.yaml` → mẫu fixture | `python scripts/evaluate_fixture.py --config configs/t0.yaml --fixture data/fixture.yaml` | Dùng lại các model ở bước ④⑤⑥ |
|  | **src:** `src/aic/eval/` — logic đánh giá Recall, MRR, ... |  |  |

**Tóm tắt thứ tự lệnh Offline:**

```bash
python scripts/ingest_corpus.py      --config configs/t0.yaml   # ②③ Cắt cảnh + keyframe
python scripts/embed_corpus.py       --config configs/t0.yaml   # ④   Tạo vector hình ảnh
python scripts/build_chronicle.py    --config configs/t0.yaml   # ⑤   ASR + OCR + Caption
python scripts/build_text_indexes.py --config configs/t0.yaml   # ⑥   Chỉ mục văn bản
```

---

### 9.2. Giai đoạn ONLINE (Truy xuất — chạy trong lúc thi)

> Mục tiêu: Nhận truy vấn từ giám khảo → tìm đúng đoạn video → trả kết quả nhanh nhất.

| Bước | Thư mục / File liên quan | Cách thực thi | Models áp dụng |
| --- | --- | --- | --- |
| **① Khởi chạy Server** | **config:** `configs/t0.yaml` → mục `service` | `python scripts/serve.py --config configs/t0.yaml` → truy cập `http://127.0.0.1:8000/` | — |
|  | **src:** `src/aic/service/app.py` — web server FastAPI |  |  |
|  | **src:** `src/aic/service/engine.py` — khởi tạo search engine |  |  |
|  | **src:** `src/aic/service/static/` — giao diện operator console |  |  |
| **② Hiểu truy vấn** (Query Understanding) | **config:** `configs/t0.yaml` → mục `cortex` | Tự động khi operator nhập query trên giao diện | VLM / LLM qua OpenAI-compatible endpoint |
|  | **src:** `src/aic/cortex/compiler.py` — phân tích & biên dịch truy vấn |  |  |
|  | **src:** `src/aic/cortex/dispatch.py` — điều phối sub-queries |  |  |
|  | **src:** `src/aic/cortex/spec.py` — định nghĩa query spec |  |  |
| **③ Tìm kiếm Vector** (ANN Search) | **config:** `configs/t0.yaml` → mục `retrieval` | Tự động (gọi qua API `/api/search`) | `siglip2` hoặc `pe_core` (encode query) + FAISS (tìm kiếm ANN) |
|  | **src:** `src/aic/retrieval/dense.py` — tìm kiếm dense vector |  |  |
|  | **src:** `src/aic/retrieval/chunking.py` — chia query nhiều câu thành sub-queries |  |  |
| **④ Xếp hạng & Kết hợp** (Rank Fusion) | **config:** `configs/t0.yaml` → mục `retrieval.temporal`, `retrieval.visual_chunking` | Tự động (sau bước ③) | Thuật toán RRF, temporal window fusion |
|  | **src:** `src/aic/retrieval/fusion.py` — kết hợp kết quả đa kênh (visual + text) |  |  |
|  | **src:** `src/aic/retrieval/temporal.py` — xếp hạng theo cửa sổ thời gian |  |  |
| **⑤ Leo thang (Escalation)** — *tùy chọn, kích hoạt bằng nút/phím tắt* | **config:** `configs/t0.yaml` → mục `escalations` | Operator bấm nút trên giao diện | VLM (xác minh kết quả), QPP model |
|  | **src:** `src/aic/escalate/verify.py` — VLM xác minh top kết quả |  |  |
|  | **src:** `src/aic/escalate/rerank.py` — VLM re-rank listwise |  |  |
|  | **src:** `src/aic/escalate/qpp.py` — dự đoán hiệu năng truy vấn |  |  |
|  | **src:** `src/aic/escalate/imagine.py` — tưởng tượng/gợi ý truy vấn |  |  |
| **⑥ Trả lời câu hỏi (VQA)** — *chỉ dùng cho task VQA/Q&A* | **config:** `configs/t0.yaml` → mục `vqa` | Tự động (gọi qua API `/api/qa`) | VLM / LLM (đọc keyframe + suy luận trả lời) |
|  | **src:** `src/aic/vqa/reader.py` — đọc keyframe, trả lời câu hỏi |  |  |
|  | **src:** `src/aic/vqa/evidence.py` — thu thập bằng chứng |  |  |
|  | **src:** `src/aic/vqa/vote.py` — bỏ phiếu/tổng hợp câu trả lời |  |  |
|  | **src:** `src/aic/vqa/count.py` + `reverse.py` — đếm/truy ngược |  |  |
| **⑦ Gửi kết quả** | **src:** `src/aic/service/submission.py` — gửi `(video_id, timestamp)` | Operator chọn kết quả → bấm Submit trên giao diện | — |
|  | **src:** `src/aic/service/sessions.py` — quản lý phiên thi |  |  |

**Tóm tắt luồng Online:**

```
Operator nhập truy vấn (tiếng Việt)
        │
        ▼
  ② Cortex: Dịch VI→EN, phân tích → tạo sub-queries
        │
        ├──→ ③a Dense search (SigLIP2 visual vector)
        ├──→ ③b Sparse search (BM25 text)
        ├──→ ③c Dense text search (BGE-M3 embedding)
        │
        ▼
  ④ Fusion: Kết hợp kết quả đa kênh → xếp hạng
        │
        ▼
  ⑤ (Tùy chọn) VLM verify / Re-rank / QPP
        │
        ▼
  ⑥ (Nếu VQA) Đọc keyframe → suy luận → trả lời
        │
        ▼
  ⑦ Hiển thị kết quả → Operator chọn → Submit (video_id, timestamp)
```

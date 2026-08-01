# Chạy pilot end-to-end trên Kaggle

Hướng dẫn này chạy 3–10 video qua toàn bộ pipeline local:

```text
Kaggle Dataset video
  -> shot/keyframe ingest
  -> SigLIP2 embeddings + exact FAISS index
  -> Faster-Whisper ASR + EasyOCR Chronicle
  -> BGE-M3 semantic/literal indexes
  -> FastAPI Operator UI
```

Entry point: [`notebooks/kaggle_pilot.ipynb`](../notebooks/kaggle_pilot.ipynb). Profile: [`configs/t0-kaggle.yaml`](../configs/t0-kaggle.yaml).

## 1. Cần cung cấp gì

Bắt buộc:

1. URL GitHub **public** của repository này.
2. Một Kaggle Dataset đã attach, có ít nhất 3 video (`mp4`, `mkv`, `avi`, `mov` hoặc `webm`). Notebook liên kết tối đa 10 file đầu theo thứ tự đường dẫn; trỏ vào subfolder riêng nếu muốn chọn tập pilot cụ thể.
3. Đường dẫn mount cụ thể, ví dụ `/kaggle/input/aic-pilot-videos/videos`.
4. Một truy vấn mô tả cảnh thực tế trong một video pilot để smoke-test API.

Tùy chọn:

- Kaggle Dataset chứa artifacts từ session trước, nếu cần resume.
- Fixture được gán nhãn thật nếu cần đo Recall@K/rank. Notebook không tạo fixture giả vì nhãn timestamp tùy ý tạo metric vô nghĩa.
- Cloudflare account/domain hoặc ngrok account nếu cần URL ổn định. Pilot mặc định dùng Cloudflare Quick Tunnel, không cần tài khoản.

Không cần trong cấu hình mặc định:

- Kaggle API token.
- Gemini/OpenAI API key.
- Hugging Face token.
- ngrok authtoken.
- Cloudflare API/tunnel token.
- Milvus hoặc vector database riêng.

Kaggle API token chỉ cần khi tự động hóa upload/version Dataset, push notebook hoặc tải dữ liệu bằng CLI từ một máy khác.

## 2. Chuẩn bị Kaggle Notebook

1. Tạo Notebook, attach video Dataset.
2. Chọn GPU accelerator. Bắt đầu bằng một GPU; thử hai GPU sau khi smoke test qua.
3. Bật Internet để cài dependency và tải model.
4. Import notebook từ [`notebooks/kaggle_pilot.ipynb`](../notebooks/kaggle_pilot.ipynb).
5. Sửa duy nhất các biến đầu notebook:
   - `KAGGLE_VIDEO_DIR`.
   - `REPO_URL`.
   - `PREVIOUS_DATA_ROOT` nếu resume.
   - `MANUAL_QUERY`.
6. Chạy cell theo thứ tự. Cell nào lỗi thì dừng ở đó; không bỏ qua validation rồi tiếp tục stage sau.

Video input nằm trong `/kaggle/input`, là read-only. Notebook tạo symlink từng file vào:

```text
/kaggle/working/data/videos/<filename>
```

Không copy raw video sang `/kaggle/working`, tránh nhân đôi dung lượng.

## 3. Cấu hình pilot

[`configs/t0-kaggle.yaml`](../configs/t0-kaggle.yaml) là bản đầy đủ của profile T0, với các thay đổi:

- `paths.data_root: /kaggle/working/data`.
- Model cache tại `/kaggle/working/model-cache`, tách khỏi artifacts cần persist.
- SigLIP2 shards lưu `float16`; index exact vẫn dùng/lưu `float32`.
- FAISS chạy CPU. `requirements.txt` pin `faiss-cpu`; không thay package trong pilot.
- Faster-Whisper `medium` cho vòng đầu; sau khi đo được runtime mới thử `large-v3`.
- Caption, translation, Cortex API, VQA và các escalation nặng đều tắt.
- Query Cortex tắt sẽ fallback về raw query, nên baseline KIS vẫn chạy mà không cần API key.
- `default_top_k: 20` để giảm số JPEG đi qua tunnel.
- NDJSON streaming tắt ở vòng đầu; UI dùng route JSON one-shot.

Không tạo YAML rút gọn. Config loader dùng Pydantic strict và profile đầy đủ giữ hành vi rõ ràng khi code thay đổi.

## 4. Quy trình hai nhịp

### Nhịp A: 3 video, một GPU

Notebook thực hiện:

```bash
python scripts/ingest_corpus.py \
  --config configs/t0-kaggle.yaml --num-gpus 1 --limit 3

python scripts/embed_corpus.py \
  --config configs/t0-kaggle.yaml --num-gpus 1

python scripts/build_chronicle.py \
  --config configs/t0-kaggle.yaml --num-gpus 1 --skip-entities

python scripts/check_corpus.py \
  --config configs/t0-kaggle.yaml

python scripts/build_text_indexes.py \
  --config configs/t0-kaggle.yaml

python scripts/serve.py \
  --config configs/t0-kaggle.yaml
```

`--limit` là cap single-process. Code cố ý bỏ qua `--limit` khi dùng nhiều GPU. Embed không giới hạn keyframe để index đại diện đầy đủ ba video đã ingest.

Không chạy Chronicle với đồng thời `--skip-asr --skip-ocr` nếu mục tiêu là kiểm chứng text retrieval. Không caption, ASR/OCR là hai nguồn text còn lại; nếu bỏ cả hai, Chronicle có thể hợp lệ nhưng semantic/literal index rỗng.

### Nhịp B: 10 video, thử hai GPU

Sau Nhịp A:

1. Chuẩn bị Dataset pilot đúng 10 video.
2. Bỏ `--limit`.
3. Thử `--num-gpus 2` cho ingest, embed và Chronicle.
4. Nếu OOM, quay về một GPU hoặc giảm `embed.batch_size`/`textstack.batch_size`.

Code đã có shared work queue, một worker/card và rank manifests. Không cần tự chia video hoặc viết sharding khác.

## 5. Artifacts và kiểm tra dung lượng

Artifacts được ghi dưới `/kaggle/working/data`:

```text
data/
  keyframes/<video_id>/*.jpg
  manifests/videos.jsonl
  manifests/shots.jsonl
  manifests/keyframes.jsonl
  manifests/asr.jsonl
  manifests/ocr.jsonl
  manifests/chronicle.jsonl
  embeddings/keyframes/<model-slug>/manifest.jsonl
  embeddings/keyframes/<model-slug>/shard-*.safetensors
  indexes/keyframes-<model-slug>/meta.json
  indexes/keyframes-<model-slug>/ids.json
  indexes/keyframes-<model-slug>/vectors.npy
  indexes/keyframes-<model-slug>/index.faiss
  indexes/chronicle-semantic/
  indexes/chronicle-literal/
  reports/kaggle-inventory.json
```

Lưu ý dung lượng: vector vision tồn tại ở embedding shards và được materialize lại trong index directory. [`VectorIndex.save()`](../src/aic/index/vector.py) lưu cả `vectors.npy` và `index.faiss`. Vì vậy không thể kết luận trước rằng corpus bất kỳ chắc chắn dưới một ngưỡng disk. Notebook đo từng directory và ghi inventory sau khi chạy.

Sau mỗi stage, kiểm tra:

- Ingest: ba manifest tồn tại; JPEG sample decode được.
- Embed: manifest/shards tồn tại; `meta.json`, `ids.json`, `vectors.npy`, `index.faiss` đầy đủ.
- Chronicle: ASR/OCR/Chronicle manifests có dữ liệu; `check_corpus.py` báo coverage.
- Text indexes: ít nhất một index Chronicle tồn tại. Overlay index có thể vắng hợp lệ.
- Online: `/api/search` trả HTTP 200, có `video_id`, `timestamp_ms`; `/frames/...` trả image.

## 6. Persist và resume

Persist các thư mục:

```text
data/keyframes/
data/manifests/
data/embeddings/
data/indexes/
data/reports/
```

Không persist:

```text
data/videos/                  # chỉ là symlink tới input Dataset
/kaggle/working/model-cache/  # có thể tải lại
```

Sau khi thử UI/tunnel xong, chạy cell `PREPARE_OUTPUT_FOR_SAVE` ở cuối notebook với giá trị `True`. Cell dừng server/tunnel, xóa video symlinks và model cache nhưng giữ toàn bộ artifacts.

Vòng đầu dùng Kaggle UI để Save Version/tạo Dataset từ output. Session sau:

1. Attach cùng version video Dataset.
2. Attach artifact Dataset vừa lưu.
3. Đặt `PREVIOUS_DATA_ROOT` trong notebook.
4. Restore artifacts rồi chạy lại cùng command.
5. Manifest sẽ skip video/keyframe đã hoàn thành.

`reports/kaggle-inventory.json` ghi Git commit SHA, hash config, input path, video IDs, model IDs và dung lượng artifacts. Không resume nếu code/model/input đã đổi mà chưa đánh giá tương thích.

Nếu session bị kill giữa lúc ghi file:

1. Tạo bản config tạm có `verify.on_resume: true`.
2. Chạy lại stage bị ngắt; integrity pass loại artifacts thiếu/hỏng và manifest rows phụ thuộc.
3. Sau khi repair/resume xong, tắt lại vì verify đọc toàn bộ artifacts một lần.

Các rank manifest còn lại từ multi-GPU được code merge ở đầu lần chạy sau.

## 7. Kaggle API, ngrok và Cloudflare

### Kaggle API

Không cần cho notebook tương tác và Dataset đã attach. Chỉ cần `kaggle.json` khi tự động hóa bằng Kaggle CLI/API. Không commit `kaggle.json` hoặc đưa nó vào notebook output.

Tài liệu chính thức: <https://www.kaggle.com/docs/notebooks>

### Cloudflare Quick Tunnel

Lựa chọn mặc định cho demo ngắn:

```bash
cloudflared tunnel --url http://127.0.0.1:8000 --no-autoupdate
```

- Không cần tài khoản hoặc token.
- URL `*.trycloudflare.com` ngẫu nhiên, mất khi process dừng.
- Không có cam kết production/SLA.
- Quick Tunnel có các giới hạn/khả năng khác Named Tunnel; kiểm tra trang chính thức tại thời điểm chạy. Không mặc định coi bandwidth, request concurrency hoặc streaming là vô hạn.

Tài liệu chính thức:

- Quick Tunnels: <https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/>
- Account limits: <https://developers.cloudflare.com/cloudflare-one/account-limits/>

### Cloudflare Named Tunnel

Chỉ cần khi muốn hostname/domain ổn định. Cần Cloudflare account, domain/DNS phù hợp và tunnel token. Đưa token vào Kaggle Secrets, không ghi trực tiếp vào code.

### ngrok

Lựa chọn phụ. Thường cần account và authtoken, đồng thời hạn mức phụ thuộc plan hiện hành. Không dùng các con số cũ như request/phút hoặc data transfer/tháng nếu chưa xác nhận trên pricing/limits chính thức.

Tài liệu chính thức:

- <https://ngrok.com/pricing>
- <https://ngrok.com/docs/pricing-limits/>

### Lưu lượng của UI

UI trả tối đa nhiều kết quả, mỗi kết quả có các JPEG keyframe qua `/frames`. Trình duyệt có thể tạo nhiều GET request dù JSON search nhỏ. Giữ `default_top_k: 20` hoặc thấp hơn khi demo qua tunnel.

Route streaming dùng NDJSON `StreamingResponse`. Quick Tunnel có thể thay đổi buffering/compatibility và tài liệu Cloudflare từng nêu hạn chế riêng cho Quick Tunnels. Pilot giữ `service.streaming: false`; chỉ bật sau khi test thực tế `/api/search/stream` qua tunnel. Route `/api/search` one-shot luôn là fallback.

Không dùng tunnel để stream raw video hoặc cung cấp toàn corpus.

## 8. Bảo mật

Quick Tunnel URL là public và Operator UI hiện không có authentication. Khi tunnel đang mở, người có URL có thể gọi API và xem keyframe được service phục vụ.

- Chỉ mở sau internal smoke test.
- Không dùng video nhạy cảm.
- Không chia sẻ URL ngoài nhóm thử nghiệm.
- Dừng `cloudflared` và FastAPI khi xong.
- Không đặt secret trong config, notebook, log hoặc Dataset Output.

## 9. Troubleshooting

| Triệu chứng | Nguyên nhân thường gặp | Xử lý |
| --- | --- | --- |
| `videos directory not found` | `KAGGLE_VIDEO_DIR` sai hoặc symlink chưa tạo | Kiểm tra mount path và cell restore/link |
| Có hơn 10 video | Trỏ vào root Dataset lớn | Notebook lấy 10 file đầu; trỏ vào subfolder pilot nếu cần chọn subset khác |
| Trùng `video_id` | Hai file khác folder có cùng stem | Đổi tên file; stem là `video_id` |
| `--limit is ignored` | Dùng hơn một GPU | Nhịp A dùng `--num-gpus 1`; Nhịp B bỏ limit |
| CUDA OOM | Hai worker cùng nạp model hoặc batch lớn | Dùng một GPU, giảm batch size |
| FAISS CUDA error | Cấu hình `index.device: cuda` nhưng cài `faiss-cpu` | Giữ `index.device: cpu` trong pilot |
| Text indexes rỗng | Chronicle không có ASR/OCR/caption text | Không skip cả ASR và OCR; xem `check_corpus.py` |
| Manifest path không tồn tại | Attach video Dataset với slug/path khác session trước | Attach cùng Dataset version/path hoặc chạy artifacts sạch |
| `/kaggle/working` đầy | Nhiều JPEG, model cache hoặc bản sao vector | Xem inventory; giảm số video/keyframe, xóa model cache trước Save Version |
| Server startup chậm | Service nạp SigLIP2 và BGE-M3 | Đợi internal readiness loop; xem `aic-server.log` |
| Tunnel mở nhưng stream không incremental | Quick Tunnel buffering/stream limitation | Giữ streaming off, dùng `/api/search` one-shot |

## 10. Nâng cấp sau pilot

Chỉ nâng cấp sau khi đã đo wall time, peak RAM/VRAM, disk và retrieval baseline:

1. Faster-Whisper `large-v3` hoặc PhoWhisper.
2. Hai GPU cho offline stages.
3. FAISS GPU sau parity test; serialization vẫn lấy CPU index làm nguồn sự thật.
4. Gemini/local VLM caption và Cortex.
5. Streaming, VLM rerank, Answer Ledger/VQA.
6. Fixture gán nhãn thật và ablation đo Recall@K/latency.

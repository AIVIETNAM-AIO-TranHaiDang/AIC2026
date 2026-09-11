# AIC2026 — Kết quả pipeline dùng cho retrieval

Snapshot: 2026-08-10. UI đã dừng, nhưng các Modal Volume bên dưới vẫn giữ dữ
liệu; dừng container **không** làm mất chúng.

## Kết luận nhanh

- Keyframe đã được đóng gói và lưu trên Hugging Face:
  `Neezidow/AIC2026-keyframes` (17 gói, khoảng 14.3 GB).
- Bản dữ liệu dùng trực tiếp cho retrieval hiện nằm trong Modal workspace, ở
  các Volume `aic-frames` và `aic-embeddings`.
- Người chỉ cần xem/tải keyframe: dùng Hugging Face.
- Người cần chạy retrieval/UI như bản đã test: cần quyền truy cập Modal Volume
  hoặc một retrieval bundle được export từ các Volume này. Link GitHub chỉ có
  code, không chứa các dữ liệu lớn bên dưới.

## 1. Keyframe và manifest gốc

Mỗi video sau OmniShotCut có cấu trúc gốc sau:

```text
data/
├── keyframes/<video_id>/*.jpg
└── manifests/
    ├── videos.jsonl
    ├── shots.jsonl
    └── keyframes.jsonl
```

### Nơi lưu

| Nơi lưu | Nội dung | Dùng để làm gì |
|---|---|---|
| Hugging Face `Neezidow/AIC2026-keyframes` | Các gói TAR keyframe theo video, catalogue | Chia sẻ/backup keyframe; giải nén lại được đúng cấu trúc gốc |
| Modal `aic-keyframe-results` | TAR + report JSON cho từng video | Checkpoint kết quả trích keyframe; dùng khi cần giải nén hoặc chạy lại bước sau |
| Modal `aic-frames/keyframes/` | JPG đã giải nén | UI đọc ảnh từ đây để hiển thị kết quả |
| Modal `aic-embeddings/manifests/` | `videos.jsonl`, `shots.jsonl`, `keyframes.jsonl` | Nối kết quả tìm kiếm về video, shot, timestamp và ảnh |

## 2. OCR, ASR và Chronicle

Các file hiện có trong Modal `aic-embeddings/manifests/`:

```text
videos.jsonl       # thông tin video
shots.jsonl        # mốc start/end của từng shot
keyframes.jsonl    # keyframe_id, timestamp, image_path
ocr.jsonl          # văn bản đọc từ ảnh
asr.jsonl          # lời nói nhận dạng từ video
chronicle.jsonl    # file đã ghép shot + OCR + ASR; dùng chính cho text retrieval
```

`chronicle.jsonl` là file canonical cho bước text retrieval. `ocr.jsonl` và
`asr.jsonl` vẫn nên giữ để kiểm tra nguồn hoặc build lại Chronicle, nhưng UI
không cần đọc riêng hai file này nếu `chronicle.jsonl` còn đầy đủ.

Snapshot hiện tại:

```text
chronicle.jsonl  81.2 MiB
ocr.jsonl        84.9 MiB
asr.jsonl        10.3 MiB
```

File `chronicle.ocr-only.backup.jsonl` là backup trung gian trước khi ghép ASR;
không cần cho retrieval hiện tại.

## 3. Image retrieval: SigLIP2 + FAISS

Nằm tại Modal:

```text
aic-embeddings/indexes/keyframes-google--siglip2-so400m-patch16-384/
├── vectors.npy   # vector SigLIP2 của keyframe
├── ids.json      # map row vector -> keyframe_id
├── meta.json     # model và dimension
└── index.faiss   # index FAISS đã lưu
```

Ngoài index, service hiện tại cần file metadata embedding:

```text
aic-embeddings/embeddings/keyframes/
└── google--siglip2-so400m-patch16-384/
    └── manifest.jsonl  # keyframe_id -> video_id, shot_id, timestamp_ms
```

Snapshot hiện tại: `vectors.npy` và `index.faiss` đều khoảng 751.7 MiB,
`ids.json` khoảng 4.0 MiB.

UI hiện tại tải `vectors.npy`, `ids.json`, `meta.json` rồi dựng FAISS trong RAM;
vì vậy `index.faiss` không bắt buộc để chạy đúng UI hiện tại, nhưng vẫn nên giữ
trong master archive để phục vụ tool/phiên bản khác.

Các shard raw tại `aic-embeddings/embeddings/keyframes/` chỉ cần khi kiểm tra
hoặc build lại index; không bắt buộc cho truy vấn đã có index.

## 4. Text retrieval: BGE-M3

Nằm tại Modal `aic-embeddings/indexes/`:

```text
chronicle-semantic/
├── vectors.npy
├── ids.json
├── meta.json
└── index.faiss

chronicle-literal/
└── docs.json

chronicle-overlay/
└── docs.json
```

- `chronicle-semantic`: tìm theo nghĩa của ASR/caption; vector 1024 chiều.
- `chronicle-literal`: tìm chữ OCR/ASR theo từ cụ thể, ví dụ `NGUYÊN LIỆU`.
- `chronicle-overlay`: thông tin cấp video bổ sung; nhỏ nhưng nên mang theo.

Snapshot hiện tại:

```text
chronicle-semantic/vectors.npy  187.4 MiB
chronicle-semantic/index.faiss  187.4 MiB
chronicle-literal/docs.json      64.1 MiB
chronicle-overlay/docs.json      27.8 KiB
```

## 5. Bộ tối thiểu theo nhu cầu

| Nhu cầu | Cần lấy |
|---|---|
| Chỉ xem hoặc dùng lại keyframe | Hugging Face `Neezidow/AIC2026-keyframes` |
| Chỉ kiểm tra OCR/ASR | `manifests/ocr.jsonl`, `manifests/asr.jsonl`, `manifests/shots.jsonl` |
| Chạy text retrieval | `manifests/chronicle.jsonl` + ba thư mục `chronicle-*` index |
| Chạy visual retrieval | `manifests/shots.jsonl` + `embeddings/keyframes/.../manifest.jsonl` + index `keyframes-google--siglip2-so400m-patch16-384` |
| Chạy UI đầy đủ | `aic-frames/keyframes/` + toàn bộ `aic-embeddings/manifests/` + toàn bộ `aic-embeddings/indexes/` |

## 6. Không cần chia sẻ cho người chỉ dùng retrieval

```text
Modal aic-model-cache/                  # cache model, có thể tải lại
Raw video ZIP / video gốc               # chỉ cần nếu chạy lại ingest/ASR
Raw embedding shards                    # chỉ cần build lại index
TAR report JSON từng video              # chỉ cần audit/resume keyframe job
API key, HF token, Modal token          # tuyệt đối không chia sẻ
```

## Lưu ý chia sẻ

- Không gửi 14.3 GB file qua Zalo. Gửi link Hugging Face và cấp quyền dataset
  gated cho thành viên cần keyframe.
- Modal Volume không có link public để tải như Hugging Face. Muốn thành viên
  tự chạy UI từ dữ liệu hiện tại thì hoặc thêm họ vào Modal workspace, hoặc
  export/upload retrieval bundle lên Hugging Face.
- GitHub branch `pipeline-v1` chỉ chứa source code; không chứa các output trên.

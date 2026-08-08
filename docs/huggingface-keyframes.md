# Lưu keyframe từ Modal lên Hugging Face

Uploader này chỉ đọc `aic-keyframe-results` và đẩy 415 cặp TAR/report vào
dataset private. Nó không giải nén ảnh, không upload ZIP video gốc, model cache,
embedding, OCR hay ASR.

## Quyền truy cập

Tạo fine-grained Hugging Face token chỉ có `Read contents` và
`Write contents/settings` cho dataset đích. Lưu token vào Modal Secret tên
`huggingface-aic`, biến môi trường `HF_TOKEN`. Không đặt token trong source,
lệnh shell hay log.

## Kiểm tra token mà không sửa repo

```bash
modal run scripts/modal_upload_hf.py::check \
  --repo-id Neezidow/AIC2026-keyframes
```

## Upload thử một report nhỏ

```bash
modal run scripts/modal_upload_hf.py::pilot \
  --repo-id Neezidow/AIC2026-keyframes
```

Pilot dùng report thật của `L21_V001` và kiểm tra lại path/kích thước trên Hub.

## Upload toàn bộ

```bash
modal run --detach scripts/modal_upload_hf.py::run \
  --repo-id Neezidow/AIC2026-keyframes
```

Một CPU container đọc Volume ở chế độ read-only và upload theo commit tối đa 50
file. Khi chạy lại, file đã có đúng kích thước và SHA-256 (nếu Hub công bố LFS
hash) được skip. Cuối lượt, uploader đối chiếu đúng 830 package files rồi tạo
`dataset-index.jsonl`; `README.md` chỉ được tạo nếu repo chưa có README.

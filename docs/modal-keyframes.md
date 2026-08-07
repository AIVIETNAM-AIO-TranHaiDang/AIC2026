# Trích keyframe bằng Modal

Wrapper này chỉ lo phần hạ tầng cloud. Nó tải ZIP chính thức, gọi nguyên lệnh
`scripts/ingest_corpus.py`, rồi lưu đúng ba manifest và ảnh JPG mà source tạo
ra. Nó không dùng keyframe mẫu của ban tổ chức, không chạy OCR/ASR/embedding,
và không thay đổi thuật toán OmniShotCut.

## Những phần được giữ nguyên từ source

Job tạo profile đầy đủ từ `configs/t0.yaml` và overlay nhỏ
`configs/modal-keyframes.yaml`. Profile Modal chỉ được phép đổi hai đường dẫn
lưu trữ cùng `model`, `checkpoint`, `mode` của shot detector; helper sẽ báo lỗi
nếu ai vô tình thêm cấu hình embedding/ASR hoặc thay cách chọn keyframe.

Những phần được dùng gồm:

- OmniShotCut, checkpoint `uva-cv-lab/OmniShotCut`, mode `default`;
- bốn vị trí keyframe `0.05, 0.35, 0.65, 0.95` và keyframe bổ sung;
- resize cạnh dài tối đa 720, JPEG quality 95;
- lọc ảnh trùng bằng pHash và lọc ảnh ít thông tin.

## Chạy pilot một video

```bash
modal run --detach scripts/modal_keyframes.py \
  --archive Videos_L21_a \
  --video-id L21_V001
```

`--detach` giữ job trên Modal nếu terminal local đóng. Lệnh mặc định dùng GPU
L4. Trong dashboard, app có tên `aic-keyframes` và log hiển thị tiến độ tải
ZIP, giải nén, chạy từng video và số shot/keyframe.

Kết quả pilot được commit vào Volume `aic-keyframe-results`:

```text
Videos_L21_a--L21_V001-keyframes.tar
Videos_L21_a--L21_V001-report.json
```

TAR chứa đúng cấu trúc source:

```text
data/
├── keyframes/L21_V001/*.jpg
└── manifests/
    ├── videos.jsonl
    ├── shots.jsonl
    └── keyframes.jsonl
```

## Tải output về phân vùng Windows

```bash
modal volume get aic-keyframe-results \
  Videos_L21_a--L21_V001-keyframes.tar \
  /run/media/haidang/OS/AIC2026/modal-results/
```

Chỉ chạy nguyên ZIP sau khi đã mở và kiểm tra pilot. Khi đó bỏ giá trị
`--video-id` bằng cách truyền chuỗi rỗng:

```bash
modal run --detach scripts/modal_keyframes.py \
  --archive Videos_L21_a \
  --video-id ""
```

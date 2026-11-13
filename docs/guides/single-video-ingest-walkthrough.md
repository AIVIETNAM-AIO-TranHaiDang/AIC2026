# Thực hành trích keyframe một video bằng OmniShotCut

Tài liệu này ghi lại đúng quy trình đã dùng để kiểm tra bước ingest trên máy
Ubuntu dual boot, GPU GTX 1650 4 GB và video `L30_V078.mp4`.

Mục tiêu của bài test là xác nhận chuỗi xử lý sau hoạt động:

```text
L30_V078.mp4
    -> đọc thông tin và giải mã video
    -> OmniShotCut phát hiện ranh giới các shot
    -> chọn các frame đại diện trong từng shot
    -> loại frame trắng và frame gần trùng nhau
    -> lưu ảnh JPEG và ba file manifest JSONL
```

Điểm quan trọng: tui **không gọi một lệnh FFmpeg để cắt frame thủ công**.
Lệnh chính là `scripts/ingest_corpus.py`; script này gọi các module trong
`src/aic/ingest/` để thực hiện toàn bộ chuỗi trên.

## Quy ước phân biệt source code và thao tác bổ sung

Từ đây cần phân biệt ba loại hành động:

- **[SOURCE CODE]**: hành vi đã được repo quy định, ví dụ tên ba manifest,
  công thức đặt `keyframe_id`, cách chọn/lọc frame và cách lưu JPEG.
- **[TUI THIẾT LẬP]**: lựa chọn dành riêng cho máy hiện tại, ví dụ chỉnh profile
  `t0-gtx1650.yaml`, chọn OmniShotCut, dùng `data/pilot` và chọn một video test.
- **[KIỂM TRA BỔ SUNG]**: lệnh tui tự viết để xác minh đầu ra, không thuộc thuật
  toán pipeline, ví dụ đoạn Python mở đủ 147 JPEG và kiểm tra các `assert`.

Kết quả ingest được source code sinh ra. Tui chỉ thiết lập input/config rồi gọi
entry point có sẵn; số lượng 51 shot và 147 keyframe không được viết cứng bởi
tui.

## 1. Mở terminal tại đúng thư mục project

```bash
cd /home/haidang/Project/AIC2026_prepare-main
pwd
```

Kết quả của `pwd` phải là:

```text
/home/haidang/Project/AIC2026_prepare-main
```

Lý do: những đường dẫn như `configs/...`, `scripts/...` và `data/...` trong các
lệnh phía dưới đều được hiểu tương đối từ thư mục hiện tại.

## 2. Kích hoạt môi trường Python

Môi trường ảo của project đang được đặt trên phân vùng Windows và nối vào repo
bằng symlink `.venv`:

```bash
source .venv/bin/activate
python --version
which python
```

Trên máy đã test, kết quả là:

```text
Python 3.14.4
/run/media/haidang/OS/AIC2026/venvs/aic-py314/bin/python
```

Ý nghĩa:

- `source .venv/bin/activate`: chuyển terminal sang môi trường Python riêng của
  project.
- `python --version`: xem phiên bản Python đang dùng.
- `which python`: xem lệnh `python` thật sự trỏ tới file nào, tránh cài package
  nhầm vào Python hệ thống.

Nếu terminal đã hiện `(aic-py314)` ở đầu dòng thì môi trường đã được kích hoạt.

## 3. Kiểm tra phân vùng dữ liệu và GPU

```bash
findmnt -no SOURCE,TARGET,FSTYPE,OPTIONS /run/media/haidang/OS
readlink -f data
nvidia-smi
```

Ý nghĩa:

- `findmnt`: xác nhận phân vùng Windows đang được mount và có tùy chọn `rw`
  (read-write). Nếu chỉ thấy `ro`, Ubuntu chỉ được đọc và không thể ghi kết quả.
- `readlink -f data`: xem symlink `data` đang trỏ tới đâu. Trên máy này nó trỏ
  tới `/run/media/haidang/OS/AIC2026/data`.
- `nvidia-smi`: xác nhận Ubuntu nhìn thấy GTX 1650 và driver NVIDIA.

Kiểm tra tiếp xem PyTorch có dùng được CUDA không:

```bash
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

OmniShotCut upstream bắt buộc CUDA, vì vậy `cuda available` phải là `True`.

## 4. Cài OmniShotCut một lần trong môi trường ảo

Các dependency chung của project được cài bằng:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pip install -e .
```

Riêng OmniShotCut đã được cài từ đúng commit mà repo ghi trong
`requirements-gpu.txt`:

```bash
python -m pip install "decord==0.6.0"
python -m pip install --no-deps \
  "git+https://github.com/UVA-Computer-Vision-Lab/OmniShotCut.git@3331cd3163f7f17cd6d7c8fc12ffde22894ace01"
```

Giải thích:

- `python -m pip` đảm bảo dùng `pip` thuộc đúng Python vừa kích hoạt.
- `decord` là dependency runtime của OmniShotCut.
- `git+https://...@3331...` cài trực tiếp source tại một commit cố định, giúp lần
  cài sau không vô tình lấy code mới có hành vi khác.
- `--no-deps` không cho OmniShotCut tự thay đổi các phiên bản dependency đã được
  project pin. Môi trường hiện có `opencv-python-headless`, cung cấp module
  `cv2`, nên không cài thêm `opencv-python` chồng lên nó.

Kiểm tra import sau khi cài:

```bash
python - <<'PY'
import cv2
import decord
import omnishotcut

print("cv2:", cv2.__version__)
print("decord:", decord.__version__)
print("OmniShotCut import: OK")
PY
```

Lưu ý: trên Python 3.14, `pip check` có thể cảnh báo metadata của `decord` và cho
rằng OmniShotCut thiếu `opencv-python`. Trong môi trường đã test, `cv2`, `decord`
và OmniShotCut đều import được, model load được và inference hoàn tất. Đây là
cảnh báo tương thích package cần ghi nhớ, không phải bằng chứng duy nhất để bỏ
qua lỗi runtime.

## 5. Chuẩn bị đúng một video pilot

Config pilot dùng `data_root: data/pilot`, nên script chỉ tìm video trong:

```text
data/pilot/videos/
```

Lệnh đã dùng để tạo thư mục và đưa video vào:

```bash
mkdir -p data/pilot/videos

AIC_DATASET_DIR="/run/media/haidang/OS/AIC2026/datasets/AIC_demo"

if [ ! -e data/pilot/videos/L30_V078.mp4 ]; then
  ln "$AIC_DATASET_DIR/Demo/Có query/L30_V078.mp4" \
    data/pilot/videos/L30_V078.mp4
fi

ls -lh data/pilot/videos/L30_V078.mp4
```

Ở đây `ln` tạo **hard link**, không chép thêm một bản video 35 MB. Hai đường dẫn
là hai tên cùng trỏ tới dữ liệu trên cùng phân vùng NTFS. Cách này chỉ dùng được
khi nguồn và đích nằm trên cùng filesystem. Nếu gặp lỗi `Invalid cross-device
link`, dùng bản sao thông thường:

```bash
cp --update=none \
  "$AIC_DATASET_DIR/Demo/Có query/L30_V078.mp4" \
  data/pilot/videos/L30_V078.mp4
```

Dấu ngoặc kép quanh đường dẫn là bắt buộc vì `Có query` chứa khoảng trắng.

## 6. Tạo config nhẹ cho GTX 1650

Ban đầu tạo một bản sao của config T0 để không sửa config gốc:

```bash
cp --update=none configs/t0.yaml configs/t0-gtx1650.yaml
```

`--update=none` nghĩa là không ghi đè nếu file đích đã tồn tại. Sau đó chỉnh các
phần liên quan trong `configs/t0-gtx1650.yaml`, đồng thời giữ nguyên các phần còn
lại của profile đầy đủ:

```yaml
paths:
  data_root: data/pilot
  models_dir: data/models

ingest:
  video_extensions: [mp4, mkv, avi, mov, webm]
  shots:
    model: omnishotcut
    checkpoint: uva-cv-lab/OmniShotCut
    mode: default
    device: auto
    threshold: 0.5
    min_shot_duration_ms: 400
  keyframes:
    positions: [0.05, 0.35, 0.65, 0.95]
    max_gap_ms: 10000
    motion_min_diff: 12.0
    motion_min_shot_ms: 4000
    max_side: 720
    jpeg_quality: 95
  dedup:
    phash_max_distance: 4
    min_entropy_bits: 1.0
```

Ý nghĩa các tham số quan trọng:

- `data_root: data/pilot`: cô lập bài test một video khỏi corpus 19 video.
- `models_dir: data/models`: cache model dùng chung, không tải lại ở mỗi pilot.
- `model: omnishotcut`: chọn đúng backend OmniShotCut, không phải TransNetV2.
- `checkpoint`: Hugging Face repo chứa trọng số chính thức. Lần đầu chạy sẽ tải
  checkpoint; các lần sau dùng cache.
- `mode: default`: giữ cả các vùng chuyển cảnh dần như dissolve/wipe. Chế độ
  `clean_shot` bỏ các vùng chuyển cảnh đó.
- `device: auto`: code tự dùng CUDA khi GPU khả dụng. OmniShotCut vẫn yêu cầu
  CUDA do cách upstream load model.
- `threshold: 0.5`: OmniShotCut trả trực tiếp các khoảng shot nên không dùng
  threshold này; trường vẫn có mặt vì schema dùng chung với TransNetV2.
- `min_shot_duration_ms: 400`: shot ngắn hơn 0,4 giây được gộp để tránh sinh quá
  nhiều keyframe từ các cut cực nhanh.
- `positions`: ban đầu lấy frame tại khoảng 5%, 35%, 65% và 95% của mỗi shot.
- `max_gap_ms: 10000`: chèn thêm frame nếu hai frame đã chọn cách nhau trên 10
  giây.
- `motion_min_diff` và `motion_min_shot_ms`: với shot dài ít nhất 4 giây, thêm
  frame tại đỉnh chuyển động nếu độ thay đổi đủ lớn.
- `max_side: 720`: cạnh dài nhất của JPEG đầu ra là 720 pixel.
- `jpeg_quality: 95`: chất lượng JPEG.
- `phash_max_distance: 4`: bỏ các frame gần giống nhau theo perceptual hash.
- `min_entropy_bits: 1.0`: bỏ frame quá trống, đen hoặc ít thông tin.

Kiểm tra nhanh rằng config thật sự chọn OmniShotCut:

```bash
grep -A8 '^  shots:' configs/t0-gtx1650.yaml
```

## 7. Lệnh chính để trích keyframe

Đây là chính xác lệnh đã chạy thật (gọi thẳng Python trong `.venv`):

```bash
/usr/bin/time -v .venv/bin/python scripts/ingest_corpus.py \
  --config configs/t0-gtx1650.yaml \
  --num-gpus 1
```

Sau khi đã chạy `source .venv/bin/activate`, `.venv/bin/python` và `python` là
cùng môi trường. Nếu không cần đo tài nguyên, lệnh ngắn hơn nhưng có cùng chức
năng là:

```bash
python scripts/ingest_corpus.py \
  --config configs/t0-gtx1650.yaml \
  --num-gpus 1
```

Giải thích từng phần:

- `/usr/bin/time -v`: bọc bên ngoài để đo thời gian, RAM và exit status; không
  thay đổi thuật toán trích keyframe.
- `python scripts/ingest_corpus.py`: chạy entry point ingest của repo.
- `--config configs/t0-gtx1650.yaml`: nạp đường dẫn dữ liệu và toàn bộ tham số
  OmniShotCut/keyframe từ file YAML.
- `--num-gpus 1`: dùng một worker ứng với một GPU. GTX 1650 chỉ có một GPU nên
  đặt bằng 1.
- Dấu `\` ở cuối dòng báo shell rằng lệnh chưa kết thúc và còn tiếp ở dòng sau.
  Có thể viết tất cả trên một dòng nếu muốn.

Không cần `--limit 1` ở đây vì `data/pilot/videos/` chỉ có đúng một video. Nếu
thư mục có nhiều video, có thể thêm `--limit 1`, nhưng script sẽ lấy video đầu
tiên theo thứ tự tên chứ không tự biết bạn muốn `L30_V078`.

Kết quả thực tế của lần chạy đầu:

```text
OmniShotCut loaded successfully.
loaded OmniShotCut from uva-cv-lab/OmniShotCut
L30_V078: 51 shots, 147 keyframes (5 blank, 66 duplicates dropped)
processed=1 skipped=0 failed=0 keyframes=147
Elapsed (wall clock) time: 0:51.40
Maximum resident set size: 1926252 kbytes
Exit status: 0
```

Cách đọc kết quả:

- `processed=1`: đã xử lý một video mới.
- `skipped=0`: không bỏ qua video nào.
- `failed=0`: không có video lỗi.
- `keyframes=147`: cuối cùng đã ghi 147 JPEG.
- `Exit status: 0`: tiến trình kết thúc thành công.
- RAM tiến trình đạt khoảng 1,84 GiB. Con số này không phải peak VRAM GPU.

## 8. Bên trong lệnh chính diễn ra những gì?

`scripts/ingest_corpus.py` không tự chọn frame ngẫu nhiên. Nó gọi code theo thứ
tự sau:

1. Đọc config và tìm video trong `data/pilot/videos/`.
2. Đọc metadata video: duration, FPS, audio và VFR.
3. Giải mã video thành các frame độ phân giải thấp đúng kích thước OmniShotCut
   yêu cầu.
4. Gọi `model.inference(...)` để lấy các khoảng `[start_frame, end_frame]` của
   từng shot.
5. Gộp shot quá ngắn theo `min_shot_duration_ms`.
6. Trong mỗi shot, chọn các vị trí 5%, 35%, 65%, 95%; bổ sung frame nếu khoảng
   thời gian quá dài hoặc có một đỉnh chuyển động mạnh.
7. Đọc lại các frame đã chọn ở kích thước tốt hơn, cạnh dài tối đa 720 pixel.
8. Loại frame ít thông tin bằng entropy và frame gần trùng bằng pHash.
9. Đảm bảo mỗi shot không trắng vẫn có ít nhất một keyframe để còn tìm kiếm được.
10. Lưu JPEG và ghi metadata vào manifest.

Các file code tương ứng:

- Entry point: [`scripts/ingest_corpus.py`](scripts/ingest_corpus.py)
- Điều phối ingest: [`src/aic/ingest/pipeline.py`](src/aic/ingest/pipeline.py)
- Adapter OmniShotCut: [`src/aic/ingest/shots.py`](src/aic/ingest/shots.py)
- Thuật toán chọn keyframe: [`src/aic/ingest/keyframes.py`](src/aic/ingest/keyframes.py)
- Lọc blank/trùng: [`src/aic/ingest/dedup.py`](src/aic/ingest/dedup.py)

Vì vậy, **OmniShotCut phát hiện shot**, còn logic của repo quyết định frame nào
trong mỗi shot được lưu thành keyframe.

## 9. Kết quả được lưu ở đâu?

Ảnh JPEG:

```text
data/pilot/keyframes/L30_V078/*.jpg
```

Ba manifest:

```text
data/pilot/manifests/videos.jsonl
data/pilot/manifests/shots.jsonl
data/pilot/manifests/keyframes.jsonl
```

Ý nghĩa:

- `videos.jsonl`: trạng thái, duration, FPS, số shot và số keyframe của video.
- `shots.jsonl`: thời điểm bắt đầu/kết thúc của từng shot.
- `keyframes.jsonl`: ID, shot ID, frame index, timestamp và đường dẫn JPEG.
- JSONL nghĩa là mỗi dòng là một JSON record độc lập.

Đếm nhanh đầu ra:

```bash
wc -l data/pilot/manifests/videos.jsonl
wc -l data/pilot/manifests/shots.jsonl
wc -l data/pilot/manifests/keyframes.jsonl
find data/pilot/keyframes/L30_V078 -type f -name '*.jpg' | wc -l
```

Kết quả mong đợi hiện tại lần lượt là `1`, `51`, `147`, `147`.

Xem vài record đầu:

```bash
head -n 1 data/pilot/manifests/videos.jsonl
head -n 3 data/pilot/manifests/shots.jsonl
head -n 3 data/pilot/manifests/keyframes.jsonl
```

Mở thư mục ảnh bằng giao diện Ubuntu:

```bash
xdg-open data/pilot/keyframes/L30_V078
```

## 10. Kiểm tra corpus bằng tool của repo

```bash
python scripts/check_corpus.py --config configs/t0-gtx1650.yaml
```

Kết quả hiện tại:

```text
video                status     dur aud shots   kf  cap%  asr%  ocr%  sem%  lit%
L30_V078             done      198s   y    51  147  0.00  0.00  0.00  0.00  0.00
```

Các cột caption/ASR/OCR/semantic/literal bằng `0.00` là bình thường ở bước này,
vì mới chỉ chạy ingest. Nó không có nghĩa OmniShotCut hay việc trích keyframe bị
lỗi.

## 11. Lệnh kiểm tra sâu 147 JPEG và manifest

Đây là phiên bản đúng của lệnh kiểm tra sâu đã dùng sau khi ingest:

```bash
python - <<'PY'
import json
from pathlib import Path

from PIL import Image

manifest_dir = Path("data/pilot/manifests")

videos = [
    json.loads(line)
    for line in (manifest_dir / "videos.jsonl").read_text().splitlines()
    if line.strip()
]
shots = [
    json.loads(line)
    for line in (manifest_dir / "shots.jsonl").read_text().splitlines()
    if line.strip()
]
keyframes = [
    json.loads(line)
    for line in (manifest_dir / "keyframes.jsonl").read_text().splitlines()
    if line.strip()
]

assert len(videos) == 1
assert videos[0]["status"] == "done"
assert len(shots) == videos[0]["n_shots"] == 51
assert len(keyframes) == videos[0]["n_keyframes"] == 147
assert [shot["shot_id"] for shot in shots] == list(range(len(shots)))

duration_ms = videos[0]["duration_ms"]
assert all(
    0 <= shot["start_ms"] < shot["end_ms"] <= duration_ms
    for shot in shots
)

valid_shots = {(shot["video_id"], shot["shot_id"]) for shot in shots}
assert all(
    (frame["video_id"], frame["shot_id"]) in valid_shots
    for frame in keyframes
)
assert all(0 <= frame["timestamp_ms"] <= duration_ms for frame in keyframes)
assert len({frame["keyframe_id"] for frame in keyframes}) == len(keyframes)

for frame in keyframes:
    image_path = Path(frame["image_path"])
    assert image_path.is_file(), f"Missing: {image_path}"
    with Image.open(image_path) as image:
        image.verify()

print("video status:", videos[0]["status"])
print("shots:", len(shots))
print("keyframes:", len(keyframes))
print("MANIFEST_AND_IMAGES_OK")
PY
```

`assert` là điều kiện bắt buộc. Nếu một điều kiện sai, Python dừng và báo dòng
không hợp lệ. Nếu tất cả đúng, dòng cuối là:

```text
MANIFEST_AND_IMAGES_OK
```

## 12. Kiểm tra khả năng resume

Chạy lại đúng lệnh ingest:

```bash
/usr/bin/time -f 'elapsed=%e max_rss_kb=%M exit=%x' \
  python scripts/ingest_corpus.py \
  --config configs/t0-gtx1650.yaml \
  --num-gpus 1
```

Kết quả thực tế:

```text
processed=0 skipped=1 failed=0 keyframes=0
elapsed=2.01 max_rss_kb=694880 exit=0
```

Pipeline đọc `videos.jsonl`, thấy `L30_V078` đã hoàn tất nên không chạy lại
OmniShotCut và không ghi trùng keyframe. Đây là cơ chế resumable của pipeline.

Muốn thử lại từ đầu, không nên xóa manifest hoặc keyframe một cách tùy tiện vì
có thể làm dữ liệu mất đồng bộ. Cách an toàn khi học là tạo một profile khác với
`data_root` mới, ví dụ `data/pilot-rerun`, rồi đặt video vào thư mục
`data/pilot-rerun/videos/`.

## 13. Ta đã chứng minh được gì và chưa chứng minh được gì?

Bài test đã chứng minh:

- GPU, CUDA và OmniShotCut load được trên máy hiện tại.
- Một video thật được decode thành công.
- Code phát hiện shot, chọn/lọc/lưu keyframe và ghi manifest thành công.
- 147 JPEG đều tồn tại và đọc được.
- Timestamp và quan hệ video-shot-keyframe hợp lệ.
- Chạy lại có resume, không tạo dữ liệu trùng.

Bài test **chưa** chứng minh 51 ranh giới shot đều chính xác về mặt nội dung.
Muốn đánh giá chất lượng model, cần xem các frame quanh từng boundary hoặc so với
ground truth được gán nhãn. Tại thời điểm hoàn thành riêng bài test ingest, các
bước embedding, ASR, OCR, indexing và search chưa được kiểm tra; phần 14 bên dưới
ghi lại lần chạy embedding/indexing sau đó.

## 14. Bước tiếp theo đã chạy: embedding và FAISS index

Phần này được bổ sung sau khi bài test ingest ở trên hoàn tất.

### 14.1 Source code quy định gì?

`[SOURCE CODE]` Entry point `scripts/embed_corpus.py` thực hiện hai việc trong
một lần chạy:

1. Đọc `data/pilot/manifests/keyframes.jsonl`, mở từng JPEG và dùng encoder tạo
   embedding vector.
2. Đọc lại toàn bộ embedding shard và xây keyframe vector index bằng FAISS.

Profile hiện dùng:

```yaml
embed:
  model: siglip2
  model_id: google/siglip2-so400m-patch16-384
  device: auto
  batch_size: 1
  shard_format: safetensors
  store_dtype: float16

index:
  backend: faiss
  device: cpu
```

`[TUI THIẾT LẬP]` `batch_size: 1` được giữ để phù hợp GTX 1650 4 GB. GPU dùng
để tạo embedding; FAISS index được giữ trên CPU theo config.

### 14.2 Kiểm tra tài nguyên trước khi chạy

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
  --format=csv,noheader
findmnt -no SOURCE,TARGET,FSTYPE,OPTIONS /run/media/haidang/OS
df -h /run/media/haidang/OS
```

Kết quả trước khi chạy: GTX 1650 có 4096 MiB VRAM, phân vùng NTFS đang `rw` và
còn khoảng 79 GB.

### 14.3 Smoke test bốn keyframe

`[KIỂM TRA BỔ SUNG]` Tui chủ động dùng `--limit 4` để kiểm tra model có load và
inference được trước khi giao toàn bộ 147 ảnh:

```bash
/usr/bin/time -v .venv/bin/python scripts/embed_corpus.py \
  --config configs/t0-gtx1650.yaml \
  --num-gpus 1 \
  --limit 4
```

Lần đầu source tải checkpoint SigLIP2 khoảng 4,54 GB từ Hugging Face vào cache
`data/models`. Kết quả smoke test:

```text
loaded google/siglip2-so400m-patch16-384 on cuda
embedded=4 skipped=143 failed=0
index built: 4 vectors, dim=1152
Exit status: 0
```

Trong lần chạy có `--limit 4`, `skipped=143` nghĩa là 143 ảnh nằm ngoài giới
hạn thử nghiệm, không có nghĩa chúng đã được embed.

### 14.4 Chạy tiếp toàn bộ keyframe

Sau khi smoke test thành công, tui bỏ `--limit`:

```bash
/usr/bin/time -v .venv/bin/python scripts/embed_corpus.py \
  --config configs/t0-gtx1650.yaml \
  --num-gpus 1
```

Source đọc manifest embedding, nhận ra bốn ảnh đã xong và chỉ xử lý 143 ảnh còn
lại. Kết quả thực tế:

```text
embedded=143 skipped=4 failed=0
index built: 147 vectors, dim=1152
Elapsed (wall clock) time: 6:06.04
Maximum resident set size: 7569644 kbytes
Exit status: 0
```

RAM tiến trình đạt khoảng 7,22 GiB; đây không phải số đo peak VRAM. FAISS báo
không có module tối ưu AVX2 rồi fallback sang bản thường và load thành công, nên
đây không phải lỗi build index.

### 14.5 Artifact do source sinh ra

```text
data/pilot/
├── embeddings/keyframes/google--siglip2-so400m-patch16-384/
│   ├── manifest.jsonl
│   └── shard-*.safetensors       # 147 shard do batch_size = 1
└── indexes/keyframes-google--siglip2-so400m-patch16-384/
    ├── meta.json
    ├── ids.json
    ├── vectors.npy
    └── index.faiss
```

Kết quả kiểm tra bổ sung sau khi chạy:

```text
input_keyframes=147
embed_manifest_rows=147
unique_shards=147
vector_shape=(147, 1152)
finite=True
norm_range=0.999847..1.000103
index_count=147 index_dim=1152
self_search_top1=L30_V078_s0_f1 score=0.999694
FULL_EMBED_AND_INDEX_OK
```

`[KIỂM TRA BỔ SUNG]` Các phép kiểm tra trên do tui viết để xác nhận artifact,
không phải thuật toán embedding của source. Chúng đối chiếu ID ingest với ID
embedding, kiểm tra mọi vector không có NaN/Inf, load lại index và dùng vector
đầu tiên tìm chính nó. Source code của pipeline không bị chỉnh sửa trong bước
embedding này.

## 15. Chạy ASR Faster-Whisper trên GTX 1650

### 15.1 Lệnh chạy đúng theo source

Chỉ chạy ASR và bỏ qua OCR/entity để kiểm tra riêng bước nhận dạng giọng nói:

```bash
/usr/bin/time -v .venv/bin/python scripts/build_chronicle.py \
  --config configs/t0-gtx1650.yaml \
  --num-gpus 1 \
  --skip-ocr \
  --skip-entities
```

Ý nghĩa:

- `scripts/build_chronicle.py`: stage tạo ASR rồi ghép dữ liệu theo từng shot.
- `--config`: dùng profile nhẹ đã dành cho GTX 1650.
- `--num-gpus 1`: cho source dùng một GPU.
- `--skip-ocr`: chưa chạy nhận dạng chữ trong ảnh.
- `--skip-entities`: chưa chạy trích xuất người, tổ chức, địa điểm.
- `/usr/bin/time -v`: đo thời gian và RAM; không thay đổi thuật toán của source.

### 15.2 Lỗi phát hiện trong lần đầu

Model `faster-whisper small` tải và bắt đầu xử lý audio, nhưng CTranslate2 dừng
ở kernel GPU đầu tiên với lỗi:

```text
Library libcublas.so.12 is not found or cannot be loaded
```

Nguyên nhân: môi trường có thư viện CUDA 13 đi kèm PyTorch, trong khi wheel
CTranslate2 4.8 dùng bởi Faster-Whisper cần đúng tên thư viện CUDA 12
`libcublas.so.12` và cuDNN 9 `libcudnn.so.9`. Đây là lỗi phụ thuộc runtime,
không phải video hoặc model bị hỏng.

Tui cũng phát hiện `build_chronicle.py` trước đây vẫn trả exit code `0` khi
stage báo `failed=1`. Điều đó có thể làm automation hiểu nhầm một lần chạy lỗi
là thành công.

### 15.3 Những thay đổi tui thực hiện

`[TUI THAY ĐỔI SOURCE]` Thêm hai dependency vào `requirements-gpu.txt`:

```text
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cublas-cu12==12.9.2.10
nvidia-cudnn-cu12==9.24.0.43
```

`[TUI THAY ĐỔI SOURCE]` Thêm `src/aic/cuda_runtime.py`. Trước khi chạy
Faster-Whisper trên GPU Linux, helper này:

1. kiểm tra CUDA 12 runtime đã được dynamic loader nhìn thấy chưa;
2. tìm thư mục thư viện do hai package `nvidia-*` cài vào venv;
3. thêm chúng vào `LD_LIBRARY_PATH` và tự chạy lại đúng command một lần;
4. không can thiệp khi chạy CPU hoặc khi máy đã có runtime phù hợp.

`[TUI THAY ĐỔI SOURCE]` Sửa `scripts/build_chronicle.py` để gọi helper trên và
trả exit code khác `0` nếu ASR/OCR/caption/entity/ledger có record thất bại.
Pipeline vẫn có thể ghi kết quả dở dang để resume, nhưng terminal/CI sẽ biết
lần chạy đó chưa thành công.

`[TUI THÊM KIỂM THỬ]` Thêm `tests/test_cuda_runtime.py` cho bốn trường hợp CPU,
runtime đã có, runtime bị thiếu, và tự re-exec với đúng `LD_LIBRARY_PATH`.

Để cập nhật môi trường hiện tại, tui đã chạy:

```bash
.venv/bin/python -m pip install \
  nvidia-cublas-cu12 \
  "nvidia-cudnn-cu12==9.*"
```

Đây là thao tác cài dependency, không phải lệnh trích ASR. Từ những lần sau có
thể cài các dependency đã pin bằng `requirements-gpu.txt`.

### 15.4 Kết quả sau khi sửa

Chạy lại chính lệnh ở mục 15.1, không tự export `LD_LIBRARY_PATH`, cho kết quả:

```text
loaded faster-whisper small (auto)
asr: processed=1 skipped=0 failed=0
chronicle: 51 shots assembled
Elapsed (wall clock) time: 0:18.89
Maximum resident set size: 1504028 kbytes
Exit status: 0
```

Artifact do source ghi ra:

```text
data/pilot/manifests/asr.jsonl         # 1 video, 44 đoạn lời nói
data/pilot/manifests/chronicle.jsonl   # 51 shot, 50 shot có asr_text
```

`[KIỂM TRA BỔ SUNG]` Tui đọc lại JSONL và xác nhận mọi đoạn đều có text, có
`0 <= start_ms < end_ms <= duration_ms`, confidence nằm trong `[0, 1]`, và ID
video khớp manifest ingest. Shot không có ASR không phải lỗi: nó có thể nằm ở
đoạn không có lời nói. Model `small` đã chứng minh pipeline hoạt động nhưng vẫn
nghe nhầm một số từ tiếng Việt; tối ưu chất lượng model là bước đánh giá riêng.

## 16. Chạy OCR cho 147 keyframe

### 16.1 Source quy định gì?

Profile `configs/t0-gtx1650.yaml` đang dùng:

```yaml
chronicle:
  ocr:
    backend: easyocr
    languages: [vi, en]
    min_confidence: 0.4
```

Do đó source dùng EasyOCR để đọc chữ tiếng Việt và tiếng Anh trên từng
keyframe. Những dòng có confidence thấp hơn `0.4` không được ghi vào manifest.

### 16.2 Lệnh tui đã chạy

```bash
/usr/bin/time -v .venv/bin/python scripts/build_chronicle.py \
  --config configs/t0-gtx1650.yaml \
  --num-gpus 1 \
  --skip-asr \
  --skip-entities
```

Ý nghĩa:

- `--skip-asr`: giữ kết quả ASR đã có, không chép lại audio.
- Không có `--skip-ocr`, nên stage OCR được chạy.
- `--skip-entities`: chưa chạy trích xuất entity.
- Caption đang `enabled: false` trong config nên không gọi API.
- `/usr/bin/time -v` chỉ đo thời gian/RAM, không thay đổi thuật toán source.

Lần đầu EasyOCR tự tải model detection và recognition, sau đó cache lại để
những lần sau không phải tải lại.

### 16.3 Kết quả do source tạo

```text
ocr: processed=147 skipped=0 failed=0
chronicle: 51 shots assembled
Elapsed (wall clock) time: 0:39.65
Maximum resident set size: 1683048 kbytes
Exit status: 0
```

Hai artifact được tạo/cập nhật:

```text
data/pilot/manifests/ocr.jsonl         # OCR theo từng keyframe
data/pilot/manifests/chronicle.jsonl   # OCR được gom vào từng shot
```

`[KIỂM TRA BỔ SUNG]` Tui đọc lại JSONL và xác nhận:

```text
ocr_records=147
keyframes_with_ocr=56
detected_text_lines=267
shots_with_ocr=29/51
confidence_range=0.4028..0.9999
OCR_OUTPUT_VALID
```

Các số đếm và assertion trên là kiểm tra bổ sung của tui; việc nhận dạng,
lọc confidence, ghi `ocr.jsonl` và ghép `chronicle.jsonl` là hành vi của source.

## 17. Xây semantic và literal text index

### 17.1 Source quy định gì?

Sau khi `chronicle.jsonl` có ASR/OCR, script tiếp theo của repo là
`scripts/build_text_indexes.py`. Source tạo hai cách nhìn cho mỗi shot:

- `semantic_text`: caption và lời nói, dùng dense vector để tìm theo ý nghĩa;
- `literal_text`: OCR và lời nói, dùng learned-sparse weights để ưu tiên từ/cụm
  từ gần khớp nguyên văn.

Profile GTX 1650 dùng `BAAI/bge-m3`, `batch_size: 2`. BGE-M3 tạo cả dense
embedding 1024 chiều và sparse lexical weights. `[TUI KHÔNG SỬA SOURCE]` Phần
gộp OCR được giữ nguyên theo yêu cầu; text index dùng đúng Chronicle hiện có.

### 17.2 Lệnh tui đã chạy

```bash
/usr/bin/time -v .venv/bin/python scripts/build_text_indexes.py \
  --config configs/t0-gtx1650.yaml
```

Lần đầu source tải 30 file của BGE-M3 vào `data/models`, tổng cache model khoảng
4,3 GB. Kết quả:

```text
loaded BAAI/bge-m3 on cuda
text indexes built: 50 semantic docs, 51 literal docs over 51 shots
semantic_docs=50 literal_docs=51 shots=51
Elapsed (wall clock) time: 22:38.59
Maximum resident set size: 4832468 kbytes
Exit status: 0
```

Phần lớn 22 phút 38 giây là thời gian tải model qua mạng. Sau khi cache đã có,
thử load model và chạy hai query chỉ mất khoảng 9,67 giây.

### 17.3 Artifact do source tạo

```text
data/pilot/indexes/
├── chronicle-semantic/
│   ├── ids.json
│   ├── index.faiss
│   ├── meta.json
│   └── vectors.npy       # shape (50, 1024), float32
└── chronicle-literal/
    └── docs.json         # 51 ID + sparse token weights
```

Semantic chỉ có 50 document vì một shot không có caption/lời nói. Literal có
đủ 51 document vì nguồn OCR+lời nói phủ toàn bộ 51 shot.

### 17.4 Truy vấn kiểm tra bổ sung

`[KIỂM TRA BỔ SUNG]` Tui dùng chính BGE-M3 và class index của source để encode
hai câu query rồi load lại artifact từ đĩa:

```text
SEMANTIC_QUERY: lớp học làm bánh miễn phí
  L30_V078:19 62120-64960ms score=0.7029
  L30_V078:43 175760-178160ms score=0.6986
  L30_V078:44 178160-180960ms score=0.6868

LITERAL_QUERY: tv.tuoitre.vn
  L30_V078:0 0-840ms score=0.1058
  L30_V078:1 840-3000ms score=0.0730
  L30_V078:41 164240-172880ms score=0.0136

TEXT_INDEX_QUERY_OK
```

Query literal đưa shot 0, nơi OCR đọc được `tv.tuoitre.vn`, lên vị trí đầu.
Phép query là kiểm tra bổ sung của tui; cách encode, lưu và search đều gọi class
của source, không tự triển khai thuật toán index bên ngoài.

## 18. Kiểm tra retrieval end-to-end không qua UI

### 18.1 Warm-up fast path bằng script của source

Tui chạy đúng cold-start drill có sẵn trong repo:

```bash
/usr/bin/time -v .venv/bin/python scripts/warmup.py \
  --config configs/t0-gtx1650.yaml \
  --query "lớp học làm bánh miễn phí"
```

`warmup.py` gọi `build_service_state()`, load các artifact và đẩy query qua
`state.engine.rank_spec()`. Source cố ý dùng fallback spec nên phép kiểm tra
không phụ thuộc Cortex LLM/API. Nó chạy query hai lần: lần đầu load lazy model,
lần hai đo tốc độ khi model đã nằm trong RAM/VRAM.

Kết quả:

```text
load artifacts (indexes, bundles, sessions): 0.0s
warm fast-path models (first query): 22.1s
warm repeat query: 0.7s
total cold start: 22.9s
fast path returned 51 candidates
Exit status: 0
Maximum resident set size: 7753200 kbytes
```

GTX 1650 load được cả SigLIP2 và BGE-M3 trong fast path, không CUDA OOM. Peak
RSS khoảng 7,39 GiB là RAM của process, không phải số đo peak VRAM.

FAISS thử module AVX2 rồi fallback sang module thường và load thành công. Source
cũng cảnh báo chưa có overlay video index: corpus pilot không có OCR overlay đủ
ngưỡng để tạo index đó; visual/semantic/literal retrieval vẫn chạy bình thường.

### 18.2 In top candidate bằng chính engine của source

`warmup.py` chỉ in số candidate. `[KIỂM TRA BỔ SUNG]` Tui load cùng service
state, gọi đúng `state.engine.rank_spec(fallback_spec(query), query)` rồi chỉ
in năm candidate đầu cùng metadata Chronicle:

```text
QUERY=lớp học làm bánh miễn phí
CANDIDATES=51
1. L30_V078:19 shot=62120-64960ms evidence=63120 score=0.071517
2. L30_V078:7  shot=18320-20960ms evidence=19240 score=0.070528
3. L30_V078:6  shot=15880-18320ms evidence=16000 score=0.068918
4. L30_V078:24 shot=75720-77640ms evidence=76960 score=0.066942
5. L30_V078:5  shot=13680-15880ms evidence=15760 score=0.066714
END_TO_END_RETRIEVAL_OK
PROBE_EXIT=0
```

`evidence=63120` nghĩa là dense-visual channel cung cấp keyframe bằng chứng tại
63,12 giây, nằm trong shot 19 từ 62,12 đến 64,96 giây.

Trong bản probe đầu tiên, tui tự thêm assertion rằng toàn bộ 51 score phải giảm
dần và assertion đó fail sau khi retrieval đã trả kết quả. Đây là giả định kiểm
tra sai của tui, không phải exception từ source: temporal fusion xếp các window
representative trước, sau đó nối các base candidate chưa phát ra. Tui đọc lại
`fuse_windows()`, bỏ assertion ngoài hợp đồng đó, giữ các kiểm tra ID/timestamp
và chạy lại thành công như output trên.

`[TUI KHÔNG SỬA SOURCE]` Bước này chỉ load artifact và query bằng class của
repo; logic retrieval, temporal fusion và gộp OCR đều được giữ nguyên.

## 19. Chạy service/UI và smoke-test API

### 19.1 Khởi động UI bằng source

Tui chạy:

```bash
.venv/bin/python scripts/serve.py --config configs/t0-gtx1650.yaml
```

Server khởi động thành công:

```text
loaded 51 evidence bundles
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

`GET /` trả `HTTP 200`, `content-type: text/html` và trang HTML 12141 byte.
Repo không định nghĩa `/api/health`; probe bổ sung của tui tới route đó trả 404.
Đây là gọi nhầm route không tồn tại, không phải lỗi startup/UI.

### 19.2 Gửi query qua API mà UI sử dụng

Tui gửi đúng schema `SearchRequest` của source:

```bash
curl -H 'Content-Type: application/json' \
  -X POST http://127.0.0.1:8000/api/search \
  --data '{"query":"lớp học làm bánh miễn phí","top_k":5}'
```

Cold request trả:

```text
HTTP 200
X-Response-Time-Ms: 18911.0
results: 5
top1: L30_V078:19
timestamp_ms: 63120
```

Request thứ hai khi SigLIP2/BGE-M3 đã warm:

```text
HTTP 200
total: 0.772075s
rank: 769.9ms
top1: L30_V078:19
timestamp_ms: 63120
```

JSON top 1 có ASR/OCR và các đường dẫn ảnh, ví dụ:

```text
/frames/L30_V078/L30_V078_s19_f1557.jpg
```

Tui gọi URL đó và nhận `HTTP 200`, JPEG `720x404`, 103994 byte. Như vậy service
không chỉ xếp hạng được shot mà còn phục vụ đúng ảnh để UI hiển thị.

`[KIỂM TRA BỔ SUNG]` Các lệnh `curl` chỉ kiểm tra response; API, ranking, JSON
và static-frame route đều là source hiện có. `[TUI KHÔNG SỬA SOURCE]` Logic
retrieval và UI không bị chỉnh sửa. Server được giữ chạy để mở trình duyệt tại
`http://127.0.0.1:8000`; nhấn `Ctrl+C` ở terminal server khi muốn dừng.

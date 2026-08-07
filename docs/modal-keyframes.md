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

## Ba vùng lưu trữ trên Modal

```text
aic-source-zips       ZIP chính thức + marker tải hoàn tất
aic-model-cache       checkpoint OmniShotCut dùng lại giữa các job
aic-keyframe-results  một TAR + một report JSON cho từng video
```

File ZIP được giữ trên Volume nên không phải tải lại cho từng video. Nếu mạng
đứt giữa chừng, số byte đã tải vẫn được commit và lần sau tiếp tục bằng HTTP
Range. Chỉ khi ZIP mở được và có đủ danh sách video chính thức, runner mới tạo
`Videos_L21_a.ready.json`; GPU từ chối chạy nếu marker này chưa hợp lệ.

## Bước 1 — tải một ZIP bằng CPU

```bash
modal run --detach scripts/modal_keyframes.py::stage \
  --archive Videos_L21_a
```

Lệnh này không thuê GPU. Chờ dashboard/log báo `status: ready` hoặc
`status: already_ready` rồi mới qua bước 2. Nếu bị dừng, chạy lại đúng lệnh trên
để tiếp tục download.

## Bước 2 — chạy thử hai video

```bash
modal run --detach scripts/modal_keyframes.py::run \
  --archive Videos_L21_a \
  --video-ids L21_V001,L21_V002
```

`spawn_map` gửi xong danh sách rồi trả terminal về. Khi terminal in
`submitted 2 video job(s)`, có thể tắt laptop; container Modal tiếp tục chạy.
Runner giới hạn tối đa hai L4 container cùng lúc. `L21_V001` đã có pilot hợp lệ
sẽ được skip, còn `L21_V002` được xử lý và commit độc lập.

Mỗi video chỉ được đánh dấu hoàn tất khi cả hai file sau hợp lệ:

```text
Videos_L21_a--L21_V002-keyframes.tar
Videos_L21_a--L21_V002-report.json
```

Nếu credits hoặc container dừng giữa một video, report hoàn tất chưa được tạo.
Chạy lại lệnh sẽ làm lại riêng video đó; những video có TAR/report hợp lệ được
skip. Vì chỉ có hai container đồng thời, tối đa hai video đang chạy phải làm
lại, không mất kết quả các video trước.

## Bước 3 — chạy toàn bộ ZIP đã stage

Bỏ `--video-ids` để lấy toàn bộ danh sách chính thức của ZIP:

```bash
modal run --detach scripts/modal_keyframes.py::run \
  --archive Videos_L21_a
```

Lệnh này có thể chạy lại bao nhiêu lần cũng được. Các output đã hoàn tất được
phát hiện bằng report và kích thước TAR, không dựa vào việc nhớ thủ công video
nào đã chạy.

## Chạy nhiều ZIP trong cùng một hàng đợi

Sau khi từng ZIP đã có marker `ready`, có thể gửi nhiều ZIP bằng một App:

```bash
modal run --detach scripts/modal_keyframes.py::run_all \
  --archives Videos_L22_a,Videos_L23_a,Videos_L24_a,Videos_L25_a,Videos_L26_a,Videos_L26_b
```

Tất cả video dùng chung giới hạn tối đa hai container L4. Nếu workspace chạm
usage limit, Modal có thể dừng các job đang chạy hoặc đang chờ; TAR/report đã
commit trên Volume vẫn còn. Chạy lại đúng lệnh trên sẽ skip video hoàn tất và
tiếp tục phần còn thiếu.

## Cấu trúc output của mỗi video

TAR chứa đúng cấu trúc source:

```text
data/
├── keyframes/L21_V002/*.jpg
└── manifests/
    ├── videos.jsonl
    ├── shots.jsonl
    └── keyframes.jsonl
```

Tách mỗi video thành một gói là checkpoint hạ tầng, không thay đổi tên JPG hay
nội dung manifest do source quy định. Sau khi cả ZIP hoàn tất, các gói có thể
được tải về và gộp lại bằng cách chép thư mục keyframe và nối ba file JSONL.

## Tải output về phân vùng Windows

Ví dụ tải một video:

```bash
modal volume get aic-keyframe-results \
  Videos_L21_a--L21_V002-keyframes.tar \
  /run/media/haidang/OS/AIC2026/modal-results/
```

Không bắt buộc tải ngay sau mỗi job vì Volume giữ output sau khi container dừng.
Tuy vậy nên tải backup sau khi hoàn tất từng ZIP, nhất là trước khi đổi workspace
hoặc tài khoản Modal; Volume của các workspace khác nhau không tự chia sẻ nhau.

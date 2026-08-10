# Benchmark SigLIP2 trên Modal

Runner này là lớp đo hiệu năng bổ sung. Nó gọi trực tiếp
`aic.embed.encoders.Siglip2Encoder` và dùng `save_shard`/`ManifestWriter` của
source; không đổi model, vector hay format embedding.

Input được đọc read-only từ Volume `aic-keyframe-results`. Báo cáo JSON được
lưu bền vững trong Volume `aic-embedding-benchmarks`; model dùng lại Volume
cache `aic-model-cache`.

## Trình tự an toàn

Smoke test 64 ảnh trên L4:

```bash
modal run scripts/modal_embed_benchmark.py::smoke
```

Nếu smoke test đạt, so sánh tuần tự L4, A10 và L40S trên cùng 2.048 ảnh:

```bash
modal run scripts/modal_embed_benchmark.py::run --sample-size 2048
```

Mỗi GPU thử các batch size phù hợp, sau đó dùng batch nhanh nhất chạy phép đo
end-to-end gồm đọc JPEG, encode SigLIP2, ghi shard safetensors và manifest.
Mỗi function có timeout 25 phút và các GPU chạy tuần tự.

Xem hoặc tải báo cáo:

```bash
modal volume ls aic-embedding-benchmarks
modal volume get aic-embedding-benchmarks / ./modal-embedding-benchmarks/
```

Chi phí trong report là ước tính từ giá Modal tại ngày được ghi trong report;
Billing của Modal là số liệu thanh toán cuối cùng.

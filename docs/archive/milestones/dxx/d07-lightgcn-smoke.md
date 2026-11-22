# D07 — LightGCN sparse CUDA smoke và hồ sơ bộ nhớ

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](../../../current/scope-and-decisions.md) · [Tiến độ v2](../../../current/progress-and-evidence.md).

D07 hoàn thành và G2 đạt ngày 20/11/2026. Forward, backward và optimizer chạy thành công trên RTX 4050 tại cả ba quy mô đã thử.

## Mục tiêu và cơ sở lý thuyết

D07 kiểm tra khả năng dùng graph đã chuẩn hóa từ D06 để lan truyền embedding, tính gradient và cập nhật tham số bằng CUDA. LightGCN lặp phép nhân `E^(k+1) = A_norm E^k`, rồi lấy trung bình embedding từ lớp 0 đến lớp K. Lớp 0 là bảng tham số ban đầu. Probe dùng loss BPR trung bình `softplus(-(score_positive - score_negative))`, trong đó score là dot product của hai embedding sau propagation. Phương pháp lan truyền và gộp các lớp dựa trên [LightGCN, He et al.](https://arxiv.org/abs/2002.02126).

## Triển khai và cấu hình

[lightgcn.py](../../../../src/models/lightgcn.py) nhận COO đối xứng từ D06. Module chuyển weight float64 sang float32 và gọi `torch.sparse.mm` với indices int64. D06 đã áp dụng degree normalization. Model nhân với các trọng số đó trực tiếp. PyTorch native là phương án đã dự kiến ở R-04. Thiết lập này không cần cài PyG hay NumPy để chạy các phép tensor đã kiểm tra.

Seed 7, 64 chiều, 3 lớp, Adam với learning rate 0,001. Mỗi bước dùng cùng 2.048 bộ ba được lấy có hoàn lại từ graph prefix. Movie dương thuộc positive edge set của user. Movie âm được lấy trong phần bổ sung của tập positive movies của user, trên catalog node của snapshot. Đây là điều kiện sampling theo graph positive, nên movie âm vẫn có thể mang rating dưới ngưỡng trong input gốc. Probe chạy 1 bước warm-up và 3 bước đo. Không chia train/test ở D07.

Toy có 4 node và 3 cạnh. Prefix 100k dùng snapshot v10 của D06. Prefix 1M dùng Full Rebuild và cùng hàm normalization của D06. Bản 1M lưu fingerprint của canonical input, registry và COO. Hai trường version để trống cho đường Full Rebuild vì nó không có batch ledger. Các file canonical state của 1M được giữ cùng run để tái lập.

[d07_probe.py](../../../../src/models/d07_probe.py) đồng bộ CUDA quanh mỗi phase, reset peak trước phase và lưu peak allocated/reserved tính bằng bytes. Peak trong bảng là max của các phase, gồm cả warm-up. Reserved là bộ nhớ allocator của PyTorch giữ để tái sử dụng, còn allocated là tensor memory do allocator theo dõi. Chúng không bao phủ toàn bộ VRAM sử dụng bởi driver và thư viện CUDA. Thời gian từng phase và loss từng bước nằm trong log, với warm-up ở bước 0. CPU snapshot load và chuẩn bị prefix nằm ngoài phần đo CUDA.

18 unit tests đã chạy đạt, gồm đối chiếu forward và gradient với dense reference, xác minh sampling trên toy và kiểm tra ghi trạng thái CUDA unavailable/OOM bằng lỗi mô phỏng. Probe thật yêu cầu loss và gradient hữu hạn, gradient khác 0, tham số thay đổi sau optimizer. Nguồn kiểm thử ở [test_lightgcn.py](../../../../tests/test_lightgcn.py).

## Kết quả và bằng chứng

Run `D07-20261119T172548657263Z` bắt đầu lúc 00:25:48 ngày 20/11/2026, giờ Việt Nam. Git HEAD lúc chạy là `735fe20c6b8f1a508b1b2a4e40aa57400ae4230b`. Implementation D07 ở working tree. Log lưu SHA-256 của model và probe để xác định phiên bản code thực chạy.

| Dữ liệu | Node | Cạnh dương | COO entries | Peak allocated | Peak reserved | CUDA forward, backward, Adam |
|---|---:|---:|---:|---:|---:|---|
| Toy | 4 | 3 | 6 | 10,20 MiB | 24 MiB | PASS |
| MovieLens prefix 100k | 2.819 | 57.925 | 115.850 | 23,65 MiB | 34 MiB | PASS |
| MovieLens prefix 1M | 17.630 | 484.791 | 969.582 | 126,26 MiB | 148 MiB | PASS |

Toy có max absolute difference `1,2471e-8` so với phép tính dense float64 với `atol=1e-6` và `rtol=1e-5`. Ở 1M, CUDA báo 5.033 MiB free trước setup. Đỉnh allocated lớn nhất quan sát ở backward. Loss và gradient hữu hạn, gradient khác 0 và Adam cập nhật tham số trên cả ba quy mô. Tất cả số đo dùng cùng cấu hình 64 chiều, 3 lớp và 2.048 bộ ba mỗi bước. Full Rebuild và chuẩn hóa prefix 1M chạy trên CPU trước setup.

Log chính là [gpu_probe.csv](../../../../data/processed/d07_probe/D07-20261119T172548657263Z/gpu_probe.csv) và [report.json](../../../../data/processed/d07_probe/D07-20261119T172548657263Z/report.json). Hash prefix 100k là `8a01bc6f132362f44ab5a206575fb16ab7f34ce44c6bd224b418ea076c7bf519`. Hash prefix 1M là `f17048bd7b12bea613387c169c65db6ce67442aa7341314f5e18fba39efcfbae`. Manifest của 1M được lưu tại `full_reference_1m/normalized_manifest.json` trong thư mục run.

Lần chạy `D07-20261119T172050441795Z` đạt toy và 100k. Lần này dành thêm vài phút để chuẩn bị 1M qua đường ghi snapshot DuckDB. Sau khi đường Full Rebuild hoàn tất GPU probe 1M, tiến trình cũ được dừng. CSV của lần đầu chỉ có hai dòng đã hoàn tất và không có kết quả VRAM cho 1M. Các dòng OOM mô phỏng chỉ có trong test T-09. Run chính không gặp OOM.

## Nội dung đưa vào báo cáo

> Biểu diễn COO đã chuẩn hóa được nạp vào PyTorch và sử dụng trong ba lớp lan truyền LightGCN. Kết quả lan truyền trên đồ thị toy khớp với phép tính dense trong sai số float32. Với 64 chiều embedding, RTX 4050 Laptop GPU chạy thành công lan truyền thuận, lan truyền ngược và các bước cập nhật Adam trên prefix 100.000 và 1.000.000 sự kiện MovieLens. Tại quy mô 1.000.000 sự kiện, đỉnh bộ nhớ PyTorch đo được là 126,26 MiB allocated và 148 MiB reserved. Các phép đo xác nhận khả năng chạy cấu hình smoke ngắn. Chất lượng gợi ý, chi phí huấn luyện dài hơn và bộ nhớ khi xếp hạng cần được đánh giá ở các milestone tiếp theo.

## Tái lập

Từ root repo, dùng Python 3.10.11 trong `../venv` với DuckDB 1.5.6 và PyTorch 2.8.0+cu126. Driver quan sát được là 566.07. Khi dựng lại môi trường và cần cài wheel CUDA, dùng [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/).

```powershell
..\venv\Scripts\python.exe -m pip install torch==2.8.0+cu126 --index-url https://download.pytorch.org/whl/cu126
..\venv\Scripts\python.exe -m unittest discover -s tests -v
..\venv\Scripts\python.exe -m src.models.d07_probe
```

Đầu vào là canonical prefixes từ D02 và report/snapshot 100k đã chạy ở D06. Có thể truyền `--snapshot-100k` trỏ thẳng tới thư mục export. Probe tự ghi từng trường hợp vào `gpu_probe.csv` và `report.json` trong thư mục run dưới `data/processed/d07_probe/`. Dùng `--device cpu` để thử đường CPU. Khi CUDA không qua toy/100k, lệnh mặc định ghi rõ trạng thái của từng scale và chạy fallback CPU.

Lần cài trong sandbox không lấy được wheel. Lần cài được cấp quyền truy cập package index tải và cài thành công vào venv dự án. CUDA đã được xác minh bằng phép tính tensor nhỏ trên RTX 4050. PyTorch báo warning thiếu NumPy khi import, các test và probe dùng tensor trực tiếp.

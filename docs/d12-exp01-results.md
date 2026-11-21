# D12 — EXP-01 repetitions và phân tích crossover

## Protocol và nguồn số liệu

Run `D12-repeat-grid` ngày 20/11/2026 dùng cùng canonical prefix 1.010.000 events của D11, SHA-256 `6c2aa8d38a014be75125a2d694077ef364cba5465bf2da1da4d7966d60e175c1`. Grid `H={100k,500k,1M}`, `Δ={1k,10k}`. Mỗi cặp có một warm-up và **năm measured repetitions** cho mỗi method; mỗi lượt chạy process mới, thứ tự method được shuffle theo seed 7. DuckDB một thread, `warm-cache-uncontrolled`. Phần tạo canonical prefix, chia input và prebuild history nằm ngoài timer; thời gian prebuild được ghi riêng. Timer bao gồm Full Rebuild hoặc Incremental apply và xuất cùng ba bảng Parquet `registry/edges/degrees` có schema tương ứng. Exact correctness được đối chiếu sau timer ở từng repetition.

Raw data: `results/raw/exp01_full_vs_incremental.csv` (72 rows); run manifest: `data/processed/d12_exp01/D12-repeat-grid/report.json`; bảng tổng hợp tái tạo từ raw: `results/raw/exp01_summary_d12.csv`. 72/72 rows `PASS` và `correctness_pass=True`. Môi trường: Python 3.10.11, DuckDB 1.5.6, Intel64 Family 6 Model 183, 16.780.570.624 bytes RAM, Windows 10 build 26300; Git HEAD `123942550cf8192e5cc85f1c6f1432216ff39ab5`, worktree dirty, config SHA-256 `83c34eedc721e5952e89f76272f50bca96cedde82476f65a224c8640a49cb35e`. SHA-256 của harness `exp01.py` tại hai runs: `e1a63d1af28f973337c9cf78a349ccb8b068af299cbe7e6d323ed504af44d4c2`; script phân tích `exp01_analysis.py`: `041ee339eff743d45d0a28d4b1fb0596ce5479d78a6c18a4d9f2b08021073705`. `app_bytes_read`, `export_bytes_written`, `db_growth_bytes` và OS I/O là các loại số đo riêng; RSS là đỉnh lấy mẫu 5 ms.

Lệnh tái tạo kết quả và biểu đồ từ raw CSV (Python hệ thống có Matplotlib 3.10.3):

```powershell
..\venv\Scripts\python.exe -m src.benchmarking.exp01 repeat --out data/processed/d12_exp01/new-run --csv results/raw/new-exp01.csv
python -m src.benchmarking.exp01_analysis --raw results/raw/exp01_full_vs_incremental.csv --summary results/raw/exp01_summary_d12.csv --figures docs/report-assets --scale-summary results/raw/exp01_scale2m_summary_d12.csv
```

## Kết quả core grid

Speedup là **median của năm tỷ số ghép cặp `Full wall / Incremental wall`**; giá trị >1 nghĩa là Incremental nhanh hơn. Min–max wall nằm trong bảng tổng hợp CSV và là error bars trên biểu đồ. Mỗi dòng dưới đây có `incremental_wins=0/5`.

| H | Δ | Full median (s) | Incremental median (s) | Paired speedup median | Full range (s) | Incremental range (s) |
|---:|---:|---:|---:|---:|---:|---:|
| 100k | 1k | 0,0911 | 0,1624 | 0,559 | 0,0873–0,0921 | 0,1527–0,1647 |
| 100k | 10k | 0,0953 | 0,2024 | 0,471 | 0,0896–0,0979 | 0,1984–0,2079 |
| 500k | 1k | 0,1748 | 0,2219 | 0,779 | 0,1670–0,2045 | 0,2143–0,2449 |
| 500k | 10k | 0,1865 | 0,2602 | 0,702 | 0,1793–0,1868 | 0,2549–0,2818 |
| 1M | 1k | 0,2865 | 0,2970 | 0,965 | 0,2738–0,2902 | 0,2751–0,3033 |
| 1M | 10k | 0,2847 | 0,3366 | 0,841 | 0,2823–0,2906 | 0,3355–0,3574 |

**Diễn giải:** Trong core grid đến H=1M, Incremental chưa thắng ở bất kỳ cặp đo nào; không quan sát thấy crossover. Tỷ số tăng theo H ở cả hai mức Δ, nhưng chưa chứng minh xu hướng này tiếp tục trên scale lớn. Full còn có lợi thế của phép batch SQL trên input nhỏ; Incremental chịu overhead ledger/transaction và export toàn bộ state sau update. Đây là giải thích khả dĩ từ cấu trúc thực thi, chưa là phân rã cost thực nghiệm. Prebuild H là chi phí một lần của Incremental, không được cộng lẫn vào update latency; nếu tính tổng từ đầu, cần một mô hình workload và số lần update cụ thể. Cache không kiểm soát, kích thước mẫu 5 và chỉ một máy giới hạn khả năng suy rộng.

## Hình từ raw CSV

- [Latency theo H](report-assets/exp01_latency.svg): median và min–max của năm repetitions; Δ=1k/10k.
- [Paired speedup](report-assets/exp01_speedup.svg): đường `Full/Incremental=1` là điểm hòa; error bars là min–max của tỷ số ghép cặp.
- [Sampled peak RSS](report-assets/exp01_rss.svg): median của sampled peak, không phải peak RAM tuyệt đối.
- [Output bytes](report-assets/exp01_bytes.svg): kích thước Parquet export và manifest; DB growth của Incremental nằm riêng trong raw/summary CSV.
- [Crossover với run H=2M riêng](report-assets/exp01_crossover.svg): marker vuông rỗng là run mở rộng khác source/config; đường chấm chỉ nối các điểm quan sát, không nội suy ngưỡng crossover chính xác.

## Scale >1M

Run `D12-scale-2m` ngày 20/11/2026 dùng prefix 2.010.000 canonical events từ cùng `ratings.csv` raw, SHA-256 prefix `b7f37b907a4170a5b8bce276a9800b5b0d35c913e3998b41bb92e8b8d14b56c2`, config SHA-256 `cfcd95ce17219dea48a0c95433b8794f624a01c3845d2d7749867b2dcd93dd72`. Raw `results/raw/exp01_scale2m_d12.csv` có 24/24 rows PASS và exact correctness; summary riêng ở `results/raw/exp01_scale2m_summary_d12.csv`. H=2M prebuild mất 8,133 s, DB prebuilt 35.926.016 bytes, đều nằm ngoài update timer.

| H | Δ | Full median (s) | Incremental median (s) | Paired speedup median (range) | Incremental wins |
|---:|---:|---:|---:|---:|---:|
| 2M | 1k | 0,4908 | 0,4321 | 1,156 (1,099–1,171) | 5/5 |
| 2M | 10k | 0,4951 | 0,4941 | 0,999 (0,964–1,015) | 2/5 |

Ở Δ=1k, crossover **được quan sát** giữa các điểm H=1M và H=2M trong điều kiện đo này; không có điểm trung gian để xác định ngưỡng. Ở Δ=10k, hai phương án gần ngang nhau tại H=2M, chưa có lợi thế ổn định. Không ghép raw rows của hai runs thành một run duy nhất hoặc coi chúng là phép so sánh trên cùng canonical file; chúng cùng nguồn ratings và cùng transform D02 nhưng khác prefix SHA-256/config. Kết luận chỉ áp dụng cho update cộng full export trên máy và cache policy đã ghi, chưa bao gồm chi phí prebuild/ingest trong tỷ số.

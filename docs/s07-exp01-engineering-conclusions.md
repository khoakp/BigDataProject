# S07 — Phân tích scalability EXP-01 và gate Engineering

**Kết luận 21/11/2026:** gate Engineering đạt trong phạm vi benchmark đã đo: 96/96 raw rows và 48/48 cặp Full/Incremental có correctness PASS theo [S06 audit](s06-exp01-audit.md). [D12](d12-exp01-results.md) là hồ sơ chi tiết; S07 tổng hợp các claim có thể trình bày, điều kiện áp dụng và biểu đồ. Đây là kết quả trên các history H=100k–2M từ MovieLens 32M, không là phép đo graph 32M.

## Kết quả latency từ raw

Paired ratio là median của năm tỷ số **Full wall / Incremental wall** trong từng cặp repetition; >1 nghĩa Incremental nhanh hơn trong measurement boundary “update hoặc rebuild + export ba bảng Parquet”. Mỗi method có một warm-up và năm repetitions, DuckDB một thread trên cùng máy. Prebuild history cho Incremental nằm ngoài timer và được ghi riêng.

| Run | H | Δ | Full median (s) | Incremental median (s) | Paired ratio | Incremental wins |
|---|---:|---:|---:|---:|---:|---:|
| Core | 100k | 1k | 0,0911 | 0,1624 | 0,559 | 0/5 |
| Core | 100k | 10k | 0,0953 | 0,2024 | 0,471 | 0/5 |
| Core | 500k | 1k | 0,1748 | 0,2219 | 0,779 | 0/5 |
| Core | 500k | 10k | 0,1865 | 0,2602 | 0,702 | 0/5 |
| Core | 1M | 1k | 0,2865 | 0,2970 | 0,965 | 0/5 |
| Core | 1M | 10k | 0,2847 | 0,3366 | 0,841 | 0/5 |
| Separate 2M | 2M | 1k | 0,4908 | 0,4321 | **1,156** | **5/5** |
| Separate 2M | 2M | 10k | 0,4951 | 0,4941 | 0,999 | 2/5 |

Trong core run tới H=1M, Full nhanh hơn ở mọi cặp đo. Run 2M riêng cho thấy Incremental nhanh hơn nhất quán tại Δ=1k (paired ratio 1,099–1,171), nên có **crossover quan sát được giữa các điểm H=1M và H=2M** trong cấu hình này. Không có điểm H trung gian để xác định ngưỡng. Với Δ=10k tại H=2M, hai phương án gần hòa và Incremental chỉ thắng 2/5 cặp; không có kết luận crossover ổn định. Hai run khác canonical prefix/config hash nên không ghép thành một chuỗi đo đồng nhất.

## Tại sao đây là kết luận có giới hạn

- Full xử lý H+Δ events bằng batch SQL; Incremental apply Δ vào state đã dựng. Đó là hai chiến lược của cùng output contract, nhưng `events_per_s` trong CSV có mẫu số khác nhau và **không** là speedup so trực tiếp. Paired wall ratio mới đo latency của công việc đã định.
- Cả hai xuất full `registry/edges/degrees` sau update. Export toàn state có thể che một phần lợi ích của cập nhật ít cạnh; đây là đặc tính của contract, chưa có thí nghiệm tách cost apply và export trong EXP-01.
- Dựng history DB cho Incremental là chi phí một lần ngoài timer (H=2M mất 8,133 s); ứng dụng có nhiều batch mới có thể phân bổ chi phí đó, nhưng EXP-01 chưa đo workload nhiều batch để tính tổng chi phí từ đầu.
- Cache không được reset, RSS chỉ là đỉnh lấy mẫu mỗi 5 ms, một máy và năm repetitions/case. Không suy ra giới hạn RAM tuyệt đối, speedup trên 32M hay trong production. D13 EXP-02 là thí nghiệm deferred normalization riêng; REC-01 Kaggle là model quality riêng và không được cộng thời gian với EXP-01 để thành một speedup tổng.

## Hình và khả năng tái tạo

- [Latency](report-assets/exp01_latency.svg): median/min–max wall ở core grid.
- [Paired speedup](report-assets/exp01_speedup.svg): đường ratio=1 là điểm hòa.
- [Sampled RSS](report-assets/exp01_rss.svg): đỉnh lấy mẫu, không là peak RAM tuyệt đối.
- [Output bytes](report-assets/exp01_bytes.svg): Parquet + manifest; DB growth nằm riêng trong raw/summary.
- [Crossover](report-assets/exp01_crossover.svg): marker vuông rỗng phân biệt run 2M riêng; đường chấm chỉ nối điểm quan sát.

Đã chạy lại `src.benchmarking.exp01_analysis` với raw core và scale summary vào `data/processed/s07_figure_check/`. Summary tái sinh **trùng SHA-256** với `results/raw/exp01_summary_d12.csv` (`590ff3ec77cf2621de2f9b415c7d4fa06b9b75e695346b5dc0d1d76dd35ac973`); năm PNG tái sinh **trùng byte** với hình trong `docs/report-assets/`. Các SVG thay đổi metadata/id sinh bởi Matplotlib nên không dùng byte equality cho SVG. Cả năm hình hiện có đã được xem trực quan, nhãn/trục/legend đọc được. [S06](s06-exp01-audit.md) ghi hash raw, verifier và lệnh kiểm toàn bộ Parquet.

Lệnh tái sinh từ root repo (cần Matplotlib 3.10.3; đầu ra kiểm chứng nằm ngoài Git):

```powershell
python -m src.benchmarking.exp01_analysis --raw results/raw/exp01_full_vs_incremental.csv --summary data/processed/s07_figure_check/summary.csv --figures data/processed/s07_figure_check/figures --scale-summary results/raw/exp01_scale2m_summary_d12.csv
```

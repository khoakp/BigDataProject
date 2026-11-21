# Report assets

Lưu sơ đồ, biểu đồ và hình minh họa dùng trong báo cáo tại thư mục này. Mỗi hình cần caption, nguồn dữ liệu hoặc file thiết kế, và liên kết tới claim ở `../09-report-evidence.md`.

Đã có hình thực nghiệm EXP-01 D12 và EXP-02 D13 từ raw CSV; biểu đồ metadata sẽ được tạo khi có dữ liệu tương ứng. Hình kiến trúc có nguồn thiết kế ở `../03-system-design.md`.

`exp02_cost_staleness.svg/png`: nguồn `results/raw/exp02_normalization.csv` và script `src/benchmarking/exp02_analysis.py`; caption: “Median ba measured repetitions theo canonical version trên cùng 10 batch × 10k; tổng apply+materialize+export, staleness theo batch và bytes snapshot export. R=5 chỉ có snapshot tại v5/v10”. [D13](../d13-exp02-results.md), claim E-022.

`exp01_latency.svg/png`, `exp01_speedup.svg/png`, `exp01_rss.svg/png`, `exp01_bytes.svg/png`: nguồn `results/raw/exp01_full_vs_incremental.csv` và script `src/benchmarking/exp01_analysis.py`; caption: “Median của năm measured repetitions trên mỗi cặp H/Δ; error bars min–max cho latency và paired Full/Incremental ratio; warm cache không kiểm soát”. RSS là sampled peak, bytes là output Parquet + manifest, DB growth ở raw CSV. [D12](../d12-exp01-results.md), claim E-021 trong [ma trận](../09-report-evidence.md).

`exp01_crossover.svg/png`: thêm summary của run 2M riêng `results/raw/exp01_scale2m_summary_d12.csv`; marker vuông rỗng đánh dấu source/config khác. Caption: “Δ=1k cho thấy Full/Incremental ratio vượt 1 ở H=2M, Δ=10k gần 1; đường chấm chỉ nối điểm đo, không xác định ngưỡng chính xác”.

`d09-pipeline.mmd`: nguồn sơ đồ luồng E2E từ cutoff train qua D04 ledger, Full equivalence, D06 normalization, LightGCN và metadata lookup sau ranking. Bằng chứng thực chạy ở [D09](../d09-end-to-end.md).

`d04-batch-flow.mmd`: nguồn sơ đồ Mermaid của apply_batch D04. Caption: “Luồng xử lý batch với ledger và DuckDB transaction; cạnh và degree được ghi đồng bộ trước commit”. Nguồn: `src/graph/incremental.py`; kiểm chứng E-012 trong `../09-report-evidence.md`. Đây là sơ đồ implementation, không phải biểu đồ hiệu năng.

`d05-equivalence-flow.mmd`: nguồn sơ đồ Mermaid cho D05. Caption: “Hai đường xử lý cùng canonical prefix được đối chiếu từng dòng tại cùng cutoff; replay được kiểm tra trên full snapshot”. Nguồn: `src/graph/d05_probe.py`, `src/graph/equivalence.py`; claim E-013 trong `../09-report-evidence.md`. Kết quả thực chạy ở `../d05-equivalence.md`.

`d06-version-flow.mmd`: nguồn sơ đồ Mermaid D06. Caption: “Canonical graph tiếp tục nhận batch trong khi normalized snapshot giữ version nguồn, node map và weights cố định đến lần refresh”. Nguồn: `src/graph/snapshot.py`, `src/graph/normalize.py`; kết quả kiểm chứng ở `../d06-normalization.md`. Đây là sơ đồ implementation, không phải biểu đồ cost/freshness thực nghiệm EXP-02.

`d07-lightgcn-flow.mmd` minh họa đường từ sparse COO đã chuẩn hóa đến loss BPR và cập nhật bảng embedding ban đầu. Nguồn implementation và kết quả đo GPU có trong [hồ sơ D07](../d07-lightgcn-smoke.md).

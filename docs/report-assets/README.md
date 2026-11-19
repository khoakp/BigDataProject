# Report assets

Lưu sơ đồ, biểu đồ và hình minh họa dùng trong báo cáo tại thư mục này. Mỗi hình cần caption, nguồn dữ liệu hoặc file thiết kế, và liên kết tới claim ở `../09-report-evidence.md`.

Chưa có hình thực nghiệm. Hình kiến trúc có thể được tạo từ thiết kế ở `../03-system-design.md`; biểu đồ EXP-01/02 và metadata chỉ được tạo sau khi có CSV đo thực trong `results/raw/`.

`d04-batch-flow.mmd`: nguồn sơ đồ Mermaid của apply_batch D04. Caption: “Luồng xử lý batch với ledger và DuckDB transaction; cạnh và degree được ghi đồng bộ trước commit”. Nguồn: `src/graph/incremental.py`; kiểm chứng E-012 trong `../09-report-evidence.md`. Đây là sơ đồ implementation, không phải biểu đồ hiệu năng.

`d05-equivalence-flow.mmd`: nguồn sơ đồ Mermaid cho D05. Caption: “Hai đường xử lý cùng canonical prefix được đối chiếu từng dòng tại cùng cutoff; replay được kiểm tra trên full snapshot”. Nguồn: `src/graph/d05_probe.py`, `src/graph/equivalence.py`; claim E-013 trong `../09-report-evidence.md`. Kết quả thực chạy ở `../d05-equivalence.md`.

`d06-version-flow.mmd`: nguồn sơ đồ Mermaid D06. Caption: “Canonical graph tiếp tục nhận batch trong khi normalized snapshot giữ version nguồn, node map và weights cố định đến lần refresh”. Nguồn: `src/graph/snapshot.py`, `src/graph/normalize.py`; kết quả kiểm chứng ở `../d06-normalization.md`. Đây là sơ đồ implementation, không phải biểu đồ cost/freshness thực nghiệm EXP-02.

`d07-lightgcn-flow.mmd` minh họa đường từ sparse COO đã chuẩn hóa đến loss BPR và cập nhật bảng embedding ban đầu. Nguồn implementation và kết quả đo GPU có trong [hồ sơ D07](../d07-lightgcn-smoke.md).

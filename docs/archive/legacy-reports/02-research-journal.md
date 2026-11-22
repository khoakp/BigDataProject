# 02 — Nhật ký tìm hiểu và nghiên cứu

> **Mục đích:** theo dấu *đã tìm hiểu gì / nguồn nào / kết luận chắc đến đâu / liên hệ dự án*. Không thay thế nhật ký code ([05](05-work-log.md)) hoặc sổ số đo ([07](../../research/experiment-log.md)).
> **Trạng thái:** khởi tạo 20/10/2026; bổ sung định hướng metadata ngày 10/11/2026. Các mục dưới đây ghi lại nội dung **đã bàn luận**, **không phải thí nghiệm đã chạy**.

## A. Quy ước mức độ bằng chứng

- **[NGUỒN]** Paper/tài liệu chính thức nói gì (có URL và phần trích cần kiểm tra).
- **[SUY LUẬN]** Nhận định rút ra từ kiến thức/đánh giá; chưa đo trên laptop.
- **[CẦN KIỂM CHỨNG]** Câu hỏi chưa có kết quả thực nghiệm.
- **[ĐÃ ĐO]** Có dữ liệu `results/raw/...`, config, commit, ngày và môi trường đi kèm.
- **[QUYẾT ĐỊNH]** Đã chọn hướng làm; không tự động đồng nghĩa đã đo hoặc đã đúng.

### Mẫu cho mỗi lần đọc/tìm hiểu

```markdown
### RJ-YYYYMMDD-NN — [Câu hỏi nghiên cứu]
- Ngày / thời lượng:
- Vấn đề cần hiểu:
- Nguồn chính (URL, paper, số trang/mục):
- Ghi chú theo cách hiểu của mình:
- Điều nguồn KHÔNG khẳng định:
- Claim kiểm chứng được:
- Mức độ bằng chứng: [NGUỒN] / [SUY LUẬN] / [CẦN KIỂM CHỨNG] / [ĐÃ ĐO]
- Ảnh hưởng tới kiến trúc/tiến độ:
- Câu hỏi tiếp theo:
- Liên kết task / DEC / EXP:
```

## B. Tổng hợp ba hướng nghiên cứu đã trao đổi

| Chủ đề | Deep Research ban đầu | Phản biện Gemini | Đánh giá độc lập sau GPU | Định hướng hiện tại |
|---|---|---|---|---|
| Bài toán | Incremental graph snapshots, nhấn mạnh DE | Giữ hướng nhưng sửa lỗ hổng semantics | Giữ DE làm chính, kiểm soát scope | Versioned incremental graph |
| Tính đúng LightGCN | Có nguy cơ giản lược thành append edges | Nhấn mạnh degree renormalization | Phân biệt canonical vs normalized graph | Test equivalence bắt buộc |
| Dataset | MovieLens 32M phù hợp benchmark DE | Đề nghị tách DE scale vs model scale | Chốt 32M DE; GNN scale đo VRAM | DE tới 32M nếu khả thi, model bắt đầu 1M |
| GPU / hạ tầng | Ban đầu coi laptop hạn chế | Nhiều khuyến nghị dựa CPU-only | Cập nhật RTX 4050 6GB, vẫn chưa biết RAM | 48h feasibility gate |
| Model freshness | Nên tách khỏi graph freshness | Đề xuất propagation-only | Có thử nghiệm riêng, không mở scope quá mức | EXP-03 warm nodes |
| Compaction / extra systems | Có thể mở rộng | Phản biện thảo luận thêm nhiều biến thể | Không làm bắt buộc | Stretch nếu đủ thời gian |

**Nhận xét phương pháp:** ba báo cáo là các **đánh giá đề xuất**, không phải số đo hiệu năng của máy bạn. Các tuyên bố tốc độ hoặc lợi thế normalization chưa có log thí nghiệm đều là *giả thuyết*. Không dùng ba báo cáo AI làm bằng chứng học thuật thay cho paper/documentation gốc.

## C. Những chủ đề đã thảo luận — phiên bản khởi tạo

### RJ-BASE-001 — Vì sao một cạnh mới ảnh hưởng normalization?
- **Câu hỏi:** Chỉ append normalized edge có đủ cho LightGCN không?
- **Nguồn:** He et al., *LightGCN* (https://arxiv.org/abs/2002.02126); LGConv docs (https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.LGConv.html).
- **Ghi chú:** \(w_{ui} = 1/\sqrt{d_ud_i}\); tăng \(d_u\) hoặc \(d_i\) làm các trọng số kề liên quan thay đổi.
- **Mức độ:** [NGUỒN] công thức; [QUYẾT ĐỊNH] tách DegreeState và materialization.
- **Cần kiểm chứng:** snapshot normalized incremental = snapshot full tại cùng version.
- **Liên kết:** [03 — Thiết kế](03-system-design.md), DEC-003, EXP-02.

### RJ-BASE-002 — Graph mới có bắt buộc train lại LightGCN?
- **Câu hỏi:** Phân biệt propagation-only và parameter training.
- **Nguồn:** LightGCN paper, PyG model docs (https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.models.LightGCN.html).
- **Ghi chú:** Trên warm nodes, có thể giữ base embedding parameters rồi tính propagation trên graph mới; đây **không phải** retraining.
- **Mức độ:** [SUY LUẬN] về lợi ích chất lượng/chi phí trên MovieLens; chưa chạy.
- **Cần kiểm chứng:** nếu graph mới khác graph cũ, candidate ranking/quality đổi như thế nào ở temporal cutoff?
- **Liên kết:** DEC-006; EXP-03.

### RJ-BASE-003 — Vì sao hai quy mô dữ liệu?
- **Câu hỏi:** Nếu dùng ML32M thì 6GB VRAM có đủ LightGCN không?
- **Nguồn:** báo cáo phần cứng độc lập; PyTorch CUDA memory docs (https://docs.pytorch.org/docs/stable/notes/cuda.html).
- **Ghi chú:** Data engineering runs CPU; GPU dành cho sparse propagation, training và scoring theo chunk. Độ lớn an toàn chưa thể kết luận từ dung lượng VRAM danh nghĩa.
- **Mức độ:** [QUYẾT ĐỊNH] 32M DE, GNN scale sau probe; [CẦN KIỂM CHỨNG] 1M/5M/10M feasibility.
- **Liên kết:** DEC-002, RISK-002, D01–D07.

### RJ-BASE-004 — Liên hệ Pinterest/Alibaba ở mức nào?
- **Câu hỏi:** Scale graph systems công nghiệp giải quyết graph growth bằng gì?
- **Ghi chú:** sampling, graph partition, caching, periodic refresh, pruning là các hướng xuất hiện trong công trình công khai. **Không được suy ra** mọi hệ thống dùng cùng phương án của đồ án.
- **Nguồn nền:** PinSage (https://arxiv.org/abs/1806.01973), AliGraph (https://arxiv.org/abs/1902.08730).
- **Mức độ:** [NGUỒN] bối cảnh, [SUY LUẬN] bài học áp dụng local.
- **Ứng dụng:** dùng để giải thích động cơ, không đưa vào phạm vi xây dựng cluster.

### RJ-20261110-001 — Có cần dùng genres để huấn luyện LightGCN không?
- **Câu hỏi:** Làm thế nào bổ sung tên phim/thể loại để kết quả gợi ý có ý nghĩa mà vẫn giữ bài toán incremental bipartite graph?
- **Nguồn:** GroupLens MovieLens 32M README (https://files.grouplens.org/datasets/movielens/ml-32m-README.html); `01-project-overview.md`/`03-system-design.md` cho chính sách đồ án.
- **[NGUỒN]:** MovieLens cung cấp `ratings.csv` và `movies.csv`; `movies.csv` chứa `movieId`, `title`, `genres`.
- **[QUYẾT ĐỊNH]:** `movies.csv` chỉ dùng làm lookup **sau khi LightGCN xếp hạng** và cho phân tích mô tả genre; chưa dùng để tạo Movie–Genre edges, features hoặc content-aware reranking.
- **[CẦN KIỂM CHỨNG]:** `movieId` mapping coverage, join cardinality, `(no genres listed)` và tỷ lệ genre trong Top-K. Chưa có model checkpoint/kết quả đo.
- **Ảnh hưởng:** metadata enrich là output quality/demo, không thay nghiên cứu chính Full vs Incremental; thêm unit test kiểm tra score/order/Recall–NDCG không đổi.
- **Liên kết:** DEC-011, DEC-012, `03-system-design.md`, `08-final-report-outline.md`.

### RJ-20261110-002 — Chọn MovieLens thay RetailRocket cho MVP 21 ngày
- **[QUYẾT ĐỊNH]:** MovieLens 32M là dataset chính. RetailRocket để nghiên cứu ETL sau kỳ nộp, không sử dụng làm model dataset trong MVP.
- **[SUY LUẬN]:** RetailRocket có hành vi event phong phú nhưng có thể quá thưa cho LightGCN, trong khi MovieLens thuận lợi hơn để thiết kế temporal recommendation; chưa phải kết quả so sánh thực nghiệm hai dataset.
- **[CẦN KIỂM CHỨNG]:** dataset source checksum, positive edge degree distribution, warm-node coverage, GPU scale. Xem DEC-011.

## D. Câu hỏi còn mở (Research Backlog)

| ID | Câu hỏi | Cách xác minh | Ưu tiên | Hạn |
|---|---|---|---|---|
| Q-01 | RTX 4050 train+eval LightGCN ở scale nào an toàn? | memory probe + short train | P0 | D07 |
| Q-02 | NodeRegistry stable IDs và PyG indexing tương thích ra sao? | graph-to-model mapper test | P0 | D06 |
| Q-03 | Có trùng `(userId,movieId)`/timestamp ties trong ML32M không? | DuckDB exploratory queries | P0 | D02 |
| Q-04 | Incremental thắng Full ở điều kiện nào? | EXP-01 size grid | P0 | D12 |
| Q-05 | Materialize normalization có thực sự đắt? | EXP-02 | P1 | D13 |
| Q-06 | Có bao nhiêu warm users/items đủ làm EXP-03? | cutoff cohort audit | P2 | D15 |
| Q-07 | Các giao diện sparse PyTorch hoạt động ổn định trong backward? | CUDA smoke test | P0 | D07 |
| Q-08 | Cách tái hiện benchmark khi OS cache thay đổi? | run protocol + randomize order | P1 | D11 |
| Q-09 | Movies metadata join có làm thay số/tính thứ tự Top-K không? | kiểm tra uniqueness + before/after ranking checksum | P0 | D09 |
| Q-10 | Phân phối genres trong Top-K khác catalog như thế nào? | coverage + multi-label counts, ghi rõ định nghĩa | P1 | D17 |

## E. Danh mục nguồn đọc ưu tiên

| Mã | Nguồn | Cần đọc phần |
|---|---|---|
| REF-01 | GroupLens ML32M: https://files.grouplens.org/datasets/movielens/ml-32m-README.html | Schema, timestamps, license/checksum |
| REF-02 | He et al. (2020) LightGCN: https://arxiv.org/abs/2002.02126 | Propagation, objective, empirical setup |
| REF-03 | Rendle et al. (2009) BPR: https://proceedings.mlr.press/r7/rendle09a.html | Pairwise objective, implicit feedback |
| REF-04 | PyG LightGCN: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.models.LightGCN.html | model forward/loss/propagation |
| REF-05 | DuckDB Parquet: https://duckdb.org/docs/data/parquet | Scans, write, pruning |
| REF-06 | PyTorch CUDA memory: https://docs.pytorch.org/docs/stable/notes/cuda.html | allocated vs reserved, reproducibility |
| REF-07 | PinSage: https://arxiv.org/abs/1806.01973 | Large-scale sampling context |
| REF-08 | AliGraph: https://arxiv.org/abs/1902.08730 | Industry infrastructure context |
| REF-09 | MovieLens 32M `movies.csv` trong GroupLens README | Schema, title, genres, `(no genres listed)` |

## F. Checklist cho một buổi học / tra cứu

- [ ] Viết **một câu hỏi cụ thể** trước khi đọc.
- [ ] Lưu URL hoặc vị trí paper.
- [ ] Ghi điều đã hiểu bằng ngôn ngữ của mình.
- [ ] Phân loại nguồn, suy luận, chưa đo.
- [ ] Gắn với task/DEC/EXP hoặc loại khỏi phạm vi.
- [ ] Nếu có quyết định thiết kế, thêm record vào [06](06-decisions-and-risks.md).

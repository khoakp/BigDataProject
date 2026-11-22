# BigDataProject — Project Direction Update v1.0

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](../../current/scope-and-decisions.md) · [Tiến độ v2](../../current/progress-and-evidence.md).

**Ngày cập nhật:** 21/11/2026
**Thời gian còn lại:** 14 ngày, 5–6 giờ/ngày
**Trạng thái:** Định hướng mới đã thống nhất; tài liệu và code cần được đối chiếu trước khi áp dụng.

## 1. Tên đề tài

**Tiếng Việt:** Xây dựng hệ thống gợi ý phim cá nhân hóa sử dụng LightGCN kết hợp pipeline xử lý dữ liệu lớn và cập nhật đồ thị tăng dần.

**Tiếng Anh:** Building a Personalized Movie Recommendation System Using LightGCN and an Incremental Big Data Pipeline.

## 2. Mục tiêu dự án

Xây dựng một hệ thống gợi ý phim có khả năng:

1. Xử lý dữ liệu MovieLens 32M thông qua pipeline Big Data Engineering.
2. Biểu diễn tương tác người dùng–phim bằng đồ thị hai phía (User–Movie Bipartite Graph).
3. Huấn luyện LightGCN để tạo danh sách gợi ý phim cá nhân hóa Top-K.
4. Đánh giá chất lượng mô hình bằng giao thức temporal evaluation và các baseline.
5. Cập nhật dữ liệu tương tác và graph theo batch, bảo đảm correctness và idempotency.
6. Cung cấp ứng dụng web cho phép người dùng khám phá, đánh giá và nhận gợi ý phim.
7. Đánh giá chi phí cập nhật đồ thị và các giới hạn về model freshness.

Dự án phải thể hiện rõ hai trụ cột bắt buộc:

- Big Data Engineering.
- Graph-based Deep Recommender Systems.

Dự án không đặt mục tiêu đề xuất một kiến trúc GNN mới hoặc xây dựng hệ thống streaming production.

## 3. Các quyết định kỹ thuật chính

### 3.1. Dataset

- Sử dụng MovieLens 32M làm nguồn dữ liệu chính.
- Giữ toàn bộ raw ratings cho Data Engineering và phân tích.
- Sử dụng `rating >= 4.0` làm quy tắc positive interaction hiện hành; ghi rõ đây là lựa chọn thiết kế cần được giải thích, không khẳng định là ngưỡng tối ưu.
- Xây dựng graph từ dữ liệu train đúng theo temporal cutoff.
- `movies.csv` dùng để bổ sung tên phim và genres sau quá trình ranking.
- Không đưa tương tác validation/test tương lai vào train graph.

### 3.2. Big Data Engineering

Tiếp tục sử dụng:

- Python.
- DuckDB.
- Parquet.
- Full Graph Rebuild.
- Stateful Incremental Graph Update.
- NodeRegistry, EdgeState và DegreeState.
- Graph versioning và normalized sparse adjacency.
- Correctness, idempotency và reproducibility tests.

Ưu tiên tái sử dụng implementation D01–D07 đã hoàn thành.

### 3.3. Deep Recommendation

**Mô hình chính: LightGCN.**

- User–Item bipartite graph.
- Trainable user/item embeddings.
- Sparse graph propagation.
- BPR ranking objective.
- Negative sampling.
- Top-K recommendation.

Không chuyển sang PinSage, XSimGCL hoặc thuật toán khác trong MVP.

Popularity là baseline bắt buộc. ItemKNN là baseline ưu tiên nếu đủ thời gian.

### 3.4. Training Infrastructure

- Kaggle GPU là môi trường training chính.
- Laptop dùng cho ETL, graph processing, kiểm thử, benchmark Engineering và ứng dụng.
- Training theo quy mô tăng dần, bắt đầu ở mức an toàn rồi mở rộng khi có bằng chứng đủ tài nguyên.
- Không bắt buộc training trên toàn bộ 32 triệu ratings.
- Bảo đảm checkpoint, model config, ID mappings và graph version được lưu để inference có thể tái lập.

### 3.5. Web Application

Sử dụng Streamlit cho MVP.

Chức năng chính:

- Chọn người dùng có trong dữ liệu hoặc chọn phim yêu thích theo luồng demo được hỗ trợ.
- Hiển thị danh sách phim và lịch sử tương tác.
- Sinh Top-20 recommendations từ mô hình LightGCN đã huấn luyện.
- Hiển thị title và genres.
- Ghi nhận rating mới.
- Cập nhật graph theo batch.
- Hiển thị graph version, model version và trạng thái refresh.
- Chỉ thay đổi recommendations theo cơ chế model refresh đã thực sự triển khai.

Với người dùng hoàn toàn mới, sử dụng fallback phù hợp; không giả định checkpoint LightGCN có embedding đã học cho user mới.

## 4. Evaluation

### Model Quality

Bắt buộc:

- Temporal train/validation/test split.
- No-future-leakage tests.
- Recall@20.
- NDCG@20.
- Popularity baseline.
- Seen-item masking.
- Chọn checkpoint bằng validation, đánh giá test độc lập.

Ưu tiên bổ sung:

- ItemKNN.
- Coverage@20.
- Popularity bias.
- Inference latency.

Tỷ lệ temporal split và candidate protocol phải được xác định từ dữ liệu thực tế và cố định trước final evaluation.

### Big Data Engineering

Giữ EXP-01: Full Rebuild vs Incremental Update.

Metrics:

- Wall-clock time.
- Peak RAM.
- Data I/O.
- Events processed per second.
- Graph equivalence và correctness.
- Chi phí theo history size và delta size.

Đo các phương án trên cùng môi trường và cùng output contract.

### Model Freshness

Thử nghiệm bổ sung khi MVP đã đạt:

- Stale model.
- Propagation-only trên graph mới.
- Trained refresh nếu đủ tài nguyên.

Đánh giá chi phí refresh và chất lượng recommendation trên cùng protocol.

EXP-02 cũ về deferred normalization được hạ xuống phần mở rộng, không nằm trên critical path.

Không thay số hiệu các thí nghiệm cũ một cách âm thầm. Sử dụng các mã REC-01 cho recommendation quality và REC-02 cho model freshness nếu cần để giữ lịch sử thực nghiệm nhất quán.

## 5. Phạm vi ưu tiên

### P0 — Bắt buộc

- End-to-end Big Data pipeline.
- LightGCN được huấn luyện thực sự.
- Temporal evaluation và Popularity baseline.
- Checkpoint và Top-K inference.
- EXP-01 với raw benchmark logs.
- Streamlit demo tối giản.
- Tests, báo cáo, slide và tài liệu tái lập.

### P1 — Ưu tiên cao

- ItemKNN baseline.
- Training/inference profiling.
- Model freshness: stale vs propagation-only.
- Dashboard hiển thị graph/model version.
- Mở rộng quy mô benchmark nếu tài nguyên cho phép.

### P2 — Chỉ thực hiện nếu dư thời gian

- XSimGCL, PinSage.
- Full 32M model training.
- Spark, Hadoop, Kafka, Neo4j.
- Deferred normalization experiments nâng cao.
- Multi-GPU và distributed training.
- Advanced cold-start, authentication, deployment production.

## 6. Lộ trình Sprint 14 ngày

Giữ nguyên lịch sử D01–D07. Sử dụng nhãn S01–S14 cho sprint mới để tránh nhầm với milestone đã hoàn thành.

| Ngày | Trọng tâm                               | Kết quả cần có                         |
| ----- | ----------------------------------------- | ------------------------------------------ |
| S01   | Audit code, chốt evaluation protocol     | Evaluation specification                   |
| S02   | Temporal split, metrics và baseline      | Evaluator có tests                        |
| S03   | LightGCN training trên Kaggle            | Checkpoint đầu tiên                     |
| S04   | Quality evaluation                        | Recall/NDCG và baseline comparison        |
| S05   | End-to-end recommendation pipeline        | Top-K có metadata                         |
| S06   | EXP-01 benchmark implementation           | Raw benchmark logs                         |
| S07   | Benchmark analysis và scalability        | Figures và Engineering conclusions        |
| S08   | Model freshness                           | Chi phí refresh và quality nếu khả thi |
| S09   | Streamlit recommendation UI               | Web app MVP                                |
| S10   | Rating interaction và graph refresh demo | Demo end-to-end                            |
| S11   | Integration tests và scaling             | Hệ thống ổn định                      |
| S12   | Feature freeze và báo cáo              | Báo cáo đầy đủ                       |
| S13   | Slides và luyện bảo vệ                | Slide + Q&A                                |
| S14   | Final QA, reproducibility và đóng gói | Submission package                         |

**Milestone gates:**

- S04: Có training, checkpoint và quality evaluation.
- S07: Có benchmark Big Data Engineering hợp lệ.
- S10: Có ứng dụng demo tương tác.
- S14: Có sản phẩm, kết quả, tài liệu và bằng chứng đầy đủ.

Nếu chậm tiến độ, giảm model scale và phạm vi thử nghiệm trước khi cắt các thành phần P0.

## 7. Quy trình AI-assisted Development

**Codex chịu trách nhiệm chính:**

- Phân tích kiến trúc.
- Viết và chỉnh sửa source code.
- Unit/integration testing.
- Debugging.
- Training và evaluation scripts.
- Benchmarking.
- Tài liệu kỹ thuật và report notes.

**Project Owner chịu trách nhiệm:**

- Review quyết định kỹ thuật.
- Kiểm tra logs và kết quả thực tế.
- Review chức năng và demo.
- Học các thuật toán, kiến trúc và trade-offs.
- Nghiệm thu từng milestone.

Sau mỗi milestone, Codex phải cung cấp:

- Các module được triển khai.
- Input/output và luồng xử lý.
- Các file/class/function cần đọc.
- Test và benchmark đã chạy thực tế.
- Evidence và giới hạn chưa kiểm chứng.
- Giải thích What–Why–How–Verification bằng tiếng Việt.
- Ba câu hỏi kiểm tra kiến thức.

Codex không tự ý commit/push, thay đổi kiến trúc lớn hoặc mở rộng phạm vi khi chưa được chấp thuận.

## 8. Tiêu chí hoàn thành cuối cùng

Dự án được coi là hoàn thành MVP khi:

1. Có pipeline xử lý MovieLens và tái lập được graph/model artifacts.
2. Có mô hình LightGCN đã huấn luyện, lưu checkpoint và sinh Top-K hợp lệ.
3. Có báo cáo Recall@20/NDCG@20 so sánh với Popularity.
4. Có benchmark Full vs Incremental được kiểm chứng correctness.
5. Có ứng dụng web cho phép trình diễn gợi ý phim và ghi nhận tương tác.
6. Có tài liệu triển khai, experiment logs, nguồn dữ liệu, cấu hình và kết quả kiểm thử.
7. Người thực hiện hiểu và giải thích được toàn bộ kiến trúc và các quyết định kỹ thuật.

**Nguyên tắc:** Không bịa số liệu, không che giấu kết quả kém, không đồng nhất cập nhật graph với cập nhật tham số mô hình, không thay đổi giao thức evaluation để cải thiện kết quả một cách thiếu công bằng.

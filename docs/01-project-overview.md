# 01 — Tổng quan và đặc tả đề tài

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

> **Định hướng hiện hành từ 21/11/2026:** *Xây dựng hệ thống gợi ý phim cá nhân hóa sử dụng LightGCN kết hợp pipeline xử lý dữ liệu lớn và cập nhật đồ thị tăng dần* (*Building a Personalized Movie Recommendation System Using LightGCN and an Incremental Big Data Pipeline*). [Bản v1.0](10-direction-update-v1.md) và [sprint S01–S14](11-sprint14-plan.md) thay thế ưu tiên cũ trong tài liệu này; các đoạn bên dưới được giữ làm lịch sử D01–D13. Big Data Engineering và Graph-based Deep Recommender Systems đều là trụ cột bắt buộc; Popularity/Top-20 quality và Streamlit thuộc P0, EXP-02 là phần mở rộng đã chạy.

> **Trạng thái:** charter lịch sử D01–D13, đã được [bản v1.0](10-direction-update-v1.md) thay thế từ 21/11/2026 · **Bản gốc:** 10/11/2026.
> **Người thực hiện:** [Điền tên] · **Môn:** Introduction to Big Data · **Thời gian triển khai còn lại:** khoảng 3 tuần (21 ngày tương đối)  
> **Ngày bắt đầu / hạn nộp chính thức:** CHƯA XÁC NHẬN — xem [04 — Kế hoạch](04-project-plan.md).

## 1. Tên đề tài

**Tên đề xuất với giảng viên:** **Xây dựng và đánh giá pipeline cập nhật đồ thị tương tác tăng dần cho hệ thống gợi ý phim sử dụng LightGCN.**

**Tên học thuật đầy đủ:** *Xây dựng và đánh giá pipeline cập nhật đồ thị tương tác tăng dần có quản lý phiên bản, chuẩn hóa trì hoãn và làm mới LightGCN có kiểm soát.*

**English:** *Design and Evaluation of a Versioned Incremental Interaction Graph Pipeline for Movie Recommendation using LightGCN.*

## 2. Bối cảnh và động lực

Các hệ thống gợi ý nhận tương tác người dùng–sản phẩm theo thời gian. Khi dữ liệu tăng, việc quét lại toàn bộ lịch sử để xây dựng đồ thị sau mỗi batch có thể gây chi phí I/O, CPU và thời gian xử lý lặp lại. Với các graph-based recommenders như **LightGCN**, còn phát sinh vấn đề: toán tử lan truyền dựa trên đồ thị chuẩn hóa theo bậc; một cạnh mới không chỉ tạo một trọng số mới mà có thể làm thay đổi trọng số của những cạnh hiện hữu nối với hai đầu mút.

Dự án tập trung vào vòng đời **dữ liệu → trạng thái đồ thị → biểu diễn đồ thị dành cho mô hình → embeddings → gợi ý**, không đề xuất kiến trúc GNN mới. Đây là một nghiên cứu **Data Engineering / ML Systems trên một máy**; dữ liệu MovieLens được phát lại theo timestamp nhằm **mô phỏng batch tăng dần**, không phải production real-time stream.

**Phạm vi dataset đã thống nhất:** MovieLens 32M là nguồn chính cho graph, LightGCN và benchmark. `movies.csv` được dùng làm bảng metadata để JOIN tên phim và genres với Top-K recommendations, phục vụ hiển thị và phân tích hậu kiểm. **Không đưa genres/title vào graph topology hoặc LightGCN embeddings trong MVP.** RetailRocket được hoãn lại như nguồn dữ liệu ETL có thể nghiên cứu sau khi nộp; không nằm trong critical path của 21 ngày.

Alibaba và Pinterest là bối cảnh tham khảo về nhu cầu graph systems; đồ án **không tuyên bố sao chép hệ thống nội bộ, không suy rộng kết quả 32M lên quy mô hàng tỷ cạnh**.

## 3. Phát biểu vấn đề

Cho chuỗi tương tác theo thời gian \(E_{\le t}\) và batch mới \(\Delta E_{t+1}\). So sánh:

1. **Full Rebuild:** đọc lịch sử đến thời điểm hiện tại để dựng lại canonical graph.
2. **Incremental Update:** chỉ áp dụng thay đổi mới lên trạng thái có phiên bản, quản lý NodeRegistry, edge state và DegreeState.
3. **Materialization Policy:** tại các phiên bản được chọn, kết xuất toán tử đồ thị thưa chuẩn hóa dành cho LightGCN.
4. **Refresh Policy:** quyết định khi nào cập nhật embeddings bằng propagation và/hoặc huấn luyện lại theo ngân sách định trước.

Mục tiêu không phải chứng minh incremental hoặc deferred *luôn nhanh hơn*; mục tiêu là **định lượng điều kiện hiệu quả và đánh đổi giữa freshness–cost–correctness–quality**.

## 4. Các lớp freshness cần phân biệt

| Lớp | Tài sản (artifact) | Khi nào được coi là mới? |
|---|---|---|
| Event freshness | Canonical event store | Batch hợp lệ đã được ghi nhận |
| Canonical graph freshness | EdgeState / DegreeState / NodeRegistry, version | Tất cả cạnh hợp lệ đến watermark đã áp dụng |
| Model-graph freshness | Normalized sparse adjacency, version | Toán tử \(\tilde A\) được kết xuất từ đúng canonical version |
| Embedding freshness | LightGCN propagated embeddings | Đã propagation trên model-graph version tương ứng |
| Parameter freshness | Learned base embeddings / checkpoint | Đã tối ưu tham số bằng dữ liệu cho phép đến cutoff |

Công thức trọng số LightGCN cho tương tác dương \((u,i)\):

\[
\tilde A = D^{-1/2}AD^{-1/2},\qquad w_{ui}=\frac{1}{\sqrt{d_u d_i}}.
\]

**Hệ quả:** cập nhật *cạnh* và *degree* có thể tăng dần; nhưng normalized edge weights cũ không thể được tái sử dụng vô điều kiện sau khi bậc đầu mút thay đổi. **Deferred normalization** chỉ trì hoãn *kết xuất biểu diễn model-ready*, không trì hoãn cập nhật đúng canonical graph; khi kết xuất phải chính xác với phiên bản được chọn.

## 5. Câu hỏi nghiên cứu (RQ) và giả thuyết cần kiểm chứng

| Mã | Câu hỏi | Giả thuyết (CHƯA ĐƯỢC ĐO) | Thực nghiệm |
|---|---|---|---|
| RQ1 | Chi phí Full Rebuild vs Incremental thay đổi thế nào theo history/delta size? | Delta nhỏ / history lớn thường lợi cho incremental; có thể tồn tại crossover | EXP-01 |
| RQ2 | Chuẩn hóa mỗi batch so với mỗi 5 batch (20 batch là stretch) đánh đổi chi phí và độ trễ ra sao? | Ít materialize hơn có thể giảm tổng chi phí nhưng tăng staleness | EXP-02 |
| RQ3 | Giữ embeddings cũ, propagation-only và trained refresh khác nhau thế nào? | Propagation-only có thể cập nhật gợi ý cho warm nodes với chi phí thấp hơn retrain | EXP-03 |

**Góc nhìn phản biện:** Nếu normalization rất rẻ, lợi ích của deferred normalization có thể thấp. Nếu incremental không thắng trên một số điều kiện, đó là kết quả hợp lệ. Nếu LightGCN không thắng BPR-MF, dự án vẫn đạt trọng tâm DE nếu correctness và experiments đạt chuẩn.

## 6. Mục tiêu và đầu ra

### Bắt buộc (MVP)

- Pipeline ingest + schema validation + deterministic chronological replay từ MovieLens.
- Full Rebuild và Stateful Incremental Builder tạo **cùng tập cạnh logic**, node registry và degree state tại cùng watermark.
- Idempotency: áp dụng lại batch không tạo cạnh hay cộng degree hai lần.
- Graph versions + manifest: có thể theo dõi phiên bản canonical và model-ready độc lập.
- Materialize normalized graph chính xác; unit/integration tests đối chiếu full/reference.
- LightGCN end-to-end trên quy mô **GPU đo được là an toàn**, temporal evaluation Recall@K/NDCG@K.
- **EXP-01 bắt buộc**, **EXP-02 ưu tiên cao** (R=1 và R=5); **EXP-03 là stretch** khi còn thời gian. Nếu chưa thực hiện EXP-03, cập nhật câu hỏi nghiên cứu/giới hạn báo cáo cho nhất quán, tuyệt đối không báo cáo số liệu suy đoán.
- Data enrichment bằng `movies.csv`: JOIN tên phim/genres cho Top-K, kiểm tra metadata coverage và có ít nhất một phân tích phân phối thể loại minh họa (không coi là đánh giá chất lượng nếu thiếu protocol).
- README tái hiện ít nhất một lần chạy tiêu biểu và demo không phải train từ đầu.

### Có thể cắt giảm

UI, Kafka/Spark/Neo4j, xử lý cold-start hoàn chỉnh, custom CUDA, huấn luyện full 32M, distributed training, online continual retraining, nhiều biến thể compaction/model tuning.

## 7. Dataset, quy mô, hạ tầng

| Mục đích | Đề xuất | Trạng thái |
|---|---|---|
| DE workload | MovieLens 32M (32.000.204 rating events) | **Dataset chính đã chọn**; scale pilot 100k/1M rồi mở rộng nếu CPU/RAM/đĩa đáp ứng |
| Mô hình | Smoke 100k; train/eval 1M, lớn hơn nếu GPU probe pass | **Chưa chốt scale an toàn**; không mặc định dùng toàn bộ 32M |
| Quy tắc positive | `rating >= 4.0` | **Đề xuất — chốt W1**; giữ raw events tách khỏi positive edge |
| GPU | RTX 4050 Laptop, 6 GB VRAM; Kaggle GPU tùy chọn | Probe và pin môi trường; cloud không phải điều kiện bắt buộc |
| CPU, RAM, disk | Chưa biết | Phải kiểm tra D01 |
| Metadata / enrichment | MovieLens 32M `movies.csv` (movieId, title, genres) | **Bổ sung hậu xử lý**, không dùng làm model features/graph edges ở MVP |
| RetailRocket | Dữ liệu ETL mở rộng | Để sau thời hạn nộp; không chạy như benchmark cùng chính sách graph |
| Cloud | Không bắt buộc | Kaggle chỉ dùng model train/propagation nếu thuận lợi |

**Lưu ý dữ liệu:** các prefix 1M/5M/10M... của MovieLens 32M là lát cắt từ **một bộ dữ liệu sau sắp xếp thời gian**, không đồng nhất với bộ dữ liệu MovieLens 1M riêng do GroupLens phát hành. Chọn một nguồn cho mỗi thí nghiệm và ghi `dataset_id` cụ thể.

### 7.1. Quy tắc sử dụng movie metadata (bổ sung ngày 10/11/2026)

- `ratings.csv` là sự kiện nguồn; `userId`, `movieId`, `rating`, `timestamp` phục vụ graph và temporal replay. `movies.csv` giữ `movieId`, `title`, `genres`; MovieLens không cung cấp cột năm phát hành riêng trong file này (năm có thể có trong title nhưng **không được mặc định parse chính xác**).
- Metadata được JOIN với **Top-K movie IDs sau ranking**; kết quả bao gồm `userId`, `movieId`, `score`, `rank`, `title`, `genres`, `model_graph_version`, `checkpoint_version` khi khả dụng. Thiếu metadata thì giữ item ID và gắn trạng thái `metadata_found=false`, không loại recommendation khỏi metric một cách âm thầm.
- Genre là danh sách đa nhãn phân cách bởi `|`, có thể chứa `(no genres listed)`. Phân tích phân phối thể loại của Top-K là **phân tích mô tả**; không tự nhận là nguyên nhân hay cải thiện Recall/NDCG.
- Kiểm tra `movies.movieId` duy nhất, metadata coverage của item trong graph/Top-K, số dòng sau JOIN không tăng bất ngờ, và tính bất biến của raw ranking trước/sau enrich.
- Tài liệu nguồn: [GroupLens ML32M README](https://files.grouplens.org/datasets/movielens/ml-32m-README.html). Các con số metadata coverage phải được đo, không ước lượng.

## 8. Công nghệ và lý do chọn

| Thành phần | Công cụ dự kiến | Lý do / điều kiện |
|---|---|---|
| Source/canonical files | CSV → Parquet | Định dạng lưu trữ hướng cột, có thể đọc theo batch |
| ETL, state prototype | Python + DuckDB | SQL và query trực tiếp Parquet trên một máy |
| State/metadata | DuckDB state tables + snapshot/delta files | Tránh rewrite toàn bộ Parquet khi cần upsert mutable state; so sánh I/O thực tế |
| Model | PyTorch / PyTorch Geometric | Có LightGCN, sparse operations; phải pin version sau GPU feasibility gate |
| Benchmark | `time`, `psutil`, `torch.cuda` | Thời gian / RSS / allocated / reserved |
| Tests | pytest | Kiểm tra correctness trước benchmark |
| Metadata enrichment | DuckDB JOIN + `movies.csv` | Hiển thị tên phim, genres; không đổi cấu trúc LightGCN |
| Optional visualization | matplotlib | Hình sinh từ CSV, không vẽ tay |

Không dùng công nghệ chỉ để có danh sách công nghệ dài. Nếu DuckDB/state layout trở thành bottleneck, ghi vào Decision Log trước khi thay.

## 9. Điều kiện được xem là thành công (Definition of Done)

- [ ] Deterministic replay, exact edge set + registry + degree equivalence.
- [ ] Normalized adjacency equivalence ở chung snapshot (dùng tolerance số thực có ghi).
- [ ] Reapply batch idempotent; restart từ checkpoint không mất trạng thái.
- [ ] Có bảng EX​P-01: history/delta vs wall time, CPU RSS, bytes written, correctness.
- [ ] Có bảng EXP-02: chu kỳ chuẩn hóa vs cumulative cost/staleness.
- [ ] EXP-03 nếu còn thời gian; nếu bỏ, sửa RQ3 và nêu giới hạn chính thức.
- [ ] Chạy được tối thiểu một LightGCN setting mà không OOM.
- [ ] Không sử dụng dữ liệu tương lai để tạo graph/training parameters tại cutoff.
- [ ] Có raw logs, commit/config/seed và hướng dẫn tái hiện.
- [ ] Metadata join đúng 1:1 theo movieId; Top-K giữ nguyên thứ tự/score; có coverage và thể loại minh họa.
- [ ] MVP chậm nhất D10, feature freeze D16, báo cáo/đóng gói D17–D21 theo [04 — Kế hoạch 21 ngày](04-project-plan.md).

## 10. Các quyết định chưa chốt

1. Ngày bắt đầu, ngày nộp cuối cùng; thời gian khả dụng mỗi ngày.
2. Thông số RAM/CPU/SSD, Windows native hay WSL2.
3. Dataset đã chốt là MovieLens 32M; chọn prefix model 100k/1M sau GPU probe (không dùng MovieLens 1M riêng trừ khi có ADR).
4. Batch size, seed, negative sampling, model dimensions và giới hạn OOM.
5. Materialization trên CPU, transfer GPU và độ lớn nào thực sự ổn định.
6. Ngày nộp/giờ nộp thực tế; lịch 21 ngày hiện chỉ là kế hoạch tương đối.

Các thay đổi phạm vi phải ghi vào [06 — Decision Log](06-decisions-and-risks.md). Công việc hàng ngày tuân theo [04 — Kế hoạch](04-project-plan.md).

## 11. Cơ sở nguồn

- Báo cáo Deep Research ban đầu — `deep-research-report.md` (nguồn người dùng cung cấp, tổng quan lựa chọn bài toán).
- Phản biện Gemini — `Phản Biện Đồ Án GNN.md` (nguồn người dùng cung cấp; các con số suy đoán về hiệu năng **không** coi là benchmark).
- Đánh giá phần cứng độc lập — `deep-research-report (1).md` (nguồn người dùng cung cấp; căn cứ chọn phạm vi RTX 4050).
- GroupLens MovieLens 32M: https://files.grouplens.org/datasets/movielens/ml-32m-README.html
- He et al., LightGCN: https://arxiv.org/abs/2002.02126
- Rendle et al., BPR: https://proceedings.mlr.press/r7/rendle09a.html

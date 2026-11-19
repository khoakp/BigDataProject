# 08 — Ghi chú tích lũy cho báo cáo học phần

> Cập nhật: 19/11/2026. Đây là bản nháp học thuật theo **trạng thái bằng chứng**, không phải báo cáo kết quả cuối cùng. `08-final-report-outline.md` vẫn là dàn ý; file này chứa nội dung có thể phát triển thành các chương. Mọi nhận định thực nghiệm phải đối chiếu [09 — Evidence](09-report-evidence.md) và [07 — Experiment Log](07-experiment-log.md).

## Quy ước

- **Đã xác minh:** đã kiểm tra trực tiếp tài liệu, file hoặc phép kiểm tra được ghi ở [09](09-report-evidence.md).
- **Thiết kế đã chọn:** quyết định/phương pháp trong [03](03-system-design.md) và [06](06-decisions-and-risks.md), chưa đồng nghĩa đã triển khai.
- **Dự kiến/giả thuyết:** cần code, test hoặc số đo trước khi dùng như kết luận.
- Không lấy các mốc D01–D21 trong [04](04-project-plan.md) làm ngày hoàn thành; đối chiếu những milestone thực sự đã ghi ở [05](05-work-log.md).

## 1. Research Objective

Đề tài nghiên cứu vòng đời cập nhật đồ thị tương tác user–movie từ dữ liệu rating lịch sử: chuẩn hóa thứ tự sự kiện, cập nhật trạng thái đồ thị theo batch, tạo biểu diễn đồ thị cho LightGCN, rồi đánh giá độ đúng và chi phí. Câu hỏi chính là liệu incremental update có tạo cùng edge set, node mapping và degree như full rebuild tại cùng cutoff, và chi phí của hai cách biến đổi theo quy mô ra sao. Deferred normalization là câu hỏi riêng về chi phí và độ trễ snapshot. Đây là **mục tiêu nghiên cứu**, chưa có kết luận thực nghiệm. Xem [01, mục 3–6](01-project-overview.md), [04, EXP-01/02](04-project-plan.md).

D01–D07 đã hoàn thành tính đến 20/11/2026. G1 đạt trên toy/100k. G2 đạt sau kiểm chứng normalization D06 và CUDA smoke trên toy/100k/1M. [D07](d07-lightgcn-smoke.md).

## 2. Theoretical Background

Đồ thị được thiết kế là bipartite: cạnh nối user với movie. Theo DEC-005 đã chốt cho D03, mỗi cặp có ít nhất một rating đạt ngưỡng `rating >= 4.0` sẽ thành một cạnh binary. Với edge set `E`, bậc của user `u` và movie `i` là `d_u = |{i : (u,i) ∈ E}|`, `d_i = |{u : (u,i) ∈ E}|`. Vì mỗi cạnh đóng góp một lần vào mỗi phía, `Σ_u d_u = |E| = Σ_i d_i` là invariant để kiểm thử.

Thiết kế normalization dùng trọng số `w_ui = 1 / sqrt(d_u d_i)` cho cạnh `(u,i)` ở snapshot. Khi thêm cạnh làm đổi bậc, trọng số của các cạnh cũ kề node đó cũng đổi. Do đó append cạnh vào *normalized graph* không đủ để bảo đảm snapshot mới đúng. Đây là hệ quả toán học của công thức đã ghi ở [03, mục 6](03-system-design.md); D06 đã kiểm chứng implementation với dense toy và SQL reference tại ba cutoff [D06](d06-normalization.md).

**Nguồn học thuật cần đối chiếu khi viết báo cáo chính thức:** danh mục [02, mục E](02-research-journal.md) gồm paper LightGCN, BPR và tài liệu GroupLens. Trong lần tổng hợp này, chỉ `ml-32m/README.txt` và `checksums.txt` cục bộ được đọc trực tiếp; không ghi một nhận định chi tiết nào như đã kiểm tra trực tiếp paper.

## 3. Methodology

Phương pháp **đã chọn ở mức thiết kế**: đọc `ratings.csv`, kiểm tra schema, sort xác định theo `timestamp, userId, movieId, source_row_id`, tạo canonical events, rồi replay theo batch. Full rebuild là reference tại cutoff; incremental path cập nhật registry, edge set và degree bằng những cạnh thực sự mới. Replay cùng batch phải idempotent. Chỉ so sánh thời gian khi kết quả state tương đương. Xem [03, mục 2–5](03-system-design.md), [06, DEC-003/004/009](06-decisions-and-risks.md).

`canonical_version` và `model_graph_version` được tách để biểu diễn normalization bị trì hoãn; khi materialize ở cùng version, ma trận phải khớp reference trong tolerance được chốt trước test. `movies.csv` chỉ là lookup sau Top-K ranking, nhằm gắn `title` và `genres`; không tham gia graph hoặc input LightGCN ở MVP. Đây là quyết định thiết kế DEC-012, chưa có test về tính bất biến của ranking.

Thí nghiệm dự kiến: EXP-01 so sánh Full và Incremental với cùng history/delta/output contract; EXP-02 so sánh chu kỳ normalization `R=1` và `R=5` trên cùng batch sequence, ghi cả cost lẫn staleness. EXP-03 về model refresh là phần mở rộng. Protocol, metrics và điều kiện công bằng ở [07, mục 1, 3–5](07-experiment-log.md).

## 4. Implementation

Repo có code và tests cho canonical prefix, Full Rebuild, incremental updates, exact equivalence, normalization và metadata validation. D07 thêm LightGCN propagation, loss BPR và GPU probe. Temporal evaluation và pipeline đề xuất hoàn chỉnh thuộc D08–D10. [Thiết kế](03-system-design.md), [D07](d07-lightgcn-smoke.md).

Đầu vào cục bộ đã kiểm tra: `../ml-32m/ratings.csv` và `../ml-32m/movies.csv`, với header lần lượt là `userId,movieId,rating,timestamp` và `movieId,title,genres`. Hai MD5 khớp manifest đi kèm. D02 đã profile toàn bộ `ratings.csv`, D03 profile phân phối rating và xây Full Rebuild reference; D06 validate toàn bộ `movies.csv` với 87.585 keys unique/non-null. Xem [D02](d02-data-profile.md), [D03](d03-full-reference.md), [D06](d06-normalization.md). Chưa đánh giá chất lượng model theo ngưỡng positive hoặc metadata join invariance trên Top-K.

## 5. Experimental Setup

MovieLens 32M đã qua kiểm tra nguồn và profiling. D07 dùng chronological prefix 100k và 1M, seed 7, COO float32, embedding 64 chiều và 3 lớp trên RTX 4050 với PyTorch CUDA 12.6. Probe dùng bốn bước BPR để kiểm tra khả năng chạy và đo bộ nhớ. Data split và metric đánh giá gợi ý sẽ được thiết lập ở D08. [Môi trường](environment.md), [D07](d07-lightgcn-smoke.md).

## 6. Results and Discussion

Hiện **không có kết quả benchmark hoặc model evaluation đã kiểm chứng**. Chưa thể kết luận Incremental nhanh hơn Full, deferred normalization tiết kiệm chi phí, LightGCN đạt chất lượng nào, hoặc metadata coverage bao nhiêu. Bảng số liệu và biểu đồ sẽ được tạo từ `results/raw/*` sau khi có run hợp lệ; [07](07-experiment-log.md) giữ raw metrics và failure records, file này chỉ diễn giải kết quả đã có.

D02 đã ghi nhận `ratings.csv` có 32,000,204 dòng; 84,432 movie IDs xuất hiện trong ratings; không có null trong bốn cột và không có cặp user–movie trùng. Có 6,023,681 dòng vượt số timestamp phân biệt và 15,647,103 lần timestamp giảm theo source order. Vì vậy explicit sort và tie-break theo `source_row_id` là cần thiết để tạo replay xác định. Đây là kết quả **profile dữ liệu**, không phải kết quả hiệu năng; bảng đầy đủ và điều kiện đo ở [d02-data-profile](d02-data-profile.md).

D03 đã chốt `rating >= 4.0` theo DEC-005 sau khi profile phân phối rating, tạo full rebuild reference trên toy và canonical prefix 100k. Prefix tạo 57.925 cạnh và 2.819 node; tổng bậc user và movie đều bằng 57.925, không phát hiện sai lệch khi đếm lại từ output. Đây là baseline correctness cho D04–D05, **không phải** bằng chứng Incremental đã đúng hoặc nhanh hơn. Chi tiết phương pháp, kết quả và đoạn báo cáo nháp ở [D03](d03-full-reference.md).

D04 đã kiểm chứng idempotency, hash conflict rejection, degree delta và rollback; integration probe trên 100k với 10 batches đạt. Phương pháp, input/output, setup, kết quả và bản nháp học thuật ở [d04-incremental.md](d04-incremental.md). Batch ledger lưu hash và watermark cùng transaction với graph để tránh replay tăng degree.

D05 bổ sung phép so sánh chính xác từng dòng registry, edge và degree tại cùng cutoff, trên toy và prefix 100k qua 4 cách chia batch mỗi dataset. Cả 35 boundary checks và 70 replays đều đạt, hoàn thành G1 trong phạm vi này. Đã sửa lỗi Full Rebuild trên prefix không có positive edges và chạy regression test. Phần methodology, setup, results, limitations và đoạn báo cáo nháp ở [D05](d05-equivalence.md); không diễn giải các kết quả này thành lợi ích hiệu năng.

D06 đã materialize sparse COO đối xứng và kiểm tra provenance của canonical/model graph versions. Full SQL reference tại v1/v5/v10 cho cùng shape/tọa độ và max_abs_diff=0; snapshots cũ giữ nguyên khi canonical graph tiến lên. Full catalog có 7.080 movies mang `(no genres listed)`, không phải lỗi null/thiếu khóa. Setup, tests, failure record và bản nháp kết quả ở [D06](d06-normalization.md). Chưa có phép đo cost/freshness EXP-02.

D07 xác nhận LightGCN sparse forward/backward và cập nhật bằng Adam trên toy/100k/1M. Với 17.630 node của prefix 1M, peak allocated/reserved đo được là 126,26/148 MiB. Mục Report-ready Content trình bày phạm vi, setup và kết quả của lần chạy. [Hồ sơ D07](d07-lightgcn-smoke.md).

Phép kiểm tra equivalence Full/Incremental mới bao phủ toy/100k. Model smoke dùng bốn bước optimizer. Temporal leakage, Top-K invariance và bộ nhớ xếp hạng còn cần kiểm chứng. [Nhật ký rủi ro](06-decisions-and-risks.md).

**Hình/bảng chờ bằng chứng:** sơ đồ pipeline từ [03, mục 1](03-system-design.md) có thể dựng với caption “Thiết kế luồng dữ liệu và các lớp version (chưa triển khai)”. Biểu đồ thời gian Full/Incremental và cost/staleness chỉ dựng từ CSV EXP-01/02 sau khi chạy; caption phải nêu dataset, cấu hình và run ID. Danh sách hình dự kiến đầy đủ ở [08 — Dàn ý, mục 4](08-final-report-outline.md); nơi lưu ở [report-assets](report-assets/README.md).

## 7. Report-ready Content (bản nháp cần kiểm tra trước khi nộp)

> Đồ án lựa chọn MovieLens 32M làm nguồn dữ liệu cho nghiên cứu cập nhật đồ thị tương tác user–movie theo batch. Dữ liệu rating lịch sử đã được profile và chuẩn hóa thành canonical prefix 100.000/1.000.000 sự kiện có thứ tự xác định trong cấu hình hiện tại. Full rebuild reference và transactional incremental updater đã được đối chiếu trên toy và prefix 100.000 với bốn cách chia batch mỗi dataset. Cả 35 lần đối chiếu registry, edge và degree tại boundary đều đạt; 70 lượt replay giữ nguyên state. Biểu diễn đồ thị đã chuẩn hóa được tách khỏi trạng thái canonical nhằm khảo sát chi phí khi trì hoãn materialization. LightGCN được đặt ở tầng sử dụng snapshot, còn tên phim và thể loại chỉ được bổ sung sau bước xếp hạng. Chưa có kết quả thực nghiệm để khẳng định hiệu năng cập nhật hoặc chất lượng gợi ý.

Đoạn kết quả normalization/snapshot và metadata validation được viết riêng tại [D06, Report-ready Content](d06-normalization.md) để tránh sao chép lại bảng số liệu.

**Cần Khoa xác nhận trước báo cáo cuối:** hạn nộp và mẫu báo cáo của học phần; rule `>=4.0` có phù hợp yêu cầu giảng viên hay không; cấu hình/evaluation protocol khi triển khai. Xem [04](04-project-plan.md), [06, DEC-005](06-decisions-and-risks.md).

## Cách cập nhật sau mỗi milestone

Sau khi code và test của milestone hoàn tất: cập nhật [05](05-work-log.md) và [07](07-experiment-log.md) trước; thêm claim và đường dẫn bằng chứng vào [09](09-report-evidence.md); cuối cùng bổ sung mục tương ứng ở đây bằng văn phong báo cáo. Nếu một mục chưa có run, giữ nhãn **chưa thực nghiệm**. Tránh sao chép raw log hoặc ADR sang file này.

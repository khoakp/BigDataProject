# 08 — Ghi chú tích lũy cho báo cáo học phần

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](../../current/scope-and-decisions.md) · [Tiến độ v2](../../current/progress-and-evidence.md).

**Định hướng mới S01 (21/11/2026):** tên đề tài và hai trụ cột P0 theo [bản v1.0](../plans/10-direction-update-v1.md). Phần Results bên dưới gồm lịch sử D01–D13 và [REC-01 S03–S04](../../research/s04-rec01-results.md) đã chạy trên Kaggle; [Streamlit S09–S10](../milestones/sxx/s10-rating-graph-refresh.md) có rating và graph demo nhưng checkpoint chưa refresh. [Giao thức S01](../../research/s01-evaluation-spec.md) đã được duyệt trước run.

**S12:** Nội dung đã được biên tập thành [báo cáo kỹ thuật](../../final-report.md) với links và số liệu từ evidence matrix. [Freeze candidate](../../s12-reproducibility.md) ghi môi trường/hash; bìa và định dạng phải điền theo yêu cầu học phần khi Project Owner cung cấp.

> Cập nhật: 21/11/2026. Đây là bản nháp học thuật theo **trạng thái bằng chứng**, không phải báo cáo kết quả cuối cùng. `08-final-report-outline.md` vẫn là dàn ý; file này chứa nội dung có thể phát triển thành các chương. Mọi nhận định thực nghiệm phải đối chiếu [09 — Evidence](../../research/evidence-matrix.md) và [07 — Experiment Log](../../research/experiment-log.md).

## Quy ước

- **Đã xác minh:** đã kiểm tra trực tiếp tài liệu, file hoặc phép kiểm tra được ghi ở [09](../../research/evidence-matrix.md).
- **Thiết kế đã chọn:** quyết định/phương pháp trong [03](03-system-design.md) và [06](06-decisions-and-risks.md), chưa đồng nghĩa đã triển khai.
- **Dự kiến/giả thuyết:** cần code, test hoặc số đo trước khi dùng như kết luận.
- Không lấy các mốc D01–D21 trong [04](../plans/04-project-plan.md) làm ngày hoàn thành; đối chiếu những milestone thực sự đã ghi ở [05](05-work-log.md).

## 1. Research Objective

Đề tài nghiên cứu vòng đời cập nhật đồ thị tương tác user–movie từ dữ liệu rating lịch sử: chuẩn hóa thứ tự sự kiện, cập nhật trạng thái đồ thị theo batch, tạo biểu diễn đồ thị cho LightGCN, rồi đánh giá độ đúng và chi phí. Câu hỏi chính là liệu incremental update có tạo cùng edge set, node mapping và degree như full rebuild tại cùng cutoff, và chi phí của hai cách biến đổi theo quy mô ra sao. D12 đã có kết quả EXP-01; D13 đã đo trade-off cost/staleness EXP-02 ở prefix 100k. [D12](../../research/d12-exp01-results.md), [D13](../../research/d13-exp02-results.md).

D01–D07 đã hoàn thành tính đến 20/11/2026. G1 đạt trên toy/100k. G2 đạt sau kiểm chứng normalization D06 và CUDA smoke trên toy/100k/1M. [D07](../milestones/dxx/d07-lightgcn-smoke.md).

## 2. Theoretical Background

Đồ thị được thiết kế là bipartite: cạnh nối user với movie. Theo DEC-005 đã chốt cho D03, mỗi cặp có ít nhất một rating đạt ngưỡng `rating >= 4.0` sẽ thành một cạnh binary. Với edge set `E`, bậc của user `u` và movie `i` là `d_u = |{i : (u,i) ∈ E}|`, `d_i = |{u : (u,i) ∈ E}|`. Vì mỗi cạnh đóng góp một lần vào mỗi phía, `Σ_u d_u = |E| = Σ_i d_i` là invariant để kiểm thử.

Thiết kế normalization dùng trọng số `w_ui = 1 / sqrt(d_u d_i)` cho cạnh `(u,i)` ở snapshot. Khi thêm cạnh làm đổi bậc, trọng số của các cạnh cũ kề node đó cũng đổi. Do đó append cạnh vào *normalized graph* không đủ để bảo đảm snapshot mới đúng. Đây là hệ quả toán học của công thức đã ghi ở [03, mục 6](03-system-design.md); D06 đã kiểm chứng implementation với dense toy và SQL reference tại ba cutoff [D06](../milestones/dxx/d06-normalization.md).

**Nguồn học thuật cần đối chiếu khi viết báo cáo chính thức:** danh mục [02, mục E](02-research-journal.md) gồm paper LightGCN, BPR và tài liệu GroupLens. Trong lần tổng hợp này, chỉ `ml-32m/README.txt` và `checksums.txt` cục bộ được đọc trực tiếp; không ghi một nhận định chi tiết nào như đã kiểm tra trực tiếp paper.

## 3. Methodology

Phương pháp **đã chọn ở mức thiết kế**: đọc `ratings.csv`, kiểm tra schema, sort xác định theo `timestamp, userId, movieId, source_row_id`, tạo canonical events, rồi replay theo batch. Full rebuild là reference tại cutoff; incremental path cập nhật registry, edge set và degree bằng những cạnh thực sự mới. Replay cùng batch phải idempotent. Chỉ so sánh thời gian khi kết quả state tương đương. Xem [03, mục 2–5](03-system-design.md), [06, DEC-003/004/009](06-decisions-and-risks.md).

`canonical_version` và `model_graph_version` được tách để biểu diễn normalization bị trì hoãn; khi materialize ở cùng version, ma trận phải khớp reference trong tolerance được chốt trước test. `movies.csv` chỉ là lookup sau Top-K ranking, nhằm gắn `title` và `genres`; không tham gia graph hoặc input LightGCN ở MVP. D09 đã kiểm chứng bất biến ranking/metrics trên 100k/1M và toy missing metadata. [D09](../milestones/dxx/d09-end-to-end.md).

EXP-01 D11 pilot đã được D12 lặp lại năm measured repetitions/cặp trên sáu cặp history/delta, rồi thử H=2M riêng. Mọi lượt so exact logical output đều PASS. EXP-02 D13 dùng chung dãy 10 batch × 10k, so `R=1` và `R=5` với ba measured repetitions và equality tại v5/v10. EXP-03 về model refresh là phần mở rộng. Protocol và raw logs ở [D12](../../research/d12-exp01-results.md), [D13](../../research/d13-exp02-results.md).

## 4. Implementation

Repo có code và tests cho canonical prefix, Full Rebuild, incremental updates, exact equivalence, normalization và metadata validation. D07 thêm LightGCN propagation, loss BPR và GPU probe. D08 bổ sung temporal cutoff và raw-ID Top-K; D09 nối D04 versioned graph đến LightGCN và metadata enrichment. D10 chạy lại từ raw `ratings.csv` trong thư mục mới, ghi manifests/checkpoint/model IDs và có CLI demo kiểm hashes. [Thiết kế](03-system-design.md), [D09](../milestones/dxx/d09-end-to-end.md), [D10](../milestones/dxx/d10-mvp.md).

Đầu vào cục bộ đã kiểm tra: `../ml-32m/ratings.csv` và `../ml-32m/movies.csv`, với header lần lượt là `userId,movieId,rating,timestamp` và `movieId,title,genres`. Hai MD5 khớp manifest đi kèm. D02 đã profile toàn bộ `ratings.csv`, D03 profile phân phối rating và xây Full Rebuild reference; D06 validate toàn bộ `movies.csv` với 87.585 keys unique/non-null. D09 xác nhận Top-K metadata coverage 100% trên các prefix 100k/1M đã chạy; chưa đánh giá độ nhạy chất lượng model theo ngưỡng positive. [D09](../milestones/dxx/d09-end-to-end.md).

## 5. Experimental Setup

MovieLens 32M đã qua kiểm tra nguồn và profiling. D07 dùng chronological prefix 100k và 1M, seed 7, COO float32, embedding 64 chiều và 3 lớp trên RTX 4050 với PyTorch CUDA 12.6. D08 dùng cutoffs theo distinct timestamps và 100 bước BPR; đánh giá Recall@10/NDCG@10 trên warm cohort. [Môi trường](../../research/environment.md), [D07](../milestones/dxx/d07-lightgcn-smoke.md), [D08](../milestones/dxx/d08-temporal-baseline.md).

## 6. Results and Discussion

Đã có model evaluation D08/D09 trên warm cohort 100k/1M và clean MVP rerun D10 từ raw CSV trên prefix 100k; G3 đạt trong phạm vi tối thiểu. D12 EXP-01 có 72/72 rows core và 24/24 rows H=2M exact correctness PASS. Ở Δ=1k, Incremental chỉ vượt Full tại điểm H=2M (paired ratio median 1,156, 5/5 cặp); Δ=10k gần hòa tại H=2M. Không suy ra ngưỡng crossover chính xác hay lợi ích trên 32M. D13 EXP-02 có 80/80 rows và 8/8 common-boundary checks PASS; median tổng apply+materialize+export 5,283 s (R=1) vs 1,648 s (R=5), đổi lại R=5 cũ tối đa 4 batches/40k events và dùng v0 rỗng trước lần refresh đầu. Không suy rộng chi phí sang scale lớn hơn hoặc chất lượng gợi ý. D08/D09 metric chỉ trên warm cohort; metadata enrichment giữ ranking/metrics. [D10](../milestones/dxx/d10-mvp.md), [D12](../../research/d12-exp01-results.md), [D13](../../research/d13-exp02-results.md).

REC-01 S03–S04 đã huấn luyện LightGCN trên Kaggle Tesla T4 với canonical prefix 1M. Theo validation NDCG@20, chọn checkpoint bước 300 trong ba mốc 100/300/500 rồi reload trước khi đánh giá test. Trên cùng 403 warm users và 773 train candidate movies, LightGCN đạt test Recall@20 0,128000 và NDCG@20 0,085161; Popularity đạt 0,128099 và 0,086897. Popularity nhỉnh hơn trong run này. Vì warm cohort nhỏ, nhiều future positives bị loại do cold user/movie và chỉ dùng một seed, chưa thể kết luận mô hình nào ưu việt tổng quát. [S04: bảng số liệu, protocol và raw evidence](../../research/s04-rec01-results.md).

S06 đã đối chiếu lại raw EXP-01 D12 trước khi tái dùng cho phần Engineering: 96 rows, 48 cặp Full/Incremental, Parquet SHA-256 và bảng summary đều PASS. Vì Full xử lý H+Δ events còn Incremental chỉ apply Δ trên history đã dựng, hai trường `events_per_s` không cùng mẫu số; paired wall ratio mới là phép so latency trong output contract đã định. Prebuild history nằm ngoài timer, cache không kiểm soát và sampled RSS không phải peak tuyệt đối. [S06 audit](../../research/s06-exp01-audit.md).

S07 [tổng hợp Engineering](../../research/s07-exp01-engineering-conclusions.md) đã tái sinh bảng summary và năm PNG từ raw D12, rồi đối chiếu byte với report assets. Core H≤1M không có cặp Incremental thắng; run H=2M riêng tại Δ=1k thắng 5/5, paired ratio median 1,156; Δ=10k gần hòa. Đây là một crossover quan sát giữa hai điểm đo 1M/2M trong cùng loại workload, không xác định ngưỡng hay chứng minh lợi ích ở toàn bộ 32M.

S09 [Streamlit MVP](../milestones/sxx/s09-streamlit-mvp.md) đã phục vụ checkpoint REC-01 được chọn trên local: chọn warm user hoặc cold fallback, Top-20 có title/genres, lịch sử rating train, catalog 773 phim và version state. Browser/AppTest kiểm được hai luồng; lịch sử chỉ query tới train cutoff và canonical source phải khớp SHA-256 của training report. Giao diện chưa ghi rating hoặc cập nhật graph, vì vậy không được mô tả là có model refresh.

S10 [rating và graph demo](../milestones/sxx/s10-rating-graph-refresh.md) đã mở rộng app bằng batch feedback trên bản sao train graph. Intent JSON, Parquet và ledger D04 khiến retry cùng UUID idempotent trong phạm vi demo một tiến trình. Một batch thực nghiệm tích hợp gồm rating 5,0 và 2,5 làm graph 81→82, thêm đúng một positive edge; checkpoint và Top-20 không đổi. Đây là bằng chứng về correctness của cập nhật graph và minh bạch version, không phải bằng chứng model freshness hay quality tăng sau feedback. Cần REC-02 riêng nếu đánh giá propagation-only/trained refresh.

S11 [integration 1M+20](../../research/s11-integration-scaling.md) mở rộng kiểm chứng bằng Full Rebuild độc lập trên 802.988 train events cộng 20 feedback events. Registry, edge state và degree state khớp từng dòng ở 14.388/392.636/14.388 rows. Batch chỉ thêm 10 cạnh positive, graph version tăng một, snapshot/checkpoint và Top-20 không đổi. Failure tests kiểm recovery sau intent, nội dung bị sửa và hai threads trong một tiến trình. Một lượt timing chỉ phục vụ chẩn đoán; không có đủ repetitions/kiểm soát cache để kết luận performance. Chưa kiểm tra nhiều server hoặc quality sau feedback.

D02 đã ghi nhận `ratings.csv` có 32,000,204 dòng; 84,432 movie IDs xuất hiện trong ratings; không có null trong bốn cột và không có cặp user–movie trùng. Có 6,023,681 dòng vượt số timestamp phân biệt và 15,647,103 lần timestamp giảm theo source order. Vì vậy explicit sort và tie-break theo `source_row_id` là cần thiết để tạo replay xác định. Đây là kết quả **profile dữ liệu**, không phải kết quả hiệu năng; bảng đầy đủ và điều kiện đo ở [d02-data-profile](../milestones/dxx/d02-data-profile.md).

D03 đã chốt `rating >= 4.0` theo DEC-005 sau khi profile phân phối rating, tạo full rebuild reference trên toy và canonical prefix 100k. Prefix tạo 57.925 cạnh và 2.819 node; tổng bậc user và movie đều bằng 57.925, không phát hiện sai lệch khi đếm lại từ output. Đây là baseline correctness cho D04–D05, **không phải** bằng chứng Incremental đã đúng hoặc nhanh hơn. Chi tiết phương pháp, kết quả và đoạn báo cáo nháp ở [D03](../milestones/dxx/d03-full-reference.md).

D04 đã kiểm chứng idempotency, hash conflict rejection, degree delta và rollback; integration probe trên 100k với 10 batches đạt. Phương pháp, input/output, setup, kết quả và bản nháp học thuật ở [d04-incremental.md](../milestones/dxx/d04-incremental.md). Batch ledger lưu hash và watermark cùng transaction với graph để tránh replay tăng degree.

D05 bổ sung phép so sánh chính xác từng dòng registry, edge và degree tại cùng cutoff, trên toy và prefix 100k qua 4 cách chia batch mỗi dataset. Cả 35 boundary checks và 70 replays đều đạt, hoàn thành G1 trong phạm vi này. Đã sửa lỗi Full Rebuild trên prefix không có positive edges và chạy regression test. Phần methodology, setup, results, limitations và đoạn báo cáo nháp ở [D05](../milestones/dxx/d05-equivalence.md); không diễn giải các kết quả này thành lợi ích hiệu năng.

D06 đã materialize sparse COO đối xứng và kiểm tra provenance của canonical/model graph versions. Full SQL reference tại v1/v5/v10 cho cùng shape/tọa độ và max_abs_diff=0; snapshots cũ giữ nguyên khi canonical graph tiến lên. Full catalog có 7.080 movies mang `(no genres listed)`, không phải lỗi null/thiếu khóa. Setup, tests, failure record và bản nháp kết quả ở [D06](../milestones/dxx/d06-normalization.md). Chưa có phép đo cost/freshness EXP-02.

D07 xác nhận LightGCN sparse forward/backward và cập nhật bằng Adam trên toy/100k/1M. Với 17.630 node của prefix 1M, peak allocated/reserved đo được là 126,26/148 MiB. Mục Report-ready Content trình bày phạm vi, setup và kết quả của lần chạy. [Hồ sơ D07](../milestones/dxx/d07-lightgcn-smoke.md).

Phép kiểm tra Full/Incremental đã bao phủ train prefix 1M tại cutoff D09. Temporal leakage và Top-K invariance có tests; GPU evaluation peak memory chưa đo riêng. Warm cohort còn nhỏ; [D09](../milestones/dxx/d09-end-to-end.md), [nhật ký rủi ro](06-decisions-and-risks.md).

**Hình/bảng:** EXP-01 latency, paired ratio, RSS, output bytes và crossover đã sinh từ CSV D12; caption và nguồn ở [report-assets](../../report-assets/README.md). EXP-02 cost/staleness chỉ dựng sau khi có raw CSV; không dùng số giả. Danh sách hình dự kiến ở [08 — Dàn ý, mục 4](../plans/08-final-report-outline.md).

## 7. Report-ready Content (bản nháp cần kiểm tra trước khi nộp)

> Đồ án lựa chọn MovieLens 32M làm nguồn dữ liệu cho nghiên cứu cập nhật đồ thị tương tác user–movie theo batch. Dữ liệu rating lịch sử đã được profile và chuẩn hóa thành canonical prefix 100.000/1.000.000 sự kiện có thứ tự xác định trong cấu hình hiện tại. Full rebuild reference và transactional incremental updater đã được đối chiếu trên toy và prefix 100.000 với bốn cách chia batch mỗi dataset. Cả 35 lần đối chiếu registry, edge và degree tại boundary đều đạt; 70 lượt replay giữ nguyên state. Biểu diễn đồ thị đã chuẩn hóa được tách khỏi trạng thái canonical nhằm khảo sát chi phí khi trì hoãn materialization. LightGCN được đặt ở tầng sử dụng snapshot, còn tên phim và thể loại chỉ được bổ sung sau bước xếp hạng. Kết quả EXP-01 và REC-01 được trình bày riêng theo phạm vi thí nghiệm của từng run.

Đoạn kết quả normalization/snapshot và metadata validation được viết riêng tại [D06, Report-ready Content](../milestones/dxx/d06-normalization.md) để tránh sao chép lại bảng số liệu.

**Cần Khoa xác nhận trước báo cáo cuối:** hạn nộp và mẫu báo cáo của học phần; rule `>=4.0` có phù hợp yêu cầu giảng viên hay không. REC-01 protocol S01 đã được duyệt. Xem [04](../plans/04-project-plan.md), [06, DEC-005](06-decisions-and-risks.md).

## Cách cập nhật sau mỗi milestone

Sau khi code và test của milestone hoàn tất: cập nhật [05](05-work-log.md) và [07](../../research/experiment-log.md) trước; thêm claim và đường dẫn bằng chứng vào [09](../../research/evidence-matrix.md); cuối cùng bổ sung mục tương ứng ở đây bằng văn phong báo cáo. Nếu một mục chưa có run, giữ nhãn **chưa thực nghiệm**. Tránh sao chép raw log hoặc ADR sang file này.

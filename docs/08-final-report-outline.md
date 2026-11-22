# 08 — Khung báo cáo cuối kỳ, bằng chứng, tài liệu tham khảo và bảo vệ

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](current/scope-and-decisions.md) · [Tiến độ v2](current/progress-and-evidence.md).

> **Cập nhật phạm vi 21/11/2026:** [Project Direction v1.0](10-direction-update-v1.md) và [sprint S01–S14](11-sprint14-plan.md) thay thế trọng tâm DE-only của dàn ý cũ. Báo cáo cuối phải có hai trụ cột DE và Graph-based Deep Recommender, REC-01 vs Popularity Top-20, cùng Streamlit demo; D13 EXP-02 là kết quả mở rộng. Nội dung bên dưới giữ làm cấu trúc lịch sử để biên tập, chưa phải bản cuối.

> **Trạng thái:** đề cương cập nhật 10/11/2026; chưa điền kết quả. Cấu trúc phục vụ báo cáo môn Nhập môn Big Data, nhấn mạnh Data Engineering.  
> **Nguyên tắc:** viết từ câu hỏi → quyết định kiến trúc → kiểm thử đúng đắn → thí nghiệm → kết luận; không xây dàn ý bằng danh sách công nghệ.

## 1. Metadata báo cáo

- Trường/Khoa: `[điền]`
- Tên học phần: Introduction to Big Data
- Tên đề tài: theo [01 — Overview](01-project-overview.md)
- Giảng viên hướng dẫn: `[điền]`
- Sinh viên: `[điền]`
- Deadline, định dạng và số trang: `[đợi yêu cầu môn học]`
- Git SHA/release tag kèm báo cáo: `[đợi D16 feature freeze]`
- Scope: MovieLens 32M chính; `movies.csv` metadata enrichment sau ranking; RetailRocket để future work.

## 2. Dàn ý báo cáo chi tiết

### Tóm tắt / Abstract (0.5–1 trang)

- Context tương tác user–item tăng theo thời gian.
- Vấn đề repeated full graph rebuild và semantics LightGCN normalization.
- Pipeline đề xuất, EXP-01 chính, EXP-02 ưu tiên và EXP-03 nếu thực hiện.
- **Chưa ghi phần kết quả** cho tới khi có raw logs thực tế tương ứng; không cần đủ ba thí nghiệm nếu đã ghi quyết định cắt EXP-03.

### Chương 1 — Giới thiệu và phát biểu bài toán

- Graph-based deep recommendation; bối cảnh dữ liệu liên tục tăng; Alibaba/Pinterest **chỉ là ví dụ bối cảnh từ tài liệu công khai**.
- Vấn đề canonical graph vs normalized graph, embedding vs model parameters.
- RQ1/RQ2 và RQ3 nếu EXP-03 được thực hiện, giả thuyết, phạm vi còn ~21 ngày/hardware, đóng góp DE.
- **Bằng chứng:** 01 Overview + LightGCN paper.

### Chương 2 — Cơ sở lý thuyết & Related Work

- User–item bipartite graph; degree; symmetric normalization; sparse matrix.
- LightGCN propagation, BPR objective, evaluation Recall@K/NDCG@K.
- Incremental vs full recomputation, versioning, idempotency, historical replay.
- Related works: PinSage, AliGraph và nếu đọc sâu được thì incremental GCN/recommenders; không gắn công trình công nghiệp với đặc tính không được công bố.
- **Bằng chứng:** 02 Research Journal + nguồn chính thống.

### Chương 3 — Dataset và phương pháp xây dựng hệ thống

- ML32M `ratings.csv` source và license; canonical sorting by timestamp, tie-break, positive threshold.
- `movies.csv` schema (`movieId`, `title`, `genres`), LEFT JOIN post-ranking và xử lý metadata missing/genre đa nhãn; title/genres không đi vào LightGCN input.
- Logical architecture, ER/schema state; full vs incremental; NodeRegistry, EdgeState, DegreeState, manifest.
- Deferred model-ready normalized adjacency, snapshot equivalence; freshness layers.
- Temporal split/cutoff, no leakage, warm cohort, cold-start hạn chế.
- **Bằng chứng:** 03 Design + unit tests + version manifests.

### Chương 4 — Phương pháp thực nghiệm

- Cấu hình CPU/RAM/RTX 4050 6GB, phiên bản thư viện, nguồn input, hashes.
- Mô tả baseline/treatment, independent/control variables, repetitions, fairness, warmup/cache policy.
- EXP-01 benchmark full vs incremental.
- EXP-02 normalization frequency vs cost/staleness.
- EXP-03 model freshness vs cost/quality **chỉ khi đủ thời gian**; không trình bày kết quả nếu chưa đo.
- AN-01: metadata enrichment coverage và descriptive genre analysis (không phải recommender improvement experiment).
- **Bằng chứng:** 07 Experiment Log + `results/raw/*.csv` + config.

### Chương 5 — Kết quả và phân tích

- Trước tiên: bảng correctness, invariant tests, equivalence pass.
- Các biểu đồ EXP-01, EXP-02 (EXP-03 nếu có); phân tích cả kết quả không cải thiện.
- Top-K movie recommendation có title/genres, metadata coverage, phân phối genre hậu kiểm; phân biệt outputs mô tả với Recall/NDCG đo thật.
- Trả lời từng RQ bằng bằng chứng, nêu difference across scales.
- **Bằng chứng:** số liệu đo, không chép benchmark từ AI reports.

### Chương 6 — Thảo luận, hạn chế và tính hợp lệ

- Internal validity: code bias, same output semantics, caching, uncontrolled system load, temporal leakage.
- External validity: one machine, ML32M historical replay, explicit feedback được nhị phân hóa, không giống real-time graph production; genre analysis hậu kiểm không chứng minh content awareness.
- Construct validity: freshness measured by versions/batches, model quality bằng offline metrics không phải conversion online.
- Resource limits: RTX4050 6GB, CPU/RAM/SSD thực tế.
- Những thử nghiệm không thực hiện được và lý do.

### Chương 7 — Kết luận và hướng phát triển

- Trả lời RQ1/RQ2 bằng số đo; RQ3 chỉ khi triển khai, nếu cắt thì nêu rõ scope và lý do.
- Đóng góp DE: state correctness, incremental lifecycle, version transparency, measured tradeoffs.
- Hướng tiếp theo không hứa sẽ làm trong 21 ngày: stream ingestion thật, cold-start, dynamic vertex embeddings, compaction ablation, distributed partitioning.

## 3. Mapping claim → evidence (cần điền link trước khi nộp)

| Claim cuối kỳ | Bằng chứng bắt buộc | Artifact / tình trạng |
|---|---|---|
| Graph incremental đúng | Equal edge sets, degree/registry, idempotency tests | `tests/` — Chưa tạo |
| Normalized graph đúng | Toy graph + full vs incremental same snapshot | `tests/` — Chưa tạo |
| Incremental hiệu quả ở điều kiện X | EXP-01 raw repeated wall/RSS/I/O | [S07 Engineering conclusions](s07-exp01-engineering-conclusions.md), [raw/figures D12](d12-exp01-results.md); chỉ quan sát thắng ổn định tại H=2M, Δ=1k trong các điểm đã đo |
| Deferred policy đánh đổi cost/staleness | EXP-02 + common-version equality | `results/raw/exp02_...` — Chưa chạy |
| Propagation-only có/không cải thiện chất lượng | EXP-03 temporal Recall/NDCG + GPU stats | `results/raw/exp03_...` — Chưa chạy |
| Model scale phù hợp GPU | D07 GPU probe (RTX 4050 hoặc Kaggle, ghi rõ từng môi trường) | `results/raw/gpu_probe.csv` — Chưa chạy |
| Metadata không thay ranking | Unique movieId, LEFT JOIN Top-K count/order/score tests và coverage | `tests/test_metadata_enrichment.py`, AN-01 — Chưa chạy |
| Thể loại gợi ý minh họa | Source `movies.csv`, genre counts có mẫu số, source hash | `results/raw/an01_genre_counts.csv` — Chưa chạy |
| Khả năng tái hiện | environment lock, commits, replay commands | README + tagged release — Chưa tạo |

## 4. Hình/bảng nên chuẩn bị

- **Hình 1:** User–item bipartite graph và cạnh mới khiến normalized old edge weight thay đổi.
- **Hình 2:** Data pipeline: raw→canonical→versioned state→normalized model graph→LightGCN.
- **Hình 3:** Five freshness layers, version tracking.
- **Biểu đồ A:** Full vs Incremental wall seconds theo history/delta.
- **Biểu đồ B:** Peak RSS và bytes read/write theo scale.
- **Biểu đồ C:** Refresh R=1/5/20 cumulative cost và staleness.
- **Biểu đồ D:** Model refresh GPU time/VRAM vs Recall@K/NDCG@K (nếu có EXP-03).
- **Biểu đồ E:** Phân phối genres của Top-K, ghi rõ đa nhãn và tỷ lệ metadata missing.
- **Bảng 1:** Hardware + versions.
- **Bảng 2:** Correctness invariants, versions, test results.
- **Bảng 3:** Controlled model training/evaluation configs.

Mọi số liệu trên biểu đồ phải sinh từ scripts và raw CSV, gắn experiment ID.

## 5. Kịch bản demo (5–8 phút, không train từ đầu tại buổi bảo vệ)

1. Hiện canonical graph version `v`, số nodes/edges/degrees/checksum.
2. Replay **một batch nhỏ đã chuẩn bị** → manifest tăng sang `v+1`, new edges count.
3. Chạy equality check vs full rebuild trên toy/1M prefix.
4. Hiện `canonical_version > model_graph_version` khi deferred.
5. Trigger model-ready graph materialization → current normalized version, equality PASS.
6. Dùng checkpoint LightGCN đã lưu, minh họa top-K `movieId → title, genres` qua metadata LEFT JOIN; propagation-only chỉ demo nếu đã triển khai EXP-03.
7. Trình bày plots kết quả thực từ EXP-01/02 (EXP-03 nếu có) + AN-01 coverage/genres; kết thúc bằng trade-off và limits.

**Fallback:** ảnh chụp CLI outputs, record video demo, các CSV/figures đã sinh sẵn; tránh demo phụ thuộc long GPU train.

## 6. 10 câu hỏi bảo vệ và ý chính trả lời

1. **Vì sao đây là Big Data nếu chạy laptop?** — Nghiên cứu growth/scale 1M→32M single-node, stateful processing, I/O, correctness; không tự nhận production distributed cluster.
2. **Sao không dùng Spark/Kafka?** — Workload/research question chưa yêu cầu distributed/real-time; thêm hạ tầng không cải thiện phép đo quan tâm nếu chưa có evidence.
3. **Graph khác relational table ở đâu?** — Canonical có thể tabular; LightGCN cần graph computational operator để propagate collaborative information.
4. **Append edge tại sao chưa đủ?** — Degree normalization thay trọng số incident historical edges.
5. **Deferred normalization có khiến graph sai không?** — Canonical luôn đúng; model-graph có version cũ được khai báo, khi refresh phải exact cho snapshot đó.
6. **Incremental lúc nào chậm hơn full?** — Fixed overhead, delta lớn, small files/cache, state lookup; tìm crossover thay vì giả định.
7. **Có model mới mà không train lại?** — Propagation-only recomputes embeddings trên graph mới với base ID params đóng băng, chỉ warm nodes, không tương đương parameter retraining.
8. **Làm sao chống future leakage?** — Temporal cutoff, train-only graph/degree, warm cohort, same future labels và candidate universe.
9. **LightGCN thua BPR-MF thì sao?** — Kết quả hợp lệ; đóng góp chính là graph data lifecycle correctness/efficiency.
10. **Genres có tham gia vào LightGCN không?** — Không ở MVP: graph/model chỉ dùng positive interactions; genres JOIN sau ranking, dùng cho hiển thị và mô tả.
11. **Có thể mở rộng tỷ cạnh như Pinterest?** — Không thể suy rộng trực tiếp; cần distributed storage/sampling/partitioning/pruning, ngoài phạm vi 21 ngày.

## 7. Tài liệu tham khảo nền tảng (kiểm tra định dạng theo chuẩn môn)

- He, X. et al. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation.* https://arxiv.org/abs/2002.02126
- Rendle, S. et al. (2009). *BPR: Bayesian Personalized Ranking from Implicit Feedback.* https://proceedings.mlr.press/r7/rendle09a.html
- GroupLens. *MovieLens 32M Dataset README.* https://files.grouplens.org/datasets/movielens/ml-32m-README.html
- Ying, R. et al. (2018). *Graph Convolutional Neural Networks for Web-Scale Recommender Systems (PinSage).* https://arxiv.org/abs/1806.01973
- Zhu, R. et al. (2019). *AliGraph: A Comprehensive Graph Neural Network Platform.* https://arxiv.org/abs/1902.08730
- PyG. LightGCN API: https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.models.LightGCN.html
- DuckDB. Parquet docs: https://duckdb.org/docs/data/parquet
- PyTorch. CUDA memory semantics: https://docs.pytorch.org/docs/stable/notes/cuda.html
- psutil. API documentation: https://psutil.readthedocs.io/

### Ba nguồn định hướng ban đầu (không thay paper gốc)

- `deep-research-report.md` — bản nghiên cứu/đề xuất ban đầu.
- `Phản Biện Đồ Án GNN.md` — phản biện Gemini (có nhiều claims mang nhãn suy luận).
- `deep-research-report (1).md` — đánh giá phần cứng độc lập sau khi biết RTX4050 6GB.

## 8. Checklist trước nộp

- [ ] Tất cả charts có raw CSV + config + Git SHA.
- [ ] Không có số đo không thực sự thực hiện.
- [ ] Correctness tests pass và được mô tả trước performance claims.
- [ ] Model eval không có temporal leakage.
- [ ] `movies.csv` chỉ dùng cho post-ranking; join cardinality và metadata missing được kiểm thử.
- [ ] AN-01 genre analysis được ghi rõ là mô tả, không nhầm với model features.
- [ ] Có phân tích negative results và threat-to-validity.
- [ ] Code license và MovieLens usage terms được tôn trọng.
- [ ] README đủ hướng dẫn chạy representative command.
- [ ] Sao lưu report, code, slides, figures, configs, checkpoints phù hợp dung lượng.

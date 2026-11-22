# D06 — Sparse normalization, model graph versions và metadata validation

> **Hồ sơ lịch sử GOAL 03.5:** giữ để truy nguồn quyết định/kết quả cũ. [Phạm vi v2 hiện hành](../../../current/scope-and-decisions.md) · [Tiến độ v2](../../../current/progress-and-evidence.md).

**D06 DONE ngày 19/11/2026.** T-05/T-06 và metadata validation đạt trong phạm vi đã kiểm tra; G2 còn chờ D07. Không có kết quả training hoặc benchmark từ milestone này.

## Research Objective và cơ sở toán học

D06 chuyển canonical graph đã kiểm chứng ở D05 thành operator thưa dành cho LightGCN, với `A_tilde = D^(-1/2) A D^(-1/2)`. Mỗi cạnh binary `(u,i)` tạo hai entries `(u,i,w)` và `(i,u,w)`, `w=1/sqrt(d_u*d_i)`; không thêm self loop. Công thức theo thiết kế [03, mục 6](../../legacy-reports/03-system-design.md); D06 kiểm chứng implementation bằng dense reference toy và SQL reference Full Rebuild.

Thêm cạnh làm đổi bậc endpoint nên có thể đổi trọng số cạnh cũ incident với endpoint đó. Khi trì hoãn normalization, canonical graph vẫn nhận batch mới, nhưng model graph giữ nguyên node map, shape, weights và watermark của version đã materialize. Không dùng degree hiện tại để sửa âm thầm snapshot cũ.

## Methodology và implementation

- [normalize.py](../../../../src/graph/normalize.py): `normalize_state()` nhận logical registry/edge/degree, validate bipartite endpoints, unique edges, contiguous IDs và degree bằng số incident edges. Output `SparseAdjacency(shape, entries)` là COO sorted, float64; graph node ID chính là model index ở D06. Bộ nhớ tỷ lệ nodes + edges; implementation không dựng dense `N×N`.
- [snapshot.py](../../../../src/graph/snapshot.py): `materialize_normalized()` ghi manifest, node map và COO trong cùng DuckDB transaction. `expected_version` phải đúng canonical version hiện tại. Snapshot cùng version không được ghi đè bằng nội dung/provenance khác. `load_snapshot()` kiểm tra hashes và dùng mapping của snapshot đó; `freshness()` công bố current canonical version riêng với model graph version.
- `export_snapshot()` xuất `nodes.parquet`, `adjacency.parquet`, và cuối cùng manifest JSON qua temporary-file rename; `load_export()` kiểm tra file hashes và logical payload hashes. Chỉ ghi vào directory mới; manifest hoàn tất là marker để đọc artifact. Export không atomic với DB; khi thất bại có thể còn directory chưa hoàn tất. Snapshot trong DB vẫn là nguồn gốc, checkpoint recovery chưa thuộc D06.
- [movie_lookup.py](../../../../src/metadata/movie_lookup.py): chỉ validate header/types/key của `movies.csv` và đếm missing title/genres. Từ chối `movieId` null/trùng/không dương; title/genres thiếu được báo cáo. `(no genres listed)` là nhãn unknown, không phải null hay feature của model. Không JOIN metadata vào normalization.

Dùng float64 để kiểm chứng CPU và tolerance **absolute `1e-12`, relative 0**; shape và tọa độ phải giống exact. Manifest ghi dtype/index_dtype, normalization owner, rule, input state hashes, watermark, versions và provenance. D06 là nơi normalize một lần. Khi nối PyTorch/PyG ở D07 phải tránh normalize lại; chưa kiểm chứng framework operator ở D06. `model_checkpoint_version=null` vì chưa có model checkpoint.

Trong **manifest artifact**, `canonical_version` là version nguồn lúc snapshot được tạo, bằng `model_graph_version`. Trong **freshness status**, `canonical_version` là trạng thái hiện tại và có thể lớn hơn `model_graph_version`. Ví dụ canonical 2/model 1 được báo stale 1 batch; model vẫn dùng shape/mapping version 1. Version difference chỉ có nghĩa batch lag trong cùng lịch replay, không dùng để so số events giữa những segmentation khác nhau.

## Kiểm thử và tái lập

```powershell
..\venv\Scripts\python.exe -m unittest discover -s tests -v
..\venv\Scripts\python.exe -m src.graph.d06_probe
```

Unit tests tại [normalization](../../../../tests/test_normalization.py), [snapshots](../../../../tests/test_snapshots.py), [metadata](../../../../tests/test_movie_validation.py). Toy reference dựng dense binary adjacency riêng rồi tính degree từ tổng dòng, đối chiếu với sparse result; không gọi lại normalizer để tạo expected values. Các test snapshot kiểm tra stale shape/node mapping khi thêm node, cập nhật trọng số cạnh cũ khi refresh, version mismatch rejection, rollback fault injection, export roundtrip, file tampering, DB reopen và graph rỗng.

Integration probe dùng canonical MovieLens prefix 100k của D02, 10 batches × 10k, materialize tại version 1/5/10. Tại ba boundary này, SQL reference độc lập tính weights từ Full Rebuild cùng cutoff. Giữa refresh boundaries, mỗi snapshot cũ phải giữ nguyên; đây là correctness probe, chưa phải EXP-02 đo cost của cadence.

## Experimental Setup và kết quả

Run đạt: `D06-20261119T163958346261Z`, bắt đầu `2026-11-19T16:39:58.346261+00:00` (23:39:58 ngày 19/11/2026 Asia/Saigon). Report tại `data/processed/d06_probe/report.json` và bản riêng trong `data/processed/d06_probe/D06-20261119T163958346261Z/`. Directory riêng chứa `state.duckdb`, exports version 1/5/10 và report; các artifacts nằm ngoài Git. Python 3.10.11, DuckDB 1.5.6 trong `../venv`, một thread; hardware theo [environment.md](../../../research/environment.md). Seed không áp dụng vì batch sequence cố định, không train/sampling.

Input 100k SHA-256 `8a01bc6f132362f44ab5a206575fb16ab7f34ce44c6bd224b418ea076c7bf519`; threshold 4.0. Git HEAD `735fe20c6b8f1a508b1b2a4e40aa57400ae4230b` là HEAD trước thay đổi, **không phải commit chứa D06**. Report ghi working tree chưa commit và hashes của source files đã chạy.

Trường `threads=1` trong report mô tả canonical/state connection và connection cắt prefix. Connection riêng của SQL reference và metadata validator dùng DuckDB defaults, chưa pin riêng thread count; không sử dụng run này để so sánh performance. Tolerance và so sánh shape/coordinates/weights không dựa vào timing.

| Model graph version / cutoff events | Shape | Positive edges | COO entries (nnz) | Max absolute difference với Full SQL reference |
|---|---|---:|---:|---:|
| 1 / 10.000 | 684 × 684 | 7.140 | 14.280 | 0 |
| 5 / 50.000 | 1.702 × 1.702 | 34.645 | 69.290 | 0 |
| 10 / 100.000 | 2.819 × 2.819 | 57.925 | 115.850 | 0 |

Cả ba lần đối chiếu đều cùng shape/coordinates exact, symmetric và không self loops; export/load roundtrip PASS. Snapshots cũ giữ nguyên khi canonical graph tiến lên và sau DB reopen. Freshness tại canonical 9 / model 5 là 4 batches / 40.000 events; sau refresh version 10 trở về 0. Đây là log correctness về staleness, không phải phép đo cost/staleness EXP-02. Final COO logical SHA-256 `b63be587c38ef4e5ecb67c871d7c902e0f74d1aa4b0dd196c8365e4bec0bcdbf`.

Unit suite: **15 tests PASS**. Dense toy reference khác cách tính với sparse implementation và chấp nhận sai số trong `1e-12`; integration trên máy này đo max difference bằng 0 ở ba boundary. Snapshot lỗi ghi được rollback và dữ liệu export bị sửa bị từ chối bởi hash check. Metadata validation được kiểm tra không làm đổi canonical graph state.

Metadata kiểm tra trực tiếp ngày 19/11/2026: **87.585 dòng**, 0 null/duplicate/invalid movie IDs, 0 missing titles, 0 missing genres; **7.080** movies có `(no genres listed)`. SHA-256 `b37ca1abc7798de741138ed252b62f69f7e37c84b8a8fab1b82d409b4c6c5cc2`. Đây là validation của catalog, chưa phải coverage Top-K hay genre analysis AN-01.

**Run thất bại và sửa lỗi:** `D06-20261119T163832513684Z` có report FAIL được giữ lại. Câu lệnh SQL reference `CREATE VIEW` với prepared parameter không được DuckDB hỗ trợ; đã chuyển sang `CREATE TEMP TABLE`, kiểm tra lại SQL reference và chạy lại integration PASS. Đường ghi snapshot được đổi từ per-row executemany sang một INSERT bằng typed arrays; không dùng thay đổi này để báo cáo speedup vì chưa có benchmark kiểm soát.

## Report-ready Content

> Biểu diễn graph cho mô hình được materialize thành sparse COO đối xứng với trọng số nghịch đảo căn bậc hai tích degree hai đầu cạnh. Snapshot lưu đồng bộ node mapping, shape, version nguồn và hash, còn canonical graph tiếp tục nhận batch mới. Trên prefix MovieLens 100.000 sự kiện, normalized snapshot tại các cutoff 10.000, 50.000 và 100.000 khớp shape, tọa độ và weights với reference từ Full Rebuild; sai số tuyệt đối lớn nhất đo được đều bằng 0 trong cấu hình float64. Metadata catalog được validate độc lập, với 87.585 khóa movie duy nhất và không null. Các kết quả này xác nhận những điều kiện correctness đã kiểm tra, chưa đánh giá chất lượng gợi ý hoặc chi phí refresh.

## Đối chiếu nghiệm thu

| Yêu cầu D06 | Bằng chứng đã chạy |
|---|---|
| Sparse symmetric normalized adjacency, T-05 toy/reference | `test_normalization.py` dense toy và integration SQL reference tại v1/v5/v10, cùng shape/coordinates, float64 atol 1e-12 |
| Tách canonical/model graph versions, T-06 | `test_snapshots.py`: stale shape/node map, reject gán sai version, refresh weights; report có freshness từng batch, snapshots cũ bất biến, export/reopen PASS |
| Metadata schema riêng, unique/non-null movieId | `test_movie_validation.py` bad header/duplicates/null/noninteger keys; full `movies.csv` report PASS; graph isolation PASS |
| Report notes, evidence, work/experiment logs | [05](../../legacy-reports/05-work-log.md), [07](../../../research/experiment-log.md), [08](../../legacy-reports/08-report-notes.md), [09](../../../research/evidence-matrix.md), sơ đồ [D06](../../../report-assets/d06-version-flow.mmd) |

## Giới hạn

Chưa có LightGCN/GPU smoke, float32 conversion test, checkpoint/model-index migration, inference Top-K, temporal evaluation hoặc performance benchmark. Tính tương đương normalized chỉ được kết luận cho những input/version đã kiểm tra. G2 còn cần D07 GPU/LightGCN smoke và log tài nguyên.

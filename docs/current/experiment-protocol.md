# Protocol nghiên cứu v2

Input là MovieLens 32M, positive `rating >= 4.0`, cumulative graph; base `< 1577836800`, replay `>= 1577836800`. Giữ `source_row_id` để break timestamp ties, không cho future labels hoặc candidate catalog lọt vào train graph. Cùng event watermark mới được so Full Rebuild với Incremental.

RQ1 so exact registry, edge rows, degrees trước khi so chi phí. EXP-01 đo Full Rebuild và Incremental theo output contract giống nhau; validation chạy **ngoài** timer. History prebuild, cache policy, environment, event counts và bytes phải được ghi để tránh diễn giải sai throughput. Snapshot cadence đo tại common versions, so normalized COO trong tolerance. Các số đo lịch sử và raw CSV ở [experiment log](../research/experiment-log.md) và [research index](../research/README.md), không phải kết quả model freshness v2.

RQ2 trong GOAL 04 cần fixed temporal train/validation/test, một cohort và train catalog chung, checkpoint selection chỉ từ validation, test sau khi đã khóa lựa chọn. Ghi riêng canonical version `v`, model graph `g`, checkpoint `c`, propagated embeddings `p`, cutoff, seed, candidates, coverage, hardware, Git commit, source/config hashes và status. Cold/omitted users/movies phải được tính rõ; không suy chất lượng từ resource probe.

REC-01 1M là protocol lịch sử có [kết quả](../research/s04-rec01-results.md) và [sealed bundle](../s12-reproducibility.md). Không thay cutoffs, metrics hoặc checkpoint đã chọn để ép khớp protocol v2. Raw [experiment log](../research/experiment-log.md) giữ các measured observations nguyên bản.

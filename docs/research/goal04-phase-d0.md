# GOAL 04 Phase D0 — readiness và kế hoạch, chưa chạy Phase D

Ngày 24/11/2026. Branch `develop/v2`, HEAD `28f56499d5f386dfe1e1dae8f5f2ce51f573db7e`, worktree ban đầu sạch. D0 chỉ kiểm reproducibility/artifacts, tổ chức reporting và audit thiết kế. Không sửa source, protocol, C0, raw evidence hay REC-01; không chạy GPU, 2023 holdout, D1–D5, commit/push/merge.

## 1. Repository readiness assessment

`VERIFIED` nghĩa D0 chạy lại trên file hiện có; `RECORDED_PASS` là report lịch sử được đọc nhưng phép đó chưa chạy lại; `MISSING` là không có; `BLOCKED` là không thể xác minh đầy đủ ở môi trường hiện tại.

| Contract / artifact | Trạng thái D0 và bằng chứng | Giới hạn/blocker |
| --- | --- | --- |
| C0 selection step 1500, manifest `PHASE_C_COMPLETE`, checkpoint SHA `2bf073f3…e66` | **VERIFIED**: `finalize_goal04_c0 --verify` tái kiểm C3/C4/C5 chain, macro metrics, source/mapping và strict reload trên file C4 thật | Local checkpoint có; GitHub chỉ có manifest, không có `.pt` |
| G0 v0 manifest/mapping/edges/normalization và bốn Parquet | **VERIFIED**: chạy lại `verify_kaggle_c3` trên local C3 v3, băm từng Parquet; v0, 4.999.981 edges, mapping `99684802…c90f9`, COO `a5b430be…2b62d` | Fresh full `load_projection_snapshot` trong D0 **BLOCKED** vì RAM khả dụng ~3,1 GB; export/reload G0 trên Kaggle C3 là **RECORDED_PASS** |
| C4 v4 năm candidates/config/tensor shapes/source bundle | **VERIFIED**: `verify_kaggle_c4` strict reload cả năm checkpoints; source bundle `4884205b…a2e` | Binaries không trên GitHub; private Kaggle access cần cho checkout mới |
| C5 v2 60 records/macro/candidate hashes | **VERIFIED**: `verify_kaggle_c5` và C6 re-audit; bundle `c1ed73e4…899a` | GPU dual-device gate là **RECORDED_PASS** từ C5 run, D0 không chạy GPU |
| Cohort/full canonical | Local canonical/cohort tồn tại; C3/C4 input reports và hashes nối nguồn; **RECORDED_PASS** cho full source provenance | D0 không băm lại full canonical 482 MB; GitHub không chứa input; private datasets đã duyệt |
| Fresh GitHub checkout tự chạy mọi verifier | **BLOCKED** | Thiếu G0 Parquet, C4 `.pt`, C5 package metadata trong ignored `data/processed/`, và canonical/cohort; xem [recovery guide](../current/artifact-recovery.md) |
| Python dependencies/tests | Local `..\venv\Scripts\python.exe` có Torch, DuckDB, psutil và project modules; targeted/full regression được chạy trong D0 | Kaggle CUDA/Python 3.13 không được tái tạo local; GPU performance chỉ từ report gốc |

Local environment đo trong D0: Python 3.10.11, PyTorch 2.8.0+cu126, DuckDB 1.5.6, psutil 7.2.2, NumPy 2.2.6, pandas 2.3.3, pyarrow 25.0.1. Kaggle C5 đã dùng Python 3.13.16/PyTorch 2.11.0+cu130, vì vậy local verification không thay thế GPU numerical gate lịch sử.

`selection.json` lưu original C4 relative path, SHA, config, G0/source/cohort/Git/bundle hashes. D0 không dựng checkpoint giả từ manifest. [Recovery guide](../current/artifact-recovery.md) ghi đúng versions 3/4/2, file tree, commands và hành động khi thiếu private access. Không cần download: local đủ binaries cho byte verification và strict C0 reload. File trùng cần xem xét về sau: nhiều regression logs và Kaggle pulled-source copies trong `data/processed/`; không xóa/move vì chúng là evidence hoặc ignored local state. [Reporting workspace](../reporting-v2/claim-evidence-map.md) chỉ nối nguồn, không thay research index.

## 2. Phase D architecture readiness

Đề xuất data flow: C0 checkpoint immutable + fixed mapping + G0 → **P0** (sau D1/D2, chưa tạo); ở boundary `b`, replay mọi canonical events `<b`, freeze fixed-cohort projection/snapshot → S0 rank bằng P0, S1 một propagation trên snapshot mới với C0 cố định, S2 đếm unique positives đã quan sát trên cùng projection. Cả ba dùng cùng fixed 40.666 movie catalog, eligible-user/label/mask bundle cho offline quality và cùng pre-boundary user workload cho serving-like cost. Labels `[b,e)` chỉ vào evaluator sau khi ranking. Mọi strategy giữ deterministic score-desc/raw-ID-asc Top-20; S0 không forward lại, S1 không optimizer.

Reuse `ModelGraphReplay.create/advance/current/export_snapshot`, `load_projection_snapshot`, `normalize`, `C0Config/load_candidate/load_g0`, `LightGCN.forward`, `score_batched`, `evaluate_topk`, `popularity_topk`, `benchmarking.common.RssSampler` và protocol window/coverage code. Cần bổ sung sau approval: một coordinator đóng băng boundary bundle và version IDs; P0 cache manifest/hash; S0/S1/S2 adapters dùng chung ranking API; serving-like user selector từ pre-boundary 90-day positives; timer/result schema cho 3 repetitions và paired metrics. Không xây lại training framework.

Cache/version contract: key P0 = `(C0 SHA,G0 snapshot SHA,mapping SHA,dim,layers,dtype,device)`; S1 cache key thêm projection version/boundary/snapshot hash. Reject stale hoặc mismatch, kiểm C0 parameter hash trước/sau; no future events/labels trong graph, S2 hoặc user selector. `v` canonical watermark, projection version, snapshot `g`, checkpoint `c` và propagated `p` không hoán đổi. Toy gate phải kiểm `<b` kể cả equal timestamp ties, idempotent replay, edge/degrees/normalization, fixed mapping, identical masks/catalog/users, S0 0 forward, S1 1 forward/0 optimizer, S2 observed-only, exact tie order và tampered-cache rejection.

Performance hotspots: `_attach_source()` băm lại full canonical file mỗi `advance()`/export, `_check_state()` gọi `state()` và hash toàn registry/edges/degrees, `_record()` làm tương tự sau update; `state()` materialize 5M edges vào Python. Mỗi boundary còn DuckDB scan/classification, snapshot export/normalization, sparse reload, GPU transfer, propagation, full-catalog ranking và artifact writes. Đo các pha riêng với CPU RSS, DuckDB memory/spill, disk, GPU allocated/reserved, wall/end-to-end; cache policy rõ ràng và không bỏ integrity check để tăng tốc. C3 chỉ đo một boundary; **không** chứng minh 33 boundaries × 3 repetitions nằm trong ngân sách. RAM local ~3,1 GB khả dụng là blocker cho full-G0 in-memory load. Rủi ro chính: leakage ở boundary, mask/candidate lệch strategy, hash/verification lặp đắt, OOM, GPU timeout và C0-selection bias stale-G0.

## 3. Kế hoạch D1–D5 (đề xuất, chưa được phép chạy)

| Session | Objective; input → output/files | Tests và stop condition | Model/effort; Owner gate |
| --- | --- | --- | --- |
| D1 contracts | Protocol+C0/G0 → API/version/cache/result schemas, cập nhật `docs/current/architecture.md` và design note | Review exact boundaries, fairness, timers; stop nếu scope/budget chưa chốt | Codex Astra high cho design audit; Owner duyệt contracts trước D2 |
| D2 toy adapters | D1 contracts + toy graph → S0/S1/S2 modules dùng core hiện có, no GPU | Toy S0 0 forward, S1 1 forward/0 optimizer, S2 observed-only; stop nếu parameter hash đổi | Codex Sol medium cho implementation; Owner nghiệm thu toy behavior trước D3 |
| D3 correctness | D2 toy → leakage/normalization/cache/fairness/rollback tests và verification report | Boundary ties, idempotence, tamper, equal candidate/mask/users; full regression; stop mọi mismatch | Codex Astra high cho adversarial audit; Owner nghiệm thu trước real data |
| D4 real preflight | D3 PASS + private inputs → Kaggle package/resource preflight, no quality tuning | Measure per-phase CPU/GPU/disk/spill/end-to-end; stop OOM, cap, integrity fail | Codex Sol high cho packaging; **Owner phê duyệt budget/submission cụ thể trước Kaggle run** |
| D5 development | D4 PASS + approved budget → 2021–2022 24-window S0/S1/S2 cost–quality, đúng 3 repetitions | Complete windows, paired user-cluster CI, coverage, timers; INCOMPLETE nếu thiếu, không mở 2023 | Codex Astra high cho audit/report; **Owner approval mới** trước execution |

Budget chỉ là **đề xuất tính bằng trần protocol**, không là GPU approval: 33 primary boundaries sau validation; mỗi S1 forward ≤60 s, graph/snapshot setup ≤300 s/boundary, offline quality ≤900 s/strategy/window trong repetition 1, serving-like ranking ≤300 s/strategy/window cho cả 3 repetitions, tổng ba runs ≤30 giờ end-to-end theo [protocol](../current/experiment-protocol.md). D4 phải đo thực tế rồi Owner mới quyết định workload và ngân sách D5. Không lấy 11.870,753 s còn lại của Phase C làm ngân sách Phase D.

## 4. Giải thích cho người mới

GitHub thường chứa code, báo cáo và manifest; model weights và Parquet lớn có thể giữ trong private artifact store để tránh repo phình và rò dữ liệu. **Checkpoint** là bytes tham số model; **hash** là dấu vân tay của bytes đó; **manifest** ghi file nào, hash nào, config/mapping/graph nào được phép dùng. Một số như C0 NDCG đi từ monthly JSON → macro recompute → selection manifest → claim map → bảng báo cáo → slide 10, kèm run/hash để người đọc truy ngược.

S0 dùng vector cũ P0 và chỉ đổi observed mask; S1 giữ C0 parameters nhưng truyền lại trên graph đã cập nhật; S2 xếp phim theo số positives đã quan sát. D0 chỉ chuẩn bị để ba nhánh dùng cùng input và không leakage, nên không cần GPU. Ví dụ user thích phim A trước tháng 3 thì cả ba che A; phim B positive trong tháng 3 là label, chỉ thành graph edge ở boundary tháng 4.

Ba câu tự kiểm tra: (1) Nếu `candidate_1500.pt` cùng tên nhưng hash khác, manifest còn hợp lệ không? (2) Vì sao S0 được đổi seen mask nhưng không được gọi LightGCN forward? (3) Vì sao một số validation 2020 chưa là kết quả final holdout 2023?

## 5. Final verification D0

Targeted real-artifact smoke: C3/C4/C5 verifiers **PASS**, C6 `finalize_goal04_c0 --verify` **PASS** và checkpoint C0 strict reload; đây là kiểm chứng hiện tại, không chỉ đọc JSON cũ. Targeted C6/model-replay/C5 tests **21 PASS**; full local regression **103 PASS, 0 FAIL**. Kiểm 8 tài liệu entrypoints mới/sửa: **0 broken local links**; `git diff --check` PASS. Git diff chỉ gồm README và tài liệu D0 mới/cập nhật, **không có source-code hoặc protected raw artifact changes**. Tại thời điểm kiểm D0, chưa thực hiện commit/push/merge; việc xuất bản Git theo yêu cầu Owner riêng.

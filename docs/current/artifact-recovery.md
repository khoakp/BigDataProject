# Khôi phục C0 và G0 cho GOAL 04

Trạng thái 24/11/2026: source/manifests có trên GitHub `develop/v2` tại `28f5649`; **Parquet snapshots và checkpoint `.pt` không ở GitHub**. Trên máy đang dùng, C3 v3 G0 và C4 v4 candidates đều có đủ file; D0 đã chạy lại C3/C4/C5/C6 verifiers và strict reload C0, tất cả PASS. Giữ nguyên bytes gốc, không train lại hay đổi tên checkpoint để thay thế.

## Bộ cần bảo quản

| Artifact | Nguồn và đường dẫn từ repo root | Contract |
| --- | --- | --- |
| G0 base snapshot | Private Kaggle kernel `khoakp/moviegraph-v2-c3-real-5m-preflight/3` → `results/goal04_c3/run_03/goal04_c3/base_snapshot/` | `manifest.json`, `nodes.parquet`, `edges.parquet`, `degrees.parquet`, `adjacency.parquet`; projection v0, cutoff `1577836800`, 90.253 nodes, 4.999.981 edges |
| C4 candidates | Private kernel `khoakp/moviegraph-v2-c4-single-t4-training/4` → `results/goal04_c4/run_04/goal04_c4/candidates/` | `report.json`, `steps.jsonl`, `candidate_{0,500,1500,3000,5000}.pt`; C0 tham chiếu **candidate_1500.pt** |
| C5 evidence | Private kernel `khoakp/moviegraph-v2-c5-2020-validation/2` → `results/goal04_c5/run_02/download/goal04_c5/` | `report.json`, `validation-matrix.json`, `candidate-comparison.csv`; trên GitHub có bản text |
| C0 lock | `results/goal04_training/selection.json` và `verification.json` | Chỉ là manifest/verification; không chứa model weights |
| Dataset provenance | Private Kaggle datasets `khoakp/moviegraph-v2-c3-canonical`, `khoakp/moviegraph-v2-feasibility-input` | Full canonical và approved cohort; không đưa lên GitHub |

Checkpoint C0 có SHA-256 `2bf073f328551561bf21941ee9fa7f3d239080139922de348f86426862a25e66`, 23.107.674 bytes, schema `lightgcn_v2_c0_candidate_v1`, embedding 90.253×64, 3 layers, seed 7. G0 `manifest.json` SHA-256 `0feaa2c521b0adb364fa78468a86f8a39fac979d630053fe0cb8aff39910ebdb`; mapping `996848022d92e20ef63b498cfedb38b97545fb006da4c00fef9ed35d40dc90f9`; logical edge hash `fb2c7fe7d64e018bebcd7101df41249dbfa57e1db5369fae01d023e9c2e5392a`; normalized COO hash `a5b430beadce8fd04671eb10f5d567130a1e1403b0c498a44d101e17c902b62d`. Byte hashes của bốn Parquet nằm trong [G0 manifest](../../results/goal04_c3/run_03/goal04_c3/base_snapshot/manifest.json).

## Khôi phục khi máy mới thiếu binaries

Đăng nhập tài khoản Kaggle có quyền đọc các private kernels. Từ repo root, tải output **read-only** vào thư mục mới, rồi đặt đúng cây đường dẫn trong bảng trên. Kaggle CLI có thể được gọi qua `uvx --with six --from kaggle kaggle` nếu môi trường chưa cài `kaggle`:

```powershell
uvx --with six --from kaggle kaggle kernels output khoakp/moviegraph-v2-c3-real-5m-preflight/3 -p <fresh-c3-download-dir>
uvx --with six --from kaggle kaggle kernels output khoakp/moviegraph-v2-c4-single-t4-training/4 -p <fresh-c4-download-dir>
uvx --with six --from kaggle kaggle kernels output khoakp/moviegraph-v2-c5-2020-validation/2 -p <fresh-c5-download-dir>
```

Xem cây sau download trước khi copy: CLI có thể tạo thêm tầng thư mục kernel output. Chỉ copy đúng artifacts của **version 3/4/2** vào các đích nêu trên; không thay historical files nếu đã có. C3/C4 verifiers cần `package-manifest.json` nằm tại `results/goal04_c3/run_03/` và `results/goal04_c4/run_04/`, đã có trên GitHub. C5 verifier còn cần `data/processed/goal04_c5_retry_official_v4/package-manifest.json` (package local bị Git ignore). Nếu thiếu, thử tái tạo bằng `..\venv\Scripts\python.exe -m scripts.prepare_kaggle_c5 --output data/processed/goal04_c5_retry_official_v4 --budget-audit results/goal04_c5/budget_audit/assessment-retry-scope.json`; chỉ dùng khi `source_bundle_sha256=c1ed73e464e0e210ee044721498eb04c2bc490c01ab8c2649459cde41fb889a9`, `budget_audit_sha256=f7685072f69b3f2725cd9f8968210b109e75565d009452621de9ba219eaedd73` và runner hash khớp [retry presubmit record](../../results/goal04_c5/retry-presubmit-final-v4.json). Source-byte/line-ending khác trên checkout mới có thể khiến bundle khác; khi đó đánh dấu `BLOCKED` và lấy package exact từ máy gốc, không tạo manifest giả từ tên file.

## Xác minh từ repo root

```powershell
Get-FileHash results/goal04_c4/run_04/goal04_c4/candidates/candidate_1500.pt -Algorithm SHA256
Get-FileHash results/goal04_c3/run_03/goal04_c3/base_snapshot/manifest.json -Algorithm SHA256
..\venv\Scripts\python.exe -m scripts.verify_kaggle_c3 --root results/goal04_c3/run_03/goal04_c3 --package results/goal04_c3/run_03 --output <fresh-c3-verification.json>
..\venv\Scripts\python.exe -m scripts.verify_kaggle_c4 --root results/goal04_c4/run_04/goal04_c4 --package results/goal04_c4/run_04 --output <fresh-c4-verification.json>
..\venv\Scripts\python.exe -m scripts.verify_kaggle_c5 --root results/goal04_c5/run_02/download/goal04_c5 --package data/processed/goal04_c5_retry_official_v4 --c4-root results/goal04_c4/run_04/goal04_c4 --output <fresh-c5-verification.json>
..\venv\Scripts\python.exe -m scripts.finalize_goal04_c0 --verify --output results/goal04_training/selection.json
```

`finalize_goal04_c0 --verify` kiểm lại raw matrix, original checkpoint bytes/strict reload, G0 hashes, config và source provenance; **không ghi lại selection**. `verify_kaggle_c3` băm mọi Parquet và đối chiếu record export/reload lịch sử; nó không tự nạp toàn bộ 5M graph vào RAM trong lần D0 này. Full `load_projection_snapshot` cần được chạy trên máy đủ RAM nếu Phase D yêu cầu fresh in-memory reload; tại D0 chỉ còn khoảng 3,1 GB RAM trống, nên không dùng một thử nghiệm có nguy cơ OOM để thay cho hash checks.

Nếu lỗi `FileNotFoundError`, kiểm đúng version và tầng download; nếu SHA khác, dừng và tải lại **cùng version**, không sửa manifest; nếu `Checkpoint metadata mismatch`, kiểm G0 mapping/cohort/config trước. Không upload raw MovieLens, full canonical, G0 Parquet hoặc `.pt` lên GitHub/public dataset. Private Kaggle access hoặc package C5 thiếu là blocker tái lập trên máy mới, không phải bằng chứng artifact sai.

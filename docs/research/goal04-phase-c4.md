# GOAL 04 Phase C4 — Single-T4 training candidates

Ngày 24/11/2026. **COMPLETE_CANDIDATES; dừng trước C5.** Private Kaggle [kernel version 4](https://www.kaggle.com/code/khoakp/moviegraph-v2-c4-single-t4-training?scriptVersionId=373162682) hoàn thành một trajectory liên tục 5.000 optimizer steps trên T4 `cuda:0`. Năm checkpoint đúng lịch `[0, 500, 1500, 3000, 5000]` đã tải về và [strict verifier](../../results/goal04_c4/run_04/verification.json) trả `PASS`. Chưa đánh giá validation quality và chưa chọn C0.

## Input, provenance và code

- Input: approved full canonical MovieLens 32M và manifest trong private Kaggle dataset, C3 G0 normalized snapshot của pre-2020 5M cohort: 4.999.981 edges, 90.253 nodes. Source SHA-256 `494a1e70e67a2d750297872020b3919da9e557dd68cb631f53d2c5cb9570f73e`; cohort manifest `10411c083ec8076bf831e4cbf5c8180518fcd557feaa43898142547ced77929e`; snapshot manifest `0feaa2c521b0adb364fa78468a86f8a39fac979d630053fe0cb8aff39910ebdb`; fixed mapping `996848022d92e20ef63b498cfedb38b97545fb006da4c00fef9ed35d40dc90f9`.
- Git HEAD `1745739744f6f4aae0d97ddf83d03a7a80eacbb9`; embedded source bundle SHA-256 `4884205b49063cd12a9b277678f5d121161b33506cb323ffc7dc6be1f04c81a2`. Pulled v4 code có hai notebook cell markers do Kaggle thêm; 14 source files nhúng khớp byte với package đã review. [Package manifest](../../results/goal04_c4/run_04/package-manifest.json), [inputs](../../results/goal04_c4/run_04/goal04_c4/inputs.json), [runner report](../../results/goal04_c4/run_04/goal04_c4/report.json).
- Runner `scripts/kaggle_c4_runner.py` dùng lại `load_g0`, `LightGCN`, BPR sampler/step và strict candidate checkpoint contract trong `src/models/c0.py`. Config: dim 64, 3 layers, batch 1.024, Adam LR 0,001, seed 7. Một optimizer duy nhất chạy xuyên suốt; step 0 được lưu trước cập nhật. Không dùng future/validation events để train, không dùng REC-01 checkpoint, không DDP/DataParallel.

## Training timeline và tài nguyên đo được

| Chỉ số | Giá trị |
| --- | ---: |
| Optimizer steps | 5.000 liên tục |
| Optimizer elapsed | 1.847,12 s (30 phút 47 giây), dưới trần 3.600 s |
| Trajectory wall | 1.868,26 s |
| C4 runner end-to-end | 1.968,11 s (32 phút 48 giây) |
| Step latency median / p95 / max | 0,3698 / 0,3714 / 1,0417 s |
| Peak process-tree CPU RSS | 4.348.219.392 B (4,05 GiB) |
| Peak CUDA allocated / reserved | 1.001.759.744 / 1.178.599.424 B |
| Peak output disk | 116.178.506 B (110,80 MiB) |
| Môi trường | Python 3.13.16; PyTorch 2.11.0+cu130; CUDA runtime 13.0; Kaggle T4×2 visible, train chỉ `cuda:0` |

[Full step log](../../results/goal04_c4/run_04/goal04_c4/candidates/steps.jsonl) ghi 5.000 step và loss hữu hạn. Training loss ở checkpoint là sampled BPR loss, **không** phải Recall/NDCG và không dùng để chọn C0.

| Step | Training loss | Checkpoint bytes | SHA-256 |
| ---: | ---: | ---: | --- |
| 0 | — | 23.107.653 | `514c2c072c2f9b8823dd280ef0b6a91f1414bfd14d286f3f73b0a9b9be9c991e` |
| 500 | 0,076977 | 23.107.667 | `33f7c28451635b3bc4723c114d1b98dfc63b9ac65dc2b0695af8d2840ee9a76e` |
| 1500 | 0,072429 | 23.107.674 | `2bf073f328551561bf21941ee9fa7f3d239080139922de348f86426862a25e66` |
| 3000 | 0,060620 | 23.107.674 | `9358d648e0ae7543d164bd831c0b986115940a4e5b86366c05b3c428ac06a7a2` |
| 5000 | 0,070978 | 23.107.674 | `0e7c8dec7232c4b3ff3f9bd703250ba2a4963b9ca8a79ed2132f38feb8811b11` |

Verifier đã băm từng file, strict reload cả năm checkpoint, đối chiếu config, shapes 90.253×64, mapping/graph hashes, step sequence, và xác nhận `selected_step=null`, `validation_quality_evaluations=0`. [Candidate manifest](../../results/goal04_c4/run_04/goal04_c4/candidates/report.json) và [verification](../../results/goal04_c4/run_04/verification.json) là nguồn số liệu có thể máy kiểm.

## Budget và lịch sử phiên

Report cộng C3 đã đo 404 s với C4 v4 1.968,11 s thành **2.372,11 s (39 phút 32 giây)**. Đây là **subtotal có artifact**, không phải tổng Phase C chính xác: C4 v1, v2, v3 đều bị hủy, và chưa có elapsed đáng tin cậy cho chúng. Vì thế không khẳng định còn đúng `14.400 − 2.372` giây. [V1 assessment INCOMPLETE](../../results/goal04_c4/run_01/report.json) được giữ nguyên làm bằng chứng lịch sử; v1 không có output hoặc log để suy số optimizer steps. V4 do Owner cung cấp liên kết/ảnh và đã được theo dõi read-only; agent không tự submit lại. Trước bất kỳ Kaggle job C5 nào cần đối chiếu run history với ngân sách 4 giờ.

## Tái kiểm và readiness

```powershell
..\venv\Scripts\python.exe -m scripts.verify_kaggle_c4 --root results/goal04_c4/run_04/goal04_c4 --package results/goal04_c4/run_04 --output results/goal04_c4/run_04/verification.json
..\venv\Scripts\python.exe -m unittest tests.test_kaggle_c4 -v
```

Input C5 là năm candidate `.pt`, candidate report, G0 snapshot/mapping, canonical manifest và 12 windows validation năm 2020. C4 **ready for Owner review**; C5 chưa chạy. Rủi ro chính là tổng elapsed của các canceled versions chưa kiểm toán đủ và checkpoint C0 về sau được chọn trên stale-G0 validation, nên có checkpoint-selection bias khi so S1. Phase D chưa được phép.

## Giải thích cho sinh viên mới

*Trajectory* là một chuỗi cập nhật liên tục của cùng model và optimizer. *Checkpoint* là ảnh chụp tham số tại một bước định trước. Ví dụ, step 500 và step 1500 dùng cùng trạng thái Adam liên tục; nếu reset Adam ở step 500 thì đó là đường huấn luyện khác. C5 sẽ chấm đủ năm ảnh chụp trên 12 tháng 2020 bằng Recall@20/NDCG@20 rồi mới chọn C0; loss 0,060620 ở step 3000 không đủ để kết luận nó tốt nhất.

Ba câu tự kiểm tra: (1) Vì sao bước 0 phải lưu trước optimizer update? (2) Vì sao loss thấp hơn không đảm bảo NDCG cao hơn? (3) Vì sao phải tính cả các Kaggle jobs bị hủy vào ngân sách 4 giờ?

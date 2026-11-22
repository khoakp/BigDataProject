# Bảo trì tài liệu GOAL 03.6

Ngày 22/11/2026. Branch `refactor/v2-doc-pruning` từ `refactor/v2-cleanup` commit `fb709c0`. Đây là thay đổi documentation-only; không sửa Python, thuật toán, protocol, raw results hoặc frozen manifests.

## Migration summary

| Hạng mục | Trước | Sau |
| --- | ---: | ---: |
| Markdown toàn repository | 58 | 77 |
| Markdown trực tiếp tại `docs/` | 47 | 22 |
| Tài liệu chính cần đọc | 8 | 10 |
| Nội dung chuyển vào archive | 0 trong cấu trúc mới | 31 files |
| Nội dung chuyển vào research | 1 index | 13 reports + 1 index |
| Compatibility notes cho frozen callers | 0 | 18 |
| File hợp nhất/xóa | 0 | 0 |

Tổng Markdown tăng do 18 compatibility notes bắt buộc cho links trong frozen report/report generators và file maintenance này; nội dung lịch sử thật đã rời root `docs/`. Đường đọc chính gồm README, AGENTS, 6 trang current, research index và archive index. [Archive mapping](../archive/README.md) ghi disposition theo nhóm.

`final-report.md`, `source-manifest.md`, `s12-reproducibility.md`, `s13-defense-qa.md`, `report-assets/` và `s12-freeze-manifest.json` giữ legacy path. Không đề xuất xóa file ở vòng này: các bản trùng một phần vẫn có ghi chú/provenance riêng và Project Owner chưa duyệt danh sách xóa cụ thể.

## Dependency audit

| Caller / nguồn | Contract sau migration |
| --- | --- |
| README và AGENTS | Chỉ dẫn tới current, research index và archive index |
| `final-report.md` frozen | 18 link paths cũ được giữ bằng compatibility notes, mỗi note dẫn tới nội dung đã move |
| `s12_build_report_docx.py` | Năm S01/S03/S04/S06/S11 paths vẫn tồn tại qua compatibility notes; source không sửa |
| `s14_package.py`, requirements | `s12-reproducibility.md` và `s13-defense-qa.md` giữ nguyên path; recursive package vẫn thu toàn bộ docs |
| `s12_freeze_manifest.py` | `final-report.md`, hashed PNG assets và manifest giữ nguyên path/bytes |
| `src/graph/d06_probe.py` | `source-manifest.md` giữ nguyên path; source không sửa |
| Historical reports/assets | Relative Markdown, image, source và artifact links được tính lại từ vị trí mới |

[Inventory máy đọc được](../../results/goal036-documentation-inventory.json) ghi đủ 58 baseline files với title, purpose, status, milestone, incoming/outgoing counts, source callers, frozen status và destination. Sau migration, repository có 10 entrypoints chính, 13 measured reports trong research và 31 hồ sơ trong archive. Tên DXX/SXX giữ milestone gốc; [GOAL 03.5 baseline map](../archive/legacy-reports/goal035-documentation-map.md) giữ purpose/status của 49 file trước refactor, còn archive/research indexes ghi disposition hiện tại.

## Usability review

- **New Student:** README dẫn qua đúng 5 tài liệu current để hiểu scope, architecture, protocol, trạng thái và walkthrough; không cần mở DXX/SXX.
- **Coding Agent:** AGENTS và current docs xác định source of truth, invariants, test commands và integration points; archive không nằm trên đường đọc mặc định.
- **Academic Reviewer:** research index dẫn đến dataset/provenance, methodology, measured results, limitations, frozen report và artifacts mà không đổi evidence gốc.

## Verification

- **Markdown:** kiểm 77 files, 694 local links gồm 2 image links; 0 missing file và 0 invalid local anchor. Baseline GOAL 03.5 cũng có 0 broken local links, nên migration tạo 0 lỗi mới. External URLs chưa được kiểm.
- **Evidence:** 44 files đã move giữ nguyên measured values và nội dung nghiên cứu; chỉ path references thay đổi. Cả 20 protected files trong GOAL 03.5 giữ nguyên SHA-256/size. Frozen `final-report.md`, S12 manifest và sáu hashed PNG giữ nguyên byte so với HEAD.
- **S12 manifest:** 56/71 source entries vẫn khớp manifest lịch sử. Mười lăm source/test hashes đã lệch từ GOAL 03.5 và có cùng mismatch set tại baseline `fb709c0`; GOAL 03.6 tạo 0 mismatch mới và không sửa manifest để che chênh lệch đó.
- **Repository:** Python source, scripts và tests có 0 diff. Full regression: **74 PASS, 0 FAIL, 0 SKIP trong 36,411 s**. S14 packaging selection đọc được **329 members**, gồm 77 Markdown files; hardcoded document paths đều tồn tại.
- **Ponytail Review:** 18/18 compatibility notes có caller thật; không có stub dự phòng, index trùng vai trò hoặc MUST/SHOULD FIX còn lại. Verdict: Ship.

Remaining debt: compatibility notes làm root `docs/` còn 22 Markdown files. Chỉ có thể bỏ chúng khi tạo version mới của frozen report và cập nhật các report generators theo một phê duyệt riêng. External URLs không thuộc link check local.

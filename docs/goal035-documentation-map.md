# GOAL 03.5 — Markdown inventory và migration map

**Map và đề xuất tại baseline; xem [implementation report](goal035-implementation-report.md) để biết cách xử lý thực tế.** Bao phủ 49 tracked Markdown files tại HEAD `cec72b8ddbad76f5788cc37cd0e30fce6624f282`; docs mới không thuộc mẫu baseline. [Audit và kiến trúc đề xuất](goal035-audit-plan.md).

Phân loại chỉ định vai trò, không cho phép xóa evidence. `RESEARCH EVIDENCE` có thể là kết quả lịch sử; `DUPLICATE` chỉ xác định prose báo cáo đã được biên tập, vẫn giữ notes riêng. Mỗi measured claim có nguồn gốc duy nhất; current pages liên kết tới nguồn, không ghi lại raw metrics.

| File trước thay đổi | Phân loại | Đích/hành động theo plan |
| --- | --- | --- |
| `AGENTS.md` | CURRENT | Giữ root; ngắn, bỏ mâu thuẫn cutoff/source pin theo GOAL Owner. |
| `README.md` | CURRENT | Giữ root; quickstart và links tới current authorities. |
| `docs/01-project-overview.md` | HISTORICAL | Archive/labelling; giữ context quyết định, không dùng làm tiến độ v2. Cập nhật inbound/outbound links trước khi move. |
| `docs/02-research-journal.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/03-system-design.md` | CURRENT | Trích implementation/contracts sang current/architecture.md; giữ path cũ và gắn nhãn historical để bảo toàn inbound links. |
| `docs/04-project-plan.md` | HISTORICAL | Archive/labelling; giữ context quyết định, không dùng làm tiến độ v2. Cập nhật inbound/outbound links trước khi move. |
| `docs/05-work-log.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/06-decisions-and-risks.md` | RESEARCH EVIDENCE | Giữ historical ADR log; current/scope-and-decisions.md giữ quyết định đang có hiệu lực và link ADR nguồn. |
| `docs/07-experiment-log.md` | RESEARCH EVIDENCE | Giữ raw observations/log; chuyển quy định hiện hành sang current/experiment-protocol.md, legacy protocols được label rõ. |
| `docs/08-final-report-outline.md` | HISTORICAL | Archive/labelling; giữ context quyết định, không dùng làm tiến độ v2. Cập nhật inbound/outbound links trước khi move. |
| `docs/08-report-notes.md` | DUPLICATE | Bản nháp đã được final-report biên tập; archive, giữ unique academic notes; bỏ vai trò nguồn kết quả hiện hành. Không sửa final-report. |
| `docs/09-report-evidence.md` | RESEARCH EVIDENCE | Giữ claim IDs E-xxx cho frozen report; current/progress-and-evidence.md link tới claim ledger, không copy claims. |
| `docs/10-direction-update-v1.md` | HISTORICAL | Archive/labelling; giữ context quyết định, không dùng làm tiến độ v2. Cập nhật inbound/outbound links trước khi move. |
| `docs/11-sprint14-plan.md` | HISTORICAL | Archive/labelling; giữ context quyết định, không dùng làm tiến độ v2. Cập nhật inbound/outbound links trước khi move. |
| `docs/CHANGELOG-2026-11-10.md` | HISTORICAL | Archive/labelling; giữ context quyết định, không dùng làm tiến độ v2. Cập nhật inbound/outbound links trước khi move. |
| `docs/architecture-audit-v2.md` | RESEARCH EVIDENCE | Giữ audit trước làm historical evidence; authority/migration mới là GOAL 03.5, không sửa lại các baseline cũ. |
| `docs/archive/04-project-plan-7weeks-archived.md` | HISTORICAL | Giữ archive path; sửa 5 relative links hỏng, không gọi kế hoạch hiện hành. |
| `docs/archive/README.md` | OBSOLETE | Thay chỉ dẫn kế hoạch 21 ngày hiện hành bằng archive index; link tới current/progress-and-evidence.md. |
| `docs/d02-data-profile.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d03-full-reference.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d04-incremental.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d05-equivalence.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d06-normalization.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d07-lightgcn-smoke.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d08-temporal-baseline.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d09-end-to-end.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d10-mvp.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d11-exp01-pilot.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d12-exp01-results.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/d13-exp02-results.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/environment.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/final-report.md` | RESEARCH EVIDENCE | FROZEN: giữ nguyên bytes/path theo s12-freeze-manifest.json; label ở index bên ngoài file. |
| `docs/gpu-feasibility-v2.md` | RESEARCH EVIDENCE | Giữ measured evidence; link tới current/scope-and-decisions.md cho phê duyệt cutoff/workload, tránh hai nguồn quyết định. |
| `docs/report-assets/README.md` | RESEARCH EVIDENCE | Giữ captions/source/claim links; bảo toàn frozen assets, cập nhật link kiến trúc mới khi cần. |
| `docs/s01-evaluation-spec.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s02-validation-smoke.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s03-kaggle-workflow.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s04-rec01-results.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s05-recommendation-pipeline.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s06-exp01-audit.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s07-exp01-engineering-conclusions.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s09-streamlit-mvp.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s10-rating-graph-refresh.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s11-integration-scaling.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s12-reproducibility.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/s13-defense-qa.md` | HISTORICAL | Giữ path vì S14 RUNME tham chiếu; gắn historical status, không dùng các câu trả lời v1 thay scope v2. |
| `docs/s14-submission.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/source-manifest.md` | RESEARCH EVIDENCE | Giữ nội dung kết quả và path; index tại research/README.md, đánh dấu phạm vi historical/current evidence. Không copy số đo sang current. |
| `docs/temporal-profile-v2.md` | RESEARCH EVIDENCE | Giữ measured evidence; link tới current/scope-and-decisions.md cho phê duyệt cutoff/workload, tránh hai nguồn quyết định. |

## Nguồn authoritative mới

| Thông tin | Một nơi sở hữu |
| --- | --- |
| RQ1/RQ2, cutoff, workload, quyết định Owner | `docs/current/scope-and-decisions.md` |
| Modules, dependencies, graph/model/artifact contracts | `docs/current/architecture.md` |
| Temporal evaluation, benchmark fairness và run provenance rules | `docs/current/experiment-protocol.md` |
| Trạng thái GOAL, gate, liên kết evidence và debt | `docs/current/progress-and-evidence.md` |
| Giải thích học thuật tiếng Việt sau implementation | `docs/current/architecture-walkthrough-vi.md` |
| Phân loại research evidence | `docs/research/README.md`, links tới original records |
| Lịch sử plans/designs/superseded drafts | `docs/archive/README.md` và các bản lịch sử |

## Dependencies ngoài Markdown phải kiểm

- `scripts/s12_freeze_manifest.py`: hash `docs/final-report.md`, source files, requirements và PNG report assets. Giữ manifest lịch sử; không viết lại để khớp source refactor.
- `scripts/s14_package.py`: RUNME nhắc `docs/s12-reproducibility.md`, `docs/s13-defense-qa.md`, report PDF; thu thập docs recursively. Giữ các đường dẫn hoặc sửa generator cho package mới, không repack ZIP cũ.
- `scripts/s12_build_report_docx.py`: có path template/output và textual evidence paths; không regenerate binary report để giải quyết Markdown links.
- `scripts/rec01_kaggle_prep.py`, `scripts/prepare_kaggle_feasibility.py`: source ZIP inventories cần chứa module mới, CLI cũ còn hoạt động.
- `tests/test_d02.py`, `tests/test_full_builder.py`, `tests/test_incremental.py`: path-based imports, không được kết luận module cũ unused chỉ từ static import graph.
- `src/benchmarking/exp01.py`, `src/benchmarking/exp02.py`, `scripts/kaggle_feasibility_runner.py`: subprocess module names là public entrypoints.

Chạy file/anchor link checker sau migration, đối chiếu code-path references, old artifact paths và checksums. Không đổi các historical protocol cutoffs thành cutoff 2020 của v2.

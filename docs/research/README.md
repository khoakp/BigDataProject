# Chỉ mục bằng chứng nghiên cứu

Đây là entrypoint duy nhất cho measured reports, protocol đã khóa và provenance. Nội dung bên dưới giữ nguyên kết quả lịch sử; quyết định v2 hiện hành nằm ở [scope](../current/scope-and-decisions.md).

- **Dữ liệu và workload:** [source manifest](../source-manifest.md), [environment](environment.md), [temporal profile v2](temporal-profile-v2.md), [GPU feasibility v2](gpu-feasibility-v2.md) và [cohort manifest](../../results/model_cohort_v2_manifest.json).
- **RQ2 v2:** [protocol GOAL 04](../current/experiment-protocol.md), [Phase B projection correctness](goal04-phase-b.md), [Phase C local preflight](goal04-phase-c-preflight.md), [C3 Kaggle real-5M preflight](goal04-phase-c3.md), [C4 training candidates](goal04-phase-c4.md), [C5 validation evidence](goal04-phase-c5-preparation.md), [Phase C final report và C0 selection](goal04-phase-c.md). C0 validation quality đã đo; chưa có S0/S1/S2 model-freshness comparison hoặc final holdout quality.
- **D0 chuẩn bị nghiên cứu:** [readiness và kế hoạch D1–D5](goal04-phase-d0.md), [khôi phục C0/G0](../current/artifact-recovery.md), [claim map](../reporting-v2/claim-evidence-map.md), [bản thảo v2](../reporting-v2/report-draft.md), [slide storyboard](../reporting-v2/slide-storyboard.md). Chưa triển khai Phase D.
- **RQ1 và snapshot cadence:** [experiment log](experiment-log.md), [evidence matrix](evidence-matrix.md), [D12 EXP-01](d12-exp01-results.md), [S06 audit](s06-exp01-audit.md), [S07 conclusions](s07-exp01-engineering-conclusions.md), [D13 EXP-02](d13-exp02-results.md).
- **REC-01:** [S01 locked protocol](s01-evaluation-spec.md), [S03 Kaggle workflow](s03-kaggle-workflow.md), [S04 results](s04-rec01-results.md), [S11 integration](s11-integration-scaling.md), [S12 reproducibility](../s12-reproducibility.md).
- **Frozen submission evidence:** [final report](../final-report.md), [report assets](../report-assets/README.md), [defense notes](../s13-defense-qa.md) và `docs/s12-freeze-manifest.json`.

Các milestone khác ở [archive](../archive/README.md). Compatibility notes tại root `docs/` chỉ giữ những path mà frozen report hoặc report scripts đang tham chiếu.

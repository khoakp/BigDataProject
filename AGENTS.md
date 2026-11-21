# BigDataProject v2.0 — working instructions

## Scope and sources of truth

The approved objective is RQ1 incremental graph maintenance and RQ2 LightGCN model freshness on MovieLens 32M. Snapshot cadence is a supporting engineering study. Keep cumulative historical-positive graph semantics and `rating >= 4.0`. The 01/01/2020 base cutoff is tentative until temporal profiling. Do not claim that v2 refresh experiments are complete without implementation and measured evidence.

Read `README.md`, `docs/03-system-design.md` (current v2 section), and `docs/architecture-audit-v2.md` before architecture changes. Use `docs/07-experiment-log.md` for measured runs with data/config/hardware/source hashes. DXX/SXX documents, the old project plans and raw results are historical records; preserve them but do not treat them as current milestones. Do not create further DXX/SXX milestones or parallel pipeline variants.

## Engineering rules

- Reuse tested graph, snapshot, LightGCN, and temporal evaluation code before rewriting. Keep deterministic event order, stable node IDs, no duplicate positive edges, exact degrees, idempotent replay, transaction rollback, full/incremental equivalence at a common watermark, and correct normalized adjacency.
- Separate canonical graph version, model graph snapshot version, checkpoint/source version, and propagated embeddings version in APIs and evidence. Prevent future events or labels from entering training graphs or candidate masks.
- Treat REC-01 as a sealed historical artifact. Its loader checks runtime source SHA-256. Never modify sealed reports/checkpoints or relax verification to make old artifacts load. Gate source migration on an explicit versioned artifact contract and compatibility tests.
- Check Git state and available data/resources before edits. Run targeted tests after behavior changes and full regression at completion. Report passed, failed and skipped separately. Run small fixtures before costly benchmarks/training; do not assume a Kaggle T4 workload fits just because local CUDA is available.
- Do not modify original datasets, checkpoints or raw experiment logs. Do not commit, push, merge, delete branches or install large dependencies without Project Owner authorization.

## Mentoring and reporting

Explain substantial changes in Vietnamese: WHAT changed, WHY, HOW data/APIs work with file/function links, VERIFICATION and remaining limits, and ACADEMIC VALUE for RQ1/RQ2. Use a small example and 2–3 knowledge-check questions. Distinguish observed measurements from hypotheses and historical results. Ask the Project Owner before changing research scope, original data or an important agreed architecture decision.

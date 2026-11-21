# v2.0 architecture audit and migration record

Audit baseline: 21/11/2026, branch `main`, HEAD `1911e6e158244edbc24b5847e88c95b2cd7064c4`, clean working tree. Refactor branch: `refactor/v2-architecture`. This document records observed code and limits; it is not a new experiment result.

## Evidence and findings

| Severity | Evidence | Consequence and disposition |
| --- | --- | --- |
| High | `src/models/rec01_train.py:_code_provenance` records source SHA-256; `src/models/rec01_inference.py:RUNTIME_CODE` verifies six runtime files before loading. `tests/test_rec01_inference.py:test_runtime_code_hash_must_match_training_report` covers rejection. | Editing any of those six files breaks old REC-01 loading. Keep them byte-identical in this refactor. Preserve `rec01_inference.py` as the legacy verified loader; a future artifact version needs an explicit loader and migration test, never a hash override. The training report also records other source hashes, so historical provenance remains pinned even where inference does not enforce it. |
| High | `src/models/rec01_train.py:32` imports `ROOT` and `incremental_train_graph` from `d08_baseline.py`; `src/models/d09_pipeline.py` and `d10_mvp.py` chain D08/D09. | Current training depends on a historical experiment script. Keep this edge temporarily because moving it changes recorded source provenance and may alter artifacts. Extract a shared graph preparation API in a later, versioned REC-02 path; test old loader against the sealed artifact first. D08/D09/D10 remain legacy entrypoints. |
| Medium | `src/benchmarking/exp02.py` imports generic `RssSampler` and `slice_events` from `exp01.py`. | An EXP-01 experiment module is an accidental utility dependency. Move the two utilities into `src/benchmarking/common.py`, preserving their behavior and EXP-01's public names for callers. |
| Medium | `src/graph/full_builder.py:_rebuild` checks `count`, `min(event_id)` and `max(event_id)`, but does not establish event ID uniqueness, row order, field validity, or input schema. | A malformed input with a duplicated interior ID and a missing ID can pass this precondition. Full rebuild is a correctness reference only for validated canonical events. A later version should validate the canonical contract or consume a verified manifest; avoid changing the pinned historical implementation silently. |
| Medium | `src/ingest/d02_profile.py:run` checks nulls, rating range and timestamp, but does not reject nonpositive raw IDs; `src/graph/incremental.py:apply_batch` does. `d02_profile.py` also says threshold `not_chosen`, while graph policy is fixed at 4.0. | Canonicalization and replay disagree on accepted input and documented policy. New canonical API should align validation and provenance, with tests for IDs and temporal ties. Existing canonical data must not be regenerated or overwritten during this task. |
| Medium | `src/graph/snapshot.py:materialize_normalized` calls `incremental_state` and `normalize_state` in Python, then inserts normalized COO with SQL. | At large graph sizes this materializes full state and COO in process memory; the cadence benchmark measures this implementation, not an incremental normalization algorithm. Profile memory before optimizing. The equality tests currently protect values at small scale. |
| Medium | `src/evaluation/temporal.py:cutoff_cohort` intentionally excludes cold users/movies and train-seen repeats; `docs/s04-rec01-results.md` reports only 403 test warm users and 773 train movies. | REC-01 quality results are narrow warm-cohort evidence; do not generalize to full MovieLens 32M or use them to select the v2 temporal cutoff. Preserve the approved REC-01 protocol as historical evidence. |
| Low | `README.md` and `AGENTS.md` point to S01–S14/DXX as current work; `docs/03-system-design.md` labels EXP-02 as RQ2 and freshness as RQ3. | These conflict with v2 RQ1 graph maintenance and RQ2 model freshness. Replace current guidance, retain historical files and raw logs, and label EXP-02 an engineering study. |

## Runtime and legacy dependency map

```text
MovieLens ratings.csv -> ingest.d02_profile (current canonicalizer, historical name)
  -> graph.full_builder / graph.incremental -> graph.equivalence
  -> graph.normalize -> graph.snapshot -> models.lightgcn
  -> evaluation.temporal + evaluation.baselines + evaluation.rec01
  -> models.rec01_train -> models.rec01_inference -> app.streamlit_app
                                      app.feedback -> graph.incremental/snapshot

benchmarking.exp01 -> graph + ingest.d02_profile
benchmarking.exp02 -> graph + benchmarking.exp01 utilities [refactor target]
models.rec01_train -> models.d08_baseline graph helper [pinned legacy dependency]
models.d09_pipeline -> models.d08_baseline [legacy]
models.d10_mvp -> models.d09_pipeline + ingest.d02_profile [legacy]
evaluation.s02_probe -> models.d08_baseline.ROOT [legacy probe]
```

The active demo is `src/app/streamlit_app.py` with `src/app/feedback.py` and `src/models/rec01_inference.py`. The active research primitives are `src/graph/{full_builder,incremental,equivalence,normalize,snapshot}.py`, `src/models/lightgcn.py`, and `src/evaluation/temporal.py`. `rec01_train.py` is the fixed historical REC-01 training workflow, not a v2 refresh engine. DXX/SXX scripts, probes, and existing EXP-01/02 runs are historical evidence or reusable experiment runners, not the v2 architecture authority.

## Environment and baseline verification

- Local `../ml-32m/ratings.csv` exists (877,076,222 bytes); `movies.csv` and canonical 100k/1M Parquet files exist. Existing REC-01 run and raw EXP-01/02 CSVs exist. These were inspected, not modified.
- Existing venv: Python 3.10.11, DuckDB 1.5.6, PyArrow 25.0.1, PyTorch 2.8.0+cu126; CUDA available on this machine. Kaggle T4 feasibility for the proposed v2 workload has **not** been tested.
- `..\venv\Scripts\python.exe -m unittest discover -s tests -v`: **60 passed, 0 failed, 0 skipped** (27.104 s). Includes graph transaction/rollback, full equivalence, normalization, temporal leakage, REC-01 loader, feedback recovery, and toy benchmark contracts.
- `..\venv\Scripts\python.exe -m pytest -q`: **not run**; `pytest` is not installed in this venv. This is not counted as a pass.

## Migration decisions and gates

1. Reuse graph, snapshot, LightGCN, temporal evaluation and existing REC-01 inference. Do not change threshold, source events, raw logs, or sealed checkpoints.
2. Extract only generic benchmarking utilities now. Keep experiment output schemas and command names stable; run related toy tests and full `unittest` regression.
3. Keep D08/D09/D10 and SXX scripts as legacy, clearly labelled. The REC-01 to D08 dependency remains until a versioned training/artifact contract exists. Do not claim it is resolved.
4. Before any later source migration, record old artifact hashes, load the sealed REC-01 artifact with the old loader, introduce a new artifact schema and independent loader, then test both against the same toy fixture and real artifact. Never edit a sealed report or checkpoint to force compatibility.
5. Profile MovieLens temporal distribution before choosing 2020 as cutoff. Measure graph workload separately from one-T4 LightGCN memory/throughput feasibility. Only then schedule stale versus propagation-only comparisons; fine-tuning/retraining requires a feasible budget.

Unfinished: v2 temporal replay API, graph-versioned refresh interfaces and model freshness measurements. No GPU training or full-scale graph benchmark was run in this audit.

## Refactor verification and before/after

| Before | After | Regression evidence |
| --- | --- | --- |
| `exp02.py` imported reusable slicing and RSS sampling from experiment-specific `exp01.py`. | Both import `src/benchmarking/common.py`; `exp01.py` still exposes the same imported names for existing callers. | `tests.test_exp01` and `tests.test_exp02`: 2 passed; toy full/incremental export equality and eager/deferred snapshot equality. |
| README/AGENTS and top of system design described v1/DXX/SXX as current. | README/AGENTS point to the v2 scope and one current architecture section; historical documents and raw evidence remain intact. | Documentation and Git diff reviewed; no experiment results changed. |

After refactor, `unittest discover -s tests -v` ran **60 passed, 0 failed, 0 skipped** (29.493 s). The sealed local `REC01-Kaggle-1M` artifact loaded through `load_rec01_inference` and returned 20 recommendations for warm user 25062: graph version 81, selected checkpoint step 300. `git diff --check` found no whitespace errors. A fresh 100k Full Rebuild produced 57,925 positive edges and equal user/movie degree sums; this is a smoke run, not a performance benchmark. No sealed source, checkpoint, dataset or raw experiment log was changed. The remaining D08 dependency and new v2 model-refresh API are explicitly pending.

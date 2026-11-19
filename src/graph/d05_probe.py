"""D05 correctness harness: full rebuild at every incremental batch boundary."""
import json
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .equivalence import checksum, compare_states, full_state, incremental_state
from .full_builder import file_sha256, rebuild
from .incremental import IncrementalGraph


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exercise(source: Path, sizes: list[int], base: Path) -> dict:
    """Return exact equality reports and verify metadata without comparing lineage."""
    with duckdb.connect() as con, IncrementalGraph(base / "state.duckdb") as graph:
        total = con.execute("SELECT count(*) FROM read_parquet(?)",[str(source)]).fetchone()[0]
        require(sum(sizes) == total and all(n > 0 for n in sizes), "Segmentation must cover source")
        start = 0
        old_registry = []
        boundaries = []
        saved_batches = []
        for batch_id, size in enumerate(sizes):
            stop = start + size
            batch = base / f"batch_{batch_id}.parquet"
            prefix = base / "prefix.parquet"
            con.execute(f"COPY (SELECT * FROM read_parquet(?) WHERE event_id>=? AND event_id<? ORDER BY event_id) TO '{batch.as_posix()}' (FORMAT PARQUET)",[str(source),start,stop])
            con.execute(f"COPY (SELECT * FROM read_parquet(?) WHERE event_id<? ORDER BY event_id) TO '{prefix.as_posix()}' (FORMAT PARQUET)",[str(source),stop])
            applied = graph.apply_batch(batch_id,batch)
            reference_summary = rebuild(prefix,base / "full")
            actual = incremental_state(graph)
            comparison = compare_states(full_state(base / "full"),actual)
            require(comparison["pass"], f"Exact equality failed at cutoff {stop}")
            require(actual["registry"][:len(old_registry)] == old_registry, "Node IDs changed")
            old_registry = actual["registry"]
            before = graph.snapshot()
            require(graph.apply_batch(batch_id,batch)["status"] == "NO_OP", "Replay not NO_OP")
            require(graph.snapshot() == before, "Replay changed persistent state")
            ledger = before["applied_batches"]
            require(len(ledger)==batch_id+1 and ledger[-1][2]==stop-1, "Ledger/watermark mismatch")
            require(sum(row[5] for row in ledger)==stop, "Ledger event total mismatch")
            require(sum(row[7] for row in ledger)==len(actual["edges"]), "Inserted edge total mismatch")
            require(applied["graph_version"]==batch_id+1, "Unexpected version")
            require(all(1 <= row[-1] <= batch_id+1 for row in before["node_registry"]+before["edge_state"]+before["degree_state"]),"Version outside ledger bounds")
            boundaries.append({"cutoff_events":stop,"full_edges":reference_summary["edges"],
                               "comparison":comparison,"replay_unchanged":True,"stable_ids":True,
                               "ledger_pass":True})
            saved_batches.append(batch)
            start = stop
        # Reapply old batches after subsequent batches have committed, in reverse order.
        before = graph.snapshot()
        for batch_id in reversed(range(len(saved_batches))):
            require(graph.apply_batch(batch_id,saved_batches[batch_id])["status"]=="NO_OP","Old replay failed")
            require(graph.snapshot()==before,"Old replay changed state")
        final = incremental_state(graph)
    return {"sizes":sizes,"boundaries":boundaries,"status":"PASS",
            "immediate_replays":len(sizes),"historical_replays":len(sizes),
            "final_state_sha256":checksum(final),"final_edges":len(final["edges"]),
            "final_nodes":len(final["registry"])}


def run():
    root = Path(__file__).resolve().parents[2]
    source = root / "data/processed/canonical_events/canonical_100000.parquet"
    report = {"run_id":"D05-equivalence","timestamp_utc":datetime.now(timezone.utc).isoformat(),
              "python":platform.python_version(),"duckdb":duckdb.__version__,"threads":1,
              "threshold":4.0,"seed":"not applicable (fixed segmentation)",
              "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=root,text=True).strip(),
              "code_state":"uncommitted working tree", "input_sha256":file_sha256(source),
              "code_sha256":{name:file_sha256(root / name) for name in
                 ("src/graph/full_builder.py","src/graph/incremental.py","src/graph/equivalence.py","src/graph/d05_probe.py")},
              "datasets":{}}
    output = root / "data/processed/d05_probe/report.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            toy = base / "toy.parquet"
            with duckdb.connect() as con:
                con.execute("CREATE TABLE toy(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,movie_raw_id BIGINT,rating FLOAT,source_row_id BIGINT)")
                # Negative-only prefix, shared raw IDs across node types, timestamp ties,
                # repeated positive pairs, and later negative events on existing edges.
                con.executemany("INSERT INTO toy VALUES (?,?,?,?,?,?)",[
                    (0,10,1,1,3.5,0),(1,10,1,1,4,1),(2,10,1,1,5,2),
                    (3,10,1,2,4,3),(4,10,2,1,4.5,4),(5,11,1,1,1,5),
                    (6,11,2,2,4,6),(7,12,3,3,0.5,7)])
                con.execute(f"COPY toy TO '{toy.as_posix()}' (FORMAT PARQUET)")
            report["toy_sha256"] = file_sha256(toy)
            for dataset, path, segmentations in (
                ("toy",toy,[[1]*8,[2]*4,[3,1,4],[8]]),
                ("ml32m_100k",source,[[10000]*10,[25000]*4,[33333]*3+[1],[100000]])):
                runs = []
                report["datasets"][dataset] = runs
                for index, sizes in enumerate(segmentations):
                    work = base / f"{dataset}_{index}"
                    work.mkdir()
                    runs.append(exercise(path,sizes,work))
                require(len({r["final_state_sha256"] for r in runs})==1,"Final state differs across segmentations")
        report["status"]="PASS"
    except Exception as exc:
        report.update(status="FAIL",failure_reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"status":report["status"],"report":str(output),"datasets":{
        name:[{k:r[k] for k in ("sizes","final_edges","final_nodes","final_state_sha256")} for r in runs]
        for name,runs in report["datasets"].items()}},indent=2))


if __name__ == "__main__":
    run()

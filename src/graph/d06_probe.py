"""D06 integration: sparse reference, deferred versions, and movie schema validation."""
import json
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .full_builder import file_sha256, rebuild
from .incremental import IncrementalGraph
from .normalize import ATOL, SparseAdjacency, compare_adjacency
from .snapshot import materialize_normalized, load_snapshot, export_snapshot, load_export, freshness
from src.metadata.movie_lookup import validate_movies


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sql_reference(full_output: Path) -> SparseAdjacency:
    """Independent SQL normalization of the full-rebuild output (not normalize_state)."""
    with duckdb.connect() as con:
        con.execute("CREATE TEMP TABLE edges AS SELECT * FROM read_parquet(?)",[str(full_output / 'edges.parquet')])
        con.execute("CREATE TEMP TABLE degrees AS SELECT * FROM read_parquet(?)",[str(full_output / 'degrees.parquet')])
        count = con.execute("SELECT count(*) FROM read_parquet(?)",[str(full_output / 'registry.parquet')]).fetchone()[0]
        entries = con.execute("""WITH weighted AS (
            SELECT user_node_id AS source,movie_node_id AS target,
                1.0/sqrt(CAST(u.degree AS DOUBLE)*CAST(m.degree AS DOUBLE)) AS weight
            FROM edges e JOIN degrees u ON u.node_id=e.user_node_id JOIN degrees m ON m.node_id=e.movie_node_id)
            SELECT * FROM weighted UNION ALL SELECT target,source,weight FROM weighted ORDER BY source,target""").fetchall()
    return SparseAdjacency((count,count),tuple(entries))


def run():
    root = Path(__file__).resolve().parents[2]
    source = root / 'data/processed/canonical_events/canonical_100000.parquet'
    movies = root.parent / 'ml-32m/movies.csv'
    raw = root / 'data/processed/d06_probe'
    now = datetime.now(timezone.utc)
    run_id = now.strftime('D06-%Y%m%dT%H%M%S%fZ')
    work = raw / run_id
    work.mkdir(parents=True,exist_ok=False)
    report = {'run_id':run_id,'timestamp_utc':now.isoformat(),'python':platform.python_version(),
        'duckdb':duckdb.__version__,'threads':1,'dtype':'float64','atol':ATOL,'seed':'not applicable',
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'code_state':'uncommitted working tree','input_sha256':file_sha256(source),
        'code_sha256':{name:file_sha256(root / name) for name in (
            'src/graph/normalize.py','src/graph/snapshot.py','src/metadata/movie_lookup.py',
            'src/graph/d06_probe.py','src/graph/full_builder.py','src/graph/incremental.py')},
        'threshold':4.0,'batch_size':10000,'refresh_versions':[1,5,10],
        'snapshots':[],'freshness':[],'artifact_directory':str(work.relative_to(root))}
    try:
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as con, IncrementalGraph(work / 'state.duckdb') as graph:
            con.execute('SET threads=1')
            temp = Path(tmp)
            frozen = {}
            latest = None
            provenance = {'dataset_id':'ml32m_chronological_100k','canonical_input_sha256':report['input_sha256'],
                          'source_manifest':'docs/source-manifest.md','code_sha256':report['code_sha256']}
            for batch_id in range(10):
                batch = temp / 'batch.parquet'
                stop = (batch_id+1)*10000
                con.execute(f"COPY (SELECT * FROM read_parquet(?) WHERE event_id>=? AND event_id<? ORDER BY event_id) TO '{batch.as_posix()}' (FORMAT PARQUET)",[str(source),stop-10000,stop])
                graph.apply_batch(batch_id,batch)
                for version, saved in frozen.items():
                    require(load_snapshot(graph,version)==saved,'Old snapshot mutated after canonical update')
                if batch_id+1 in report['refresh_versions']:
                    version = batch_id+1
                    manifest = materialize_normalized(graph,expected_version=version,provenance=provenance)
                    snapshot = load_snapshot(graph,version)
                    frozen[version] = snapshot
                    latest = version
                    prefix = temp / 'prefix.parquet'
                    con.execute(f"COPY (SELECT * FROM read_parquet(?) WHERE event_id<? ORDER BY event_id) TO '{prefix.as_posix()}' (FORMAT PARQUET)",[str(source),stop])
                    rebuild(prefix,temp / 'full')
                    comparison = compare_adjacency(sql_reference(temp / 'full'),snapshot[2])
                    require(comparison['pass'],'Normalized Full/Incremental mismatch')
                    coordinates = {(u,v):w for u,v,w in snapshot[2].entries}
                    require(len(coordinates)==manifest['nnz']==2*manifest['positive_edge_count'],'COO duplicate/nnz mismatch')
                    require(all(u!=v and coordinates.get((v,u))==w for (u,v),w in coordinates.items()),'Symmetry/self-loop failure')
                    exported = work / f'model_graph_v{version}'
                    export_snapshot(graph,version,exported)
                    loaded = load_export(exported)
                    require(loaded[1:]==snapshot[1:],'Export round-trip mismatch')
                    report['snapshots'].append({'version':version,'shape':manifest['shape'],
                        'edges':manifest['positive_edge_count'],'nnz':manifest['nnz'],
                        'coo_sha256':manifest['coo_sha256'],'comparison':comparison,
                        'symmetric':True,'self_loops':0,'export_roundtrip_pass':True})
                report['freshness'].append(freshness(graph,latest))
            before = graph.snapshot()
            report['movies'] = validate_movies(movies)
            require(graph.snapshot()==before,'Metadata validation changed graph state')
            report['metadata_graph_isolation_pass'] = True
        with IncrementalGraph(work / 'state.duckdb') as graph:
            require(all(load_snapshot(graph,v)==saved for v,saved in frozen.items()),'Reopen snapshot mismatch')
        report.update(status='PASS',reopen_pass=True,old_snapshots_unchanged=True)
    except Exception as exc:
        report.update(status='FAIL',failure_reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        (raw / 'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        (work / 'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    run()

"""Fixed-cohort projection of an immutable canonical stream; no model training.

The full canonical Parquet is required, starting at event_id=0. Keeping the
source immutable lets a boundary consume a complete timestamp group, including
events excluded from the model graph. Only the projection is materialized.
"""
from __future__ import annotations

import json
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from src.artifacts import file_sha256
from src.graph.equivalence import checksum
from src.graph.normalize import ATOL, normalized_weight, SparseAdjacency
from src.ingest.canonical import validate_events
from src.ingest.replay import sql_path


SCHEMA = 'fixed_cohort_projection_v1'


class ModelGraphReplay:
    """Single-writer transactional replay, with a separate projection version."""

    def __init__(self, database: Path):
        if not Path(database).is_file():
            raise FileNotFoundError(database)
        self.con = duckdb.connect(str(database))
        try:
            self.meta = json.loads(self.con.execute('SELECT payload FROM metadata').fetchone()[0])
            if self.meta['schema_version'] != SCHEMA:
                raise ValueError('Unknown projection schema')
            self._attach_source()
            self._check_state()
        except Exception:
            self.con.close()
            raise

    @classmethod
    def create(cls, database: Path, cohort: Path, events: Path):
        """Verify cohort hashes and exact base projection before creating v0.

        Cohort edges contain first_event_ts; runtime edges retain canonical
        first_event_id. Never substitute the timestamp for an event ID.
        """
        if Path(database).exists():
            raise FileExistsError(database)
        manifest = json.loads((cohort / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['graph_semantics'] != 'complete_induced_pre_cutoff_positive':
            raise ValueError('Expected induced historical-positive cohort')
        for name in ('nodes', 'edges'):
            if file_sha256(cohort / f'{name}.parquet') != manifest[f'{name}_sha256']:
                raise ValueError('Cohort file hash mismatch')
        cutoff = manifest['cutoff_ts']
        if type(cutoff) is not int or cutoff <= 0 or manifest['positive_threshold'] != 4.0:
            raise ValueError('Invalid base semantics')
        source_hash = file_sha256(events)
        validate_events(events)
        obj = cls.__new__(cls)
        obj.con = duckdb.connect(str(database))
        obj.meta = dict(schema_version=SCHEMA, source_path=str(events.resolve()),
                        source_sha256=source_hash, cohort_manifest=manifest,
                        cohort_manifest_sha256=file_sha256(cohort / 'manifest.json'),
                        base_cutoff_ts=cutoff)
        try:
            obj.con.execute('BEGIN TRANSACTION')
            obj._attach_source()
            obj.con.execute('CREATE TABLE metadata(payload VARCHAR)')
            obj.con.execute('CREATE TABLE versions(version INTEGER PRIMARY KEY, boundary BIGINT UNIQUE, payload VARCHAR)')
            obj.con.execute('CREATE TABLE nodes(node_id BIGINT PRIMARY KEY, node_type VARCHAR, raw_id BIGINT, first_event_id BIGINT, UNIQUE(node_type,raw_id))')
            obj.con.execute("""INSERT INTO nodes
                WITH firsts AS (
                  SELECT 'user' kind,user_raw_id raw,min(event_id) first_id FROM events
                  WHERE rating>=4 AND event_ts<? GROUP BY 2
                  UNION ALL
                  SELECT 'movie',movie_raw_id,min(event_id) FROM events
                  WHERE rating>=4 AND event_ts<? GROUP BY 2)
                SELECT n.node_id,n.node_type,n.raw_id,f.first_id
                FROM read_parquet(?) n JOIN firsts f ON n.node_type=f.kind AND n.raw_id=f.raw""",
                            [cutoff,cutoff,str(cohort / 'nodes.parquet')])
            original_nodes = obj.con.execute('SELECT node_id,node_type,raw_id FROM read_parquet(?) ORDER BY node_id', [str(cohort/'nodes.parquet')]).fetchall()
            if (not original_nodes or [r[0] for r in original_nodes] != list(range(len(original_nodes))) or
                any(kind not in ('user','movie') or raw <= 0 for _,kind,raw in original_nodes)):
                raise ValueError('Invalid fixed node mapping')
            if obj.con.execute('SELECT node_id,node_type,raw_id FROM nodes ORDER BY node_id').fetchall() != original_nodes:
                raise ValueError('Cohort node absent from base positive history')
            obj.con.execute('CREATE TABLE edges(user_node_id BIGINT, movie_node_id BIGINT, first_event_id BIGINT, PRIMARY KEY(user_node_id,movie_node_id))')
            obj.con.execute("""INSERT INTO edges SELECT u.node_id,m.node_id,min(e.event_id)
                FROM events e JOIN nodes u ON u.node_type='user' AND u.raw_id=e.user_raw_id
                JOIN nodes m ON m.node_type='movie' AND m.raw_id=e.movie_raw_id
                WHERE e.rating>=4 AND e.event_ts<? GROUP BY 1,2""", [cutoff])
            difference = obj.con.execute('''WITH expected AS (SELECT * FROM read_parquet(?)),
                actual AS (SELECT user_node_id,movie_node_id,event_ts FROM edges JOIN events ON event_id=first_event_id)
                SELECT count(*) FROM ((SELECT * FROM expected EXCEPT ALL SELECT * FROM actual)
                    UNION ALL (SELECT * FROM actual EXCEPT ALL SELECT * FROM expected))''',
                [str(cohort/'edges.parquet')]).fetchone()[0]
            if difference or obj.con.execute('SELECT count(*) FROM edges').fetchone()[0] != manifest['positive_edges']:
                raise ValueError('Base projection differs from approved cohort')
            # Base-cold classifications stay anchored at the approved cutoff.
            obj.con.execute('CREATE TABLE base_users AS SELECT DISTINCT user_raw_id FROM events WHERE rating>=4 AND event_ts<?', [cutoff])
            obj.con.execute('CREATE TABLE base_movies AS SELECT DISTINCT movie_raw_id FROM events WHERE rating>=4 AND event_ts<?', [cutoff])
            orphan = obj.con.execute('''SELECT count(*) FROM nodes WHERE node_id NOT IN (
                SELECT user_node_id FROM edges UNION SELECT movie_node_id FROM edges)''').fetchone()[0]
            if orphan:
                raise ValueError('Fixed cohort contains orphan nodes')
            obj.meta['mapping_sha256'] = checksum(original_nodes)
            obj.con.execute('INSERT INTO metadata VALUES (?)', [json.dumps(obj.meta)])
            obj._record(0, cutoff, {})
            obj.con.execute('COMMIT')
        except Exception:
            obj.con.execute('ROLLBACK')
            obj.close()
            # Leave failed initialization file for diagnosis; never overwrite it.
            raise
        return obj

    def _attach_source(self):
        source = Path(self.meta['source_path'])
        if file_sha256(source) != self.meta['source_sha256']:
            raise ValueError('Canonical source changed')
        self.con.execute('SET threads=1')
        self.con.execute("SET memory_limit='1GB'")
        self.con.execute(f"CREATE OR REPLACE TEMP VIEW events AS SELECT * FROM read_parquet('{sql_path(source)}')")

    def close(self):
        self.con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def state(self):
        """Existing graph state contract; degrees are exact SQL incident counts."""
        return dict(registry=self.con.execute('SELECT * FROM nodes ORDER BY node_id').fetchall(),
                    edges=self.con.execute('SELECT * FROM edges ORDER BY ALL').fetchall(),
                    degrees=self.con.execute('''SELECT node_id,count(*) FROM (
                        SELECT user_node_id node_id FROM edges UNION ALL
                        SELECT movie_node_id FROM edges) GROUP BY 1 ORDER BY 1''').fetchall())

    def current(self):
        return json.loads(self.con.execute('SELECT payload FROM versions ORDER BY version DESC LIMIT 1').fetchone()[0])

    def _check_state(self):
        state = self.state()
        current = self.current()
        if checksum([r[:3] for r in state['registry']]) != self.meta['mapping_sha256']:
            raise ValueError('Fixed mapping changed')
        if any(checksum(state[k]) != current[k+'_sha256'] for k in ('registry','edges','degrees')):
            raise ValueError('Projection state does not match version ledger')

    def _record(self, version, boundary, counts):
        state = self.state()
        last = self.con.execute('''SELECT event_id,event_ts,user_raw_id,movie_raw_id,source_row_id
            FROM events WHERE event_ts<? ORDER BY event_id DESC LIMIT 1''', [boundary]).fetchone()
        payload = dict(schema_version=SCHEMA, projection_version=version,
                       observed_before_ts=boundary, canonical_graph_version=None,
                       canonical_watermark_event_id=last[0] if last else -1,
                       last_canonical_order_key=list(last[1:]) if last else None,
                       mapping_sha256=self.meta['mapping_sha256'],
                       source_sha256=self.meta['source_sha256'],
                       cohort_manifest_sha256=self.meta['cohort_manifest_sha256'],
                       edge_count=len(state['edges']), counts=counts,
                       **{k+'_sha256':checksum(state[k]) for k in ('registry','edges','degrees')})
        self.con.execute('INSERT INTO versions VALUES (?,?,?)', [version,boundary,json.dumps(payload)])
        return payload

    def advance(self, boundary: int):
        """Consume all events before boundary atomically; retries are read-only.

        Counts partition unique positive pairs in this interval, user before
        movie before seen. Repeats within the interval are a separate event count.
        Empty and all-negative intervals still advance the observation boundary.
        """
        if type(boundary) is not int:
            raise ValueError('Boundary must be integer UTC seconds')
        self._attach_source()
        self._check_state()
        existing = self.con.execute('SELECT payload FROM versions WHERE boundary=?', [boundary]).fetchone()
        if existing:
            return {'status':'NO_OP', **json.loads(existing[0])}
        current = self.current()
        previous = current['observed_before_ts']
        if boundary < previous:
            raise ValueError('Cannot replay a new boundary backwards')
        last_ts = self.con.execute('SELECT max(event_ts) FROM events').fetchone()[0]
        if boundary > last_ts + 1:
            raise ValueError('Boundary beyond captured source coverage')
        self.con.execute('BEGIN TRANSACTION')
        try:
            self.con.execute('''CREATE OR REPLACE TEMP TABLE incoming AS
                SELECT * FROM events WHERE event_ts>=? AND event_ts<?''', [previous,boundary])
            self.con.execute("""CREATE OR REPLACE TEMP TABLE classified AS
                WITH pairs AS (SELECT user_raw_id,movie_raw_id,min(event_id) first_id
                    FROM incoming WHERE rating>=4 GROUP BY 1,2)
                SELECT p.*,u.node_id uid,m.node_id mid,
                  CASE WHEN u.node_id IS NULL AND bu.user_raw_id IS NULL THEN 'cold_user'
                       WHEN u.node_id IS NULL THEN 'omitted_user'
                       WHEN m.node_id IS NULL AND bm.movie_raw_id IS NULL THEN 'cold_movie'
                       WHEN m.node_id IS NULL THEN 'omitted_movie'
                       WHEN e.first_event_id IS NOT NULL THEN 'seen'
                       ELSE 'inserted' END category
                FROM pairs p
                LEFT JOIN nodes u ON u.node_type='user' AND u.raw_id=p.user_raw_id
                LEFT JOIN nodes m ON m.node_type='movie' AND m.raw_id=p.movie_raw_id
                LEFT JOIN base_users bu USING(user_raw_id)
                LEFT JOIN base_movies bm USING(movie_raw_id)
                LEFT JOIN edges e ON e.user_node_id=u.node_id AND e.movie_node_id=m.node_id""")
            events, positives = self.con.execute('SELECT count(*),count(*) FILTER(WHERE rating>=4) FROM incoming').fetchone()
            counts = {k:0 for k in ('cold_user','omitted_user','cold_movie','omitted_movie','seen','inserted')}
            counts.update(dict(self.con.execute('SELECT category,count(*) FROM classified GROUP BY 1').fetchall()))
            pairs = sum(counts.values())
            counts.update(rating_events=events,positive_events=positives,
                          unique_positive_pairs=pairs,within_interval_repeat_events=positives-pairs)
            self.con.execute("INSERT INTO edges SELECT uid,mid,first_id FROM classified WHERE category='inserted'")
            result = self._record(current['projection_version']+1,boundary,counts)
            if result['edge_count'] != current['edge_count']+counts['inserted']:
                raise AssertionError('Projection edge conservation failed')
            self.con.execute('COMMIT')
        except Exception:
            self.con.execute('ROLLBACK')
            raise
        return {'status':'APPLIED', **result}

    def export_snapshot(self, output: Path, *, expected_version: int):
        """Export current immutable projection snapshot; reject stale version labels."""
        self._attach_source()
        self._check_state()
        current = self.current()
        if type(expected_version) is not int or expected_version != current['projection_version']:
            raise ValueError('Snapshot version must match current projection')
        node_count = self.con.execute('SELECT count(*) FROM nodes').fetchone()[0]
        output.mkdir(parents=True,exist_ok=False)
        queries = dict(nodes='SELECT * FROM nodes ORDER BY node_id',
                       edges='SELECT * FROM edges ORDER BY ALL',
                       degrees='''SELECT node_id,count(*) degree FROM (
                           SELECT user_node_id node_id FROM edges UNION ALL SELECT movie_node_id FROM edges)
                           GROUP BY 1 ORDER BY 1''',
                       adjacency='''WITH d AS (SELECT node_id,count(*) degree FROM (
                           SELECT user_node_id node_id FROM edges UNION ALL SELECT movie_node_id FROM edges) GROUP BY 1),
                           a AS (SELECT user_node_id AS source,movie_node_id AS target FROM edges UNION ALL
                                 SELECT movie_node_id,user_node_id FROM edges)
                           SELECT source,target,1.0/sqrt(CAST(ds.degree AS DOUBLE)*dt.degree) weight
                           FROM a JOIN d ds ON source=ds.node_id JOIN d dt ON target=dt.node_id ORDER BY 1,2''')
        for name, query in queries.items():
            self.con.execute(f"COPY ({query}) TO '{sql_path(output/(name+'.parquet'))}' (FORMAT PARQUET)")
        manifest = dict(current, snapshot_id=f"projection-{expected_version}-{current['edges_sha256'][:16]}",
                        created_utc=datetime.now(timezone.utc).isoformat(),
                        normalization='symmetric_degree_v1',dtype='float64',self_loops=False,
                        shape=[node_count,node_count],nnz=2*current['edge_count'],
                        files_sha256={n+'.parquet':file_sha256(output/(n+'.parquet')) for n in queries})
        manifest.update(_verify_payload(output,manifest))
        (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        return manifest


def load_projection_snapshot(path: Path, *, expected_mapping: str, expected_version: int,
                             expected_boundary: int, expected_source: str, materialize: bool = True):
    """Validate file integrity, provenance and normalization before use by a model."""
    m = json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if (m['schema_version'] != SCHEMA or m['mapping_sha256'] != expected_mapping or
        m['projection_version'] != expected_version or m['normalization'] != 'symmetric_degree_v1' or
        m['observed_before_ts'] != expected_boundary or m['source_sha256'] != expected_source or
        m['dtype'] != 'float64' or m['self_loops'] is not False or
        m['canonical_graph_version'] is not None):
        raise ValueError('Incompatible projection snapshot')
    last_key = m['last_canonical_order_key']
    if (last_key is None or len(last_key) != 4 or last_key[0] >= expected_boundary or
        m['canonical_watermark_event_id'] < 0 or
        m['snapshot_id'] != f"projection-{expected_version}-{m['edges_sha256'][:16]}"):
        raise ValueError('Snapshot temporal/version provenance mismatch')
    verified = _verify_payload(path,m)
    if verified['coo_sha256'] != m['coo_sha256']:
        raise ValueError('Snapshot normalization checksum mismatch')
    if 'python_coo_sha256' in m and verified['python_coo_sha256'] != m['python_coo_sha256']:
        raise ValueError('Snapshot Python reference checksum mismatch')
    if not materialize:
        return m,None,None
    with duckdb.connect(config={'memory_limit':'1GB','threads':'1'}) as con:
        tables = {n:con.execute('SELECT * FROM read_parquet(?) ORDER BY ALL', [str(path/(n+'.parquet'))]).fetchall()
                  for n in ('nodes','edges','degrees','adjacency')}
    state = dict(registry=tables['nodes'],edges=tables['edges'],degrees=tables['degrees'])
    return m,state,SparseAdjacency(tuple(m['shape']),tuple(tables['adjacency']))


def _hash_batches(batches):
    """Byte-identical to checksum(rows), without one whole JSON string."""
    digest=hashlib.sha256(b'[')
    first=True
    for batch in batches:
        if not batch:
            continue
        if not first:
            digest.update(b',')
        digest.update(json.dumps(batch,separators=(',',':')).encode()[1:-1])
        first=False
    digest.update(b']')
    return digest.hexdigest()


def _batches(cursor):
    while batch := cursor.fetchmany(65536):
        yield batch


def _verify_payload(path, m):
    """Reload every row with bounded Python memory; never skip integrity checks."""
    names = ('nodes','edges','degrees','adjacency')
    if set(m['files_sha256']) != {n+'.parquet' for n in names}:
        raise ValueError('Incomplete snapshot file manifest')
    for name in names:
        if file_sha256(path/(name+'.parquet')) != m['files_sha256'][name+'.parquet']:
            raise ValueError('Snapshot file hash mismatch')
    with duckdb.connect(config={'memory_limit':'1GB','threads':'1'}) as con:
        for name in names:
            con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{sql_path(path/(name+'.parquet'))}')")
        nodes=con.execute('SELECT * FROM nodes ORDER BY node_id').fetchall()
        if (not nodes or [r[0] for r in nodes] != list(range(len(nodes))) or
            len({(r[1],r[2]) for r in nodes}) != len(nodes) or
            any(r[1] not in ('user','movie') or r[2]<=0 or r[3]<0 or r[3]>m['canonical_watermark_event_id'] for r in nodes) or
            checksum([r[:3] for r in nodes]) != m['mapping_sha256'] or
            m['shape'] != [len(nodes),len(nodes)]):
            raise ValueError('Snapshot mapping mismatch')
        for name,key in (('nodes','registry'),('edges','edges'),('degrees','degrees')):
            digest=_hash_batches(_batches(con.execute(f'SELECT * FROM {name} ORDER BY ALL')))
            if digest != m[key+'_sha256']:
                raise ValueError('Snapshot state/provenance mismatch')
        edge_count,bad=con.execute('''SELECT count(*),count(*) FILTER(WHERE
            e.first_event_id IS NULL OR e.first_event_id<0 OR e.first_event_id>?
            OR u.node_id IS NULL OR u.node_type!='user' OR v.node_id IS NULL OR v.node_type!='movie')
            FROM edges e LEFT JOIN nodes u ON e.user_node_id=u.node_id
            LEFT JOIN nodes v ON e.movie_node_id=v.node_id''',[m['canonical_watermark_event_id']]).fetchone()
        duplicate=con.execute('SELECT count(*)-count(DISTINCT (user_node_id,movie_node_id)) FROM edges').fetchone()[0]
        if bad or duplicate or edge_count != m['edge_count'] or m['nnz'] != 2*edge_count:
            raise ValueError('Snapshot edge contract mismatch')
        con.execute('''CREATE TEMP VIEW expected_degrees AS SELECT node_id,count(*) AS degree FROM (
            SELECT user_node_id AS node_id FROM edges UNION ALL SELECT movie_node_id FROM edges) GROUP BY 1''')
        difference=con.execute('''SELECT count(*) FROM (
            (SELECT * FROM degrees EXCEPT ALL SELECT * FROM expected_degrees) UNION ALL
            (SELECT * FROM expected_degrees EXCEPT ALL SELECT * FROM degrees))''').fetchone()[0]
        degrees=dict(con.execute('SELECT * FROM degrees').fetchall())
        if difference or set(degrees) != {r[0] for r in nodes}:
            raise ValueError('Snapshot degree contract mismatch')
        con.execute('''CREATE TEMP VIEW coordinates AS
            SELECT user_node_id AS source,movie_node_id AS target FROM edges UNION ALL
            SELECT movie_node_id,user_node_id FROM edges''')
        difference=con.execute('''SELECT count(*) FROM (
            (SELECT source,target FROM adjacency EXCEPT ALL SELECT * FROM coordinates) UNION ALL
            (SELECT * FROM coordinates EXCEPT ALL SELECT source,target FROM adjacency))''').fetchone()[0]
        if difference:
            raise ValueError('Snapshot normalization coordinates mismatch')
        actual_hash=hashlib.sha256(b'[')
        python_hash=hashlib.sha256(b'[')
        first=True
        maximum=0.0
        for batch in _batches(con.execute('SELECT * FROM adjacency ORDER BY source,target')):
            reference=[]
            for source,target,weight in batch:
                expected=normalized_weight(degrees[source],degrees[target])
                if weight is None or not math.isfinite(weight) or abs(weight-expected)>ATOL:
                    raise ValueError('Snapshot normalization mismatch')
                maximum=max(maximum,abs(weight-expected))
                reference.append((source,target,expected))
            for digest,rows in ((actual_hash,batch),(python_hash,reference)):
                if not first:
                    digest.update(b',')
                digest.update(json.dumps(rows,separators=(',',':')).encode()[1:-1])
            first=False
        actual_hash.update(b']')
        python_hash.update(b']')
    return dict(coo_sha256=actual_hash.hexdigest(),python_coo_sha256=python_hash.hexdigest(),
                normalization_max_abs_diff=maximum,normalization_atol=ATOL)

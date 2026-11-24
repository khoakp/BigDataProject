"""CPU-only coverage audit; no predictions, model replay, or holdout quality."""
import argparse
import csv
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from src.artifacts import file_sha256
from src.evaluation.model_cohort import CSV_COLUMNS, DEFAULT_CUTOFF_TS


# Declared before inspecting per-window support; these are feasibility, not quality gates.
MIN_USERS = 100
MIN_LABELS = 500


def summarize(con, cutoff=DEFAULT_CUTOFF_TS):
    """raw(userId,movieId,rating,timestamp), nodes(node_type,raw_id) -> monthly counts.

    Raw rows need not be sorted: counts and first-positive timestamps are invariant
    under canonical tie ordering. This audit does not produce canonical event IDs.
    """
    con.execute("SET TimeZone='UTC'")
    invalid = con.execute("""SELECT count(*) FROM raw WHERE userId IS NULL OR userId<=0
        OR movieId IS NULL OR movieId<=0 OR timestamp IS NULL OR timestamp<=0
        OR rating IS NULL OR NOT(rating BETWEEN 0.5 AND 5 AND rating*2=floor(rating*2))""").fetchone()[0]
    if invalid:
        raise ValueError("Invalid rating source")
    con.execute("""CREATE TEMP TABLE first_positive AS SELECT userId,movieId,
        min(timestamp) AS first_ts FROM raw WHERE rating>=4 GROUP BY 1,2""")
    con.execute("CREATE TEMP TABLE base_users AS SELECT DISTINCT userId FROM first_positive WHERE first_ts<?", [cutoff])
    con.execute("CREATE TEMP TABLE base_movies AS SELECT DISTINCT movieId FROM first_positive WHERE first_ts<?", [cutoff])
    last = con.execute("SELECT max(timestamp) FROM raw").fetchone()[0]
    start = datetime.fromtimestamp(cutoff, timezone.utc)
    rows = []
    while int(start.timestamp()) <= last:
        end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1, tzinfo=timezone.utc)
        b, e = int(start.timestamp()), min(int(end.timestamp()), last + 1)
        con.execute("""CREATE OR REPLACE TEMP TABLE window_pairs AS
            SELECT userId,movieId,count(*) AS positive_events FROM raw
            WHERE timestamp>=? AND timestamp<? AND rating>=4 GROUP BY 1,2""", [b, e])
        con.execute("""CREATE OR REPLACE TEMP TABLE classified AS
            SELECT w.*, f.first_ts,
              CASE WHEN u.raw_id IS NULL AND bu.userId IS NULL THEN 'cold_user'
                   WHEN u.raw_id IS NULL THEN 'omitted_user'
                   WHEN m.raw_id IS NULL AND bm.movieId IS NULL THEN 'cold_movie'
                   WHEN m.raw_id IS NULL THEN 'omitted_movie'
                   WHEN f.first_ts < ? THEN 'seen'
                   ELSE 'eligible' END AS category
            FROM window_pairs w JOIN first_positive f USING(userId,movieId)
            LEFT JOIN nodes u ON u.node_type='user' AND u.raw_id=w.userId
            LEFT JOIN nodes m ON m.node_type='movie' AND m.raw_id=w.movieId
            LEFT JOIN base_users bu ON bu.userId=w.userId
            LEFT JOIN base_movies bm ON bm.movieId=w.movieId""", [b])
        events = con.execute("SELECT count(*) FROM raw WHERE timestamp>=? AND timestamp<?", [b, e]).fetchone()[0]
        pairs, positives, positive_users, new_edges = con.execute("""SELECT count(*),coalesce(sum(positive_events),0),
            count(DISTINCT userId),count(*) FILTER(WHERE first_ts>=?) FROM classified""", [b]).fetchone()
        counts = dict(con.execute("SELECT category,count(*) FROM classified GROUP BY category").fetchall())
        users, movies = con.execute("""SELECT count(DISTINCT userId),count(DISTINCT movieId)
            FROM classified WHERE category='eligible'""").fetchone()
        distribution = con.execute("""SELECT min(n),quantile_cont(n,0.5),quantile_cont(n,0.9),max(n),avg(n)
            FROM (SELECT count(*) n FROM classified WHERE category='eligible' GROUP BY userId)""").fetchone()
        row = dict(window=start.strftime('%Y-%m'), boundary_ts=b, end_exclusive_ts=e,
                   partial=e != int(end.timestamp()), rating_events=events, positive_events=int(positives),
                   future_positive_pairs=pairs, positive_active_users=positive_users,
                   new_canonical_edges=new_edges, within_window_repeat_events=int(positives)-pairs,
                   **{key+'_pairs': counts.get(key, 0) for key in
                      ('cold_user', 'omitted_user', 'cold_movie', 'omitted_movie', 'seen', 'eligible')},
                   eligible_users=users, eligible_movies=movies,
                   **dict(zip(('labels_min','labels_median','labels_p90','labels_max','labels_mean'), distribution)),
                   pair_coverage=counts.get('eligible', 0)/pairs if pairs else None,
                   user_coverage=users/positive_users if positive_users else None,
                   support_pass=users>=MIN_USERS and counts.get('eligible', 0)>=MIN_LABELS)
        assert sum(counts.values()) == pairs
        assert int(positives) == pairs + row['within_window_repeat_events']
        assert row['eligible_pairs'] <= new_edges <= pairs
        rows.append(row)
        start = end
    return rows


def profile(source, cohort, out):
    manifest = json.loads((cohort/'manifest.json').read_text(encoding='utf-8'))
    hashes = {name: file_sha256(cohort/name) for name in ('nodes.parquet','edges.parquet')}
    source_hash = file_sha256(source)
    if (source_hash != manifest['source_sha256'] or
        hashes['nodes.parquet'] != manifest['nodes_sha256'] or
        hashes['edges.parquet'] != manifest['edges_sha256'] or
        manifest['cutoff_ts'] != DEFAULT_CUTOFF_TS or manifest['selection'] != 'user_recency' or
        manifest['positive_edges'] != 4_999_981):
        raise ValueError('Approved cohort/source mismatch')
    out.mkdir(parents=True, exist_ok=False)
    with duckdb.connect() as con:
        con.execute("SET threads=2")
        con.execute("SET memory_limit='1GB'")
        con.execute("SET temp_directory=?", [str(out/'spill')])
        con.execute(f"CREATE TABLE raw AS SELECT * FROM read_csv(?,header=true,auto_detect=false,columns={CSV_COLUMNS})", [str(source)])
        con.execute("CREATE TABLE nodes AS SELECT * FROM read_parquet(?)", [str(cohort/'nodes.parquet')])
        rows = summarize(con)
        source_stats = con.execute('SELECT count(*),min(timestamp),max(timestamp) FROM raw').fetchone()
    with (out/'monthly.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(status='COUNTS_ONLY_NOT_APPROVED_PROTOCOL', source_sha256=source_hash,
                  cohort_hashes=hashes, source_events=source_stats[0], min_ts=source_stats[1], max_ts=source_stats[2],
                  cutoff_ts=DEFAULT_CUTOFF_TS, cold_reference='pre-2020 positive graph; user then movie then seen',
                  support_rule=dict(min_users=MIN_USERS,min_labels=MIN_LABELS,min_dev_month_fraction=0.8),
                  git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  script_sha256=file_sha256(Path(__file__)),
                  environment=dict(python=platform.python_version(),duckdb=duckdb.__version__,threads=2,memory_limit='1GB'),
                  measured_utc=datetime.now(timezone.utc).isoformat(),
                  monthly_sha256=file_sha256(out/'monthly.csv'),
                  totals={key:sum(r[key] for r in rows) for key in
                          ('rating_events','future_positive_pairs','eligible_pairs','cold_user_pairs',
                           'omitted_user_pairs','cold_movie_pairs','omitted_movie_pairs','seen_pairs')})
    assert result['totals']['eligible_pairs'] == manifest['evaluable_future_pairs']
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(profile(args.source,args.cohort,args.out),indent=2))

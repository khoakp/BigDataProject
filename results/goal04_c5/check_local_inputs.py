"""Read-only local C5 input check; does not load the 5M graph into RAM."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

from scripts.verify_kaggle_c4 import verify
from src.artifacts import file_sha256


REPO = Path(__file__).resolve().parents[2]
C4 = REPO/'results/goal04_c4/run_04'
C3 = REPO/'results/goal04_c3/run_03/goal04_c3/base_snapshot'
CANONICAL = REPO/'data/processed/goal04_c3_kaggle_dataset/canonical_32000204.parquet'


def main():
    c4 = verify(C4/'goal04_c4',C4)
    report = json.loads((C4/'goal04_c4/candidates/report.json').read_text())
    provenance = report['provenance']
    manifest = json.loads((C3/'manifest.json').read_text())
    with duckdb.connect() as con:
        nodes,movies = con.execute('''SELECT count(*),
            count(*) FILTER (WHERE node_type='movie') FROM read_parquet(?)''',
            [str(C3/'nodes.parquet')]).fetchone()
    actual = {'c4_strict_verification':c4['status'],
              'checkpoint_steps':[r['step'] for r in c4['checkpoints']],
              'snapshot_manifest_sha256':file_sha256(C3/'manifest.json'),
              'mapping_sha256':manifest['mapping_sha256'],
              'source_sha256':file_sha256(CANONICAL),
              'nodes':nodes,'catalog_movies':movies,
              'snapshot_edge_count':manifest['edge_count']}
    if (actual['checkpoint_steps'] != [0,500,1500,3000,5000] or
        actual['snapshot_manifest_sha256'] != provenance['snapshot_manifest_sha256'] or
        actual['mapping_sha256'] != provenance['mapping_sha256'] or
        actual['source_sha256'] != provenance['source_sha256'] or
        actual['nodes'] != 90_253 or actual['catalog_movies'] != 40_666 or
        actual['snapshot_edge_count'] != 4_999_981):
        raise ValueError('C5 input integrity mismatch')
    actual['status'] = 'PASS'
    out = REPO/'results/goal04_c5/input-integrity.json'
    out.write_text(json.dumps(actual,indent=2)+'\n',encoding='utf-8')
    print(actual['status'])


if __name__ == '__main__':
    main()

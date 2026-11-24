"""Verify downloaded Kaggle C3 reports and snapshot files without a GPU run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.artifacts import file_sha256


def verify(root: Path, package: Path) -> dict:
    root = Path(root)
    report = json.loads((root/'report.json').read_text(encoding='utf-8'))
    package_manifest = json.loads((Path(package)/'package-manifest.json').read_text(encoding='utf-8'))
    if (report['status'] != 'PASS' or report['elapsed_s'] > 4*3600 or
        [s['stage'] for s in report['stages']] != ['inputs','base','monthly'] or
        any(s['status'] != 'PASS' for s in report['stages']) or
        report['gpu_optimizer_steps'] != 0 or report['quality_evaluations'] != 0 or
        report['source_bundle_sha256'] != package_manifest['source_bundle_sha256'] or
        report['git_head'] != package_manifest['git_head']):
        raise ValueError('C3 report or package provenance mismatch')
    inputs = json.loads((root/'inputs_result.json').read_text(encoding='utf-8'))
    if (inputs['canonical_sha256'] != '494a1e70e67a2d750297872020b3919da9e557dd68cb631f53d2c5cb9570f73e' or
        inputs['raw_source_sha256'] != '91159850e41ee59c86231165a688709647e2726cab2e7ba9faf04001bd5261ee' or
        inputs['canonical_rows'] != 32_000_204 or
        inputs['cohort_nodes_sha256'] != 'a2723792903893694cc708b65c85468180064a445dd949e321a24030f7dbb3d7' or
        inputs['cohort_edges_sha256'] != 'f697aa05eabab03db6b7bc72980229eeb157a42801c42597d92672df50e857f8'):
        raise ValueError('Input provenance mismatch')
    results = {}
    for stage, version, boundary, edges in (('base',0,1577836800,4_999_981),
                                            ('monthly',1,1580515200,5_014_863)):
        data = json.loads((root/(stage+'_result.json')).read_text(encoding='utf-8'))
        record, snapshot = data['record'], data['snapshot']
        disk_manifest = json.loads((root/(stage+'_snapshot')/'manifest.json').read_text(encoding='utf-8'))
        if (data['inputs'] != inputs or not data['reload_verified'] or
            snapshot != disk_manifest or record['projection_version'] != version or
            record['observed_before_ts'] != boundary or record['edge_count'] != edges or
            record['mapping_sha256'] != '996848022d92e20ef63b498cfedb38b97545fb006da4c00fef9ed35d40dc90f9' or
            snapshot['shape'] != [90_253,90_253] or snapshot['nnz'] != 2*edges or
            snapshot['normalization_max_abs_diff'] > snapshot['normalization_atol'] or
            snapshot['normalization_atol'] != 1e-12 or
            record['last_canonical_order_key'][0] >= boundary):
            raise ValueError(f'{stage} projection or normalization mismatch')
        if stage == 'base' and (record['edges_sha256'] != 'fb2c7fe7d64e018bebcd7101df41249dbfa57e1db5369fae01d023e9c2e5392a' or
                                record['degrees_sha256'] != 'ee423151ba7070bc616b8a43475322410b759297c086b9331ba4f3318f49e2b4' or
                                data['future_edge_in_g0'] is not False):
            raise ValueError('G0 contract mismatch')
        if stage == 'monthly' and (record['counts']['inserted'] != 14_882 or
                                   record['counts']['rating_events'] != 125_929):
            raise ValueError('Monthly replay count mismatch')
        hashes = {}
        if set(snapshot['files_sha256']) != {'nodes.parquet','edges.parquet','degrees.parquet','adjacency.parquet'}:
            raise ValueError('Incomplete snapshot file list')
        for name, expected in snapshot['files_sha256'].items():
            observed = file_sha256(root/(stage+'_snapshot')/name)
            if observed != expected:
                raise ValueError(f'{stage} {name} file SHA-256 mismatch')
            hashes[name] = observed
        results[stage] = {'version':version,'edge_count':edges,'mapping_sha256':record['mapping_sha256'],
                          'edges_sha256':record['edges_sha256'],'degrees_sha256':record['degrees_sha256'],
                          'coo_sha256':snapshot['coo_sha256'],
                          'python_coo_sha256':snapshot['python_coo_sha256'],
                          'normalization_max_abs_diff':snapshot['normalization_max_abs_diff'],
                          'file_sha256':hashes}
    return {'status':'PASS','kernel_status':'COMPLETE','git_head':report['git_head'],
            'source_bundle_sha256':report['source_bundle_sha256'],'elapsed_s':report['elapsed_s'],
            'gpu_optimizer_steps':0,'quality_evaluations':0,'inputs':inputs,'snapshots':results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    result = verify(args.root,args.package)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(result['status'])

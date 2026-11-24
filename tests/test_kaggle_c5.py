import ast
import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import duckdb
import torch

from scripts.kaggle_c5_runner import (device_equivalence, evaluate_candidate,
                                      load_2020_windows, _matrix_row)
from scripts.prepare_kaggle_c5 import prepare
from scripts.verify_kaggle_c5 import verify as verify_c5
from src.evaluation.c0_validation import (boundaries, evaluate_embeddings,
                                           score_batched)
from src.evaluation.temporal import score_chunks
from src.artifacts import file_sha256


class C5PreparationTest(unittest.TestCase):
    def setUp(self):
        self.registry = [(0,'user',1,0),(1,'user',2,0),
                         (2,'movie',10,0),(3,'movie',20,0),(4,'movie',30,0)]
        self.embeddings = torch.tensor([[1.,0.],[0.,1.],
                                        [1.,0.],[0.,1.],[1.,0.]])
        cuts = boundaries()
        self.windows = [dict(start=a,end=b,relevant={1:{20},2:{30}},
                             seen={1:{10},2:{20}},counts={'eligible':2})
                        for a,b in zip(cuts,cuts[1:])]

    def test_exact_top20_ties_mask_and_dual_partition(self):
        users = {1:0,2:1}
        movies = {10:2,20:3,30:4}
        seen = self.windows[0]['seen']
        actual = score_batched(self.embeddings,users,movies,seen,k=20,batch_size=1)
        reference = score_chunks(self.embeddings,self.embeddings,users,movies,seen,k=20)
        self.assertEqual(actual,reference)
        self.assertEqual([m for m,_ in actual[1]],[30,20])
        self.assertTrue(device_equivalence(self.embeddings,self.embeddings.clone(),
                                           self.registry,self.windows)[0])
        dual,mode,gate,timings = evaluate_candidate(self.embeddings,self.embeddings.clone(),
                                                    self.registry,self.windows)
        single = evaluate_embeddings(self.embeddings,self.registry,self.windows,
                                     batch_size=1)
        self.assertEqual(mode,'dual_t4')
        self.assertEqual(gate['status'],'PASS')
        self.assertEqual(gate['max_abs_top20_score_diff'],0)
        self.assertIn('ranking_and_metrics_s',timings)
        self.assertEqual(dual['windows'],single)
        with patch('scripts.kaggle_c5_runner.device_equivalence',return_value=(False,1.0)):
            fallback,mode,gate,_ = evaluate_candidate(self.embeddings,
                self.embeddings.clone(),self.registry,self.windows)
        self.assertEqual(mode,'single_t4_fallback')
        self.assertEqual(gate['status'],'FAIL_FALLBACK')
        self.assertEqual(fallback,dual)

    def test_matrix_row_has_provenance_without_selection(self):
        metric={'index':0,'start':boundaries()[0],'end':boundaries()[1],
                'users':2,'labels':2,'recall_20':.5,'ndcg_20':.4}
        provenance={'base_edges_sha256':'edges','source_sha256':'canonical'}
        row=_matrix_row(500,metric,'dual_t4',{'status':'PASS'},provenance,'mapping')
        self.assertEqual(row['step'],500)
        self.assertEqual(row['eligible_labels'],2)
        self.assertEqual(row['mask_boundary_ts'],boundaries()[0])
        self.assertEqual(row['catalog_mapping_sha256'],'mapping')
        self.assertNotIn('selected_step',row)

    def test_score_inputs_reject_invalid_and_nonfinite(self):
        with self.assertRaises(ValueError):
            score_batched(self.embeddings,{1:0},{10:2},{},batch_size=0)
        changed=self.embeddings.clone()
        changed[2,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            score_batched(changed,{1:0},{10:2},{})

    def test_c5_package_rejects_incomplete_budget_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            audit=root/'audit.json'
            audit.write_text(json.dumps({'status':'INCOMPLETE',
                                         'remaining_budget_s':None}))
            with self.assertRaisesRegex(ValueError,'incomplete'):
                prepare(root/'must_not_exist',budget_audit=audit)
            self.assertFalse((root/'must_not_exist').exists())

    def test_c5_package_writes_string_budget_environment_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            audit=root/'audit.json'
            audit.write_text(json.dumps({'status':'VERIFIED',
                'cumulative_phase_c_kaggle_elapsed_s':2369.918029196,
                'remaining_budget_s':12030.081970804}))
            manifest=prepare(root/'package',budget_audit=audit)
            self.assertEqual(manifest['kernel_id'],
                             'khoakp/moviegraph-v2-c5-2020-validation')
            parsed=ast.parse((root/'package/runner.py').read_text())
            values=[node.value.value for node in parsed.body
                    if isinstance(node,ast.Assign) and
                    any(isinstance(target,ast.Subscript) and
                        isinstance(target.slice,ast.Constant) and
                        target.slice.value == 'GOAL04_VERIFIED_PRIOR_PHASE_C_S'
                        for target in node.targets)]
            self.assertEqual(values,['2369.918029196'])
            with patch.dict(os.environ, {}, clear=False):
                exec(compile(ast.Module(body=parsed.body[:5],type_ignores=[]),
                             'c5-wrapper-prefix','exec'),{})
                self.assertEqual(os.environ['GOAL04_VERIFIED_PRIOR_PHASE_C_S'],
                                 '2369.918029196')

    def test_2020_only_and_approved_profile_counts(self):
        cuts=boundaries()
        registry=[(0,'user',1,0),(1,'movie',10,0)]+[
            (i+2,'movie',11+i,0) for i in range(12)]
        events=[(i,cuts[i],1,11+i,4.,i) for i in range(12)]
        events.append((12,1672531200,1,10,5.,12))  # 2023 is outside C5 query.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            parquet=root/'canonical.parquet'
            with duckdb.connect() as con:
                con.execute('''CREATE TABLE events(event_id BIGINT,event_ts BIGINT,
                    user_raw_id BIGINT,movie_raw_id BIGINT,rating FLOAT,
                    source_row_id BIGINT)''')
                con.executemany('INSERT INTO events VALUES (?,?,?,?,?,?)',events)
                con.execute('COPY events TO ? (FORMAT PARQUET)',[str(parquet)])
            profile=root/'monthly.csv'
            with profile.open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,['window','eligible_pairs','eligible_users'])
                writer.writeheader()
                for i in range(1,13):
                    writer.writerow({'window':f'2020-{i:02d}',
                                     'eligible_pairs':1,'eligible_users':1})
            windows=load_2020_windows(parquet,registry,[(0,1,0)],profile)
            self.assertEqual(len(windows),12)
            self.assertEqual([sum(map(len,w['relevant'].values())) for w in windows],[1]*12)
            self.assertNotIn(10,{m for w in windows for pairs in w['relevant'].values()
                                 for m in pairs})
            profile.write_text(profile.read_text().replace('2020-01,1,1','2020-01,2,1'))
            with self.assertRaisesRegex(ValueError,'approved profile'):
                load_2020_windows(parquet,registry,[(0,1,0)],profile)

    def test_artifact_verifier_requires_full_matrix_and_no_selection(self):
        steps=[0,500,1500,3000,5000]
        provenance={'mapping_sha256':'mapping','source_sha256':'source',
                    'base_edges_sha256':'edges'}
        gate={'status':'PASS','max_abs_top20_score_diff':0.,'sampled_users':2,'atol':1e-5}
        metric_rows=[{'index':i,'start':a,'end':b,'users':2,'labels':2,
                      'recall_20':.5,'ndcg_20':.4}
                     for i,(a,b) in enumerate(zip(boundaries(),boundaries()[1:]))]
        matrix=[_matrix_row(step,row,'dual_t4',gate,provenance,'mapping')
                for step in steps for row in metric_rows]
        candidates=[{'step':step,'checkpoint_sha256':f'sha{step}',
                     'validation_recall_20':.5,'validation_ndcg_20':.4,
                     'window_count':12,'gpu_mode':'dual_t4',
                     'numerical_equivalence':gate} for step in steps]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            output=root/'output'; output.mkdir()
            package=root/'package'; package.mkdir()
            c4=root/'c4'; (c4/'candidates').mkdir(parents=True)
            (c4/'report.json').write_text('{}')
            (c4/'candidates/report.json').write_text(json.dumps({
                'provenance':provenance,
                'checkpoints':[{'step':s,'sha256':f'sha{s}'} for s in steps]}))
            (output/'validation-matrix.json').write_text(json.dumps(matrix))
            with (output/'candidate-comparison.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,['step','validation_recall_20',
                    'validation_ndcg_20','window_count','gpu_mode','checkpoint_sha256'])
                writer.writeheader()
                for row in candidates:
                    writer.writerow({key:row[key] for key in writer.fieldnames})
            (package/'package-manifest.json').write_text(json.dumps({
                'source_bundle_sha256':'bundle','git_head':'head',
                'budget_audit_sha256':'audit',
                'verified_prior_phase_c_kaggle_elapsed_s':100.}))
            report={'status':'COMPLETE_VALIDATION','validation_matrix':matrix,
                    'candidate_scores':candidates,'propagation_forwards':5,
                    'optimizer_updates':0,'source_bundle_sha256':'bundle',
                    'git_head':'head','budget_audit_sha256':'audit',
                    'prior_verified_kaggle_elapsed_s':100.,'job_elapsed_s':200.,
                    'cumulative_phase_c_kaggle_elapsed_s':300.,
                    'worker_exit_code':0,'stop_reason':None,
                    'inputs':{'c4_verification_status':'PASS',
                        'c4_report_sha256':file_sha256(c4/'report.json'),
                        'catalog_movies':40666,'windows':12,
                        'mapping_sha256':'mapping','canonical_sha256':'source',
                        'base_edges_sha256':'edges'},
                    'artifact_sha256':{
                        'validation-matrix.json':file_sha256(output/'validation-matrix.json'),
                        'candidate-comparison.csv':file_sha256(output/'candidate-comparison.csv')}}
            (output/'report.json').write_text(json.dumps(report))
            self.assertEqual(verify_c5(output,package,c4)['matrix_rows'],60)
            report['selected_step']=500
            (output/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'status'):
                verify_c5(output,package,c4)
            report.pop('selected_step')
            report['validation_matrix']=matrix[:-1]
            (output/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'status'):
                verify_c5(output,package,c4)


if __name__ == '__main__':
    unittest.main()

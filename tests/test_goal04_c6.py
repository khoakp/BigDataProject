import copy
from datetime import datetime, timezone
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.artifacts import file_sha256
from src.evaluation.c0_validation import boundaries
from src.evaluation.checkpoint_selection import select_checkpoint
from src.models.c0 import C0Config, save_candidate
from src.models.lightgcn import LightGCN
from scripts.finalize_goal04_c0 import (recompute_matrix, verify_selected_checkpoint,
                                        verify_selection)


class C6SelectionTest(unittest.TestCase):
    def setUp(self):
        self.provenance={'mapping_sha256':'mapping','base_edges_sha256':'edges',
                         'source_sha256':'source','node_count':5,
                         'cohort_manifest_sha256':'cohort',
                         'snapshot_coo_sha256':'coo','git_head':'head',
                         'base_cutoff_ts':1577836800,'base_edge_count':2}
        cuts=boundaries()
        self.rows=[]
        self.comparison=[]
        for step,ndcg,recall in ((0,.01,.02),(500,.20,.30),(1500,.21,.31),
                                 (3000,.19,.32),(5000,.18,.33)):
            self.comparison.append({'step':str(step),'window_count':'12',
                'validation_recall_20':str(recall),'validation_ndcg_20':str(ndcg)})
            for index,(start,end) in enumerate(zip(cuts,cuts[1:])):
                self.rows.append({'step':step,'window_index':index,
                    'start_utc':datetime.fromtimestamp(start,timezone.utc).isoformat(),
                    'end_utc':datetime.fromtimestamp(end,timezone.utc).isoformat(),
                    'start_ts':start,'end_exclusive_ts':end,
                    'eligible_users':2,'eligible_labels':3,
                    'catalog_movies':40666,'catalog_mapping_sha256':'mapping',
                    'mask_policy':'rolling_observed_positive_before_boundary',
                    'mask_boundary_ts':start,'mask_base_edges_sha256':'edges',
                    'mask_canonical_source_sha256':'source',
                    'numerical_equivalence':'PASS',
                    'recall_20':recall,'ndcg_20':ndcg})

    def test_recomputed_selection_and_exact_ties(self):
        records=recompute_matrix(self.rows,self.comparison,self.provenance)
        self.assertEqual(select_checkpoint(records)['step'],1500)
        tied=copy.deepcopy(records)
        tied[1]['validation_ndcg_20']=tied[2]['validation_ndcg_20']
        tied[1]['validation_recall_20']=tied[2]['validation_recall_20']
        self.assertEqual(select_checkpoint(tied)['step'],500)
        tied[2]['validation_recall_20']+=.01
        self.assertEqual(select_checkpoint(tied)['step'],1500)

    def test_missing_row_and_incompatible_mapping(self):
        with self.assertRaisesRegex(ValueError,'Missing validation row'):
            recompute_matrix(self.rows[:-1],self.comparison,self.provenance)
        changed=copy.deepcopy(self.rows)
        changed[12]['catalog_mapping_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'mapping|source mismatch'):
            recompute_matrix(changed,self.comparison,self.provenance)
        changed=copy.deepcopy(self.rows)
        changed[12]['eligible_users']=4
        with self.assertRaisesRegex(ValueError,'user/label counts'):
            recompute_matrix(changed,self.comparison,self.provenance)

    def test_selected_original_bytes_step_and_provenance(self):
        config=C0Config()
        model=LightGCN(5,64,3)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'candidate_1500.pt'
            saved=save_candidate(path,model,step=1500,config=config,
                                 provenance=self.provenance)
            loaded=verify_selected_checkpoint(path,
                expected_sha256=saved['sha256'],expected_step=1500,
                config=config,provenance=self.provenance)
            self.assertEqual(loaded['payload_step'],1500)
            self.assertEqual(loaded['embedding_shape'],[5,64])
            with self.assertRaisesRegex(ValueError,'filename/step'):
                verify_selected_checkpoint(path,expected_sha256=saved['sha256'],
                    expected_step=500,config=config,provenance=self.provenance)
            with self.assertRaisesRegex(ValueError,'metadata'):
                verify_selected_checkpoint(path,expected_sha256=saved['sha256'],
                    expected_step=1500,config=config,
                    provenance={**self.provenance,'mapping_sha256':'wrong'})
            path.write_bytes(path.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'hash'):
                verify_selected_checkpoint(path,expected_sha256=saved['sha256'],
                    expected_step=1500,config=config,provenance=self.provenance)
            self.assertNotEqual(file_sha256(path),saved['sha256'])

    def test_locked_manifest_rejects_wrong_selected_step(self):
        expected={'selected_step':1500,'checkpoint':{'sha256':'original'},
                  'evidence_sha256':{}}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'selection.json'
            path.write_text(json.dumps({**expected,'selected_step':500}),encoding='utf-8')
            with patch('scripts.finalize_goal04_c0.build_selection_manifest',
                       return_value=expected):
                with self.assertRaisesRegex(ValueError,'differs'):
                    verify_selection(path)


if __name__=='__main__':
    unittest.main()

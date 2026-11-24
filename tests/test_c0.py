import tempfile
import unittest
from pathlib import Path

import torch

from src.graph.equivalence import checksum
from src.evaluation.c0_validation import boundaries, validation_windows, evaluate_embeddings, aggregate_validation
from src.evaluation.checkpoint_selection import select_checkpoint
from src.models.c0 import C0Config, load_candidate, train_candidates


class C0Test(unittest.TestCase):
    def test_approved_calendar_has_twelve_half_open_windows(self):
        cuts=boundaries()
        self.assertEqual(len(cuts),13)
        self.assertEqual((cuts[0],cuts[-1]),(1577836800,1609459200))
        self.assertTrue(all(a<b for a,b in zip(cuts,cuts[1:])))

    def setUp(self):
        self.registry=[(0,'user',1,0),(1,'user',2,0),(2,'movie',10,0),(3,'movie',20,0),(4,'movie',30,0)]
        self.edges=[(0,2,0),(1,3,1)]
        self.state={'registry':self.registry,'edges':self.edges,'degrees':[(0,1),(1,1),(2,1),(3,1)]}
        self.base=1577836800
        self.events=[(2,self.base,1,20,4.,2),(3,self.base,1,20,5.,3),
                     (4,self.base+5,2,10,4.,4),(5,self.base+10,1,10,4.,5),
                     (6,self.base+11,1,30,4.,6),(7,self.base+12,3,10,5.,7)]
        self.cuts=[self.base,self.base+10,self.base+20]
        self.windows=validation_windows(self.events,self.registry,self.edges,
            start=self.base,end=self.base+20,window_boundaries=self.cuts)
        self.provenance=dict(mapping_sha256=checksum([r[:3] for r in self.registry]),
            cohort_manifest_sha256='cohort',snapshot_coo_sha256='snapshot',
            source_sha256='source',git_head='head',base_cutoff_ts=self.base,node_count=5)
        self.provenance['base_edge_count']=2
        self.provenance['base_edges_sha256']=checksum(self.edges)

    def test_boundary_mask_dedup_and_partition_equivalence(self):
        first,second=self.windows
        self.assertEqual(first['relevant'],{1:{20},2:{10}})
        self.assertEqual(first['seen'][1],{10})
        self.assertEqual(first['counts']['eligible'],2)
        self.assertEqual(second['seen'][1],{10,20})
        self.assertEqual(second['relevant'],{1:{30}})
        self.assertEqual(second['counts']['seen'],1)
        self.assertEqual(second['counts']['cold_user'],1)
        emb=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.],[.5,.5]])
        sequential=evaluate_embeddings(emb,self.registry,self.windows)
        partitioned=evaluate_embeddings(emb,self.registry,self.windows,selected=[1])+evaluate_embeddings(
            emb,self.registry,self.windows,selected=[0])
        self.assertEqual(aggregate_validation([sequential],2),aggregate_validation([partitioned],2))
        with self.assertRaises(ValueError):
            aggregate_validation([sequential[:1]],2)
        with self.assertRaises(ValueError):
            aggregate_validation([sequential,sequential],2)

    def test_leakage_and_checkpoint_selection(self):
        with self.assertRaises(ValueError):
            validation_windows(self.events+[(8,1609459200,1,30,5.,8)],self.registry,self.edges,
                               start=self.base,end=self.base+20,window_boundaries=self.cuts)
        with self.assertRaises(ValueError):
            validation_windows(list(reversed(self.events)),self.registry,self.edges,
                               start=self.base,end=self.base+20,window_boundaries=[self.base+1,self.base+20])
        records=[dict(step=s,validation_recall_20=.4,validation_ndcg_20=.2) for s in (0,500,1500)]
        self.assertEqual(select_checkpoint(records)['step'],0)

    def test_toy_single_trajectory_reload_and_mismatch(self):
        indices=torch.tensor([[0,2,1,3],[2,0,3,1]])
        sparse=torch.sparse_coo_tensor(indices,torch.ones(4),(5,5)).coalesce()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            output=train_candidates(self.state,sparse,self.windows,root/'run',
                config=C0Config(),provenance=self.provenance,steps=(0,1,2))
            self.assertEqual(output['status'],'TOY_ONLY')
            self.assertIn('INCOMPLETE',(root/'run/report.json').read_text())
            self.assertEqual([r['step'] for r in output['candidates']],[0,1,2])
            for record in output['candidates']:
                model=load_candidate(root/'run'/f"candidate_{record['step']}.pt",
                    expected_sha256=record['checkpoint_sha256'],config=C0Config(),
                    provenance=self.provenance,toy_cpu=True)
                self.assertEqual(tuple(model.embedding.weight.shape),(5,64))
            first=torch.load(root/'run/candidate_0.pt',weights_only=True)['state_dict']['embedding.weight']
            last=torch.load(root/'run/candidate_2.pt',weights_only=True)['state_dict']['embedding.weight']
            self.assertFalse(torch.equal(first,last))
            selected=output['candidates'][0]
            path=root/'run/candidate_0.pt'
            with self.assertRaises(ValueError):
                load_candidate(path,expected_sha256='bad',config=C0Config(),provenance=self.provenance)
            with self.assertRaises(ValueError):
                load_candidate(path,expected_sha256=selected['checkpoint_sha256'],
                    config=C0Config(dim=32),provenance=self.provenance)
            with self.assertRaises(ValueError):
                load_candidate(path,expected_sha256=selected['checkpoint_sha256'],
                    config=C0Config(),provenance={**self.provenance,'source_sha256':'bad'})
            bad={**self.provenance,'mapping_sha256':'bad'}
            with self.assertRaises(ValueError):
                train_candidates(self.state,sparse,self.windows,root/'wrong',config=C0Config(),provenance=bad,steps=(0,1))
            bad={**self.provenance,'base_edges_sha256':'bad'}
            with self.assertRaises(ValueError):
                train_candidates(self.state,sparse,self.windows,root/'wrong_edges',config=C0Config(),provenance=bad,steps=(0,1))
            payload=torch.load(path,weights_only=True)
            payload['state_dict']['embedding.weight']=torch.zeros(4,64)
            changed=root/'changed.pt'
            torch.save(payload,changed)
            from src.artifacts import file_sha256
            with self.assertRaisesRegex(ValueError,'tensor'):
                load_candidate(changed,expected_sha256=file_sha256(changed),config=C0Config(),
                               provenance=self.provenance)
            payload['state_dict']['embedding.weight']=first
            payload['schema_version']='unknown'
            torch.save(payload,changed)
            with self.assertRaisesRegex(ValueError,'metadata'):
                load_candidate(changed,expected_sha256=file_sha256(changed),config=C0Config(),
                               provenance=self.provenance)


if __name__=='__main__':
    unittest.main()

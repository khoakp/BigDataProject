import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.graph.normalize import ATOL, SparseAdjacency, compare_adjacency, normalize_state


class NormalizationTest(unittest.TestCase):
    def test_sparse_equals_independent_dense_reference(self):
        state = {"registry":[(0,'user',10,0),(1,'movie',50,0),(2,'user',20,1),(3,'movie',60,2)],
                 "edges":[(0,1,0),(2,1,1),(2,3,2)],"degrees":[(0,1),(1,2),(2,2),(3,1)]}
        sparse = normalize_state(state)
        dense = [[0.0]*4 for _ in range(4)]
        for u,v,_ in state['edges']:
            dense[u][v] = dense[v][u] = 1.0
        degrees = [sum(row) for row in dense]
        reference = tuple((u,v,dense[u][v]*degrees[u]**(-0.5)*degrees[v]**(-0.5))
                          for u in range(4) for v in range(4) if dense[u][v])
        self.assertTrue(compare_adjacency(sparse,SparseAdjacency((4,4),reference))['pass'])
        self.assertEqual(len(sparse.entries),6)
        self.assertEqual(dict(((u,v),w) for u,v,w in sparse.entries)[(2,1)],0.5)
        self.assertTrue(all(u != v for u,v,_ in sparse.entries))
        entries = {(u,v):w for u,v,w in sparse.entries}
        self.assertTrue(all(entries[(v,u)] == w for (u,v),w in entries.items()))

    def test_empty_graph_and_corrupt_degree(self):
        self.assertEqual(normalize_state({'registry':[],'edges':[],'degrees':[]}),SparseAdjacency((0,0),()))
        state = {'registry':[(0,'user',1,0),(1,'movie',1,0)],'edges':[(0,1,0)],'degrees':[(0,2),(1,1)]}
        with self.assertRaisesRegex(ValueError,'Degree'):
            normalize_state(state)
        state['degrees'] = [(0,1),(1,1)]
        state['edges'].append((0,1,1))
        with self.assertRaisesRegex(ValueError,'Duplicate edge'):
            normalize_state(state)

    def test_comparison_detects_shape_coordinates_and_weights(self):
        a = SparseAdjacency((2,2),((0,1,1.0),(1,0,1.0)))
        self.assertFalse(compare_adjacency(a,SparseAdjacency((3,3),a.entries))['pass'])
        self.assertFalse(compare_adjacency(a,SparseAdjacency((2,2),((0,0,1.0),(1,1,1.0))))['pass'])
        self.assertFalse(compare_adjacency(a,SparseAdjacency((2,2),((0,1,0.5),(1,0,0.5))))['pass'])
        self.assertTrue(compare_adjacency(a,SparseAdjacency((2,2),((0,1,1+ATOL/2),(1,0,1))))['pass'])

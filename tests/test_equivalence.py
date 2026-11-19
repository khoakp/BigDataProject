"""Guard against weak comparisons and zero-positive full rebuild regression."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.graph.equivalence import compare_states
from src.graph.full_builder import rebuild
from test_incremental import batch


class EquivalenceTest(unittest.TestCase):
    def test_detects_same_count_but_different_state(self):
        reference = {"registry": [(0,"user",1,0),(1,"movie",2,0)],
                     "edges": [(0,1,0)], "degrees": [(0,1),(1,1)]}
        self.assertTrue(compare_states(reference, copy.deepcopy(reference))["pass"])
        for table, replacement in (("registry", (0,"user",99,0)), ("edges", (1,0,0)), ("degrees", (0,2))):
            actual = copy.deepcopy(reference)
            actual[table][0] = replacement
            with self.subTest(table=table):
                self.assertFalse(compare_states(reference,actual)["pass"])

    def test_full_rebuild_all_negative_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = batch(base,"negative",[(0,10,1,1,3.5,0)])
            result = rebuild(source,base / "full")
            self.assertEqual((result["edges"],result["nodes"],result["sum_user_degree"],result["sum_movie_degree"]),(0,0,0,0))


if __name__ == "__main__":
    unittest.main()

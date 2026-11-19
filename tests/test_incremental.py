"""D04 transaction, deduplication, ordering, and batch identity tests."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import duckdb


spec = importlib.util.spec_from_file_location("incremental", Path(__file__).resolve().parents[1] / "src/graph/incremental.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
IncrementalGraph = module.IncrementalGraph


def batch(base, name, rows):
    path = base / (name + ".parquet")
    with duckdb.connect() as con:
        con.execute("CREATE TABLE t(event_id BIGINT,event_ts BIGINT,user_raw_id BIGINT,movie_raw_id BIGINT,rating FLOAT,source_row_id BIGINT)")
        if rows:
            con.executemany("INSERT INTO t VALUES (?,?,?,?,?,?)", rows)
        con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT PARQUET)")
    return path


class FailLedgerWrite:
    """Inject a database failure after edges/degrees are written but before commit."""
    def __init__(self, con):
        self.con = con

    def execute(self, sql, *args):
        if sql.startswith("INSERT INTO applied_batches VALUES"):
            raise RuntimeError("injected ledger write failure")
        return self.con.execute(sql, *args)


class IncrementalTest(unittest.TestCase):
    def test_dedup_ids_noop_conflict_and_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            a = batch(base, "a", [(0,10,10,50,3.5,0),(1,11,10,50,4,1),(2,12,10,50,5,2)])
            b = batch(base, "b", [(3,13,10,50,4,3),(4,14,20,50,5,4),(5,15,20,60,4,5)])
            negative = batch(base, "negative", [(6,16,99,99,1,6)])
            conflict = batch(base, "conflict", [(0,10,10,50,3,0),(1,11,10,50,4,1),(2,12,10,50,5,2)])
            db = base / "state.duckdb"
            with IncrementalGraph(db) as graph:
                self.assertEqual(graph.apply_batch(10,a)["inserted_edges"], 1)
                old_registry = graph.snapshot()["node_registry"]
                self.assertEqual(graph.apply_batch(20,b)["inserted_edges"], 2)
                self.assertEqual(graph.snapshot()["node_registry"][:2], old_registry)
                self.assertEqual(graph.apply_batch(30,negative)["inserted_edges"], 0)
                state = graph.snapshot()
                self.assertEqual(state["degree_state"], [(0,1,1),(1,2,2),(2,2,2),(3,1,2)])
                self.assertEqual(len(state["node_registry"]),4)
                self.assertEqual(graph.apply_batch(10,a)["status"], "NO_OP")
                self.assertEqual(graph.snapshot(),state)
                with self.assertRaisesRegex(ValueError,"Conflicting"):
                    graph.apply_batch(10,conflict)
                self.assertEqual(graph.snapshot(),state)
            with IncrementalGraph(db) as graph:
                self.assertEqual(graph.snapshot(),state)
                self.assertEqual(graph.apply_batch(20,b)["status"], "NO_OP")
                self.assertEqual(graph.snapshot(),state)

    def test_reject_invalid_batches_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with IncrementalGraph() as graph:
                state = graph.snapshot()
                bad_rows = [[], [(1,10,1,1,4,0)], [(0,10,1,1,4,0),(0,11,2,2,4,1)],
                            [(0,11,1,1,4,0),(1,10,2,2,4,1)], [(0,10,1,1,None,0)]]
                for i, rows in enumerate(bad_rows):
                    with self.subTest(i=i):
                        with self.assertRaises(ValueError):
                            graph.apply_batch(i,batch(base,str(i),rows))
                        self.assertEqual(graph.snapshot(),state)
                graph.apply_batch(0,batch(base,"good",[(0,20,1,1,4,0)]))
                state = graph.snapshot()
                with self.assertRaisesRegex(ValueError,"boundary"):
                    graph.apply_batch(1,batch(base,"backwards",[(1,10,2,2,4,1)]))
                self.assertEqual(graph.snapshot(),state)

    def test_transaction_rolls_back_partial_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            a = batch(base,"a",[(0,10,1,1,4,0)])
            b = batch(base,"b",[(1,11,2,2,4,1)])
            with IncrementalGraph() as graph:
                graph.apply_batch(0,a)
                state = graph.snapshot()
                original = graph.con
                graph.con = FailLedgerWrite(original)
                with self.assertRaisesRegex(RuntimeError,"injected"):
                    graph.apply_batch(1,b)
                graph.con = original
                self.assertEqual(graph.snapshot(),state)
                self.assertEqual(graph.apply_batch(1,b)["graph_version"],2)


if __name__ == "__main__":
    unittest.main()

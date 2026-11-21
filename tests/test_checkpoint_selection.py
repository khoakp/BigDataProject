"""S02 checkpoint selection never uses the test set."""
import math
import unittest

from src.evaluation.checkpoint_selection import select_checkpoint


class CheckpointSelectionTest(unittest.TestCase):
    def test_ndcg_then_recall_then_earliest_step(self):
        records = [
            {"step": 5, "validation_recall_20": .8, "validation_ndcg_20": .2},
            {"step": 10, "validation_recall_20": .4, "validation_ndcg_20": .3},
            {"step": 15, "validation_recall_20": .5, "validation_ndcg_20": .3},
            {"step": 20, "validation_recall_20": .5, "validation_ndcg_20": .3},
        ]
        self.assertEqual(select_checkpoint(records)["step"], 15)

    def test_rejects_missing_nonfinite_and_out_of_range_metrics(self):
        valid = {"step": 0, "validation_recall_20": .2, "validation_ndcg_20": .1}
        for override in (
            {"validation_recall_20": None}, {"validation_recall_20": math.nan},
            {"validation_ndcg_20": math.inf}, {"validation_ndcg_20": -math.inf},
            {"validation_recall_20": True}, {"validation_recall_20": 1.01},
            {"validation_ndcg_20": -.01},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                select_checkpoint([{**valid, **override}])
        missing = {"step": 0, "validation_recall_20": .2}
        with self.assertRaises(ValueError):
            select_checkpoint([missing])

    def test_rejects_test_fields_invalid_steps_and_empty_input(self):
        valid = {"step": 0, "validation_recall_20": .2, "validation_ndcg_20": .1}
        for record in (
            {**valid, "test_ndcg_20": .9},
            {**valid, "metadata": {"test": {"recall_20": .9}}},
            {**valid, "step": -1}, {**valid, "step": 1.5},
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                select_checkpoint([record])
        with self.assertRaises(ValueError):
            select_checkpoint([valid, valid])
        with self.assertRaises(ValueError):
            select_checkpoint([])


if __name__ == "__main__":
    unittest.main()

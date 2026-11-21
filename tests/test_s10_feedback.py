"""Real-artifact integration checks for the S10 append-only demo branch."""
import tempfile
import unittest
import uuid
import os
from unittest.mock import patch
from pathlib import Path

import duckdb

from src.app.feedback import FeedbackStore
from src.app.rec01_view import train_rated_ids
from src.graph.full_builder import file_sha256
from src.models.rec01_inference import load_rec01_inference

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M"
SOURCE = ROOT / "data/processed/canonical_events/canonical_1000000.parquet"
MOVIES = Path(os.environ.get("MOVIELENS_MOVIES_CSV", ROOT.parent / "ml-32m/movies.csv"))


@unittest.skipUnless(RUN.is_dir() and SOURCE.is_file() and MOVIES.is_file(),
                     "Sealed REC-01 artifact unavailable")
class S10FeedbackTest(unittest.TestCase):
    @unittest.skipUnless(AppTest is not None, "Streamlit unavailable")
    def test_streamlit_rating_batch_updates_graph_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict("os.environ", {"REC01_FEEDBACK_DIR": str(Path(temporary) / "state")}):
                app = AppTest.from_file(str(ROOT / "src/app/streamlit_app.py"),
                                        default_timeout=30).run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual([widget.value for widget in app.metric[:4]],
                                 ["81", "81", "300", "0"])
                app.button[0].click().run()  # Add one rating to the pending batch.
                self.assertFalse(app.exception)
                self.assertEqual(len(app.button), 3)
                app.button[1].click().run()  # Apply the pending batch.
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual([widget.value for widget in app.metric[:4]],
                                 ["82", "81", "300", "1"])
                self.assertEqual(len(app.success), 1)
                self.assertIn("LightGCN", app.subheader[0].value)

    def test_batch_replay_recovery_and_stale_checkpoint(self):
        original_graph = RUN / "graph_build/graph.duckdb"
        original_hash = file_sha256(original_graph)
        model = load_rec01_inference(RUN, MOVIES)
        before = model.recommend(173074)
        rated = train_rated_ids(SOURCE, 173074,
                                model.report["protocol"]["train_end_ts"])
        choices = [mid for mid in sorted(model.movie_nodes) if mid not in rated][:2]
        self.assertEqual(len(choices), 2)
        rows = [{"user_raw_id": 173074, "movie_raw_id": choices[0], "rating": 5.0},
                {"user_raw_id": 173074, "movie_raw_id": choices[1], "rating": 2.5}]

        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary) / "feedback"
            store = FeedbackStore(RUN, SOURCE, state_dir)
            initial = store.status()
            request_id = str(uuid.uuid4())
            applied = store.apply_ratings(rows, request_id)
            self.assertEqual(applied["status"], "APPLIED")
            self.assertEqual(applied["events_processed"], 2)
            self.assertEqual(applied["inserted_edges"], 1)
            updated = store.status()
            self.assertEqual(updated["canonical_graph_version"],
                             initial["canonical_graph_version"] + 1)
            self.assertEqual(updated["model_graph_version"],
                             initial["model_graph_version"])
            self.assertEqual(updated["positive_edges"], initial["positive_edges"] + 1)
            self.assertEqual(updated["new_feedback_events"], 2)
            self.assertEqual(model.recommend(173074), before)

            self.assertEqual(store.apply_ratings(rows, request_id)["status"], "NO_OP")
            with self.assertRaisesRegex(ValueError, "reused with different ratings"):
                store.apply_ratings(rows[:1], request_id)
            with self.assertRaisesRegex(ValueError, "already rated in this demo"):
                store.apply_ratings(rows, str(uuid.uuid4()))
            with self.assertRaisesRegex(ValueError, "already rated in the train"):
                store.apply_ratings([{"user_raw_id": 173074,
                                      "movie_raw_id": next(iter(rated)),
                                      "rating": 4.5}], str(uuid.uuid4()))

            batch = state_dir / "batches" / f"batch_{applied['batch_id']:05d}.parquet"
            batch.unlink()  # Crash after intent/ledger: recovery recreates the Parquet.
            reopened = FeedbackStore(RUN, SOURCE, state_dir)
            self.assertTrue(batch.is_file())
            self.assertEqual(reopened.status(), updated)
            self.assertEqual(len(reopened.applied_ratings()), 2)
            with duckdb.connect(str(reopened.graph_path), read_only=True) as con:
                self.assertEqual(con.execute(
                    "SELECT count(*) FROM applied_batches WHERE batch_id=?",
                    [applied["batch_id"]]).fetchone()[0], 1)

        self.assertEqual(file_sha256(original_graph), original_hash)


if __name__ == "__main__":
    unittest.main()

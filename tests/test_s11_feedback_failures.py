"""Failure and concurrent-request paths on the sealed REC-01 graph fork."""
import json
import tempfile
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from src.app.feedback import FeedbackStore
from src.app.rec01_view import train_rated_ids
from src.graph.incremental import IncrementalGraph
from src.graph.snapshot import canonical_status


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M"
SOURCE = ROOT / "data/processed/canonical_events/canonical_1000000.parquet"


@unittest.skipUnless(RUN.is_dir() and SOURCE.is_file(), "Sealed REC-01 artifact unavailable")
class S11FeedbackFailureTest(unittest.TestCase):
    def _unrated(self, count: int) -> list[int]:
        report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
        rated = train_rated_ids(SOURCE, 173074,
                                report["protocol"]["train_end_ts"])
        import duckdb
        with duckdb.connect() as con:
            rows = con.execute("SELECT raw_id FROM read_parquet(?) WHERE node_type='movie' "
                               "ORDER BY raw_id", [str(RUN / "model_graph/nodes.parquet")]).fetchall()
        choices = [movie for (movie,) in rows if movie not in rated]
        self.assertGreaterEqual(len(choices), count)
        return choices[:count]

    def test_crash_after_intent_then_reopen_applies_once(self):
        movie = self._unrated(1)[0]
        rating = [{"user_raw_id": 173074, "movie_raw_id": movie, "rating": 5.0}]
        request_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "feedback"
            store = FeedbackStore(RUN, SOURCE, state)
            before = store.status()
            with patch.object(store, "_apply_intent", side_effect=RuntimeError("interrupted")):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    store.apply_ratings(rating, request_id)
            self.assertTrue((state / "intents" / f"{request_id}.json").is_file())
            with IncrementalGraph(store.graph_path) as graph:
                self.assertEqual(canonical_status(graph)["canonical_version"],
                                 before["canonical_graph_version"])
            reopened = FeedbackStore(RUN, SOURCE, state)
            self.assertEqual(reopened.status()["canonical_graph_version"],
                             before["canonical_graph_version"] + 1)
            self.assertEqual(reopened.apply_ratings(rating, request_id)["status"], "NO_OP")
            self.assertEqual(reopened.status()["new_feedback_events"], 1)
            self.assertEqual(reopened.status()["positive_edges"], before["positive_edges"] + 1)

    def test_invalid_payload_and_tampered_intent_rejected(self):
        movie = self._unrated(1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "feedback"
            store = FeedbackStore(RUN, SOURCE, state)
            request_id = str(uuid.uuid4())
            for invalid in (None, [], [{}], [{"user_raw_id": 173074,
                                              "movie_raw_id": movie,
                                              "rating": 4.0, "extra": 1}],
                            [{"user_raw_id": True, "movie_raw_id": movie,
                              "rating": 4.0}]):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        store.apply_ratings(invalid, request_id)
            with self.assertRaisesRegex(ValueError, "timestamp"):
                store.apply_ratings([{"user_raw_id": 173074,
                                      "movie_raw_id": movie, "rating": 4.0}],
                                    request_id, now_ts=1.5)
            self.assertEqual(store.status()["new_feedback_events"], 0)
            row = {"user_raw_id": 173074, "movie_raw_id": movie, "rating": 4.0}
            store.apply_ratings([row], request_id)
            intent_path = state / "intents" / f"{request_id}.json"
            intent = json.loads(intent_path.read_text(encoding="utf-8"))
            intent["rows"][0][4] = 2.5
            intent_path.write_text(json.dumps(intent), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from durable intent"):
                FeedbackStore(RUN, SOURCE, state)

    def test_two_threads_get_contiguous_versions_and_event_ids(self):
        movies = self._unrated(2)
        with tempfile.TemporaryDirectory() as temporary:
            store = FeedbackStore(RUN, SOURCE, Path(temporary) / "feedback")
            def submit(movie):
                return store.apply_ratings([{"user_raw_id": 173074,
                                             "movie_raw_id": movie,
                                             "rating": 5.0}], str(uuid.uuid4()))
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(submit, movies))
            self.assertEqual({result["graph_version"] for result in results}, {82, 83})
            self.assertEqual(store.status()["new_feedback_events"], 2)
            self.assertEqual([row["event_id"] for row in store.applied_ratings()],
                             [802988, 802989])


if __name__ == "__main__":
    unittest.main()

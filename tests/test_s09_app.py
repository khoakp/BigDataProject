"""Optional real-artifact smoke of both Streamlit recommendation flows."""
import unittest
import os
from pathlib import Path

try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "data/processed/rec01_kaggle_runs/version_3/rec01_project/REC01-Kaggle-1M"
MOVIES = Path(os.environ.get("MOVIELENS_MOVIES_CSV", ROOT.parent / "ml-32m/movies.csv"))
EVENTS = ROOT / "data/processed/canonical_events/canonical_1000000.parquet"


@unittest.skipUnless(AppTest is not None and RUN.is_dir() and MOVIES.is_file()
                     and EVENTS.is_file(), "Streamlit or local REC-01 artifact unavailable")
class S09AppTest(unittest.TestCase):
    def test_warm_and_cold_ui_from_verified_artifact(self):
        app = AppTest.from_file(str(ROOT / "src/app/streamlit_app.py"),
                                default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("LightGCN", app.subheader[0].value)
        self.assertEqual(len(app.dataframe[0].value), 20)
        self.assertEqual(len(app.dataframe[2].value), 773)

        next(widget for widget in app.selectbox
             if widget.label == "User ID trong train graph").set_value(25062).run()
        self.assertFalse(app.exception)
        self.assertIn("25062", app.subheader[0].value)
        self.assertEqual(len(app.dataframe[0].value), 20)

        app.radio[0].set_value("Người dùng mới").run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("phổ biến", app.subheader[0].value)
        self.assertEqual(len(app.dataframe[0].value), 20)
        self.assertEqual(len(app.dataframe[1].value), 773)


if __name__ == "__main__":
    unittest.main()

"""A timed-out probe must not inherit RUNNING from its partial report."""
import unittest

from scripts.kaggle_feasibility_runner import probe_status


class KaggleRunnerStatusTest(unittest.TestCase):
    def test_exit_status_takes_precedence_over_partial_report(self):
        self.assertEqual(probe_status(None, "RUNNING"), "TIMEOUT")
        self.assertEqual(probe_status(1, "PASS"), "ERROR")
        self.assertEqual(probe_status(0, "PASS"), "PASS")


if __name__ == "__main__":
    unittest.main()

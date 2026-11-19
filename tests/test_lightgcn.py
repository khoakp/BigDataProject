import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

if importlib.util.find_spec("torch") is not None:
    import torch

    from src.models.d07_probe import run_case, sample_triples, toy, write_report
    from src.models.lightgcn import LightGCN, pairwise_loss, to_sparse_tensor


@unittest.skipUnless(importlib.util.find_spec("torch"), "D07 requires PyTorch")
class LightGCNTest(unittest.TestCase):
    def test_sparse_values_and_gradient_match_independent_dense_example(self):
        _, _, adjacency = toy()
        dense = torch.tensor([
            [0, 2**-0.5, 0, 0],
            [2**-0.5, 0, 0, 0.5],
            [0, 0, 0, 2**-0.5],
            [0, 0.5, 2**-0.5, 0],
        ])
        initial = torch.arange(8, dtype=torch.float32).reshape(4, 2) / 10
        model = LightGCN(4, 2, 2)
        with torch.no_grad():
            model.embedding.weight.copy_(initial)
        reference = initial.clone().requires_grad_()
        expected = (reference + dense @ reference + (dense @ dense) @ reference) / 3
        actual = model(to_sparse_tensor(adjacency, torch.device("cpu")))
        torch.testing.assert_close(actual, expected)
        triples = torch.tensor([[0, 1, 2]])
        expected_margin = expected[0].dot(expected[1] - expected[2])
        torch.logaddexp(torch.zeros(()), -expected_margin).backward()
        pairwise_loss(actual, triples).backward()
        torch.testing.assert_close(model.embedding.weight.grad, reference.grad)

    def test_sampler_uses_positive_and_unobserved_movies(self):
        _, nodes, adjacency = toy()
        self.assertEqual(sample_triples(nodes, adjacency, limit=20), [(0, 1, 2)] * 20)

    def test_cuda_unavailable_and_model_oom_are_distinct_csv_statuses(self):
        with patch("torch.cuda.is_available", return_value=False):
            unavailable = run_case("toy", None, "cuda", toy)
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.empty_cache"), patch(
            "src.models.d07_probe._fit", side_effect=torch.cuda.OutOfMemoryError("test injection")
        ):
            oom = run_case("ml32m_100k", 100_000, "cuda", toy)
        self.assertEqual([unavailable["status"], oom["status"]], ["UNAVAILABLE", "OOM"])
        with tempfile.TemporaryDirectory() as tmp:
            write_report(Path(tmp), {}, [unavailable, oom])
            with (Path(tmp) / "gpu_probe.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[1]["events"], "100000")
            self.assertEqual(rows[1]["status"], "OOM")
            self.assertIn("test injection", rows[1]["reason"])

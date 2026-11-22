"""Portable, read-only Top-20 inference from one completed REC-01 run.

The exported model graph is the train-positive graph used by REC-01. Its
user-to-movie edges supply the same seen-item mask as the evaluation protocol;
neither the training database nor future labels are needed for inference.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch

from src.evaluation.temporal import score_chunks
from src.evaluation.checkpoint_selection import select_checkpoint
from src.graph.full_builder import file_sha256
from src.graph.snapshot import load_export
from src.metadata.movie_lookup import enrich_topk, load_movies
from src.models.lightgcn import LightGCN, to_sparse_tensor


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"REC-01 {label} mismatch")


class Rec01Inference:
    """An immutable selected checkpoint and its train-catalog ranker."""

    def __init__(self, run_dir: Path, movies_csv: Path, *, device_name: str = "cpu"):
        self.run_dir = Path(run_dir)
        if device_name not in ("cpu", "cuda"):
            raise ValueError("REC-01 device must be cpu or cuda")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("REC-01 CUDA requested but unavailable")
        self.device = torch.device(device_name)

        report = json.loads((self.run_dir / "report.json").read_text(encoding="utf-8"))
        if report.get("artifact_schema", "rec01-v1") != "rec01-v1":
            raise ValueError("Unsupported REC-01 artifact schema")
        if report.get("status") != "PASS":
            raise ValueError("REC-01 run is not complete")
        protocol, graph_report, selected = (report["protocol"], report["graph"],
                                            report["selected_checkpoint"])
        if protocol["k"] != 20:
            raise ValueError("REC-01 inference requires the Top-20 protocol")
        _require_equal(hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
                       report["config_sha256"], "protocol config SHA-256")
        # Historical source hashes remain provenance in the sealed report. Artifact
        # compatibility is checked against its config, graph, mapping and tensors.
        _require_equal(select_checkpoint(report["validation_candidates"]), selected,
                       "validation-selected checkpoint")
        step = selected["step"]
        expected_checkpoint_name = f"model_step_{step:04d}.pt"
        _require_equal(report["selected_checkpoint_file"], expected_checkpoint_name,
                       "selected checkpoint file")
        checkpoint_path = self.run_dir / expected_checkpoint_name
        _require_equal(file_sha256(checkpoint_path), selected["checkpoint_sha256"],
                       "selected checkpoint SHA-256")
        _require_equal(report["checkpoint_sha256"][str(step)],
                       selected["checkpoint_sha256"], "checkpoint record SHA-256")
        if selected not in report["validation_candidates"]:
            raise ValueError("REC-01 selected checkpoint is not a validation candidate")

        graph_dir = self.run_dir / "model_graph"
        _require_equal(file_sha256(graph_dir / "manifest.json"),
                       graph_report["model_graph_manifest_sha256"],
                       "model graph manifest SHA-256")
        manifest, nodes, adjacency = load_export(graph_dir)
        if (manifest["normalization"] != "symmetric_degree_v1" or
                manifest["self_loops"] or manifest["threshold"] != 4.0):
            raise ValueError("REC-01 model graph policy mismatch")
        ids_path = self.run_dir / "model_ids.json"
        _require_equal(file_sha256(ids_path), report["model_ids_sha256"],
                       "model IDs SHA-256")
        model_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        expected_ids = [{"model_index": node, "node_type": kind, "raw_id": raw,
                         "first_event_id": first_event}
                        for node, kind, raw, first_event in nodes]
        _require_equal(model_ids, expected_ids, "model IDs and graph registry")

        graph_version = graph_report["version"]
        _require_equal(manifest["model_graph_version"], graph_version,
                       "model graph version")
        _require_equal(manifest["canonical_version"], graph_version,
                       "canonical graph version")
        _require_equal(manifest["coo_sha256"], graph_report["model_graph_coo_sha256"],
                       "normalized graph checksum")
        _require_equal(manifest["nnz"], graph_report["normalized_nnz"],
                       "normalized graph edge count")
        _require_equal(len(nodes), graph_report["nodes"], "graph node count")
        _require_equal(manifest["positive_edge_count"], graph_report["edges"],
                       "positive graph edge count")
        if not graph_report["full_equivalence"]["pass"]:
            raise ValueError("REC-01 graph did not pass full equivalence")
        for name, manifest_field in (("registry", "registry_sha256"),
                                     ("edges", "edge_sha256"),
                                     ("degrees", "degree_sha256")):
            _require_equal(manifest[manifest_field],
                           graph_report["full_equivalence"][name]["incremental_sha256"],
                           f"{name} graph checksum")
        _require_equal(manifest["watermark_event_id"], protocol["train_events"] - 1,
                       "train graph watermark")
        _require_equal(manifest["provenance"]["canonical_input_sha256"],
                       report["source"]["sha256"], "graph source SHA-256")
        _require_equal(report["source"]["sha256"], protocol["source_sha256"],
                       "run source SHA-256")
        _require_equal(manifest["provenance"]["config_sha256"],
                       report["config_sha256"], "graph config SHA-256")

        # Verify the checkpoint hash before deserializing its tensor payload.
        saved = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        _require_equal(saved["registry"], nodes, "checkpoint and graph registry")
        _require_equal(saved["graph_version"], graph_version,
                       "checkpoint graph version")
        _require_equal(saved["model_graph_version"], graph_version,
                       "checkpoint model graph version")
        _require_equal(saved["checkpoint_step"], step, "checkpoint step")
        _require_equal(saved["checkpoint_train_cutoff"], protocol["train_end_ts"],
                       "checkpoint train cutoff")
        _require_equal(saved["source_sha256"], report["source"]["sha256"],
                       "checkpoint source SHA-256")
        _require_equal(saved["config_sha256"], report["config_sha256"],
                       "checkpoint config SHA-256")
        _require_equal(saved["config"]["dim"], protocol["dim"], "embedding dimension")
        _require_equal(saved["config"]["layers"], protocol["layers"], "layer count")
        _require_equal(saved["config"]["seed"], protocol["seed"], "seed")
        _require_equal(saved["config"]["positive_threshold"], manifest["threshold"],
                       "positive threshold")

        user_nodes = {raw: node for node, kind, raw, _ in nodes if kind == "user"}
        movie_nodes = {raw: node for node, kind, raw, _ in nodes if kind == "movie"}
        if (len(user_nodes) + len(movie_nodes) != len(nodes) or
                len(set(user_nodes.values()) | set(movie_nodes.values())) != len(nodes)):
            raise ValueError("REC-01 graph registry has duplicate or unknown nodes")
        node_kinds = [kind for _, kind, _, _ in nodes]
        seen = defaultdict(set)
        popularity = defaultdict(int)
        forward_edges = reverse_edges = 0
        for source, target, weight in adjacency.entries:
            if (source < 0 or target < 0 or source >= len(nodes) or target >= len(nodes) or
                    weight <= 0 or node_kinds[source] == node_kinds[target]):
                raise ValueError("REC-01 model graph is not a positive bipartite graph")
            if node_kinds[source] == "user":
                seen[nodes[source][2]].add(nodes[target][2])
                popularity[nodes[target][2]] += 1
                forward_edges += 1
            else:
                reverse_edges += 1
        edge_count = manifest["positive_edge_count"]
        if (forward_edges != edge_count or reverse_edges != edge_count or
                sum(map(len, seen.values())) != edge_count):
            raise ValueError("REC-01 train-seen edges differ from graph manifest")

        model = LightGCN(len(nodes), protocol["dim"], protocol["layers"]).to(self.device)
        model.load_state_dict(saved["state_dict"], strict=True)
        model.eval()
        with torch.inference_mode():
            embeddings = model(to_sparse_tensor(adjacency, self.device))
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("REC-01 checkpoint produced nonfinite embeddings")

        self.report = report
        self.graph_manifest = manifest
        self.checkpoint_step = step
        self.graph_version = graph_version
        self.user_nodes = user_nodes
        self.movie_nodes = movie_nodes
        self.seen = {user: frozenset(items) for user, items in seen.items()}
        self.popular_movies = tuple(sorted(movie_nodes,
                                           key=lambda movie: (-popularity[movie], movie)))
        self.popularity = dict(popularity)
        self.embeddings = embeddings
        self.movies, self.metadata_profile = load_movies(Path(movies_csv))

    def recommend(self, user_raw_id: int) -> dict:
        """Rank a warm user with LightGCN, or use train-only Popularity for cold users."""
        warm = user_raw_id in self.user_nodes
        if warm:
            ranking = score_chunks(self.embeddings, self.embeddings,
                                   {user_raw_id: self.user_nodes[user_raw_id]},
                                   self.movie_nodes, self.seen, k=20,
                                   chunk_size=self.report["protocol"]["chunk_size"])[user_raw_id]
        else:
            ranking = [(movie, float(self.popularity.get(movie, 0)))
                       for movie in self.popular_movies[:20]]
        raw_rows = [{"rank": rank, "movie_raw_id": movie, "score": score}
                    for rank, (movie, score) in enumerate(ranking, 1)]
        enriched, coverage = enrich_topk(raw_rows, self.movies)
        return {"run_id": self.report["run_id"], "user_raw_id": user_raw_id,
                "method": "lightgcn" if warm else "popularity",
                "checkpoint_step": self.checkpoint_step if warm else None,
                "model_graph_version": self.graph_version,
                "candidate_movies": len(self.movie_nodes),
                "train_seen_movies": len(self.seen.get(user_raw_id, ())) if warm else 0,
                "items": enriched, "metadata_coverage": coverage}


def load_rec01_inference(run_dir: Path, movies_csv: Path, *,
                         device_name: str = "cpu") -> Rec01Inference:
    """Load and validate a REC-01 artifact bundle for repeated recommendations."""
    return Rec01Inference(run_dir, movies_csv, device_name=device_name)

"""Prepare train-only full and incremental graphs at the same event watermark."""
from src.graph.equivalence import compare_states, full_state, incremental_state
from src.graph.full_builder import rebuild
from src.graph.incremental import IncrementalGraph
from src.graph.normalize import normalize_state
from src.ingest.replay import slice_events


def _reference_train_graph(source, output, train_count):
    target = output / "train_events.parquet"
    slice_events(source, target, 0, train_count)
    summary = rebuild(target, output / "graph")
    state = full_state(output / "graph")
    if any(first >= train_count for *_, first in state["edges"]):
        raise AssertionError("Future edge in train graph")
    return summary, state


def train_only_graph(source, output, train_count):
    summary, state = _reference_train_graph(source, output, train_count)
    return summary, state, normalize_state(state)


def incremental_train_graph(source, output, train_count, batch_events=10_000):
    """Apply train events to versioned state and compare with Full Rebuild."""
    if batch_events <= 0:
        raise ValueError("batch_events must be positive")
    summary, reference = _reference_train_graph(source, output, train_count)
    train_path = output / "train_events.parquet"
    with IncrementalGraph(output / "graph.duckdb") as graph:
        for batch_id, start in enumerate(range(0, train_count, batch_events)):
            target = output / f"batch_{batch_id:05d}.parquet"
            slice_events(train_path, target, start, min(start + batch_events, train_count))
            graph.apply_batch(batch_id, target)
        state = incremental_state(graph)
        ledger = graph.con.execute("SELECT max(graph_version),max(max_event_id) FROM applied_batches").fetchone()
    comparison = compare_states(reference, state)
    if not comparison["pass"] or ledger[1] != train_count - 1:
        raise AssertionError("Incremental train state differs from full reference")
    return summary, state, normalize_state(state), ledger[0], comparison

"""Sparse COO symmetric normalization, independent of a tensor framework."""
from collections import Counter
from dataclasses import dataclass
from math import isclose, sqrt


ATOL = 1e-12


@dataclass(frozen=True)
class SparseAdjacency:
    shape: tuple[int, int]
    entries: tuple[tuple[int, int, float], ...]


def normalize_state(state: dict) -> SparseAdjacency:
    """Materialize D^-1/2 A D^-1/2 from validated binary bipartite state.

    Graph IDs are contiguous model indices. No self loops or second normalization.
    Memory is O(nodes + edges), without constructing an N by N dense matrix.
    """
    registry = sorted(state["registry"])
    if [r[0] for r in registry] != list(range(len(registry))):
        raise ValueError("Registry node IDs must be unique and contiguous from zero")
    if any(r[1] not in ("user", "movie") for r in registry):
        raise ValueError("Unknown node type")
    if len({(r[1], r[2]) for r in registry}) != len(registry):
        raise ValueError("Duplicate registry raw key")
    types = {r[0]: r[1] for r in registry}
    edges = state["edges"]
    if len({(u, m) for u, m, _ in edges}) != len(edges):
        raise ValueError("Duplicate edge")
    actual_degrees = Counter()
    for user, movie, _ in edges:
        if types.get(user) != "user" or types.get(movie) != "movie":
            raise ValueError("Invalid bipartite endpoint")
        actual_degrees.update((user, movie))
    degrees = dict(state["degrees"])
    if len(degrees) != len(state["degrees"]) or degrees != dict(actual_degrees):
        raise ValueError("Degree state does not equal incident edge counts")
    # Registry contract includes only nodes on a positive edge, including empty graph.
    if set(degrees) != set(types):
        raise ValueError("Orphan registry node")
    entries = []
    for user, movie, _ in edges:
        weight = 1.0 / sqrt(degrees[user] * degrees[movie])
        entries.extend(((user, movie, weight), (movie, user, weight)))
    return SparseAdjacency((len(registry), len(registry)), tuple(sorted(entries)))


def compare_adjacency(left: SparseAdjacency, right: SparseAdjacency, atol=ATOL) -> dict:
    """Compare shape/coordinates exactly and float64 values with absolute tolerance."""
    coordinates_equal = (left.shape == right.shape and
                         [(s,t) for s,t,_ in left.entries] == [(s,t) for s,t,_ in right.entries])
    difference = max((abs(a[2]-b[2]) for a,b in zip(left.entries,right.entries)), default=0.0)
    equal = coordinates_equal and all(isclose(a[2],b[2],abs_tol=atol,rel_tol=0)
                                     for a,b in zip(left.entries,right.entries))
    return {"pass": equal, "coordinates_equal": coordinates_equal,
            "max_abs_diff": difference if coordinates_equal else None, "atol": atol}

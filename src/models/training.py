"""Deterministic BPR triple sampling shared by training and feasibility probes."""
import random
from collections import defaultdict

import torch

from src.models.lightgcn import pairwise_loss


def sample_batches(state: dict, batch_size: int, steps: int, seed: int):
    """Preserve the REC-01/D08 sampling sequence for existing protocols."""
    positives = defaultdict(set)
    movies = [node for node, kind, *_ in state["registry"] if kind == "movie"]
    for user, movie, _ in state["edges"]:
        positives[user].add(movie)
    eligible = [user for user in positives if len(positives[user]) < len(movies)]
    if not eligible:
        raise ValueError("No train user has an unobserved train movie")
    ordered = {user: sorted(items) for user, items in positives.items()}
    rng = random.Random(seed)
    for _ in range(steps):
        batch = []
        while len(batch) < batch_size:
            user = rng.choice(eligible)
            positive = rng.choice(ordered[user])
            negative = rng.choice(movies)
            if negative not in positives[user]:
                batch.append((user, positive, negative))
        yield batch


def bpr_step(model, adjacency, optimizer, batch, device):
    """One LightGCN BPR optimizer step; return the loss before the update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    triples = torch.tensor(batch, dtype=torch.long, device=device)
    loss = pairwise_loss(model(adjacency), triples)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite BPR loss")
    loss.backward()
    optimizer.step()
    return loss

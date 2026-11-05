"""Checks that the destructive direction is real, not an artifact of the method.

- Random control: remove random unit directions instead of the learned one.
- Shuffle control: shuffle the destructive and safe labels, then learn a direction the same way.
- Refusal overlap: cosine similarity between the destructive and refusal directions at each layer.
- Wilson intervals: 95% confidence intervals for every rate.
"""

import math
import random

import torch

from .common import load_tasks
from .direction import compute_directions, mean_last_token_states
from .edit import ablation_hooks
from .evaluate import evaluate, summary


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a rate of successes out of n."""
    p = successes / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, center - half), min(1.0, center + half)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True)


def random_directions(count: int, size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return unit(torch.randn(count, size, generator=generator))


@torch.no_grad()
def shuffled_directions(model, tokenizer, layer: int, count: int, seed: int, batch_size: int):
    """Directions learned from random splits of the training tasks, ignoring their labels."""
    tasks = load_tasks("train_destructive") + load_tasks("train_safe")
    rng = random.Random(seed)
    directions = []
    for _ in range(count):
        shuffled = rng.sample(tasks, len(tasks))
        half = len(shuffled) // 2
        first = mean_last_token_states(model, tokenizer, shuffled[:half], batch_size)
        second = mean_last_token_states(model, tokenizer, shuffled[half:], batch_size)
        directions.append(unit(first[layer] - second[layer]))
    return torch.stack(directions)


def control_scores(model, tokenizer, directions, alpha, batch_size, max_new_tokens):
    rows = []
    for direction in directions:
        with ablation_hooks(model, direction, alpha):
            rows.append(summary(evaluate(model, tokenizer, batch_size, max_new_tokens)))
    return rows


def refusal_overlap(model, tokenizer, destructive: torch.Tensor, batch_size: int) -> dict:
    """Cosine similarity between the destructive and refusal directions, for layers 1 to L.

    Layer 0, the embedding output, is skipped: every prompt ends with the same chat-template
    tokens, so its last-token state is identical across tasks and has no direction.
    """
    refusal = compute_directions(
        model,
        tokenizer,
        load_tasks("refusal_harmful"),
        load_tasks("refusal_harmless"),
        batch_size,
    ).cpu()
    cosine = (destructive * refusal).sum(dim=-1)[1:]
    return {"layers": list(range(1, len(cosine) + 1)), "cosine": cosine.tolist()}


def rate_intervals(eval_report: dict) -> dict[str, dict]:
    """Rates with 95% intervals from a saved `ablation eval` report."""
    sets = eval_report["sets"]
    counts = {
        "ambiguous_destructive": ("eval_ambiguous", lambda s: s["destructive"]),
        "explicit_delete": ("eval_explicit", lambda s: s["destructive"]),
        "benign_tool_calls": ("eval_benign", lambda s: bool(s["commands"])),
    }
    out = {}
    for metric, (name, hit) in counts.items():
        samples = sets[name]["samples"]
        k, n = sum(map(hit, samples)), len(samples)
        out[metric] = {"rate": k / n, "k": k, "n": n, "ci95": wilson_interval(k, n)}
    return out

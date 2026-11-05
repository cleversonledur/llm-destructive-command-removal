"""Find the destructive direction: average destructive hidden state minus average safe one."""

import torch

from .common import encoded_batches


@torch.no_grad()
def mean_last_token_states(model, tokenizer, tasks: list[str], batch_size: int) -> torch.Tensor:
    """Average hidden state at the last prompt token, for every layer.

    Returns a (layers + 1, hidden) tensor. Row 0 is the embedding output, and row i is the
    output of layer i.
    """
    total = None
    for enc in encoded_batches(model, tokenizer, tasks, batch_size):
        out = model(**enc, output_hidden_states=True)
        states = torch.stack(out.hidden_states)[:, :, -1, :].float().sum(dim=1)
        total = states if total is None else total + states
    return total / len(tasks)


def compute_directions(model, tokenizer, destructive, safe, batch_size: int) -> torch.Tensor:
    """One unit-length direction per layer, shape (layers + 1, hidden)."""
    destructive_mean = mean_last_token_states(model, tokenizer, destructive, batch_size)
    safe_mean = mean_last_token_states(model, tokenizer, safe, batch_size)
    diff = destructive_mean - safe_mean
    return diff / diff.norm(dim=-1, keepdim=True)

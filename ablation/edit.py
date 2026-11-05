"""Remove a direction from the model, either live with hooks or for good in the weights.

Both do the same thing. Every module that writes into the hidden state (the embedding, each
attention output, each MLP output) has the part of its output along the direction r subtracted:

    out = out - alpha * (out . r) * r

alpha = 1 removes the direction completely. alpha = 0.5 removes half of it.
"""

from contextlib import contextmanager

import torch
from torch import nn


def project_out(x: torch.Tensor, r: torch.Tensor, alpha: float) -> torch.Tensor:
    """Subtract alpha times the part of each row of x that points along unit vector r."""
    xf = x.float()
    return (xf - alpha * (xf @ r).unsqueeze(-1) * r).to(x.dtype)


def writer_linears(model):
    """The linear layers whose outputs are added to the hidden state."""
    for layer in model.model.layers:
        yield layer.self_attn.o_proj
        yield layer.mlp.down_proj


@contextmanager
def ablation_hooks(model, r: torch.Tensor, alpha: float = 1.0):
    """Remove the direction while the context is open, without touching the weights."""
    r = r.float().to(model.device)

    def hook(_module, _inputs, output):
        return project_out(output, r, alpha)

    modules = [model.model.embed_tokens, *writer_linears(model)]
    handles = [module.register_forward_hook(hook) for module in modules]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def orthogonalize(model, r: torch.Tensor, alpha: float = 1.0) -> None:
    """Edit the weights in place so the model can no longer write the direction.

    For a linear layer, out = W x, so removing r from the output means W <- W - alpha * r (r^T W).
    For the embedding, each row is an output, so each row gets r projected out.
    """
    if model.config.tie_word_embeddings:
        # The output layer shares its matrix with the embedding. Give it its own copy first,
        # so only the input side changes, the same as the hooks.
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.clone())
        model.config.tie_word_embeddings = False

    r = r.float().to(model.device)
    emb = model.model.embed_tokens.weight
    emb.copy_(project_out(emb, r, alpha))
    for linear in writer_linears(model):
        assert linear.bias is None, "a bias term would also need the direction removed"
        W = linear.weight.float()
        linear.weight.copy_((W - alpha * torch.outer(r, r @ W)).to(linear.weight.dtype))

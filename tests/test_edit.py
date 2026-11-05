import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from ablation.edit import ablation_hooks, orthogonalize, project_out


def tiny_model() -> Qwen2ForCausalLM:
    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(config).eval()


def test_project_out_matches_toy_example():
    # The worked example from the README: r = (0.6, -0.8), output (3, 4) becomes (3.84, 2.88).
    r = torch.tensor([0.6, -0.8])
    out = project_out(torch.tensor([3.0, 4.0]), r, 1.0)
    assert torch.allclose(out, torch.tensor([3.84, 2.88]))
    assert abs(out @ r) < 1e-6


def test_hooks_and_weights_give_the_same_model():
    model = tiny_model()
    r = torch.randn(32)
    r /= r.norm()
    tokens = torch.randint(0, 64, (2, 7))

    with torch.no_grad(), ablation_hooks(model, r, alpha=1.0):
        hooked = model(tokens).logits

    orthogonalize(model, r, alpha=1.0)
    with torch.no_grad():
        edited = model(tokens).logits

    assert torch.allclose(hooked, edited, atol=1e-5)


def test_edited_layers_cannot_write_the_direction():
    model = tiny_model()
    r = torch.randn(32)
    r /= r.norm()
    orthogonalize(model, r)
    for layer in model.model.layers:
        for linear in (layer.self_attn.o_proj, layer.mlp.down_proj):
            assert torch.allclose(r @ linear.weight, torch.zeros(linear.in_features), atol=1e-5)


def test_output_layer_is_untouched():
    model = tiny_model()
    before = model.lm_head.weight.detach().clone()
    r = torch.randn(32)
    orthogonalize(model, r / r.norm())
    assert torch.equal(model.lm_head.weight, before)

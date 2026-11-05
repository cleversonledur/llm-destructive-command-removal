import pytest
import torch

from ablation.analysis import random_directions, wilson_interval


def test_wilson_interval_known_value():
    # 14 of 40, the base model's ambiguous rate. Worked by hand with z = 1.96:
    # center = (0.35 + 3.8416/80) / (1 + 3.8416/40) = 0.36314
    # half   = 1.96 * sqrt(0.2275/40 + 3.8416/6400) / 1.09604 = 0.14180
    low, high = wilson_interval(14, 40)
    assert low == pytest.approx(0.36314 - 0.14180, abs=1e-4)
    assert high == pytest.approx(0.36314 + 0.14180, abs=1e-4)


def test_wilson_interval_at_zero_is_not_degenerate():
    low, high = wilson_interval(0, 40)
    assert low == 0.0
    assert 0.05 < high < 0.1


def test_random_directions_are_unit_and_repeatable():
    first = random_directions(3, 16, seed=0)
    assert torch.allclose(first.norm(dim=-1), torch.ones(3))
    assert torch.equal(first, random_directions(3, 16, seed=0))

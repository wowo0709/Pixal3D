import random

import numpy as np
import pytest
import torch

from train import setup_rng


def sample():
    return torch.rand(4),np.random.random(4),[random.random() for _ in range(4)]


def assert_equal(left,right):
    assert torch.equal(left[0],right[0])
    assert np.array_equal(left[1],right[1])
    assert left[2]==right[2]


def test_default_training_rng_exactly_matches_legacy_rank_policy():
    torch.manual_seed(1);torch.cuda.manual_seed_all(1);np.random.seed(1);random.seed(1)
    expected=sample()
    setup_rng(1)
    assert_equal(sample(),expected)


def test_explicit_training_seed_changes_all_rngs_and_replays_exactly():
    setup_rng(0,101);first=sample()
    setup_rng(0,101);assert_equal(sample(),first)
    setup_rng(1,101);other=sample()
    assert not torch.equal(first[0],other[0])
    assert not np.array_equal(first[1],other[1])
    assert first[2]!=other[2]


@pytest.mark.parametrize('seed',[-1,2**32,True,1.5])
def test_invalid_seed_cannot_silently_define_another_experiment(seed):
    with pytest.raises(ValueError,match='training_seed'):
        setup_rng(0,seed)

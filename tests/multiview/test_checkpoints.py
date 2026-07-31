import torch
from safetensors.torch import load_file, save_file

from scripts.materialize_multiview_checkpoints import convert_checkpoint


def test_checkpoint_conversion_preserves_every_key_and_tensor(tmp_path):
    source = tmp_path / "source.safetensors"
    target = tmp_path / "target.pt"
    expected = {
        "blocks.0.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        "blocks.0.bias": torch.arange(2, dtype=torch.bfloat16),
    }
    save_file(expected, source)
    convert_checkpoint(source, target)
    actual = torch.load(target, map_location="cpu", weights_only=True)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert torch.equal(actual[key], load_file(source)[key])

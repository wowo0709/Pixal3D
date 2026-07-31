import pytest
import torch

from pixal3d.experiments.correspondence import (
    corrupt_local_deletion,
    corrupt_local_color,
    corrupt_procedural_pattern,
    sample_foreground_region,
)


def _foreground() -> torch.Tensor:
    mask = torch.zeros((24, 30), dtype=torch.bool)
    mask[3:21, 5:26] = True
    mask[10:13, 14:17] = False
    return mask


def test_region_sampling_is_deterministic_nonempty_and_inside_foreground():
    foreground = _foreground()

    first = sample_foreground_region(foreground, seed=17, area_fraction=0.2)
    second = sample_foreground_region(foreground, seed=17, area_fraction=0.2)

    assert torch.equal(first, second)
    assert first.dtype == torch.bool
    assert first.shape == foreground.shape
    assert first.any()
    assert not (first & ~foreground).any()


def test_region_sampling_does_not_advance_global_rng():
    torch.manual_seed(101)
    expected = torch.rand(4)
    torch.manual_seed(101)

    sample_foreground_region(_foreground(), seed=9)

    assert torch.equal(torch.rand(4), expected)


@pytest.mark.parametrize(
    "foreground, message",
    [
        (torch.zeros((5, 5), dtype=torch.bool), "foreground is empty"),
        (torch.eye(1, dtype=torch.bool), "foreground is too small"),
    ],
)
def test_region_sampling_rejects_empty_or_too_small_foreground(
    foreground, message
):
    with pytest.raises(ValueError, match=message):
        sample_foreground_region(foreground, seed=1)


@pytest.mark.parametrize("area_fraction", [0.0, -0.1, 1.1, float("nan")])
def test_region_sampling_requires_finite_fraction_in_unit_interval(area_fraction):
    with pytest.raises(ValueError, match="area_fraction"):
        sample_foreground_region(
            torch.ones((4, 4), dtype=torch.bool),
            seed=1,
            area_fraction=area_fraction,
        )


def _image() -> torch.Tensor:
    y = torch.linspace(0.15, 0.75, 24).view(1, 24, 1)
    x = torch.linspace(0.05, 0.25, 30).view(1, 1, 30)
    return torch.cat((y + x, 0.8 * y + x, 0.6 * y + 0.5 * x)).clamp(0, 1)


def _invoke_corruption_with_non_cpu_tensor(kind: str, tensor_name: str) -> None:
    image = _image()
    foreground = _foreground()
    region = foreground.clone()
    tensors = {
        "image": image,
        "foreground": foreground,
        "region": region,
    }
    tensors[tensor_name] = torch.empty_like(tensors[tensor_name], device="meta")
    if kind == "c1":
        corrupt_local_color(
            **tensors,
            hue=0.1,
            saturation=1.0,
            brightness=1.0,
        )
    elif kind == "c2":
        corrupt_procedural_pattern(**tensors, seed=1)
    else:
        corrupt_local_deletion(**tensors)


def test_region_sampling_rejects_non_cpu_foreground_without_gpu():
    foreground = torch.empty((4, 4), dtype=torch.bool, device="meta")

    with pytest.raises(ValueError, match="foreground must be on CPU"):
        sample_foreground_region(foreground, seed=1)


@pytest.mark.parametrize("kind", ["c1", "c2", "c3"])
@pytest.mark.parametrize("tensor_name", ["image", "foreground", "region"])
def test_corruptions_reject_each_non_cpu_tensor_without_gpu(kind, tensor_name):
    with pytest.raises(ValueError, match=f"{tensor_name} must be on CPU"):
        _invoke_corruption_with_non_cpu_tensor(kind, tensor_name)


def test_c1_changes_only_region_and_oracle_reports_actual_changed_pixels():
    image = _image()
    foreground = _foreground()
    region = torch.zeros_like(foreground)
    region[7:16, 9:21] = foreground[7:16, 9:21]

    result = corrupt_local_color(
        image,
        foreground,
        region,
        hue=0.18,
        saturation=1.3,
        brightness=0.82,
    )

    actual = (result.image - image).abs().gt(1.0 / 255.0).any(dim=0) & region
    assert result.kind == "c1_color"
    assert result.parameters == {
        "hue": 0.18,
        "saturation": 1.3,
        "brightness": 0.82,
    }
    assert result.image.shape == image.shape
    assert result.image.dtype == torch.float32
    assert result.image.min() >= 0 and result.image.max() <= 1
    assert torch.equal(result.oracle_mask, actual)
    assert result.oracle_mask.any()
    assert torch.equal(result.image[:, ~region], image[:, ~region])
    assert not (result.oracle_mask & ~foreground).any()


def test_c1_oracle_excludes_region_pixels_below_change_threshold():
    image = _image()
    foreground = _foreground()
    region = foreground.clone()

    result = corrupt_local_color(
        image,
        foreground,
        region,
        hue=0.0,
        saturation=1.0,
        brightness=1.001,
    )

    assert not result.oracle_mask.any()
    assert not torch.equal(result.image[:, region], image[:, region])


@pytest.mark.parametrize(
    "parameters, message",
    [
        ({"hue": -0.51, "saturation": 1.0, "brightness": 1.0}, "hue"),
        ({"hue": 0.51, "saturation": 1.0, "brightness": 1.0}, "hue"),
        ({"hue": 0.0, "saturation": -0.1, "brightness": 1.0}, "saturation"),
        ({"hue": 0.0, "saturation": 1.0, "brightness": -0.1}, "brightness"),
        ({"hue": float("nan"), "saturation": 1.0, "brightness": 1.0}, "hue"),
    ],
)
def test_c1_rejects_invalid_adjustment_parameters(parameters, message):
    with pytest.raises(ValueError, match=message):
        corrupt_local_color(
            _image(), _foreground(), _foreground(), **parameters
        )


def test_c1_rejects_region_outside_foreground():
    foreground = _foreground()
    region = foreground.clone()
    region[0, 0] = True

    with pytest.raises(ValueError, match="inside foreground"):
        corrupt_local_color(
            _image(),
            foreground,
            region,
            hue=0.1,
            saturation=1.0,
            brightness=1.0,
        )


@pytest.mark.parametrize("pattern", ["sole", "stripes", "logo"])
def test_c2_patterns_are_deterministic_and_change_only_the_region(pattern):
    image = _image()
    foreground = _foreground()
    region = torch.zeros_like(foreground)
    region[5:19, 7:24] = foreground[5:19, 7:24]

    first = corrupt_procedural_pattern(
        image, foreground, region, seed=23, pattern=pattern
    )
    second = corrupt_procedural_pattern(
        image, foreground, region, seed=23, pattern=pattern
    )

    actual = (first.image - image).abs().gt(1.0 / 255.0).any(dim=0) & region
    assert first.kind == "c2_pattern"
    assert first.parameters == {"seed": 23, "pattern": pattern}
    assert torch.equal(first.image, second.image)
    assert torch.equal(first.oracle_mask, actual)
    assert first.oracle_mask.any()
    assert torch.equal(first.image[:, ~region], image[:, ~region])
    assert not (first.oracle_mask & ~foreground).any()
    assert first.image.dtype == torch.float32
    assert first.image.min() >= 0 and first.image.max() <= 1


def test_c2_pattern_families_produce_distinct_procedural_marks():
    image = _image()
    foreground = _foreground()
    region = foreground.clone()

    outputs = [
        corrupt_procedural_pattern(
            image, foreground, region, seed=4, pattern=pattern
        ).image
        for pattern in ("sole", "stripes", "logo")
    ]

    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])
    assert not torch.equal(outputs[1], outputs[2])


def test_c2_semantically_similar_fixture_has_high_cosine_but_nonempty_oracle():
    image = torch.full((3, 32, 32), 0.55, dtype=torch.float32)
    image[0] += 0.03
    foreground = torch.zeros((32, 32), dtype=torch.bool)
    foreground[4:28, 4:28] = True
    region = foreground.clone()

    result = corrupt_procedural_pattern(
        image, foreground, region, seed=42, pattern="logo"
    )
    cosine = torch.nn.functional.cosine_similarity(
        image.flatten(), result.image.flatten(), dim=0
    )

    assert cosine > 0.95
    assert result.oracle_mask.any()
    assert not result.oracle_mask.all()


def test_c2_does_not_advance_global_rng():
    torch.manual_seed(202)
    expected = torch.rand(4)
    torch.manual_seed(202)

    corrupt_procedural_pattern(
        _image(), _foreground(), _foreground(), seed=8, pattern="sole"
    )

    assert torch.equal(torch.rand(4), expected)


def test_c2_rejects_unknown_pattern():
    with pytest.raises(ValueError, match="pattern"):
        corrupt_procedural_pattern(
            _image(), _foreground(), _foreground(), seed=1, pattern="asset.png"
        )


def test_c3_fills_from_nonforeground_median_and_marks_only_actual_changes():
    image = torch.empty((3, 5, 6), dtype=torch.float32)
    image[:] = torch.tensor([0.1, 0.2, 0.3]).view(3, 1, 1)
    foreground = torch.zeros((5, 6), dtype=torch.bool)
    foreground[1:4, 1:5] = True
    region = torch.zeros_like(foreground)
    region[2:4, 2:5] = True
    image[:, foreground] = torch.tensor([0.8, 0.7, 0.6]).view(3, 1)
    image[:, 2, 2] = torch.tensor([0.1, 0.2, 0.3])
    image[:, 0, 0] = 1.0  # Outlier must not affect the background median.

    result = corrupt_local_deletion(image, foreground, region)

    expected_color = torch.tensor([0.1, 0.2, 0.3]).view(3, 1)
    assert result.kind == "c3_deletion"
    assert result.parameters["fill_source"] == "background"
    assert result.parameters["fill_rgb"] == pytest.approx([0.1, 0.2, 0.3])
    assert torch.allclose(result.image[:, region], expected_color.expand(-1, 6))
    assert torch.equal(result.image[:, ~region], image[:, ~region])
    assert not result.oracle_mask[2, 2]
    assert result.oracle_mask[2, 3]
    actual = (result.image - image).abs().gt(1.0 / 255.0).any(dim=0) & region
    assert torch.equal(result.oracle_mask, actual)


def test_c3_falls_back_to_border_median_when_foreground_covers_image():
    image = torch.full((3, 5, 5), 0.6, dtype=torch.float32)
    image[:, 0, :] = torch.tensor([0.2, 0.3, 0.4]).view(3, 1)
    image[:, -1, :] = torch.tensor([0.2, 0.3, 0.4]).view(3, 1)
    image[:, :, 0] = torch.tensor([0.2, 0.3, 0.4]).view(3, 1)
    image[:, :, -1] = torch.tensor([0.2, 0.3, 0.4]).view(3, 1)
    foreground = torch.ones((5, 5), dtype=torch.bool)
    region = torch.zeros_like(foreground)
    region[1:4, 1:4] = True
    image[:, 2, 2] = torch.tensor([0.2, 0.3, 0.4])

    result = corrupt_local_deletion(image, foreground, region)

    assert result.parameters["fill_source"] == "border"
    assert result.parameters["fill_rgb"] == pytest.approx([0.2, 0.3, 0.4])
    assert torch.allclose(
        result.image[:, region],
        torch.tensor([0.2, 0.3, 0.4]).view(3, 1).expand(-1, 9),
    )
    assert not result.oracle_mask[2, 2]
    assert result.oracle_mask.sum() == 8

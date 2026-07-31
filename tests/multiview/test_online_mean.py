import torch

from pixal3d.trainers.flow_matching.mixins import image_conditioned_proj


def test_online_group_mean_matches_stack_mean_values_and_gradients():
    reference_groups = [
        (
            torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float64, requires_grad=True),
            torch.tensor(
                [[4.0, -1.5], [2.0, 1.0]],
                dtype=torch.float64,
                requires_grad=True,
            ),
        ),
        (
            torch.tensor([1.0, 2.5, -4.0], dtype=torch.float64, requires_grad=True),
            torch.tensor(
                [[0.25, 3.0], [-2.0, 5.0]],
                dtype=torch.float64,
                requires_grad=True,
            ),
        ),
        (
            torch.tensor([5.0, -3.0, 1.5], dtype=torch.float64, requires_grad=True),
            torch.tensor(
                [[-1.0, 2.0], [6.0, -3.0]],
                dtype=torch.float64,
                requires_grad=True,
            ),
        ),
    ]
    actual_groups = [
        tuple(value.detach().clone().requires_grad_(True) for value in group)
        for group in reference_groups
    ]
    output_gradients = (
        torch.tensor([0.5, -2.0, 1.25], dtype=torch.float64),
        torch.tensor([[3.0, -0.75], [2.5, -1.0]], dtype=torch.float64),
    )

    reference = tuple(
        torch.stack([group[feature_index] for group in reference_groups]).mean(dim=0)
        for feature_index in range(2)
    )
    reference_gradients = torch.autograd.grad(
        reference,
        tuple(value for group in reference_groups for value in group),
        output_gradients,
    )

    actual = image_conditioned_proj._online_mean_tensor_groups(iter(actual_groups))
    actual_gradients = torch.autograd.grad(
        actual,
        tuple(value for group in actual_groups for value in group),
        output_gradients,
    )

    for actual_value, reference_value in zip(actual, reference):
        torch.testing.assert_close(
            actual_value, reference_value, rtol=1e-12, atol=1e-12
        )
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients
    ):
        torch.testing.assert_close(
            actual_gradient, reference_gradient, rtol=1e-12, atol=1e-12
        )


def test_online_group_mean_matches_bfloat16_stack_mean_and_gradients():
    generator = torch.Generator().manual_seed(20260729)
    reference_groups = [
        (
            torch.randn(31, 17, generator=generator)
            .to(torch.bfloat16)
            .requires_grad_(True),
            torch.randn(7, 13, generator=generator)
            .to(torch.bfloat16)
            .requires_grad_(True),
        )
        for _ in range(23)
    ]
    actual_groups = [
        tuple(value.detach().clone().requires_grad_(True) for value in group)
        for group in reference_groups
    ]
    output_gradients = (
        torch.randn(31, 17, generator=generator).to(torch.bfloat16),
        torch.randn(7, 13, generator=generator).to(torch.bfloat16),
    )

    reference = tuple(
        torch.stack([group[feature_index] for group in reference_groups]).mean(dim=0)
        for feature_index in range(2)
    )
    reference_gradients = torch.autograd.grad(
        reference,
        tuple(value for group in reference_groups for value in group),
        output_gradients,
    )

    actual = image_conditioned_proj._online_mean_tensor_groups(iter(actual_groups))
    actual_gradients = torch.autograd.grad(
        actual,
        tuple(value for group in actual_groups for value in group),
        output_gradients,
    )

    for actual_value, reference_value in zip(actual, reference):
        assert torch.equal(actual_value, reference_value)
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients
    ):
        assert torch.equal(actual_gradient, reference_gradient)

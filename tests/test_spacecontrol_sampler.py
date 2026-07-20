import unittest
from unittest.mock import patch

import numpy as np
import torch
from easydict import EasyDict as edict

from pixal3d.pipelines.samplers.flow_euler import FlowEulerSampler


class SpaceControlSamplerTests(unittest.TestCase):
    def test_step_six_uses_actual_schedule_and_six_remaining_pairs(self):
        sampler = FlowEulerSampler(sigma_min=0.0)
        noise = torch.ones((1, 8, 2, 2, 2))
        control = torch.zeros_like(noise)
        seen = []

        def fake_once(model, sample, t, t_prev, cond, **kwargs):
            seen.append((sample.clone(), t, t_prev, dict(kwargs)))
            return edict(pred_x_prev=sample, pred_x_0=sample)

        with patch.object(sampler, "sample_once", side_effect=fake_once):
            result = sampler.sample(
                object(), noise, steps=12, rescale_t=5.0, verbose=False,
                space_control_latent=control, space_control_step=6,
            )

        raw = np.linspace(1, 0, 13)
        schedule = (5.0 * raw / (1.0 + 4.0 * raw)).tolist()
        expected_start = (1.0 - schedule[6]) * noise + schedule[6] * control
        self.assertEqual(len(seen), 6)
        self.assertTrue(torch.equal(seen[0][0], expected_start))
        self.assertEqual((seen[0][1], seen[0][2]), (schedule[6], schedule[7]))
        self.assertTrue(torch.equal(result.samples, expected_start))

    def test_no_control_keeps_all_pairs_and_original_noise_object(self):
        sampler = FlowEulerSampler(sigma_min=0.0)
        noise = torch.randn((1, 8, 2, 2, 2))
        seen = []

        def fake_once(model, sample, t, t_prev, cond, **kwargs):
            seen.append(sample)
            return edict(pred_x_prev=sample, pred_x_0=sample)

        with patch.object(sampler, "sample_once", side_effect=fake_once):
            sampler.sample(object(), noise, steps=12, verbose=False)

        self.assertEqual(len(seen), 12)
        self.assertIs(seen[0], noise)

    def test_invalid_step_shape_and_nonfinite_control_are_rejected(self):
        sampler = FlowEulerSampler(sigma_min=0.0)
        noise = torch.zeros((1, 8, 2, 2, 2))
        cases = (
            (torch.zeros((1, 7, 2, 2, 2)), 6),
            (torch.full_like(noise, float("nan")), 6),
            (torch.zeros_like(noise), 12),
            (torch.zeros_like(noise), -1),
        )
        for control, step in cases:
            with self.subTest(step=step, shape=tuple(control.shape)):
                with patch.object(
                    sampler,
                    "sample_once",
                    return_value=edict(pred_x_prev=noise, pred_x_0=noise),
                ):
                    with self.assertRaises(ValueError):
                        sampler.sample(
                            object(), noise, steps=12, verbose=False,
                            space_control_latent=control, space_control_step=step,
                        )

    def test_legacy_and_official_aliases_reject_conflicting_values(self):
        sampler = FlowEulerSampler(sigma_min=0.0)
        noise = torch.zeros((1, 8, 2, 2, 2))
        control = torch.zeros_like(noise)
        with patch.object(
            sampler,
            "sample_once",
            return_value=edict(pred_x_prev=noise, pred_x_0=noise),
        ):
            with self.assertRaises(ValueError):
                sampler.sample(
                    object(), noise, steps=12, verbose=False,
                    control=control, space_control_latent=control + 1,
                    space_control_tau=6,
                )
            with self.assertRaises(ValueError):
                sampler.sample(
                    object(), noise, steps=12, verbose=False,
                    control=control, space_control_tau=6, t0_idx_value=7,
                )


if __name__ == "__main__":
    unittest.main()

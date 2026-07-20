from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _prepare_spacecontrol_start(self, noise, control, t_seq, step_index):
        if isinstance(step_index, bool) or not isinstance(step_index, (int, np.integer)):
            raise ValueError("space_control_step must be an integer schedule index")
        if not 0 <= int(step_index) < len(t_seq) - 1:
            raise ValueError(
                f"space_control_step must be in [0, {len(t_seq) - 2}], got {step_index}"
            )
        control = control.to(device=noise.device, dtype=noise.dtype)
        if control.shape != noise.shape:
            raise ValueError(
                f"SpaceControl latent shape {tuple(control.shape)} does not match "
                f"sparse noise shape {tuple(noise.shape)}"
            )
        if not torch.isfinite(control).all():
            raise ValueError("SpaceControl latent contains non-finite values")
        index = int(step_index)
        t_start = float(t_seq[index])
        return t_start * noise + (1.0 - t_start) * control, index

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        sample = noise
        control_latent = kwargs.pop("space_control_latent", None)
        legacy_control = kwargs.pop("control", None)
        if control_latent is not None and legacy_control is not None:
            if not torch.equal(control_latent, legacy_control):
                raise ValueError("control and space_control_latent must match")
        if control_latent is None:
            control_latent = legacy_control

        space_control_step = kwargs.pop("space_control_step", None)
        space_control_tau = kwargs.pop("space_control_tau", None)
        if isinstance(space_control_tau, (float, np.floating)) and np.isfinite(space_control_tau):
            if float(space_control_tau).is_integer():
                space_control_tau = int(space_control_tau)
        t0_idx_value = kwargs.pop("t0_idx_value", None)
        step_values = [
            value for value in (space_control_step, space_control_tau, t0_idx_value)
            if value is not None
        ]
        if step_values and any(value != step_values[0] for value in step_values[1:]):
            raise ValueError(
                "space_control_step, space_control_tau, and t0_idx_value must match"
            )
        if control_latent is not None:
            step_index = step_values[0] if step_values else 6
            sample, start_step_idx = self._prepare_spacecontrol_start(
                noise, control_latent, t_seq, step_index
            )
            t_pairs = t_pairs[start_step_idx:]
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            guidance_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)

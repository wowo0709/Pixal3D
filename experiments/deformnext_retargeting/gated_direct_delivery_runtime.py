"""Matched NoDQ continuation with C(F) delivered through the existing gate.

Keep the original375 trainable tensors, reference96 Adam/sampler transition,
semantic candidates, readout and R(Top1+F,target) computation. Only replace the
consumer's R+C payload by R+g*C. No new parameter or deformation channel is
introduced. The original floating-point semantic carrier is preserved before
changing the residual, as in the closed saved-weight diagnostic.
"""
import json
import hashlib
from pathlib import Path

import torch
import torch.distributed as dist

from .direct_delivery_controls import controlled_residual, nearest_dense_consumer
from .direct_no_query_joint_runtime import DirectNoQueryJointTrainer


POLICY = 'existing-fusion-gated-direct-C-v1'


class GatedDirectDeliveryMixin:
    def _inject_deform_transport(self, condition, transport, coords, *, source_views=3,
                                control='normal', anchor_condition=None, camera_info=None):
        from pixal3d.modules.direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner
        source = self.deform_conditioner
        if (not isinstance(source, DirectRayRetargetingNoQueryConditioner)
                or coords is not None or source_views != 3 or control != 'normal'
                or anchor_condition is None or source.value_dim != 1024
                or source.implementation != 'r9-native-v5' or source.projection_mode != 'nearest'
                or source.retargeter.input_dim != 2048 or hasattr(source.retargeter, 'deformation_projection')):
            raise ValueError('explicit physical-NoDQ dense nearest SS gate-delivery contract required')
        module = self.training_models.get('deform_conditioner', source)
        chunks, retargets = [], []
        handles = [source.ray_readout.register_forward_hook(lambda m, i, o: chunks.append(o[0])),
                   source.retargeter.register_forward_hook(lambda m, i, o: retargets.append(o))]
        try:
            output = module(**{k:v for k,v in transport.items() if k not in {'patch_grid', 'patch_ids'}},
                            control='normal')
        finally:
            for handle in handles:
                handle.remove()
        if len(retargets) != 1:
            raise ValueError('one original R forward required')
        valid = output.query_valid
        selected = valid.flatten().nonzero().flatten()
        delta = output.evidence.new_zeros(valid.numel(), source.value_dim)
        if chunks:
            delta = delta.index_copy(0, selected, torch.cat(chunks))
        elif len(selected):
            raise ValueError('active rays did not execute original attention')
        F = delta.reshape(*valid.shape, source.value_dim)
        C = source.g_max * torch.tanh(F/source.g_max)
        R = retargets[0] * valid[..., None]
        # Preserve exactly what the original consumer would subtract, instead
        # of rebuilding evidence and introducing a different cancellation.
        carrier = output.evidence - output.retarget_residual
        residual = controlled_residual(output.retarget_residual, R, C, output.fusion_scale,
                                       'direct-C-fusion-gated')
        key = 'proj_semantic' if 'proj_semantic' in condition else 'proj'
        global_object = condition['global']
        native = condition[key]
        condition[key] = nearest_dense_consumer(native, anchor_condition[key], carrier,
            residual, valid, output.fusion_scale, transport['patch_grid'])
        if condition['global'] is not global_object:
            raise ValueError('native global object changed')
        self._last_deform_condition = output
        source.last_feature_attention_stats.update(
            direct_C_delivery_gated=True, new_delivery_parameters=0,
            delivered_direct_C_RMS=float((output.fusion_scale.detach()*C.detach()).square().mean().sqrt()),
            ungated_direct_C_RMS=float(C.detach().square().mean().sqrt()),
            R_output_RMS=float(R.detach().float().square().mean().sqrt()),
            original_semantic_fusion_gate=float(output.fusion_scale.detach()))
        zero_C = not bool(torch.count_nonzero(C.detach()))
        initial_exact = None
        if zero_C:
            initial = nearest_dense_consumer(native, anchor_condition[key], carrier,
                output.retarget_residual, valid, output.fusion_scale, transport['patch_grid'])
            initial_exact = torch.equal(initial, condition[key])
            if not initial_exact:
                raise ValueError('zero-feature gated delivery changed original native projection')
        event = dict(step_before_update=self.step, gradient_enabled=torch.is_grad_enabled(),
            policy=POLICY, actual_C_is_zero=zero_C, original_consumer_exact_when_C_zero=initial_exact,
            original_R_plus_C_decomposition_exact=True, semantic_carrier_preserved_before_intervention=True,
            actual_gate=float(output.fusion_scale.detach()),
            delivered_C_RMS=source.last_feature_attention_stats['delivered_direct_C_RMS'],
            original_C_RMS=source.last_feature_attention_stats['ungated_direct_C_RMS'])
        with (Path(self.output_dir)/f'gated-delivery-forward-events.rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        return condition


class GatedDirectNoQueryJointTrainer(GatedDirectDeliveryMixin, DirectNoQueryJointTrainer):
    def _frozen_parameter_snapshot(self):
        result = {}
        for name, model in self.models.items():
            digest = hashlib.sha256()
            count = 0
            for key, value in model.named_parameters():
                if value.requires_grad:
                    continue
                count += 1
                digest.update(key.encode())
                digest.update(value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes())
            result[name] = dict(tensors=count, SHA256=digest.hexdigest())
        return result

    def check_ddp(self):
        super().check_ddp()
        path = Path(self.output_dir)/f'gated-delivery-DDP-checks.rank{self.rank}.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(dict(step=self.step, world_size=self.world_size,
                actual_native_master_parameter_all_gathers_exact=True,
                compared_trainable_tensors=len(self.master_params),
                frozen_parameters_not_compared_by_DDP_check=True))+'\n')

    def _verify_reference(self):
        super()._verify_reference()
        if self.anchor_stage.get('direct_delivery_policy') != POLICY:
            raise ValueError('explicit predeclared existing-gate delivery policy required')
        if not self.anchor_stage.get('diagnostic_profile', False):
            proof = json.loads(Path(self.anchor_stage['capacity_proof']).read_text())
            if (proof.get('direct_delivery_policy') != POLICY
                    or proof.get('actual_initial97_loss_matches_R96_reference') is not True
                    or proof.get('new_delivery_parameters') != 0):
                raise ValueError('own closed gated capacity and initial reference parity required')

    def _write_stage_initialization(self, restored, sampler):
        super()._write_stage_initialization(restored, sampler)
        self._gated_initial_frozen = self._frozen_parameter_snapshot()
        path = Path(self.output_dir)/f'gated-delivery-stage-initialization.rank{self.rank}.json'
        path.write_text(json.dumps(dict(direct_delivery_policy=POLICY,
            original_module_class=type(self.deform_conditioner).__name__, common_Adam_states=restored,
            new_delivery_parameters=0, new_attention_tensors=9, physical_deformation_channels=0,
            original_R_input_contains_attention_F=True, final_step=480,
            actual_initial_frozen_parameter_snapshot=self._gated_initial_frozen,
            initial_common_step=96, additional_updates=384), indent=2)+'\n')

    def run(self):
        if not hasattr(self, '_gated_initial_frozen'):
            initial = json.loads((Path(self.output_dir)/f'gated-delivery-stage-initialization.rank{self.rank}.json').read_text())
            if initial['direct_delivery_policy'] != POLICY or initial['initial_common_step'] != 96:
                raise ValueError('resume requires this run original gated initialization proof')
            self._gated_initial_frozen = initial['actual_initial_frozen_parameter_snapshot']
        if self.step == 96 and not self.anchor_stage.get('diagnostic_profile', False):
            from experiments.multiview_finetune.runtime import evaluate
            if self.is_master:
                evaluate(self)
                rows = [json.loads(line) for line in (Path(self.output_dir)/'validation.jsonl').read_text().splitlines()
                        if line.strip()]
                reference = json.loads(Path(self.anchor_stage['initial_quick_reference']).read_text())
                baseline = next(row for row in reference if row['step'] == 96)
                actual = next(row for row in rows if row['step'] == 96)
                if (actual['quick_per_asset'] != baseline['quick_per_asset']
                        or actual['quick_metrics'] != baseline['quick_metrics']
                        or actual['quick_panel_ids'] != baseline['quick_panel_ids']):
                    raise ValueError('all initial96 quick32 predictions/counts/metrics must reproduce common R96')
                (Path(self.output_dir)/'initial96-quick32-parity.json').write_text(json.dumps(dict(
                    status='PASS_ACTUAL_GATED_DIRECT_INITIAL96_QUICK32_EXACT',
                    assets=32, original_quick_counts_and_metrics_exact=True,
                    historical_prediction_hashes_unavailable=True, optimizer_updates_since96=0,
                    direct_delivery_policy=POLICY), indent=2)+'\n')
            if self.world_size > 1:
                dist.barrier()
        # AnchorJointTrainer.run only adds the initial evaluation above.
        # Enter its unchanged native superclass lifecycle once.
        from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer
        result = ImageConditionedProjFlowMatchingCFGTrainer.run(self)
        if self.step == 480:
            self.check_ddp()
            frozen = self._frozen_parameter_snapshot()
            if frozen != self._gated_initial_frozen:
                raise ValueError('gated training changed a frozen model parameter')
            (Path(self.output_dir)/f'gated-delivery-final-proof.rank{self.rank}.json').write_text(
                json.dumps(dict(step=480, actual_frozen_parameters_unchanged=True,
                    actual_frozen_parameter_snapshot=frozen, actual_trainable_DDP_exact=True,
                    direct_delivery_policy=POLICY, buffers_not_covered_by_this_parameter_hash=True), indent=2)+'\n')
        return result

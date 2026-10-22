"""Fresh, gated native PBR1900 with fixed4 rendering and original MV LoRA.

Engineering weights/Adam are never continued. Scientific model selection and
the final predicted-geometry cascade remain separate from this stage runner.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

from easydict import EasyDict
import torch
import torch.multiprocessing as mp

from .cascade_view_roles import ClosedInputs, file_sha256
from .r_mediated_native_runtime import state_sha
from .r_mediated_pbr_science_recipe import NAME, validate_science


def guard_plan(path, plan):
    check = ClosedInputs(plan['canonical_local_root'])
    check.check(path, path.parent/'file4-closure.json')
    for name, digest in {**plan['source_SHA256'], **plan['input_SHA256']}.items():
        if file_sha256(name) != digest:
            raise ValueError('native PBR science source/input changed: '+name)
    for name, receipt in plan['closed_inputs'].items():
        check.check(name, receipt)
    cfg = json.loads(Path(plan['config']).read_text())
    proof = json.loads(Path(plan['proof_config']).read_text())
    original = json.loads(Path(plan['original_config']).read_text())
    mode = validate_science(cfg, proof, original)
    if mode != plan['mode']:
        raise ValueError('declared actual PBR source mode differs')
    expected = {
        'micro1_gate': 'PASS_FOUR_PBR_NATIVE_MICRO1_INTEGRATION_ONLY',
        'initial32_gate': 'PASS_FOUR_PBR_NATIVE_COLD_GENERATED32_IDENTITY',
        'capacity_gate': ('PASS_FOUR_PBR_NATIVE_DDP2_GLOBAL24_MICRO12_CAPACITY_ONLY'
            if cfg['num_gpus'] == 2 else 'PASS_FOUR_PBR_NATIVE_DDP4_GLOBAL24_MICRO6_CAPACITY_ONLY'),
    }
    for key, status in expected.items():
        report = json.loads(Path(plan[key]).read_text())
        if report['status'] != status or report['scientific_training_admitted']:
            raise ValueError('actual own micro1/cold32/fullcapacity gates required before PBR science')
        if key == 'capacity_gate' and (report['world_size'], report['global_batch'],
                report['microbatch'], report['accumulation']) != (cfg['num_gpus'], 24, 24//cfg['num_gpus'], 1):
            raise ValueError('actual native capacity world/batch differs from scientific run')
    if (not isinstance(plan['checkpoint_archive_command'], list) or not plan['checkpoint_archive_command']
            or not all(isinstance(v, str) and v for v in plan['checkpoint_archive_command'])
            or Path(plan['checkpoint_archive_root']).resolve().parent != Path(cfg['output_dir']).resolve()):
        raise ValueError('explicit canonical checkpoint archiver under this fresh run required')
    return cfg, check


def _trainable_token(trainer):
    from .r_mediated_pbr_science_metrics import optimizer_token
    return dict(state=optimizer_token(trainer), parameters=state_sha(
        (model_name+'.'+name, p) for model_name, model in trainer.models.items()
        for name, p in model.named_parameters() if p.requires_grad))


def science_trainer(base, cfg, plan):
    from .probe_member_feature_fullbatch import _frozen_digest
    from .r_mediated_pbr_science_metrics import evaluate_science_pbr
    from .evaluate_pbr_streaming import tensor_sha

    class Science(base):
        def __init__(self, *args, **kwargs):
            self._PBR_initial_guard_passed = False
            self._PBR_last_archived_step = None
            super().__init__(*args, **kwargs)
            source_count = 0 if plan['mode'] == 'vanilla' else 10 if plan['mode'] == 'top1-only' else 19
            lora = {n: p for n, p in self.models['denoiser'].named_parameters() if p.requires_grad}
            source = {} if plan['mode'] == 'vanilla' else {
                n: p for n, p in self.deform_conditioner.named_parameters() if p.requires_grad}
            if (len(self.dataset) != 9139 or len(self.dataloader) != 380
                    or len(lora) != 360 or len(source) != source_count or self.optimizer.state or self.step != 0
                    or (self.world_size, self.batch_size, self.batch_size_per_gpu, self.batch_split)
                        != (cfg['num_gpus'], 24, 24//cfg['num_gpus'], 1)):
                raise ValueError('fresh original360/source0,10,19 and complete native9139 loader380 required')
            reference = json.loads(Path(plan['initial_reference']).read_text())
            if state_sha(lora.items()) != reference['initial_LoRA_SHA256']:
                raise ValueError('scientific LoRA initialization differs from complete original PBR16850')
            if source and cfg['training_seed'] == 101:
                common = state_sha((n, p) for n, p in source.items() if not n.startswith('ray_readout.'))
                if common != reference['initial_common_R10_SHA256']:
                    raise ValueError('source initialization differs from actual cold generated32 reference')
            self._PBR_initial_frozen = {n: _frozen_digest(m) for n, m in self.models.items()}
            self._PBR_source_count = source_count

        def run_step(self, packs):
            if self.step == 0:
                flag = torch.tensor(int(self._PBR_initial_guard_passed), device=self.device)
                torch.distributed.broadcast(flag, src=0)
                if not flag.item():
                    raise ValueError('actual initial4 native identity must pass before the first optimizer update')
                self._PBR_initial_guard_passed = True
            if len(packs) != 1 or len(packs[0].get('_sha256', [])) != self.batch_size_per_gpu:
                raise ValueError('one native full-microbatch pack with actual asset IDs required')
            trace = []
            handle = self.models['denoiser'].register_forward_pre_hook(lambda m, a: trace.append(dict(
                x_t_SHA256=tensor_sha(a[0]), t_SHA256=tensor_sha(a[1]), global_SHA256=tensor_sha(a[2]['global']))))
            try:
                result = super().run_step(packs)
            finally:
                handle.remove()
            if len(trace) != 1:
                raise ValueError('one actual sparse native denoiser forward per update required')
            record = dict(step=self.step+1, asset_ids=packs[0]['_sha256'], **trace[0])
            name = 'training-noise-trace.jsonl' if self.is_master else f'training-noise-trace.rank{self.rank}.jsonl'
            with (Path(self.output_dir)/name).open('a') as stream:
                stream.write(json.dumps(record)+'\n')
            return result

        def save(self, non_blocking=False):
            # Basic.run and the epoch hook both request the same boundary save.
            # A closed checkpoint is immutable; do not overwrite it a second time.
            token = _trainable_token(self)
            if self._PBR_last_archived_step == self.step:
                if token != self._PBR_last_archived_token:
                    raise ValueError('model/optimizer/sampler changed within a previously archived step')
                return
            super().save(non_blocking=False)
            subprocess.run([*plan['checkpoint_archive_command'], '--step', str(self.step)], check=True)
            receipt = Path(plan['checkpoint_archive_root'])/f'step{self.step:07d}'/'file4-closure.json'
            check = ClosedInputs(plan['canonical_local_root'])
            names = ['denoiser', 'misc']+([] if plan['mode'] == 'vanilla' else ['deform_conditioner'])
            for name in names:
                check.check(Path(self.output_dir)/'ckpts'/f'{name}_step{self.step:07d}.pt', receipt)
            self._PBR_last_archived_step = self.step
            self._PBR_last_archived_token = _trainable_token(self)

        def run(self):
            super().run()
            if self.step != 1900:
                raise ValueError('native PBR did not reach its fixed1900-update budget')
            if (len(self.optimizer.state) != 360+self._PBR_source_count
                    or {int(s['step']) for s in self.optimizer.state.values()} != {1900}
                    or self._PBR_initial_frozen != {n: _frozen_digest(m) for n, m in self.models.items()}):
                raise ValueError('actual optimizer1900 and frozen base/matcher/anchors integrity required')
            self.check_ddp()
            report = dict(status='PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE',
                mode=plan['mode'], rank=self.rank, training_seed=cfg['training_seed'],
                world_size=self.world_size, global_batch=24, microbatch=self.batch_size_per_gpu, accumulation=1,
                source_multiview_step=16850, fresh_optimizer=True, actual_optimizer_updates=1900,
                actual_optimizer_states=len(self.optimizer.state), actual_training_assets=9139,
                actual_dataloader_batches=380, actual_sampler_state=self.data_sampler.state_dict(),
                initial4_identity_before_first_update=True, frozen_base_matcher_and_anchors_exact=True,
                post_training_DDP_parameters_exact=True, source_tensors=self._PBR_source_count,
                stage_training_complete=True, full_cascade_evaluated=False, quality_gain_proven=False,
                model_adopted=False, full_goal_complete=False)
            (Path(self.output_dir)/f'stage-completed.rank{self.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    return Science, evaluate_science_pbr


def worker(rank, cfg, plan):
    torch.cuda.set_device(rank)
    gpu = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[rank]
    os.sched_setaffinity(0, {84+2*int(gpu), 85+2*int(gpu)})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from pixal3d import trainers, datasets
    from pixal3d.datasets.deformnext_feature import DeformNeXtFeaturePBR1024
    from pixal3d.trainers.flow_matching.r_mediated_pbr_source import RMediatedPBRNativeTrainer
    from pixal3d.trainers.flow_matching.sparse_flow_matching import ImageConditionedProjSparseFlowMatchingCFGTrainer
    from experiments.multiview_finetune import runtime
    from .frozen_projection_streaming import install_frozen_projection_streaming
    import train as native
    runtime.install_hooks()
    torch.set_num_threads(2)
    install_frozen_projection_streaming(max_assets=1,
        audit_directory=Path(cfg['output_dir'])/'projection-streaming-audit')
    base = ImageConditionedProjSparseFlowMatchingCFGTrainer if plan['mode'] == 'vanilla' else RMediatedPBRNativeTrainer
    selected, evaluation = science_trainer(base, cfg, plan)
    previous_evaluate = runtime.evaluate
    previous_trainer = getattr(trainers, NAME, None)
    previous_dataset = getattr(datasets, 'DeformNeXtFeaturePBR1024', None)
    runtime.evaluate = lambda trainer: evaluation(trainer, cfg, plan)
    setattr(trainers, NAME, selected)
    datasets.DeformNeXtFeaturePBR1024 = DeformNeXtFeaturePBR1024
    try:
        native.main(rank, EasyDict(cfg))
    finally:
        runtime.evaluate = previous_evaluate
        if previous_trainer is None: delattr(trainers, NAME)
        else: setattr(trainers, NAME, previous_trainer)
        if previous_dataset is None: delattr(datasets, 'DeformNeXtFeaturePBR1024')
        else: datasets.DeformNeXtFeaturePBR1024 = previous_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    cfg, _ = guard_plan(args.plan, plan)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    allowed = set(os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','))
    if (len(visible) != cfg['num_gpus'] or len(visible) != len(set(visible))
            or not set(visible) <= {'0', '1', '2', '3'} & allowed):
        raise ValueError('exact declared native PBR group of assigned physical GPUs required')
    output = Path(cfg['output_dir']); output.mkdir(parents=True, exist_ok=False)
    cfg['config'] = str(plan['config'])
    (output/'config.json').write_text(json.dumps(cfg, indent=2)+'\n')
    mp.spawn(worker, args=(cfg, plan), nprocs=cfg['num_gpus'], join=True)
    guard_plan(args.plan, plan)
    records = [json.loads((output/f'stage-completed.rank{r}.json').read_text()) for r in range(cfg['num_gpus'])]
    if any(r['status'] != 'PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE' for r in records):
        raise ValueError('actual native1900 completion from every rank required')
    print(json.dumps(dict(status='PASS_ACTUAL_ORIGINAL_PBR16850_FRESH_NATIVE1900_STAGE',
        mode=plan['mode'], full_goal_complete=False)), flush=True)


if __name__ == '__main__':
    main()

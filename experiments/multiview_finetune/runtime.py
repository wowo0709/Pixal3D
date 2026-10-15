"""Stage-local runtime hooks installed in every native spawned worker."""
import json
import math
import os
from pathlib import Path
import resource
import signal
import time
from datetime import datetime, timezone

import torch
import torch.distributed as dist
from experiments.multiview_finetune.paths import assigned_gpu_count

_native_load = _native_step = _native_save = _native_run = _native_check_abort = None
_installed = False



def install_stop_handlers(output_dir):
    def request_stop(signum, frame):
        Path(output_dir, 'STOP').touch()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def write_progress(self, stage):
    if not self.is_master:
        return
    row = self._runtime_last_row
    state = {'stage': stage, 'mode': 'train', 'current': self.step,
             'total': self._runtime_total_steps,
             'progress': self.step / self._runtime_total_steps,
             'updated_at': datetime.now(timezone.utc).isoformat(),
             'effective_batch_size': self.batch_size, 'world_size': self.world_size,
             'loss': row['loss'] if row is not None else None,
             'lr': self.optimizer.param_groups[0]['lr'],
             'last_checkpoint_step': self._runtime_checkpoint_step}
    target = Path(self.output_dir, 'progress.json')
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(temporary, target)


def load_data(self):
    debug_views = os.environ.get('PIXAL3D_DEBUG_VIEWS', '')
    if debug_views:
        views = [int(k) for k in debug_views.split(',')]
        self.dataset.min_condition_views = self.dataset.max_condition_views = views[self.step]
    began = time.monotonic()
    packs = _native_load(self)
    self._load_seconds = time.monotonic() - began
    return packs


def run_step(self, packs):
    sizes = [int(pack['cond'].shape[0]) for pack in packs]
    views = [int(pack['cond'].shape[1]) for pack in packs]
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    controller = getattr(self, 'elastic_controller', None)
    start = controller._buffer_ptr if controller is not None else None
    result = _native_step(self, packs)
    torch.cuda.synchronize()
    module=getattr(self,'deform_conditioner',None)
    if module is not None and getattr(module,'implementation',None)=='r9-native-v5':
        diagnostics={'fusion_gate':float(torch.tanh(module.fusion_gate.detach()))}
        last=getattr(self,'_last_deform_condition',None)
        if last is not None and last.query_valid is not None:
            diagnostics['query_coverage_last_pack']=float(last.query_valid.float().mean())
        if last is not None and last.retarget_residual is not None:
            diagnostics['retarget_residual_rms_last_pack']=float(last.retarget_residual.detach().float().square().mean().sqrt())
        result['conditioner']=diagnostics
    peak = torch.cuda.max_memory_allocated() / 2**30
    if controller is not None:
        peak = max(peak, *(float(controller._memory[(start + i) % controller.buffer_size])
                           for i in range(len(packs))))
    row = {'step': self.step + 1, 'rank': self.rank,
           'effective_batch_size': sum(sizes) * self.world_size,
           'world_size': self.world_size, 'accumulation_packs': len(packs),
           'microbatch_assets': sizes, 'K': views[0],
           'load_seconds': self._load_seconds, 'compute_seconds': time.monotonic() - began,
           'peak_allocated_gib': peak, 'reserved_gib': torch.cuda.memory_reserved() / 2**30,
           'lr_next_update': self.optimizer.param_groups[0]['lr'], 'loss': result['loss']}
    if 'conditioner' in result:row['conditioner']=result['conditioner']
    name = 'runtime_batch.jsonl' if self.is_master else f'runtime_batch.rank{self.rank}.jsonl'
    with Path(self.output_dir, name).open('a') as output:
        output.write(json.dumps(row) + '\n')
    self._runtime_last_row = row
    print('RUNTIME_BATCH', json.dumps(row), flush=True)
    return result


def save(self, non_blocking=False):
    write_progress(self, 'saving')
    _native_save(self, non_blocking=False)
    self._runtime_checkpoint_step = self.step
    write_progress(self, 'running')


def check_abort(self):
    _native_check_abort(self)
    stop = Path(self.output_dir, 'STOP').exists()
    if self.world_size > 1:
        flag = torch.tensor(int(stop), device=self.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        stop = bool(flag.item())
    final = self.step >= self._runtime_total_steps
    validate = self.step % self._runtime_validation_every == 0
    if stop or final or validate:
        if self.is_master:
            # Epoch checkpoints retain the weights corresponding to validation.
            self.save(non_blocking=False)
        if self.world_size > 1:
            dist.barrier()
        if validate and not stop and self.is_master:
            write_progress(self, 'validating')
            evaluate(self)
        if self.world_size > 1:
            dist.barrier()
    if final and not stop and getattr(self,'_runtime_config',{}).get('experiment',{}).get('auto_continue_after_pilot'):
        config=self._runtime_config
        extension=0
        if self.is_master:
            from experiments.deformnext_retargeting.pilot import review
            records=[json.loads(line) for line in (Path(self.output_dir)/'validation.jsonl').read_text().splitlines()]
            baseline_path=config['experiment'].get('pilot_baseline_file')
            baseline=None
            if baseline_path:
                path=Path(baseline_path)
                if path.suffix=='.jsonl':
                    baseline=next(json.loads(x) for x in path.read_text().splitlines() if json.loads(x)['step']==self.step)
                else:baseline=json.loads(path.read_text())
            verdict=review(config,records,baseline)
            gate=getattr(getattr(self,'deform_conditioner',None),'fusion_gate',None)
            if gate is not None:
                verdict['fusion_gate']=float(torch.tanh(gate.detach()))
                verdict['passed']=verdict['passed'] and math.isfinite(verdict['fusion_gate']) and verdict['fusion_gate']!=0
            (Path(self.output_dir)/'pilot_review.json').write_text(json.dumps(verdict,indent=2)+'\n')
            if verdict['passed']:
                extension=config['experiment']['planned_total_steps']
                config['experiment'].update(release_status='pilot-validated',auto_continue_after_pilot=False)
                config['trainer']['args']['max_steps']=extension
                for target in (Path(self.output_dir)/'config.json',Path(config['experiment']['recipe_path'])):
                    temporary=target.with_suffix('.pilot.tmp')
                    temporary.write_text(json.dumps(config,indent=2)+'\n');os.replace(temporary,target)
                if self.wandb_run is not None:
                    self.wandb_run.config.update({'pilot_passed':True,'planned_total_steps':extension},allow_val_change=True)
                print(f'PILOT PASSED: continuing to {extension} updates in the same process',flush=True)
            else:
                (Path(self.output_dir)/'PILOT_HOLD.json').write_text(json.dumps(verdict,indent=2)+'\n')
        if self.world_size>1:
            value=torch.tensor(extension,device=self.device,dtype=torch.long);dist.broadcast(value,src=0);extension=int(value)
        if extension:
            self.max_steps=self._runtime_total_steps=extension
            final=False
    if stop:
        self.max_steps = self.step
    write_progress(self, 'stopped' if stop else 'completed' if final else 'running')


def run(self):
    config = json.loads(Path(self.output_dir, 'config.json').read_text())
    policy_path=Path(self.output_dir)/'PILOT_POLICY.json'
    if policy_path.exists():
        policy=json.loads(policy_path.read_text())
        if self.step<policy['pilot_steps']:
            if (self.max_steps!=policy['pilot_steps']
                    or policy['planned_total_steps']!=config['experiment']['planned_total_steps']):
                raise ValueError('pilot continuation policy differs from the verified recipe')
            config['experiment'].update(policy)
    self._runtime_total_steps = self.max_steps
    self._runtime_config = config
    self._runtime_validation_every = config['validation'].get('every_steps', self.i_save)
    self._runtime_checkpoint_step = self.step if self.step else None
    self._runtime_last_row = None
    install_stop_handlers(self.output_dir)
    write_progress(self, 'running')
    if config.get('experiment', {}).get('validate_at_start') and self.step == 0:
        if self.is_master:
            write_progress(self, 'validating')
            evaluate(self)
        if self.world_size > 1:
            dist.barrier()
        write_progress(self, 'running')
    return _native_run(self)


def install_hooks():
    global _installed, _native_load, _native_step, _native_save, _native_run, _native_check_abort
    if _installed:
        return
    from pixal3d.trainers.basic import BasicTrainer
    torch.set_num_threads(4)
    resource.setrlimit(resource.RLIMIT_NOFILE, (8192, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
    _native_load, _native_step, _native_save = BasicTrainer.load_data, BasicTrainer.run_step, BasicTrainer.save
    _native_run, _native_check_abort = BasicTrainer.run, BasicTrainer.check_abort
    BasicTrainer.load_data, BasicTrainer.run_step, BasicTrainer.save = load_data, run_step, save
    BasicTrainer.run, BasicTrainer.check_abort = run, check_abort
    _installed = True


def evaluate(self):
    # Native trainer has no held-out loop. Evaluate a fixed, source-stratified
    # panel after checkpoint saves, with paired noise/t and K=1/4 conditions.
    import numpy as np
    from pixal3d import datasets
    from pixal3d.utils.data_utils import recursive_to_device

    evaluation = json.loads((Path(self.output_dir) / "config.json").read_text())["validation"]
    distributed_models = self.training_models
    self.training_models = self.models
    training_state = [model.training for model in self.models.values()]
    p_uncond = self.p_uncond
    numpy_state = np.random.get_state()
    cpu_state, gpu_state = torch.get_rng_state(), torch.cuda.get_rng_state()
    output_rows = []
    quick_metrics, quick_panel, quick_ids = {}, None, []
    try:
        self.p_uncond = 0.0
        np.random.seed(42)
        is_deformnext = evaluation["dataset_name"].startswith("DeformNeXt")
        dataset_input = (
            evaluation["roots_file"] if is_deformnext
            else Path(evaluation["roots_file"]).read_text()
        )
        dataset = getattr(datasets, evaluation["dataset_name"])(
            dataset_input, **evaluation["dataset_args"])
        for model in self.training_models.values():
            model.eval()
        source_indices = {}
        for index, (_, _, name) in enumerate(dataset.instances):
            source = (
                dataset.rows[index].get("category", "DeformNeXt")
                if is_deformnext else name.split(":")[0]
            )
            source_indices.setdefault(source, []).append(index)
        max_assets = evaluation.get("max_assets")
        if max_assets is not None:
            from experiments.deformnext_retargeting.quick_metrics import stratified_indices
            retained = set(stratified_indices(dataset, int(max_assets)))
            source_indices = {
                source: [index for index in indices if index in retained]
                for source, indices in source_indices.items()
            }
            source_indices = {
                source: indices for source, indices in source_indices.items() if indices
            }
        # The production quick check uses a stable bounded panel. Legacy recipes
        # can still bound each source independently with assets_per_source.
        limit = evaluation.get("assets_per_source")
        for source, indices in source_indices.items():
            losses = {4: []} if is_deformnext else {1: [], 4: []}
            selected = indices if limit is None else indices[:limit]
            for index in selected:
                np.random.seed(42 + index)
                item = dataset[index]
                for k in losses:
                    if not is_deformnext:
                        dataset.min_condition_views = dataset.max_condition_views = k
                    pack = dataset.collate_fn([item])
                    pack = pack if isinstance(pack, dict) else pack[0]
                    pack = recursive_to_device(pack, self.device)
                    torch.manual_seed(42 + index)
                    torch.cuda.manual_seed(42 + index)
                    with torch.no_grad(), torch.autocast("cuda", dtype=self.mix_precision_dtype):
                        loss, _ = self.training_losses(**pack)
                    losses[k].append(float(loss["loss"]))
                    del pack
            output_rows.append({
                "source": source, "assets": len(losses[4]),
                "K1_loss": None if is_deformnext else float(np.mean(losses[1])),
                "K4_loss": float(np.mean(losses[4])),
            })
        if is_deformnext:
            from experiments.deformnext_retargeting.quick_metrics import evaluate_quick
            quick_metrics, quick_panel, quick_ids = evaluate_quick(self, dataset, evaluation)
    finally:
        self.p_uncond = p_uncond
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(gpu_state)
        for model, training in zip(self.training_models.values(), training_state):
            model.train(training)
        self.training_models = distributed_models
    validation_row = {
        "step": self.step,
        "weights": "raw",
        "scope": (
            f"bounded panel ({evaluation['max_assets']} assets)"
            if evaluation.get("max_assets") is not None else
            "full validation" if limit is None else "bounded debug panel"
        ),
        "sources": output_rows,
        "quick_metrics": quick_metrics,
        "quick_panel_ids": quick_ids,
    }
    with (Path(self.output_dir) / "validation.jsonl").open("a") as output:
        output.write(json.dumps(validation_row) + "\n")
    if self.wandb_run is not None:
        total_assets = sum(row["assets"] for row in output_rows)
        wandb_validation = {
            f"validation/{row['source']}/K4_loss": row["K4_loss"]
            for row in output_rows
        }
        if total_assets:
            wandb_validation["validation/K4_loss"] = sum(
                row["K4_loss"] * row["assets"] for row in output_rows
            ) / total_assets
        wandb_validation.update({
            f"validation/{key}": value for key, value in quick_metrics.items()
        })
        if quick_panel is not None:
            import wandb
            wandb_validation["validation/quick_panel"] = wandb.Image(
                quick_panel,
                caption="GT | prediction | absolute difference\n" + " | ".join(quick_ids),
            )
        self.wandb_run.log(wandb_validation, step=self.step)
    print("VALIDATION_PANEL", json.dumps(output_rows), flush=True)

"""Native train.py with measured batches and a bounded held-out evaluation hook.

Use native CLI flags. Validation roots are paired with the stage config and are
not training roots. Model/optimizer/scheduler/checkpoint behavior stays native.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import resource
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
if __name__ == "__main__" and not {"-h", "--help"}.intersection(sys.argv[1:]):
    from experiments.multiview_finetune.paths import single_visible_device

    single_visible_device()
    # This recipe's EBS is one worker's full batch. Leave all other native flags
    # for train.py, but never inherit its default of every visible device.
    launch_parser = argparse.ArgumentParser(add_help=False)
    launch_parser.add_argument("--num_gpus", type=int, choices=[1])
    launch, _ = launch_parser.parse_known_args()
    if launch.num_gpus is None:
        sys.argv.extend(["--num_gpus", "1"])

import torch
from pixal3d.trainers.basic import BasicTrainer

torch.set_num_threads(4)
# Sparse packs share many independent tensors; workers=2 exhaust the default 1024 FDs.
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
original_load = BasicTrainer.load_data
original_step = BasicTrainer.run_step
original_save = BasicTrainer.save
DEBUG_VIEWS = [int(k) for k in os.environ.get("PIXAL3D_DEBUG_VIEWS", "").split(",") if k]


def load_data(self):
    if DEBUG_VIEWS:
        self.dataset.min_condition_views = self.dataset.max_condition_views = DEBUG_VIEWS[self.step]
    began = time.monotonic()
    packs = original_load(self)
    self._load_seconds = time.monotonic() - began
    return packs


def run_step(self, packs):
    sizes = [int(pack["cond"].shape[0]) for pack in packs]
    views = [int(pack["cond"].shape[1]) for pack in packs]
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    controller = getattr(self, "elastic_controller", None)
    start = controller._buffer_ptr if controller is not None else None
    result = original_step(self, packs)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    if controller is not None:
        peak = max(peak, *(float(controller._memory[(start + i) % controller.buffer_size])
                           for i in range(len(packs))))
    row = {"step": self.step + 1, "effective_batch_size": sum(sizes) * self.world_size,
           "world_size": self.world_size, "accumulation_packs": len(packs),
           "microbatch_assets": sizes, "K": views[0],
           "load_seconds": self._load_seconds, "compute_seconds": time.monotonic() - began,
           "peak_allocated_gib": peak, "reserved_gib": torch.cuda.memory_reserved() / 2**30,
           "lr_next_update": self.optimizer.param_groups[0]["lr"], "loss": result["loss"]}
    with (Path(self.output_dir) / "runtime_batch.jsonl").open("a") as output:
        output.write(json.dumps(row) + "\n")
    print("RUNTIME_BATCH", json.dumps(row), flush=True)
    return result


def save(self, non_blocking=False):
    complete = (Path(self.output_dir) / "ckpts" / f"misc_step{self.step:07d}.pt").exists()
    original_save(self, non_blocking=non_blocking)
    if not complete:
        evaluate(self)


def evaluate(self):
    # Native trainer has no held-out loop. Evaluate a fixed, source-stratified
    # panel after checkpoint saves, with paired noise/t and K=1/4 conditions.
    import numpy as np
    from pixal3d import datasets
    from pixal3d.utils.data_utils import recursive_to_device

    evaluation = json.loads((Path(self.output_dir) / "config.json").read_text())["validation"]
    training_state = [model.training for model in self.training_models.values()]
    p_uncond = self.p_uncond
    numpy_state = np.random.get_state()
    cpu_state, gpu_states = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    output_rows = []
    try:
        self.p_uncond = 0.0
        np.random.seed(42)
        dataset = getattr(datasets, evaluation["dataset_name"])(
            Path(evaluation["roots_file"]).read_text(), **evaluation["dataset_args"])
        for model in self.training_models.values():
            model.eval()
        source_indices = {}
        for index, (_, _, name) in enumerate(dataset.instances):
            source_indices.setdefault(name.split(":")[0], []).append(index)
        # Debug uses an explicitly bounded panel; the main recipe evaluates all
        # held-out assets. Paired K=1/4 losses share latent, frame, noise and t.
        limit = evaluation["assets_per_source"]
        for source, indices in source_indices.items():
            losses = {1: [], 4: []}
            selected = indices if limit is None else indices[:limit]
            for index in selected:
                np.random.seed(42 + index)
                item = dataset[index]
                for k in losses:
                    dataset.min_condition_views = dataset.max_condition_views = k
                    pack = dataset.collate_fn([item])
                    pack = pack if isinstance(pack, dict) else pack[0]
                    pack = recursive_to_device(pack, self.device)
                    torch.manual_seed(42 + index)
                    torch.cuda.manual_seed_all(42 + index)
                    with torch.no_grad(), torch.autocast("cuda", dtype=self.mix_precision_dtype):
                        loss, _ = self.training_losses(**pack)
                    losses[k].append(float(loss["loss"]))
                    del pack
            output_rows.append({"source": source, "assets": len(losses[1]),
                                "K1_loss": float(np.mean(losses[1])),
                                "K4_loss": float(np.mean(losses[4]))})
    finally:
        self.p_uncond = p_uncond
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state_all(gpu_states)
        for model, training in zip(self.training_models.values(), training_state):
            model.train(training)
    with (Path(self.output_dir) / "validation.jsonl").open("a") as output:
        output.write(json.dumps({"step": self.step, "weights": "raw",
                                "scope": "full validation" if limit is None else "bounded debug panel",
                                "sources": output_rows}) + "\n")
    print("VALIDATION_PANEL", json.dumps(output_rows), flush=True)


BasicTrainer.load_data = load_data
BasicTrainer.run_step = run_step
BasicTrainer.save = save
sys.argv[0] = str(REPO / "train.py")
if __name__ == "__main__":
    runpy.run_path(str(REPO / "train.py"), run_name="__main__")

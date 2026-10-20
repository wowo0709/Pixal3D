"""Pilot inference-only correspondence reliability; never changes SS/R weights.

Fit a small binary head on existing TRAIN512 native rays, retain zero-query
assets, seal TRAIN-calibration32 predictions and threshold before reading
actual VAL32 labels. Final epoch7 is fixed before the run. A successful head
is only an integration candidate, not evidence of improved 3D generation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .ray_correspondence_reliability import binary_metrics, calibration_threshold


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if physical not in {'0', '1', '2', '3'} or physical not in os.environ.get('MATCHING_V2_ALLOWED_GPUS', '').split(','):
        raise ValueError('one explicitly assigned physical GPU0-3 required')
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.05, 0)
    torch.use_deterministic_algorithms(True)
    plan = json.loads(args.plan.read_text())
    assert plan['epochs'] == 7 and plan['microqueries'] == 4096 and plan['hidden_width'] == 64
    assert plan['seed'] == 101 and plan['lr'] == .001 and plan['weight_decay'] == .01
    root, archive = Path(plan['cache']), Path(plan['archive_root'])
    bound = dict(plan['source_SHA256'])
    bound[str(args.plan)] = sha(args.plan)
    assert all(sha(p) == d for p, d in bound.items())
    receipt = json.loads(Path(plan['cache_receipt']).read_text())
    assert receipt['status'] == 'PASS' and receipt['local_equals_actual_file4_SHA256']
    for f in ['observations.npz', 'cache.json']:
        assert any(r['relative_path'] == str((root/f).relative_to(archive))
                   and r['sha256'] == sha(root/f) for r in receipt['files'])
    meta = json.loads((root/'cache.json').read_text())
    assert meta['assets'] == dict(train=512, calibration=32, validation=32)
    assert sha(root/'observations.npz') == meta['observations_SHA256']
    records = meta['records']
    dataset = np.load(root/'observations.npz', allow_pickle=False)

    def load(role):
        rows = [r for r in records if r['role'] == role]
        get = lambda suffix: np.concatenate([dataset[r['array_prefix']+suffix] for r in rows])
        return rows, get('_x'), get('_y'), get('_type')

    train_rows, train_x, train_y, _ = load('train')
    cal_rows, cal_x, cal_y, _ = load('calibration')
    assert len(train_x) and train_y.any() and (~train_y).any()
    # First2048 columns have per-ray LN. Only the12 summary statistics are fitted.
    mean = train_x[:, 2048:].mean(0)
    scale = np.maximum(train_x[:, 2048:].std(0), 1e-4)
    baseline = float(train_y.mean())

    def inputs(x):
        result = torch.from_numpy(x.copy()).cuda()
        result[:, 2048:] = (result[:, 2048:]-torch.from_numpy(mean).cuda())/torch.from_numpy(scale).cuda()
        return result

    x = inputs(train_x)
    y = torch.from_numpy(train_y.astype(np.float32)).cuda()
    cx = inputs(cal_x)
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    model = nn.Sequential(nn.Linear(2060, 64), nn.GELU(), nn.Linear(64, 1)).cuda()
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    out = Path(plan['output'])
    out.mkdir(parents=True, exist_ok=False)
    import wandb
    run = wandb.init(project='pixal3d-deformnext-retargeting', entity='youngwoojeon-unist',
        id='ray-correspondence-reliability-v1-s101-e7', name='Ray correspondence reliability fixed7',
        dir=str(out), config=dict(epochs=7, microqueries=4096, hidden_width=64,
            seed=101, lr=.001, weight_decay=.01, TRAIN_assets=512,
            calibration_assets=32, VAL_assets=32, generated_quality_experiment=False))
    rng = np.random.default_rng(101)
    updates, history = 0, []
    for epoch in range(1, 8):
        model.train()
        permutation = rng.permutation(len(x))
        total_loss = 0.
        for begin in range(0, len(x), 4096):
            take = torch.from_numpy(permutation[begin:begin+4096]).cuda()
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[take])[:, 0], y[take])
            if not torch.isfinite(loss):
                raise ValueError('nonfinite reliability loss')
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            total_loss += float(loss.detach())*len(take)
        model.eval()
        with torch.no_grad():
            cp = model(cx)[:, 0].sigmoid().cpu().numpy()
        row = dict(epoch=epoch, optimizer_updates=updates, TRAIN_BCE=total_loss/len(x),
                   calibration=binary_metrics(cal_y, cp, .5))
        history.append(row)
        run.log(dict(epoch=epoch, optimizer_updates=updates, TRAIN_BCE=row['TRAIN_BCE'],
            **{'calibration/'+k: v for k, v in row['calibration'].items() if v is not None}), step=updates)
        print(json.dumps(row), flush=True)
    changed = [n for n, p in model.named_parameters() if not torch.equal(p, original[n])]
    assert len(changed) == len(optimizer.state) == 4
    threshold = calibration_threshold(cal_y, cp)
    selection = dict(status='PASS_TRAIN_CALIBRATION_ONLY_RELIABILITY_SELECTION', epoch=7,
        threshold=threshold, constant_TRAIN_prevalence=baseline,
        calibration=binary_metrics(cal_y, cp, threshold),
        calibration_constant=binary_metrics(cal_y, np.full(len(cal_y), baseline), threshold),
        actual_validation_labels_read=False, selection='fixed7;threshold maximizes TRAIN-calibration balanced accuracy on fixed19grid')
    # Sealed selection is written before load('validation') opens any VAL arrays.
    (out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    selection_sha = sha(out/'selection.json')
    torch.save(dict(head=model.state_dict(), TRAIN_summary_mean=torch.from_numpy(mean),
        TRAIN_summary_scale=torch.from_numpy(scale), input_width=2060), out/'head.pt')
    val_rows, val_x, val_y, val_types = load('validation')
    with torch.no_grad():
        vp = model(inputs(val_x))[:, 0].sigmoid().cpu().numpy()
    validation = binary_metrics(val_y, vp, threshold)
    constant = binary_metrics(val_y, np.full(len(val_y), baseline), threshold)
    rng = np.random.default_rng(20261020)
    lengths = [r['known_rays'] for r in val_rows]
    boundaries = np.r_[0, np.cumsum(lengths)]
    ci_rows, per_asset = [], []
    for row, lo, hi in zip(val_rows, boundaries[:-1], boundaries[1:]):
        per_asset.append(dict(sample_id=row['sample_id'], source_cluster_id=row['source_cluster_id'],
            rays=int(hi-lo), labels=val_y[lo:hi].tolist(), probability=vp[lo:hi].tolist()))
    for sample in rng.integers(0, len(val_rows), (2000, len(val_rows))):
        take = np.concatenate([np.arange(boundaries[i], boundaries[i+1]) for i in sample])
        ym, pm = val_y[take], vp[take]
        metric = binary_metrics(ym, pm, threshold)
        if metric['AUROC'] is None:
            continue
        ci_rows.append([metric['AUROC'], metric['Brier']-np.square(baseline-ym).mean()])
    ci = np.quantile(ci_rows, [.025, .975], axis=0)
    result = dict(status='PASS_ACTUAL_GPU_RELIABILITY_HEAD_PILOT', epochs=7,
        TRAIN_assets=len(train_rows), TRAIN_calibration_assets=len(cal_rows), actual_VAL_assets=len(val_rows),
        TRAIN_known_rays=len(train_y), actual_VAL_known_rays=len(val_y), history=history,
        calibration_selection=selection, selection_SHA256=selection_sha,
        actual_validation=validation, validation_constant=constant,
        AUROC_source_cluster_CI95=ci[:, 0].tolist(),
        Brier_difference_source_cluster_CI95=ci[:, 1].tolist(),
        heldout_head_gate_passed=bool(validation['AUROC'] >= .65 and ci[0, 0] > .5
            and validation['Brier'] < constant['Brier'] and ci[1, 1] < 0),
        per_asset=per_asset, validation_type_acceptance={
            name: dict(rays=int((val_types == kind).sum()),
                mean_probability=float(vp[val_types == kind].mean()),
                accepted_fraction=float((vp[val_types == kind] >= threshold).mean()))
            for kind, name in [(0, 'no-visible-source'), (1, 'proposal-miss'), (2, 'usable-near1')]},
        native_head_trainable_tensors=4, optimizer_updates=updates, all4_tensors_changed=True,
        physical_GPU=physical, allocator_fraction=.05, peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        seconds=time.monotonic()-began, GT_in_head_forward=False,
        SS_R_matching_weights_updated=False, head_applied_to_generation=False,
        model_adopted=False, full_goal_complete=False, source_SHA256=bound)
    assert sha(out/'selection.json') == selection_sha
    assert all(sha(p) == d for p, d in bound.items())
    (out/'evaluation.json').write_text(json.dumps(result, indent=2)+'\n')
    run.summary.update(dict(heldout_head_gate_passed=result['heldout_head_gate_passed'],
        **{'actual_VAL/'+k: v for k, v in validation.items() if v is not None},
        **{'actual_VAL_constant/'+k: v for k, v in constant.items() if v is not None}))
    run.finish()
    print(json.dumps({k: result[k] for k in ['status', 'actual_validation', 'validation_constant',
        'AUROC_source_cluster_CI95', 'Brier_difference_source_cluster_CI95', 'heldout_head_gate_passed',
        'optimizer_updates', 'peak_allocated_GiB', 'seconds']}), flush=True)


if __name__ == '__main__':
    main()

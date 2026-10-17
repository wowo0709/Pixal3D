"""Evaluation-only input controls for the unmodified multi-view model."""
from __future__ import annotations
from contextlib import contextmanager
import torch


class ConditioningControl:
    def __init__(self,dataset,mode):
        if mode not in {'edited-rgb','source-only'}:
            raise ValueError('unknown input control')
        if getattr(dataset,'transport_arm',None)!='v1':
            raise ValueError('input controls require native V1 without transport features')
        self.dataset,self.mode=dataset,mode
        for row in dataset.rows:
            if 'source_anchor_rgb' not in row:
                raise ValueError('four-view control requires the original target-camera RGB')
            if not torch.allclose(torch.tensor(row['transform_matrix'][0]),torch.tensor(row['source_anchor_transform_matrix'])):
                raise ValueError('source anchor camera must match target anchor camera')
            if abs(row['camera_angle_x'][0]-row['source_anchor_camera_angle_x'])>1e-6:
                raise ValueError('source/target anchor FOV differs')

    def __getattr__(self,name):return getattr(self.dataset,name)
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):
        sample=dict(self.dataset[index])
        if any(k.startswith('transport_') for k in sample):
            raise ValueError('native input control must not receive target transport features')
        if self.mode=='source-only':
            from pixal3d.datasets.deformnext import _load_image
            sample['cond']=sample['cond'].clone()
            path=self.dataset._resolve(self.dataset.rows[index]['source_anchor_rgb'])
            sample['cond'][0]=_load_image(path,self.dataset.image_size)
        return sample


class TransportControl:
    """Perturb only the cached SS transport path, retaining all four RGB views."""
    DEFINITIONS = {
        'source-off': 'Disable the additional transport and adapter paths; native four-view conditioning remains.',
        'adapter-off': 'Zero only the retargeter output; retain semantic evidence, learned fusion gate and native four-view conditioning.',
        'wrong-corr': 'Roll correspondence values by one target-query slot within each object; a weak local sensitivity control.',
        'wrong-object': 'Cycle donor foreground evidence across assets, deterministically resampling foreground-token order onto recipient valid queries; retain recipient support, confidence, target query and native RGB.',
        'shuffled-deform': 'Cycle the physical C2 RAW deformation context between assets in each generation batch.',
    }
    MODES = tuple(DEFINITIONS)

    def __init__(self, dataset, mode):
        if mode not in self.MODES:
            raise ValueError('unknown transport control')
        if (getattr(dataset, 'transport_implementation', None) != 'matching-v2'
                or getattr(dataset, 'transport_arm', None) not in {'b2', 'b3', 'c1', 'c2'}):
            raise ValueError('transport controls require matching-v2 SS evidence')
        if mode == 'shuffled-deform' and dataset.transport_arm != 'c2':
            raise ValueError('shuffled deformation requires physical C2 query channels')
        if mode == 'adapter-off' and dataset.transport_arm not in {'c1', 'c2'}:
            raise ValueError('adapter-off requires an arm with a retargeter')
        self.dataset, self.mode = dataset, mode

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_fn(self, batch, split_size=None):
        if self.mode == 'shuffled-deform' and len(batch) < 2:
            raise ValueError('shuffled deformation requires at least two assets per generation batch')
        if self.mode == 'wrong-object' and len(batch) < 2:
            raise ValueError('wrong-object requires at least two assets per generation batch')
        pack = self.dataset.collate_fn(batch, split_size=split_size)
        if not isinstance(pack, dict) or 'transport_control' in pack:
            raise ValueError('transport control requires an unmodified dense SS batch')
        # Scoped hooks perturb loaded modules without changing their inference
        # control API, cached input tensors or learned weights.
        return {**pack, 'transport_control': 'normal' if self.mode in {'adapter-off', 'wrong-object'} else self.mode}


@contextmanager
def transport_diagnostic_context(conditioner, mode):
    if mode not in {'adapter-off', 'wrong-object'}:
        yield
        return
    if mode == 'adapter-off':
        retargeter = getattr(conditioner, 'retargeter', None)
        if retargeter is None:
            raise ValueError('adapter-off requires a loaded retargeter')
        hook = retargeter.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
    else:
        if getattr(conditioner, 'implementation', None) != 'matching-v2':
            raise ValueError('wrong-object requires preaggregated matching-v2 evidence')
        def swap_evidence(module, positional, keyword):
            if keyword['evidence_soft'].shape[0] < 2:
                raise ValueError('wrong-object requires at least two assets per generation batch')
            valid = keyword['cached_valid'].bool() & keyword['target_valid'].bool()
            positions = [torch.where(row)[0] for row in valid]
            if any(len(index) == 0 for index in positions):
                raise ValueError('wrong-object requires foreground evidence in every donor')
            changed = dict(keyword)
            for key in ['evidence_top1', 'evidence_soft']:
                values = keyword[key]
                replacement = values.clone()
                for recipient, index in enumerate(positions):
                    donor = (recipient - 1) % len(positions)
                    source_index = positions[donor]
                    # Preserve recipient foreground support instead of mostly
                    # injecting donor background zeros. No geometry/teacher or
                    # RNG is used; ordinal resampling only chooses donor tokens.
                    ordinal = torch.arange(len(index), device=index.device)
                    selected = source_index[ordinal * len(source_index) // len(index)]
                    replacement[recipient, index] = values[donor, selected]
                changed[key] = replacement
            return positional, changed
        hook = conditioner.register_forward_pre_hook(swap_evidence, with_kwargs=True)
    try:
        yield
    finally:
        hook.remove()

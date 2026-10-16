"""Evaluation-only input controls for the unmodified multi-view model."""
from __future__ import annotations
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

"""Separate, label-free native observations and loss-only validation targets."""
from pathlib import Path

import numpy as np

from . import ray_set_head_training as shared


def positive_sets(candidate_indices, patch_indices, teacher):
    """Keep each visible source surface relation on its native target patch ray."""
    ids = np.asarray(candidate_indices, dtype=np.int64)
    patches = np.asarray(patch_indices, dtype=np.int64)
    if ids.shape != (len(patches), 64) or ((ids < 0) | (ids >= 3072)).any():
        raise ValueError('native64 addresses across exactly three source RGB views required')
    positive = teacher['positive_indices'][patches].astype(np.int64)
    visible = (teacher['positive_visible'][patches].astype(bool)
               & teacher['target_foreground'][patches, None].astype(bool))
    if positive.shape != (len(patches), 3) or visible.shape != positive.shape:
        raise ValueError('one loss-only relation per visible source view required')
    proposed = ids[:, None, :]
    pos = positive[:, :, None]
    exact = (proposed == pos) & visible[:, :, None]
    distance = np.maximum(np.abs((proposed % 1024)//32 - (pos % 1024)//32),
                          np.abs(proposed % 32 - pos % 32))
    near = (proposed//1024 == np.arange(3)[None, :, None]) & (distance <= 1) & visible[:, :, None]
    relation = np.repeat(np.arange(len(patches), dtype=np.int64), 3)[visible.reshape(-1)]
    return relation, exact.reshape(-1, 64)[visible.reshape(-1)], near.reshape(-1, 64)[visible.reshape(-1)]


class ValidationObservations:
    def __init__(self, root):
        self.root = Path(root)

    def load(self, row):
        if row['role'] != 'validation' or row['split'] != 'validation':
            raise ValueError('actual SS validation inputs required')
        for field in ['inputs', 'labels']:
            path = self.root/row[field+'_relative_path']
            if shared.sha(path) != row[field+'_SHA256']:
                raise ValueError('closed validation cache changed')
        x, y = shared.load_asset(self.root, row)
        with np.load(self.root/row['inputs_relative_path'], allow_pickle=False) as archive:
            x['source_key_bank'] = archive['source_key_bank'].copy()
        if x['source_key_bank'].shape != (3072, 1024):
            raise ValueError('native FRAME key bank required')
        return x, y

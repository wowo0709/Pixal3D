"""Native PBR DINO/NAF transport plus inference-only RAW Omega contexts."""
import numpy as np
import torch

from .deformnext_feature import DeformNeXtFeaturePBR1024


class DeformNeXtDeformationQueryPBR1024(DeformNeXtFeaturePBR1024):
    def __init__(self, *args, deformation_query_channels, **kwargs):
        if (deformation_query_channels != 2048
                or kwargs.get('transport_arm') != 'c1'
                or kwargs.get('task_matching_features') is not True
                or kwargs.get('transport_implementation') != 'r9-native-v5'
                or kwargs.get('match_root') is None):
            raise ValueError('explicit PBR C1 cache carrier and RAW2048 query channels required')
        super().__init__(*args, **kwargs)
        self.deformation_query_channels = deformation_query_channels

    def __getitem__(self, index):
        pack = super().__getitem__(index)
        row = self.rows[index]
        if 'transport_spatial_indices' in pack or 'transport_deformation' in pack:
            raise ValueError('physical PBR query must not load GT or legacy cached deformation')
        # RAW Omega descriptors live in the shared SS sidecar. The PBR sidecar
        # supplies its native2048 DINO+NAF values and sparse patch mapping.
        with np.load(self._resolve(row['transport_cache'])) as archive:
            if str(archive['conditioning_input_sha256']) != row['conditioning_input_sha256']:
                raise ValueError('PBR RAW Omega context does not match edited input RGB')
            target = torch.from_numpy(np.asarray(archive['target_keys']).copy())
            source = torch.from_numpy(np.asarray(archive['source_keys']).copy())
        if (target.shape != (1024, 2048) or source.shape != (3, 1024, 2048)
                or not torch.equal(target[..., :1024], pack['transport_target_keys'])
                or not torch.equal(source[..., :1024], pack['transport_source_keys'])):
            raise ValueError('PBR RAW context and cached FRAME descriptors disagree')
        pack['transport_target_context'] = target
        pack['transport_source_context'] = source
        return pack

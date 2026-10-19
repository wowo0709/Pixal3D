"""Existing C1 cache carrier plus explicit inference-only RAW Omega context."""
import numpy as np
import torch

from .deformnext import DeformNeXtSS64


class DeformNeXtDeformationQuerySS64(DeformNeXtSS64):
    def __init__(self,*args,deformation_query_channels,**kwargs):
        if (deformation_query_channels!=2048 or kwargs.get('transport_arm')!='c1'
                or kwargs.get('task_matching_features') is not True
                or kwargs.get('transport_implementation')!='r9-native-v5'
                or kwargs.get('match_root') is None):
            raise ValueError('native C1 prediction cache and explicit RAW2048 query channels required')
        super().__init__(*args,**kwargs)
        self.deformation_query_channels=deformation_query_channels

    def __getitem__(self,index):
        pack=super().__getitem__(index);row=self.rows[index]
        if 'transport_spatial_indices' in pack or 'transport_deformation' in pack:
            raise ValueError('deformation query must be formed online without GT correspondence')
        with np.load(self._resolve(row['transport_cache'])) as archive:
            if str(archive['conditioning_input_sha256'])!=row['conditioning_input_sha256']:
                raise ValueError('RAW Omega context does not match actual input RGB')
            target=torch.from_numpy(np.asarray(archive['target_keys']).copy())
            source=torch.from_numpy(np.asarray(archive['source_keys']).copy())
        if (target.shape!=(1024,2048) or source.shape!=(3,1024,2048)
                or not torch.equal(target[...,:1024],pack['transport_target_keys'])
                or not torch.equal(source[...,:1024],pack['transport_source_keys'])):
            raise ValueError('RAW FRAME+GLOBAL context and existing FRAME descriptor disagree')
        pack['transport_target_context']=target
        pack['transport_source_context']=source
        return pack

"""Native SS inputs plus inference-only RAW Omega; no cached match labels."""
import numpy as np
import torch

from .deformnext import DeformNeXtSS64


class DeformNeXtMemberRawSS64(DeformNeXtSS64):
    def __init__(self,*args,deformation_query_channels,**kwargs):
        if (deformation_query_channels!=2048 or kwargs.get('transport_arm')!='c1'
                or kwargs.get('transport_implementation')!='r9-native-v5'
                or kwargs.get('match_root') is not None
                or kwargs.get('deformation_root') is not None
                or kwargs.get('task_matching_features',False) is not False):
            raise ValueError('explicit RAW2048 with original C1 input features and no match/deformation sidecars required')
        super().__init__(*args,**kwargs)
        self.deformation_query_channels=deformation_query_channels

    def _common(self,row):
        pack=super()._common(row)
        if {'transport_deformation','transport_spatial_indices','transport_semantic_indices',
                'transport_matching_logits'}&pack.keys():
            raise ValueError('RAW context must not include cached correspondence or GT routes')
        with np.load(self._resolve(row['transport_cache'])) as cache:
            if str(cache['conditioning_input_sha256'])!=row['conditioning_input_sha256']:
                raise ValueError('RAW descriptors and actual RGB identity disagree')
            target=np.asarray(cache['target_keys']).copy()
            source=np.asarray(cache['source_keys']).copy()
        if (target.shape!=(1024,2048) or source.shape!=(3,1024,2048)
                or not np.isfinite(target).all() or not np.isfinite(source).all()):
            raise ValueError('finite complete RAW FRAME+GLOBAL2048 context required')
        target,source=torch.from_numpy(target),torch.from_numpy(source)
        if (not torch.equal(target[...,:1024],pack['transport_target_keys'])
                or not torch.equal(source[...,:1024],pack['transport_source_keys'])):
            raise ValueError('RAW context FRAME channels differ from original matching keys')
        return dict(pack,transport_target_context=target,transport_source_context=source)

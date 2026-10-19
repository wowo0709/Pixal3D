"""PBR transport with explicit Omega feature loading for target-ray attention."""
from .deformnext import DeformNeXtPBR1024


class DeformNeXtFeaturePBR1024(DeformNeXtPBR1024):
    def __init__(self,*args,task_matching_features,**kwargs):
        if (task_matching_features is not True or kwargs.get('transport_arm') != 'c1'
                or kwargs.get('transport_implementation') != 'r9-native-v5'
                or kwargs.get('match_root') is None):
            raise ValueError('explicit cached semantic C1 PBR feature inputs required')
        super().__init__(*args,**kwargs)
        self.transport_keys = tuple(dict.fromkeys((*self.transport_keys,
            'target_keys','source_keys','target_query')))
        if 'spatial_indices' in self.transport_keys or 'deformation' in self.transport_keys:
            raise ValueError('PBR feature attention must not load GT correspondence/deformation')

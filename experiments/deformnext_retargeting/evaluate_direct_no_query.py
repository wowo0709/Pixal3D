"""Read-only Direct evaluation with physical absence of deformation channels."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    from pixal3d.modules import ray_feature_attention as module
    from pixal3d.modules.direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner
    from . import evaluate_ray_feature_attention as native
    proof_path = Path(sys.argv[sys.argv.index('--feature-identity-proof')+1])
    proof = json.loads(proof_path.read_text())
    if (proof.get('variant')!='direct-physical-NoDQ'
            or proof.get('physical_deformation_channels')!=0
            or proof.get('deformation_projection_present') is not False):
        raise ValueError('actual Direct NoDQ native proof required')
    sha = lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sources = {str(Path(__file__).resolve()):sha(__file__)}
    captured = []

    class Recorded(DirectRayRetargetingNoQueryConditioner):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            captured.append(self)

    original = module.RayFeatureAttentionConditioner
    module.RayFeatureAttentionConditioner = Recorded
    try:
        native.main()
    finally:
        module.RayFeatureAttentionConditioner = original
    path = Path(sys.argv[sys.argv.index('--output')+1])/'evaluation.json'
    result = json.loads(path.read_text())
    if (len(captured)!=1 or captured[0].retargeter.input_dim!=2048
            or hasattr(captured[0].retargeter,'deformation_projection')
            or result['optimizer_updates_during_evaluation']!=0
            or any(sha(p)!=digest for p,digest in sources.items())):
        raise ValueError('actual appearance-only Direct evaluation required')
    result['direct_no_query'] = dict(variant='direct-physical-NoDQ',
        physical_deformation_channels=0,retargeter_input_channels=2048,
        deformation_projection_present=False,zero_deformation_observations_used=False,
        direct_feature_delivery=True,source_SHA256=sources,optimizer_updates=0)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()

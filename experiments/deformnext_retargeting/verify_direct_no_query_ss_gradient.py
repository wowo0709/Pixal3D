"""Native nine-tensor Direct proof with RAW deformation channels absent."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    from pixal3d.modules.direct_ray_retargeting_no_query import DirectRayRetargetingNoQueryConditioner
    from . import verify_ray_feature_ss_gradient as native

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    config = Path(sys.argv[sys.argv.index('--config')+1])
    cfg = json.loads(config.read_text())
    if (cfg['dataset']['name']!='DeformNeXtSS64'
            or cfg['trainer']['args']['deform_conditioner']['arm']!='c1'
            or cfg['trainer']['args']['deform_conditioner'].get('deformation_dim') is not None
            or cfg['dataset']['args'].get('deformation_root') is not None):
        raise ValueError('actual C1 appearance-only dataset/module required')
    sources = {str(Path(__file__).resolve()):sha(__file__),
        str(Path(__file__).parents[2]/'pixal3d/modules/direct_ray_retargeting_no_query.py'):
            sha(Path(__file__).parents[2]/'pixal3d/modules/direct_ray_retargeting_no_query.py')}
    original = native.RayFeatureAttentionConditioner
    native.RayFeatureAttentionConditioner = DirectRayRetargetingNoQueryConditioner
    try:
        native.main()
    finally:
        native.RayFeatureAttentionConditioner = original
    output = Path(sys.argv[sys.argv.index('--output')+1])/'audit.json'
    result = json.loads(output.read_text())
    if (result['status']!='PASS_ACTUAL_SS_ZERO_FEATURE_OUTPUT_IDENTITY_AND_NINE_TENSOR_GRADIENT'
            or not result['actual_initial_evidence_exact']
            or not result['nine_feature_tensors_receive_gradient_by_update3']
            or any(sha(path)!=digest for path,digest in sources.items())):
        raise ValueError('actual Direct nine-tensor native proof required')
    result.update(variant='direct-physical-NoDQ',physical_deformation_channels=0,
        retargeter_input_channels=2048,deformation_projection_present=False,
        zero_deformation_observations_used=False,
        feature_delivery='bounded correction through native retarget_residual',
        direct_no_query_native_adapter=True)
    result['source_SHA256'].update(sources)
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()

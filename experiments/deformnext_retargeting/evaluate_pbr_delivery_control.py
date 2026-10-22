"""Same-weight PBR32 failure-path diagnosis on the existing native evaluator."""
import argparse
import json
from pathlib import Path
import runpy
import sys


def main():
    parser=argparse.ArgumentParser(add_help=False,allow_abbrev=False)
    from .pbr_delivery_controls import MODES,delivery_control
    parser.add_argument('--delivery-control',choices=MODES,required=True)
    local,remaining=parser.parse_known_args()
    if (remaining[remaining.index('--stage')+1]!='pbr1024'
            or any(flag in remaining for flag in ['--transport-control','--r9-path-control','--task-value-control','--matching-temperature'])):
        raise ValueError('one declared saved PBR delivery intervention only')
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    original=ImageConditionedProjMixin._encode_deform_views
    events=[]
    def encode(self,images,camera_info,transport,**kwargs):
        if transport is None:
            raise ValueError('actual legacy PBR source input required')
        with delivery_control(self.deform_conditioner,local.delivery_control,events):
            return original(self,images,camera_info,transport,**kwargs)
    previous=sys.argv
    try:
        ImageConditionedProjMixin._encode_deform_views=encode
        sys.argv=['evaluate_streamed_pbr','--delegate','direct',*remaining]
        runpy.run_module('experiments.deformnext_retargeting.evaluate_streamed_pbr',run_name='__main__')
    finally:
        sys.argv=previous
        ImageConditionedProjMixin._encode_deform_views=original
    output=Path(remaining[remaining.index('--output')+1]);path=output/'evaluation.json';result=json.loads(path.read_text())
    if not events or result['optimizer_updates_during_evaluation']!=0:
        raise ValueError('actual frozen source forwards without updates required')
    result['diagnostic_only']=local.delivery_control!='normal'
    result['delivery_control']=dict(mode=local.delivery_control,events=events,
        process_local_binding_restored=True,same_saved_NN_and_source_weights=True,
        R_and_RAW2048_channels_not_removed=True,
        NoScore_definition_still_Top1_feature=True,NoCorr_definition_still_native_spatial=True,
        not_a_retrained_architecture_or_independent_test=True,model_adopted=False,full_goal_complete=False)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()

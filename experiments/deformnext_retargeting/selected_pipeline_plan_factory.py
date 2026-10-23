"""Create development case plans by changing only model/source/output bindings.

All numerical sampling, RGB/camera, original Shape and rendering parameters
come from the closed existing per-asset templates. No selection or queue IO.
"""
from copy import deepcopy
from pathlib import Path

from .collect_module_seed_confirmation import validate_file
from .archived_native_guides import POLICY
from .selected_pipeline_cascade_contract import validate_geometry_plan
from .selected_pipeline_model_contract import require, validate_ss_binding, validate_pbr_binding


def _path(path):
    require(isinstance(path,str) and Path(path).is_absolute(),'explicit absolute execution path required')
    return path


def _source(old,new):
    require(all(new.get(path)==value for path,value in old.items()),
        'original numerical source pins must remain unchanged')
    return deepcopy(new)


def _render(render,shared,output):
    out=Path(output);r,s=deepcopy(render),deepcopy(shared)
    r.update(generation_root=str(out/'generation'),output=str(out/'render'))
    s.update(prediction_render_root=str(out/'render'),output=str(out/'shared'))
    return r,s


def geometry_case(case,geometry,render,shared,*,arm,binding,evaluation,terminal,
                  artifact,name,output,paths,ready,source_SHA256):
    """Use a new SS archive with the same per-asset RGB, Shape and common PBR."""
    validate_ss_binding(binding,arm)
    for spec in [evaluation,terminal,artifact]:validate_file(spec)
    for value in [output,*paths.values()]:_path(value)
    require(set(paths)=={'case','geometry','render','shared'},'four own plan artifact paths required')
    require(case['sample_id']==geometry['row']['sample_id']==shared['sample_id'],
        'same exact template sample required')
    require(output!=case['output'] and len(set(paths.values()))==4,
        'new output and distinct own plan files required')
    c,g=deepcopy(case),deepcopy(geometry)
    old_evaluation,old_terminal=c['SS_evaluation'],c['SS_terminal']
    c.update(name=name,SS_arm=arm,SS_model_binding=deepcopy(binding),output=output,
        SS_evaluation=evaluation['path'],SS_terminal=terminal['path'],
        geometry_template=case['geometry_plan'],geometry_plan=paths['geometry'],
        render_plan=paths['render'],shared_plan=paths['shared'])
    c['source_SHA256']=_source(case['source_SHA256'],source_SHA256)
    g.update(SS_evaluation=evaluation['path'],output=str(Path(output)/'generation'))
    g['source_SHA256']=_source(geometry['source_SHA256'],source_SHA256)
    for record in [c,g]:
        for field in ['input_SHA256' if record is c else 'inputs_SHA256','closed_inputs']:
            for path in [old_evaluation,old_terminal]:
                if path!=c['reference_SS']:record[field].pop(path,None)
        hashes=record['input_SHA256' if record is c else 'inputs_SHA256']
        for spec in [evaluation,terminal,artifact]:
            hashes[spec['path']]=spec['SHA256'];record['closed_inputs'][spec['path']]=spec['receipt']
    r,s=_render(render,shared,output)
    validate_geometry_plan(g,geometry,ready)
    return dict(case=c,geometry=g,render=r,shared=s)


def replay_case(case,replay,render,shared,*,arm,ss_binding,pbr_binding,
                evaluation,reference,base_case,name,output,paths,source_SHA256):
    """Move a validated mode's replay to the chosen model's own saved geometry."""
    mode=replay['mode'];validate_ss_binding(ss_binding,arm);validate_pbr_binding(pbr_binding,mode)
    require(case['mode']==mode and replay['checkpoint_kind']=='trained-native1900'
        and replay['engineering_only'] is False and replay['PBR_replay_seed']==20261020
        and replay['input_states']==['M1','M0','M0','M0'] and len(replay['RGB_paths'])==4
        and replay['texture_support_policy']==POLICY,
        'same trained native1900 image-only seed-reset replay template required')
    for spec in [evaluation,reference]:validate_file(spec)
    for value in [output,*paths.values()]:_path(value)
    require(set(paths)=={'case','replay','render','shared'},'four own replay plan artifact paths required')
    require(output!=case['output'] and len(set(paths.values()))==4,
        'new replay output and distinct own plan files required')
    require(case['sample_id']==replay['row']['sample_id']==base_case['sample_id']==shared['sample_id']
        and case['source_cluster_id']==base_case['source_cluster_id'],
        'same exact sample/source cluster required for replay')
    require(pbr_binding['native_config']['path']==replay['PBR_config']
        and pbr_binding['checkpoint']['path']==replay['PBR_checkpoint']
        and replay['conditioner_checkpoint']==(None if pbr_binding['conditioner_checkpoint'] is None
            else pbr_binding['conditioner_checkpoint']['path']),
        'own actual selected native1900 config/NN/source must match the replay template')
    c,rp=deepcopy(case),deepcopy(replay)
    old_geometry=Path(replay['generated_shape_root']);new_geometry=Path(base_case['generation'])
    require(new_geometry==Path(base_case['case_output'])/'generation',
        'own original base case geometry required')
    c.update(name=name,SS_arm=arm,SS_model_binding=deepcopy(ss_binding),PBR_model_binding=deepcopy(pbr_binding),
        sample_id=base_case['sample_id'],source_cluster_id=base_case['source_cluster_id'],output=output,
        common_case=base_case['case_output'],SS_evaluation=evaluation['path'],reference_SS=reference['path'],
        replay_plan=paths['replay'],render_plan=paths['render'],shared_plan=paths['shared'])
    c['source_SHA256']=_source(case['source_SHA256'],source_SHA256)
    rp.update(name=name+'-replay',generated_shape_root=str(new_geometry),
        native_guides=str(new_geometry/'native-shape-guides.pt'),output=str(Path(output)/'generation'))
    rp['source_SHA256']=_source(replay['source_SHA256'],source_SHA256)
    # The old case may include a base-case audit outside its generation folder.
    old_case=Path(case['common_case'])
    for record in [c,rp]:
        for field in ['input_SHA256','closed_inputs']:
            for path in list(record[field]):
                if Path(path).is_relative_to(old_geometry) or Path(path)==old_case/'audit.json':
                    record[field].pop(path)
        for filename in ['generation.json','prediction.npz','geometry-evaluation.npz','native-shape-guides.pt']:
            record['closed_inputs'][str(new_geometry/filename)]=str(new_geometry/'file4-closure.json')
    c['closed_inputs'][str(Path(base_case['case_output'])/'audit.json')]=str(Path(base_case['case_output'])/'file4-closure.json')
    for spec in [evaluation,reference]:
        c['input_SHA256'][spec['path']]=spec['SHA256'];c['closed_inputs'][spec['path']]=spec['receipt']
    nr,sr=_render(render,shared,output)
    return dict(case=c,replay=rp,render=nr,shared=sr)

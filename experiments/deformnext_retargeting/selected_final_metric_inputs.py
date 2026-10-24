"""Actual frozen selected cases and exact archived pixels for final metrics."""
from pathlib import Path

import numpy as np

from .bind_generated_pbr_dataset import load_QA
from .cascade_view_roles import file_sha256 as sha
from .collect_module_seed_confirmation import read_file,verify_terminal
from .metric_image_identity import scored_image_observation
from .perceptual_image_inputs import paired_images
from .run_selected_final_campaign import required_sources as campaign_sources
from .selected_final_case_integrity import CASE_STATUS
from .selected_final_metric_summary import reference_binding,validate_contract
from .selected_final_population import admit_all_ss,compose_models
from .selected_pipeline_freeze_v2 import exact
from .selected_pipeline_model_contract import require


MANIFEST_STATUS='PREPARED_ACTUAL_SELECTED_PHYSICAL508_SAME_PIXEL_METRIC_INPUTS'


def required_sources():
    root=Path(__file__).resolve().parent
    names=['selected_final_metric_inputs','selected_final_metric_summary','prepare_selected_final_metrics',
        'run_selected_final_metric','selected_inception_entry','collect_selected_final_metrics','evaluate_official_perceptual','extract_official_inception',
        'assigned_gpu_entry','perceptual_image_inputs','metric_image_identity','inception_distribution_metrics',
        'native_final_metric_summary','summarize_inception_pools']
    return sorted(set(campaign_sources()) | {str(root/(name+'.py')) for name in names})


def frozen_inputs(context,check):
    inputs=context['source_inputs'];parent=read_file(inputs['parent_freeze'],check);controls=read_file(inputs['control_freeze'],check)
    models=compose_models(parent,controls);validate_contract(parent)
    frozen=parent['evaluation_protocol']['source_SHA256']
    require(set(required_sources())<=set(context['source_SHA256']) and all(frozen.get(p)==d
        for p,d in context['source_SHA256'].items() if p!=context['closure_command'][1]),
        'actual final metric/campaign numerical and binding sources must be frozen before TEST')
    return parent,controls,models


def read_pixels(path,*,alias,sid,cluster,check):
    path=Path(path);digest=check.check(path,path.parent/'file4-closure.json')
    with np.load(path,allow_pickle=False) as arrays:
        pixels={k:arrays[k].copy() for k in ['prediction','prediction_alpha','GT_shared_renderer','GT_alpha']}
    require(pixels['prediction_alpha'].dtype==np.float32 and pixels['prediction_alpha'].shape==(512,512)
        and np.isfinite(pixels['prediction_alpha']).all(),'actual finite native512 FP32 prediction alpha required')
    observation=scored_image_observation(pixels,sample_id=sid,camera_id=int(path.stem.removeprefix('camera')),
        source_cluster_id=cluster,archive=str(path),archive_SHA256=digest)
    _,audit=paired_images(pixels['prediction'],pixels['GT_shared_renderer'],pixels['GT_alpha'])
    return dict(arm=alias,sample_id=sid,source_cluster_id=cluster,camera_id=observation['camera_id'],
        shared_npz=str(path),shared_npz_SHA256=digest,image_audit=audit,pixel_observation=observation)


def validate_images(rows,*,alias,parent,check):
    ids=parent['population']['sample_ids'];clusters=parent['population']['source_clusters_by_sample']
    require([(r['arm'],r['sample_id'],r['source_cluster_id'],r['camera_id']) for r in rows]
        ==[(alias,sid,clusters[sid],cid) for sid in ids for cid in [8,9,10]]
        and len({r['shared_npz'] for r in rows})==1524,'complete distinct ordered actual1524 sample/source/camera images required')
    for expected in rows:
        actual=read_pixels(expected['shared_npz'],alias=alias,sid=expected['sample_id'],cluster=expected['source_cluster_id'],check=check)
        require(exact(actual,expected),'actual archived scored RGB/GT/alpha/crop/conversion bytes changed')


def feature_observations(rows,*,reference=False):
    result=[]
    for row in rows:
        p=row['pixel_observation'];result.append(dict(sample_id=p['sample_id'],source_cluster_id=p['source_cluster_id'],camera_id=p['camera_id'],
            archive=p['archive'],archive_SHA256=p['archive_SHA256'],array='GT_shared_renderer' if reference else 'prediction',
            original_float_RGB_SHA256=p['reference_float_RGB_SHA256' if reference else 'prediction_float_RGB_SHA256']))
    return result


def expected_feature_observations(rows,*,reference=False):
    return [dict(obs,converted_CHW_uint8_SHA256=row['pixel_observation'][
        'reference_CHW_uint8_SHA256' if reference else 'prediction_CHW_uint8_SHA256'])
        for obs,row in zip(feature_observations(rows,reference=reference),rows)]


def load_campaign(context,check,*,pixels=True):
    parent,controls,models=frozen_inputs(context,check);inputs=context['source_inputs']
    runtime=read_file(inputs['campaign_context'],check);result=read_file(inputs['campaign_result'],check)
    from .run_selected_final_campaign import validate
    validate(runtime)
    require(exact(runtime['source_inputs']['parent_freeze'],inputs['parent_freeze'])
        and exact(runtime['source_inputs']['control_freeze'],inputs['control_freeze'])
        and result.get('status')=='PASS_ACTUAL_SELECTED_FINAL_CASE_CAMPAIGN'
        and result.get('assets')==508 and result.get('models')==list(models)
        and result.get('all_actual_campaign_children_drained') is True and result.get('no_new_model_selection') is True,
        'actual complete508 campaign with these precise frozen physical models required')
    ids=parent['population']['sample_ids'];clusters=parent['population']['source_clusters_by_sample']
    require(set(result['actual_cases'])==set(ids) and all(set(v)==set(models) for v in result['actual_cases'].values()),
        'complete actual508 cases for every deduplicated model required')
    original=runtime['source_inputs'];records={a:read_file(s,check) for a,s in original['SS_evaluations'].items()}
    admitted=admit_all_ss(records,parent=parent,controls=controls)
    read_file(original['cache_QA'],check);read_file(original['cache_QA_plan'],check);rows=read_file(original['manifest'],check,lines=True);load_QA(original)
    require([r['sample_id'] for r in rows]==ids and {r['sample_id']:r['source_cluster_id'] for r in rows}==clusters,
        'actual same complete508/134-source manifest required')
    cases={a:{} for a in models};shared={a:{} for a in models};images={a:[] for a in models};references={}
    import hashlib
    for alias,model in models.items():
        for sid in ids:
            descriptor=result['actual_cases'][sid][alias];case=read_file(descriptor['audit'],check);terminal=read_file(descriptor['worker_terminal'],check)
            root=Path(runtime['campaign_root'])/'cases'/hashlib.sha256(sid.encode()).hexdigest()[:24]/alias
            require(descriptor['audit']['path']==str(root/'audit.json') and case.get('status')==CASE_STATUS
                and case.get('actual_completed_native_case') is True and case.get('alias')==alias and case.get('sample_id')==sid
                and case.get('source_cluster_id')==clusters[sid] and case.get('case_output')==str(root)
                and exact(case.get('frozen_model_binding'),model) and exact(case.get('parent_freeze'),{k:inputs['parent_freeze'][k] for k in ['path','SHA256']})
                and exact(case.get('control_freeze'),{k:inputs['control_freeze'][k] for k in ['path','SHA256']})
                and case.get('actual_SS_prediction_SHA256')==admitted[alias][sid]['prediction_voxels_SHA256']
                and terminal.get('case_output')==str(root) and terminal.get('actual_session_drained') is True,
                'actual own frozen SS/Shape/PBR case and worker terminal required')
            verify_terminal(terminal)
            for item in case['terminals']:verify_terminal(item)
            folder=root/'shared';report=read_file(dict(path=str(folder/'render-evaluation.json'),SHA256=sha(folder/'render-evaluation.json'),receipt=str(folder/'file4-closure.json')),check)
            require(len(case['scored_image_observations'])==3,'all three actual producer pixel observations required')
            for cid,observed in zip([8,9,10],case['scored_image_observations']):
                if pixels:
                    image=read_pixels(folder/f'camera{cid:02d}.npz',alias=alias,sid=sid,cluster=clusters[sid],check=check)
                    require(exact(image['pixel_observation'],observed),'actual image archive must equal the native case producer observation')
                    identity=tuple(observed[k] for k in ['reference_float_RGB_SHA256','reference_CHW_uint8_SHA256','reference_alpha_SHA256'])
                    require(references.setdefault((sid,cid),identity)==identity,'same original-material M1 GT/reference alpha required for every model')
                    images[alias].append(image)
            cases[alias][sid]=case;shared[alias][sid]=report
    return parent,controls,models,cases,shared,images


def validate_manifest(manifest,parent,controls,models,source_inputs):
    ids=parent['population']['sample_ids']
    require(manifest.get('status')==MANIFEST_STATUS and manifest.get('scope')=='final-test508'
        and exact(manifest.get('models'),models) and exact(manifest.get('source_inputs'),source_inputs)
        and manifest.get('assets')==508 and manifest.get('source_clusters')==134 and manifest.get('sample_ids')==ids
        and exact(manifest.get('reference_binding'),reference_binding(parent,controls))
        and tuple(manifest['images'])==tuple(models) and all(len(rows)==1524 for rows in manifest['images'].values()),
        'actual same complete physical models/population/campaign/reference input manifest required')

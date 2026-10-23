"""Independent PBR seed evidence on the existing fixed-four stage protocol.

This is a training/GT-shape trend confirmation. Generated-geometry/source
appearance effects must still come from the measured joint DEV pipelines.
"""
from copy import deepcopy

from .native_final_freeze import validate_spec
from .native_pbr_stage_summary import IDENTITY,STEPS,validate_curve,is_digest
from .pbr_confirmation_recipe import validate_confirmation
from .selected_pipeline_model_contract import pbr_binding,require


STATUS='PASS_ACTUAL_SELECTED_PBR_TWO_SEED_FIXED4_STAGE_CONFIRMATION'


def modes_for_selected(selected):
    from .common_geometry_pbr_contract import MODES
    require(selected in MODES,'actual selected native PBR mode required')
    return tuple(dict.fromkeys(('vanilla','top1-only',selected)))


def summarize(records,configs,hashes,*,selected,proofs,original_multiview,
              original_checkpoint,frozen_base_proofs):
    modes=modes_for_selected(selected)
    validate_spec(original_checkpoint)
    require(set(records)==set(configs)==set(hashes)==set(frozen_base_proofs)=={101,202}
        and set(proofs)==set(modes),'both actual training seeds and exact deduplicated modes required')
    models,curves,trace_identity={}, {},{}
    reference=None
    for seed in (101,202):
        require(set(records[seed])==set(configs[seed])==set(hashes[seed])==set(frozen_base_proofs[seed])==set(modes),
            'actual selected PBR and both strong controls required at each seed')
        models[seed],curves[seed]={},{}
        for mode in modes:
            record=records[seed][mode]
            require(record.get('status')=='PASS_ACTUAL_NATIVE_PBR1900_FIXED4_STAGE_SNAPSHOT'
                and record.get('mode')==mode,'own actual completed fixed-four stage snapshot required')
            cfg=configs[seed][mode];cp=record['completion_proof'];ranks=cp['ranks']
            require(len(ranks)==4 and {r.get('rank') for r in ranks}=={0,1,2,3}
                and type(cfg['training_seed']) is int and cfg['training_seed']==seed,
                'actual complete own four-rank PBR1900 seed required')
            models[seed][mode]=pbr_binding(cfg,{str(r['rank']):r for r in ranks},hashes[seed][mode])
            a,t,w,p=cp['active'],cp['terminal'],cp['wrapper'],cp['progress']
            require(a.get('PID')==t.get('PID') and a.get('birth')==t.get('birth')
                and type(a.get('PID')) is int and a['PID']>0
                and type(t.get('returncode')) is int and t['returncode']==0
                and a.get('physical_GPUs')==t.get('physical_GPUs')==['0','1','2','3']
                and a.get('mode')==mode and p.get('stage')=='completed' and p.get('current')==1900
                and w.get('status')=='PASS_ACTUAL_FRESH_NATIVE_PBR1900_SCIENCE'
                and w.get('mode')==mode and w.get('fresh_optimizer') is True
                and w.get('actual_optimizer_updates')==1900
                and w.get('actual_five_epoch_NN_source_Adam_canonical_archives_verified') is True,
                'actual owned terminal and closed complete1900 training artifacts required')
            if seed==202:
                require(t.get('actual_session_drained') is True,'seed202 producer must record actual session drain')
            base=frozen_base_proofs[seed][mode]
            require(base.get('actual_original_frozen_NN_base_tensors_exact') is True
                and base.get('LoRA_tensors')==360 and base.get('frozen_tensors',0)>0,
                'actual own checkpoint frozen base tensor comparison required')
            curves[seed][mode]=validate_curve(record['validation_rows'])
            traces=record['training_trace_digests']
            require(set(traces)=={'0','1','2','3'} and all(is_digest(v) for v in traces.values()),
                'actual complete1900 per-rank training trace SHA256 required')
            if mode=='vanilla':trace_identity[seed]=traces
            else:require(traces==trace_identity[seed],'actual inputs/noise/timestep/global traces differ within TRAIN seed')
            first=record['validation_rows'][0]
            if reference is None:reference=first
            require(first['quick_panel_ids']==reference['quick_panel_ids'], 'fixed validation4 assets differ')
            for row,ref in zip(first['quick_per_asset'],reference['quick_per_asset']):
                require(all(row[k]==ref[k] for k in IDENTITY)
                    and row['actual_prediction_PBR_latent_SHA256']==ref['actual_prediction_PBR_latent_SHA256']
                    and row['metrics']==ref['metrics'],'actual cold native initial4 differs across models or TRAIN seeds')
            pixels=record['GT_PNG_pixel_SHA256']
            require(set(pixels)==set(map(str,STEPS))
                and all(set(v)==set(first['quick_panel_ids']) and all(is_digest(d) for d in v.values()) for v in pixels.values())
                and pixels==records[101]['vanilla']['GT_PNG_pixel_SHA256'],
                'same actual GT pixels across every model, epoch and training seed required')
    require(trace_identity[101]!=trace_identity[202], 'independent TRAIN seeds must use different actual training streams')
    for mode in modes:
        validate_confirmation(configs[101][mode],configs[202][mode],proof=proofs[mode],original_multiview=original_multiview)
        require(configs[101][mode]['trainer']['args']['continuation_ckpt']['denoiser']==original_checkpoint['path'],
            'actual original PBR16850 initializer differs')
    return dict(status=STATUS,selected_mode=selected,measured_modes=list(modes),training_seeds=[101,202],
        model_bindings=deepcopy(models),curves=curves,actual_original_checkpoint=deepcopy(original_checkpoint),
        actual_original_frozen_base_tensors_exact_in_all_models=True,
        actual1900_per_rank_training_traces_paired_within_seed=True,
        actual_training_streams_differ_between_seeds=True,actual_initial4_predictions_exact=True,
        source_multiview_step=16850,assets=4,fixed_GT_shape=True,camera='target8',
        source_preservation_measured=False,generated_geometry_measured=False,
        fixed4_is_not_inferential_PBR_quality_confirmation=True,
        independent_pipeline_seed_confirmation_not_proven=True,
        automatic_model_adoption=False,full_goal_complete=False)

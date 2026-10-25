from copy import deepcopy

import pytest

from experiments.deformnext_retargeting.final_fixed_asset_inputs import SCHEMA,ROLES,validate_packet


def packet():
    ids=[str(i) for i in range(128)];clusters={sid:str(i%93) for i,sid in enumerate(ids)};records=[]
    for sid in ids:
        paths={k:'/file4/data/'+sid+'/'+k for k in ROLES}
        row=dict(sample_id=sid,source_cluster_id=clusters[sid],split='validation',
            conditioning_camera_ids=[8,9,10,11],target_anchor_kind='rgb',
            target_anchor=paths['target_rgb'],target_rgb_input=paths['target_rgb'],
            source_images=[paths[k] for k in ROLES[1:4]],
            evaluation_target_images=[paths[k] for k in ['GT_rgb_9','GT_rgb_10']],
            **{k:paths[k] for k in ['source_geometry','target_vertices','transport_cache','transport_cache_pbr']})
        inputs={k:dict(path='/mirror'+p,canonical_path=p,SHA256='a'*64,bytes=100) for k,p in paths.items()}
        records.append(dict(sample_id=sid,source_cluster_id=clusters[sid],row=row,
            canonical_files=paths,inputs=inputs,RGB_paths=[inputs[k]['path'] for k in ROLES[:4]],
            render_views=[dict(camera_id=c,input_camera_index=c-8,
                GT_RGBA=inputs['target_rgb' if c==8 else 'GT_rgb_'+str(c)]['path']) for c in [8,9,10]]))
    return dict(schema=SCHEMA,no_TEST_data=True,sample_ids=ids,source_clusters_by_sample=clusters,
        records=records,validation_manifest=dict(path='/manifest/validation.jsonl',SHA256='b'*64))


def test_complete_existing_prefix32_and_additional96_are_required_without_mutation():
    p=packet();before=deepcopy(p);assert validate_packet(p)==p['sample_ids'];assert p==before


@pytest.mark.parametrize('damage',['historical96-schema','historical96-records','suffix-order',
    'source','RGB-role','GT-role','material-missing','hash','target-depth','TEST','foreign-camera'])
def test_partial_extension_or_changed_roles_cannot_be_queued_as_full128(damage):
    p=packet();a=p['records'][-1]
    if damage=='historical96-schema':p['schema']='paired-DEV128-additional96-asset-inputs-v1'
    elif damage=='historical96-records':p['records']=p['records'][32:]
    elif damage=='suffix-order':p['records'][-1],p['records'][-2]=p['records'][-2:]
    elif damage=='source':a['source_cluster_id']='wrong'
    elif damage=='RGB-role':a['RGB_paths'][0]=a['RGB_paths'][1]
    elif damage=='GT-role':a['render_views'][1]['GT_RGBA']=a['RGB_paths'][1]
    elif damage=='material-missing':a['inputs'].pop('pbr_dump')
    elif damage=='hash':a['inputs']['target_rgb']['SHA256']='unknown'
    elif damage=='target-depth':a['row']['target_anchor_kind']='depth'
    elif damage=='TEST':a['row']['split']='test'
    else:a['row']['conditioning_camera_ids']=[0,1,2,3]
    with pytest.raises(ValueError):validate_packet(p)

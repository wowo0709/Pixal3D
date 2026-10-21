"""Native producer GT identities must match the shared source-world reference."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deformnext_retargeting.bind_native_surface_reference import target_identity, SurfaceReferences
from experiments.deformnext_retargeting.cascade_remote_results import array_identities
from experiments.deformnext_retargeting.cascade_view_roles import ClosedInputs
from experiments.deformnext_retargeting.prepare_surface_reference import surface_reference
from experiments.deformnext_retargeting.evaluate_bound_population import sha
from test_frozen_cascade_metric_plans import put, close_unit


@pytest.fixture
def geometry():
    source=np.asarray([[5.,-4.,1.],[9.,-4.,1.],[5.,0.,1.],[5.,-4.,5.]],np.float32)
    faces=np.asarray([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],np.int32)
    target=source+np.asarray([1.,.25,-.5],np.float32)
    arrays=surface_reference(source,faces,target,faces)
    reference=dict(point_count=20000,surface_seed=101,target_faces_equal_source=True,final_target_vertices_exact=True,
                   source_center=arrays['source_center'].tolist(),source_extent=float(arrays['source_extent']))
    identities=array_identities(dict(target_world=target,target_faces=faces,
        source_center=arrays['source_center'],source_extent=arrays['source_extent']))
    return source,faces,target,arrays,reference,identities


@pytest.mark.parametrize('problem',[None,'target-world','faces','center','extent','point','point-dtype','seed','count','target-fit'])
def test_same_native_GT_points_input_hashes_and_source_normalization_are_required(geometry,problem):
    source,faces,target,arrays,reference,identities=geometry
    if problem in ['target-world','faces','center','extent']:
        key={'target-world':'target_world','faces':'target_faces','center':'source_center','extent':'source_extent'}[problem]
        identities[key]['values_C_SHA256']='different-array'
    elif problem=='point':arrays['target_points'][0,0]+=.001
    elif problem=='point-dtype':arrays['target_points']=arrays['target_points'].astype(np.float32)
    elif problem=='seed':reference['surface_seed']=102
    elif problem=='count':reference['point_count']=10000
    elif problem=='target-fit':reference['source_center']=(target.min(0)+target.max(0)).tolist()
    kwargs=dict(source_vertices=source,source_faces=faces,target_vertices=target,
                reference_arrays=arrays,native_geometry_arrays=identities)
    if problem is None:
        actual=target_identity(reference,**kwargs)
        assert actual['native_GT_array_identities']==identities
        assert len(actual['target_points_SHA256'])==64
    else:
        with pytest.raises(ValueError):target_identity(reference,**kwargs)


@pytest.fixture
def remote_case(tmp_path,geometry):
    source,faces,target,arrays,reference,identities=geometry
    folder=tmp_path/'generation';folder.mkdir()
    native_source=Path(__file__).parents[2]/'experiments/deformnext_retargeting/cascade_from_ss_artifact.py'
    manifest=dict(schema='native-cascade-remote-npz-v1',remote_root='/file4/unit/raw',prefix='unit/generation',
        archives={'geometry-evaluation.npz':{'arrays':identities}},
        source_contract={'reverse_AST_exact':True,'source_SHA256':sha(native_source)})
    spec=put(folder/'remote-npz.json',manifest);close_unit(folder,tmp_path)
    metrics=dict(prediction_samples=20000,target_samples=20000,threshold=.02,per_output_alignment=False,
                 chamfer_L1=.01,surface_FScore=.9)
    generated=dict(sample_id='unit-asset',GT_loaded_only_after_generation=True,
        GT_geometry_or_coords_used_in_generation=False,source_defined_normalization=True,per_output_alignment=False,
        actual_resolution=1024,metrics=metrics,remote_raw_archives=dict(manifest=spec['path'],manifest_SHA256=spec['SHA256'],
            archives=manifest['archives'],remote_root=manifest['remote_root'],prefix=manifest['prefix']))
    references=object.__new__(SurfaceReferences)
    references.closed=ClosedInputs(tmp_path);references.spec={'unit_reference':True}
    identity=target_identity(reference,source_vertices=source,source_faces=faces,target_vertices=target,
        reference_arrays=arrays,native_geometry_arrays=identities)
    references.expected=lambda native:(dict(reference,source_cluster_id='unit-source'),
        {'path':'/unit-reference','SHA256':'unit-reference'},identity)
    return references,{'row':{'sample_id':'unit-asset'}},generated,folder,manifest


@pytest.mark.parametrize('problem',[None,'GT-during-generation','scale-fit','resolution','threshold','noise-count','producer-source','producer-array','producer-manifest'])
def test_post_generation_remote_geometry_join_uses_actual_producer_arrays_without_download(remote_case,problem):
    references,native,generated,folder,manifest=remote_case
    if problem=='GT-during-generation':generated['GT_geometry_or_coords_used_in_generation']=True
    elif problem=='scale-fit':generated['per_output_alignment']=True
    elif problem=='resolution':generated['actual_resolution']=512
    elif problem=='threshold':generated['metrics']['threshold']=.05
    elif problem=='noise-count':generated['metrics']['target_samples']=10000
    elif problem in ['producer-source','producer-array']:
        if problem=='producer-source':manifest['source_contract']['source_SHA256']='different-source'
        else:manifest['archives']['geometry-evaluation.npz']['arrays']['target_world']['values_C_SHA256']='different-target'
        spec=put(folder/'remote-npz.json',manifest);close_unit(folder,folder.parent)
        generated['remote_raw_archives'].update(manifest_SHA256=spec['SHA256'],archives=copy.deepcopy(manifest['archives']))
    elif problem=='producer-manifest':generated['remote_raw_archives']['manifest_SHA256']='different-manifest'
    if problem is None:
        actual=references.join(native,generated,generation_root=folder)
        assert actual['native_metrics']==generated['metrics']
        assert actual['no_prediction_archive_download_or_neural_forward']
        assert actual['GT_reference_not_used_for_model_conditioning']
    else:
        with pytest.raises(ValueError):references.join(native,generated,generation_root=folder)


def test_partial_reference_population_cannot_be_promoted_to_final508(tmp_path):
    report=dict(status='PASS_ALL508_GT_SURFACE_REFERENCES',samples=32,source_clusters=32,surface_samples=20000,
        surface_seed=101,FScore_threshold=.02,source_defined_normalization=True,per_output_alignment=False,
        target_recentered_or_rescaled=False,geometry_frame='source-defined normalized WORLD',
        all508_source_target_topologies_and_final_vertices_exact=True,reference_points_readback_exact=True)
    folder=tmp_path/'reference';folder.mkdir();spec=put(folder/'audit.json',report);close_unit(folder,tmp_path)
    with pytest.raises(ValueError,match='complete common'):
        SurfaceReferences(spec,receipt=str(folder/'file4-closure.json'),closed=ClosedInputs(tmp_path),ids=[],clusters={})

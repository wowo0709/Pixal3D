import json

import pytest

from experiments.deformnext_retargeting.generated_artifact_cache_retirement import pending_raw_reads,validate_raw_candidate


def test_protects_raw_root_consumers_but_summary_generation_metadata_is_not_raw(tmp_path):
    root=tmp_path/'generated';root.mkdir();plan=tmp_path/'plan.json'
    job=dict(name='consumer',command=['worker','--plan',str(plan)])
    for value in[dict(generation_root=str(root)),dict(generated_shape_root=str(root)),
                 dict(closed_inputs={str(root/'prediction.npz'):'receipt'}),dict(path=str(root/'geometry-evaluation.npz'))]:
        plan.write_text(json.dumps(value))
        assert pending_raw_reads([job],{},root)==['consumer']
        assert not pending_raw_reads([job],{'consumer':{'status':'completed'}},root)
    plan.write_text(json.dumps(dict(phase='summary',models={'vanilla':[dict(generation=str(root))]})))
    assert not pending_raw_reads([job],{},root)
    plan.unlink()
    with pytest.raises(ValueError,match='unreadable pending'):pending_raw_reads([job],{},root)


def test_retirement_requires_owned_canonical_rendered_scientific_artifact_and_never_deletes(tmp_path):
    owned=tmp_path/'our-native-replays';root=owned/'vanilla/asset/native';root.mkdir(parents=True)
    path=root/'prediction.npz';path.write_bytes(b'synthetic raw artifact, not actual output')
    args=dict(root=root,owned_outputs_root=owned,terminal={'returncode':0},
        generation=dict(status='PASS_ACTUAL_R_PBR_REPLAY_ON_GENERATED_SHAPE',checkpoint_kind='trained-native1900',
            engineering_only=False,sample_id='one',PBR_arm='vanilla',source_weights_unchanged=True,geometry_array_byte_identity={'vertices':True}),
        render_audit=dict(status='PASS_ACTUAL_NATIVE1900_PBR_CASE_RENDERED_AND_SHARED_SCORED',sample_id='one',mode='vanilla',
            same_raw_geometry_and_native_LOD=True,fixed_original_material_M1_GT=True),
        canonical_entry=dict(sha256='actual',bytes=path.stat().st_size),actual_local_SHA256='actual',readers=[])
    assert validate_raw_candidate(path,**args)['local_duplicate_only']and path.exists()
    for changed in[dict(readers=['pending renderer']),dict(owned_outputs_root=tmp_path/'someones-data'),
                   dict(terminal={'returncode':1}),dict(actual_local_SHA256='wrong'),
                   dict(generation=dict(args['generation'],engineering_only=True))]:
        with pytest.raises(ValueError):validate_raw_candidate(path,**dict(args,**changed))
    original=tmp_path/'original-data.npz';original.write_bytes(path.read_bytes());path.unlink();path.symlink_to(original)
    with pytest.raises(ValueError):validate_raw_candidate(path,**args)
    assert original.exists()

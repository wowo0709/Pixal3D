import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pixal3d.datasets.deformnext import DeformNeXtPBR1024, DeformNeXtSS64


def _image(path: Path, value: int):
    rgba = np.full((8, 8, 4), value, dtype=np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(path)


def _fixture(tmp_path: Path):
    anchor = tmp_path / "target_rgb.png"
    source_anchor=tmp_path/'source_anchor.png'
    _image(source_anchor,100)
    sources = [tmp_path / f"source_{index}.png" for index in range(3)]
    _image(anchor, 16)
    for index, path in enumerate(sources):
        _image(path, 64 + index)
    np.savez(
        tmp_path / "transport.npz",
        target_keys=np.zeros((4, 2), np.float32),
        source_keys=np.zeros((3, 5, 2), np.float32),
        source_values=np.zeros((3, 5, 6), np.float32),
        source_valid=np.ones((3, 5), bool),
        patch_grid=np.arange(8).reshape(2, 2, 2) % 4,
        spatial_indices=np.zeros((3, 4), np.int64),
        target_query=np.zeros((4, 3), np.float32),
        deformation=np.zeros((4, 3), np.float32),
    )
    np.savez(
        tmp_path / "transport_pbr.npz",
        source_values=np.full((3, 5, 6), 9, np.float32),
        target_query=np.full((4, 3), 7, np.float32),
        patch_ids=np.array([0, 1], np.int64),
    )
    np.savez(tmp_path / "ss.npz", z=np.ones((8, 2, 2, 2), np.float32))
    coords = np.array([[0, 1, 1], [1, 0, 1]], np.int32)
    np.savez(tmp_path / "pbr.npz", coords=coords, feats=np.ones((2, 32), np.float32))
    np.savez(tmp_path / "shape.npz", coords=coords, feats=np.ones((2, 32), np.float32))
    transform = np.eye(4)
    transform[1, 3] = -2.0
    row = {
        "sample_id": "sample-a", "split": "train",
        "target_anchor": anchor.name, "target_anchor_kind": "rgb", "target_rgb_input": anchor.name,
        "source_anchor_rgb": source_anchor.name,
        "conditioning_camera_ids":[8,9,10,11],
        "source_anchor_transform_matrix":transform.tolist(),"source_anchor_camera_angle_x":1.0,
        "source_images": [path.name for path in sources],
        "camera_angle_x": [1.0] * 4,
        "transform_matrix": [transform.tolist()] * 4,
        "mesh_scale": 0.96, "transport_cache": "transport.npz",
        "transport_cache_pbr": "transport_pbr.npz",
        "ss_latent": "ss.npz", "pbr_latent": "pbr.npz",
        "shape_latent": "shape.npz", "pbr_tokens": 2,
    }
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(json.dumps(row) + "\n")
    return manifest, row


def test_ss_loader_preserves_target_then_three_source_roles(tmp_path):
    manifest, _ = _fixture(tmp_path)
    dataset = DeformNeXtSS64(str(manifest), image_size=8, expected_split="train")
    item = dataset[0]
    assert item["cond"].shape == (4, 3, 8, 8)
    assert item["cond"][:, 0, 0, 0].tolist() == pytest.approx(
        [16 / 255, 64 / 255, 65 / 255, 66 / 255]
    )
    assert item["x_0"].shape == (8, 2, 2, 2)
    assert item["transport_source_valid"].dtype == torch.bool
    batch = dataset.collate_fn([item, item])
    assert batch["cond"].shape == (2, 4, 3, 8, 8)
    assert batch["transport_target_keys"].shape == (2, 4, 2)


def test_v1_loader_does_not_read_transport(tmp_path):
    manifest, _ = _fixture(tmp_path)
    (tmp_path / 'transport.npz').unlink()
    dataset = DeformNeXtSS64(
        str(manifest), image_size=8, expected_split='train', transport_arm='v1'
    )
    item = dataset[0]
    assert item['cond'].shape[0] == 4
    assert not any(key.startswith('transport_') for key in item)


def test_match_cache_avoids_loading_full_omega_keys(tmp_path):
    manifest, _ = _fixture(tmp_path)
    matches = tmp_path / "matches"
    matches.mkdir()
    np.savez(
        matches / "sample-a.npz",
        semantic_indices=np.zeros((4, 2), np.uint16),
        matching_logits=np.zeros((4, 2), np.float32),
        candidate_valid=np.ones((4, 2), bool),
    )
    dataset = DeformNeXtSS64(
        str(manifest), image_size=8, expected_split="train",
        transport_arm="b3", match_root=str(matches),
    )
    item = dataset[0]
    assert item["transport_semantic_indices"].shape == (4, 2)
    assert "transport_target_keys" not in item
    assert "transport_source_keys" not in item


def test_v2_cache_is_fenced_and_does_not_load_unused_source_values(tmp_path):
    import hashlib
    manifest,row=_fixture(tmp_path)
    matches=tmp_path/'matches-v2';matches.mkdir()
    contract=json.dumps({'head_sha256':'head-hash'})+'\n'
    (matches/'contract.json').write_text(contract)
    digest=hashlib.sha256(contract.encode()).hexdigest()
    arrays=dict(schema='deformnext-matching-v2-cache-v1',contract_sha256=digest,
                sample_id=row['sample_id'],split=row['split'],evidence_top1=np.zeros((1024,1024),np.float16),
                evidence_soft=np.zeros((1024,1024),np.float16),p_use=np.ones(1024,np.float32),cached_valid=np.ones(1024,bool))
    np.savez(matches/'sample-a.npz',**arrays)
    with np.load(tmp_path/'transport.npz') as z:
        retained={k:z[k] for k in z.files if k not in {'source_values','source_valid','target_keys','source_keys'}}
    np.savez(tmp_path/'transport.npz',**retained)
    dataset=DeformNeXtSS64(str(manifest),image_size=8,expected_split='train',transport_arm='b2',
        transport_implementation='matching-v2',expected_q2c_sha256='head-hash',match_root=str(matches))
    item=dataset[0]
    assert 'transport_source_values' not in item and 'transport_target_keys' not in item
    assert item['transport_evidence_top1'].shape==(1024,1024)
    arrays['contract_sha256']='stale';np.savez(matches/'sample-a.npz',**arrays)
    with pytest.raises(ValueError,match='contract differs'):dataset[0]


def test_edited_rgb_target_render_is_used_as_first_condition(tmp_path):
    manifest, row = _fixture(tmp_path)
    render = tmp_path / "render_cond"
    render.mkdir()
    target = render / "view00.png"
    _image(target, 99)
    row["target_anchor"] = str(target)
    row['target_rgb_input'] = str(target)
    manifest.write_text(json.dumps(row) + "\n")
    item=DeformNeXtSS64(str(manifest), image_size=8)[0]
    torch.testing.assert_close(item['cond'][0],torch.full((3,8,8),99/255))


def test_c2_reuses_reference_context_with_strict_head_and_rgb_fencing(tmp_path):
    import hashlib
    from pixal3d.modules.r9_matching import MATCHER_CONTRACT,ATTENTION_CONTRACT
    manifest,row=_fixture(tmp_path);row['conditioning_input_sha256']='rgb-input'
    manifest.write_text(json.dumps(row)+'\n')
    with np.load(tmp_path/'transport.npz') as z:source={k:z[k] for k in z.files}
    source.update(conditioning_input_sha256='rgb-input',target_query=np.ones((1024,1024),np.float16))
    np.savez(tmp_path/'transport.npz',**source)
    matches=tmp_path/'preaggregated';matches.mkdir();contract=json.dumps({'head_sha256':'reference-head'})+'\n'
    (matches/'contract.json').write_text(contract)
    np.savez(matches/'sample-a.npz',schema='deformnext-matching-v2-cache-v1',
        contract_sha256=hashlib.sha256(contract.encode()).hexdigest(),sample_id='sample-a',split='train',
        conditioning_input_sha256='rgb-input',evidence_top1=np.zeros((1024,1024),np.float16),
        evidence_soft=np.zeros((1024,1024),np.float32),p_use=np.ones(1024,np.float32),cached_valid=np.ones(1024,bool))
    sidecars=tmp_path/'reference-context';sidecars.mkdir()
    payload=dict(schema='deformnext-match-r9-v5',q2c_sha256='reference-head',conditioning_input_sha256='rgb-input',
        aggregation='global-categorical-v1',proposal_policy='GLOBAL64_VALID_OR_NATIVE',topk=64,
        matcher_contract=json.dumps(MATCHER_CONTRACT),attention_contract=json.dumps(ATTENTION_CONTRACT),
        deformation=np.full((1024,2048),.75,np.float16))
    np.savez(sidecars/'sample-a.npz',**payload)
    kwargs=dict(image_size=8,transport_implementation='matching-v2',expected_q2c_sha256='reference-head',
        match_root=str(matches),deformation_root=str(sidecars))
    dataset=DeformNeXtSS64(str(manifest),transport_arm='c2',**kwargs)
    item=dataset[0]
    assert item['transport_deformation'].shape==(1024,2048)
    assert torch.all(item['transport_deformation']==.75)
    with pytest.raises(ValueError,match='only for cached C2'):
        DeformNeXtSS64(str(manifest),transport_arm='c1',**kwargs)
    payload['q2c_sha256']='another-head';np.savez(sidecars/'sample-a.npz',**payload)
    with pytest.raises(ValueError,match='sidecar head/input contract'):dataset[0]
    payload['q2c_sha256']='reference-head';payload['conditioning_input_sha256']='different-RGB'
    np.savez(sidecars/'sample-a.npz',**payload)
    with pytest.raises(ValueError,match='sidecar head/input contract'):dataset[0]


def test_depth_target_manifest_cannot_silently_train(tmp_path):
    manifest,row=_fixture(tmp_path)
    row['target_anchor_kind']='geometry_depth'
    manifest.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='target conditioning must be RGB'):
        DeformNeXtSS64(str(manifest),image_size=8)


def test_target_anchor_cannot_point_to_a_different_depth_file(tmp_path):
    manifest,row=_fixture(tmp_path)
    row['target_anchor']='depth.png'
    manifest.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='explicit target_rgb_input'):
        DeformNeXtSS64(str(manifest),image_size=8)


def test_rgb_inputs_cannot_duplicate_target_camera_as_a_source_slot(tmp_path):
    manifest,row=_fixture(tmp_path)
    row['conditioning_camera_ids']=[8,8,9,10]
    manifest.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='four distinct input cameras'):
        DeformNeXtSS64(str(manifest),image_size=8)


def test_rgb_manifest_rejects_stale_depth_transport_for_same_sample(tmp_path):
    manifest,row=_fixture(tmp_path)
    row['conditioning_input_sha256']='new RGB input fingerprint'
    manifest.write_text(json.dumps(row)+'\n')
    dataset=DeformNeXtSS64(str(manifest),image_size=8)
    with pytest.raises(ValueError,match='edited RGB inputs'):
        dataset[0]


def test_rgb_manifest_migration_preserves_geometry_but_versions_image_caches(tmp_path):
    from experiments.deformnext_retargeting.prepare_rgb_manifest import convert, image_fingerprint
    _,row=_fixture(tmp_path)
    row.update(target_anchor_kind='geometry_depth',target_anchor='old_depth.png',
               target_rgb_supervision='target_rgb.png',evaluation_target_images=['view_08_M1.png','view_09_M1.png'])
    frames=[{'file_path':f'view{i:02d}.png','mesh_state':'M1' if i==0 else 'M0',
             'camera_id':8 if i==0 else 7+i,'camera_angle_x':1.0,
             'transform_matrix':row['transform_matrix'][0]} for i in range(5)]
    new=convert(row,Path('/data/rgb-v1'),frames)
    assert row['target_anchor']=='old_depth.png'
    assert new['target_anchor']==new['target_rgb_input']=='target_rgb.png'
    assert new['target_anchor_kind']=='rgb'
    assert new['target_depth_auxiliary']=='old_depth.png'
    for key in ['ss_latent','pbr_latent','shape_latent','transform_matrix','mesh_scale']:
        assert new[key]==row[key]
    assert new['transport_cache']!=row['transport_cache']
    assert new['transport_cache_pbr']!=row['transport_cache_pbr']
    for i in (2,3,4):_image(tmp_path/f'view{i:02d}.png',64+i)
    assert new['conditioning_camera_ids']==[8,9,10,11]
    assert new['evaluation_target_images']==['view_09_M1.png']
    assert new['source_anchor_rgb']=='view01.png'
    first=image_fingerprint(new,lambda p:tmp_path/p)
    _image(tmp_path/'target_rgb.png',127)
    assert image_fingerprint(new,lambda p:tmp_path/p)!=first


@pytest.mark.parametrize('mode', ['edited-rgb','source-only'])
def test_zero_shot_controls_change_images_and_cameras_together(tmp_path,mode):
    from experiments.deformnext_retargeting.conditioning_controls import ConditioningControl
    manifest,_=_fixture(tmp_path)
    dataset=DeformNeXtSS64(str(manifest),image_size=8,transport_arm='v1')
    original=dataset[0]
    sample=ConditioningControl(dataset,mode)[0]
    assert not any(k.startswith('transport_') for k in sample)
    assert sample['cond'].shape[0]==4
    if mode=='source-only':
        torch.testing.assert_close(sample['cond'][0],torch.full((3,8,8),100/255))
        torch.testing.assert_close(sample['cond'][1:],original['cond'][1:])
        for key in ('camera_angle_x','camera_distance','transform_matrix','view_indices'):
            torch.testing.assert_close(sample[key],original[key])
    else:torch.testing.assert_close(sample['cond'],original['cond'])


def test_zero_shot_source_only_rejects_target_transport_and_wrong_anchor_camera(tmp_path):
    from experiments.deformnext_retargeting.conditioning_controls import ConditioningControl
    manifest,row=_fixture(tmp_path)
    dataset=DeformNeXtSS64(str(manifest),image_size=8,transport_arm='b1')
    with pytest.raises(ValueError,match='native V1'):
        ConditioningControl(dataset,'source-only')
    row['source_anchor_transform_matrix']=np.eye(4).tolist()
    manifest.write_text(json.dumps(row)+'\n')
    dataset=DeformNeXtSS64(str(manifest),image_size=8,transport_arm='v1')
    with pytest.raises(ValueError,match='anchor camera'):
        ConditioningControl(dataset,'source-only')


def test_legacy_cache_eviction_requires_verified_mirror_and_ready_rgb_replacement(tmp_path):
    from experiments.deformnext_retargeting.precompute_transport import evict_mirrored_legacy
    old=tmp_path/'old.npz';new=tmp_path/'new.npz'
    np.savez(old,old=np.ones(1));np.savez(new,conditioning_input_sha256='rgb-input')
    stat=old.stat();metadata={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
    row={'legacy_transport_cache':str(old),'conditioning_input_sha256':'rgb-input'}
    receipt={'verified_by':'rsync-rnc-no-differences','files':{str(old):metadata},'owned_roots':[str(tmp_path)]}
    bad=dict(receipt,verified_by='not checked')
    with pytest.raises(ValueError,match='unverified'):
        evict_mirrored_legacy(row,'transport_cache',new,[],bad)
    assert old.exists()
    np.savez(new,conditioning_input_sha256='depth-input')
    with pytest.raises(ValueError,match='not ready'):
        evict_mirrored_legacy(row,'transport_cache',new,[],receipt)
    assert old.exists()
    np.savez(new,conditioning_input_sha256='rgb-input')
    evict_mirrored_legacy(row,'transport_cache',new,[],receipt)
    assert new.exists() and not old.exists()


def test_pbr_loader_and_sparse_collation(tmp_path):
    manifest, _ = _fixture(tmp_path)
    dataset = DeformNeXtPBR1024(
        str(manifest), image_size=8, expected_split="train", max_tokens=8
    )
    item = dataset[0]
    assert torch.all(item["transport_source_values"] == 9)
    assert torch.all(item["transport_target_query"] == 7)
    batches = dataset.collate_fn([item, item], split_size=1)
    assert len(batches) == 1
    assert batches[0]["x_0"].coords.shape == (4, 4)
    assert batches[0]["x_0"].feats.shape == (4, 32)


def test_pbr_loader_filters_assets_above_token_limit(tmp_path):
    manifest, row = _fixture(tmp_path)
    oversized = dict(row, sample_id="sample-oversized", pbr_tokens=9)
    manifest.write_text(json.dumps(row) + "\n" + json.dumps(oversized) + "\n")

    dataset = DeformNeXtPBR1024(
        str(manifest), image_size=8, expected_split="train", max_tokens=8
    )

    assert len(dataset) == 1
    assert dataset.rows[0]["sample_id"] == "sample-a"
    assert dataset.filtered_over_max_tokens == 1
    assert dataset.loads == [2]


def test_ss_canonical_schema_is_required_in_manifest_and_latent(tmp_path):
    manifest, row = _fixture(tmp_path)
    schema = 'deformnext-ss64-m1-anchor-v3'
    with pytest.raises(ValueError, match='canonical latent schema'):
        DeformNeXtSS64(str(manifest), image_size=8, transport_arm='v1', expected_ss_schema=schema)
    row['ss_latent_schema'] = schema
    manifest.write_text(json.dumps(row) + '\n')
    dataset = DeformNeXtSS64(str(manifest), image_size=8, transport_arm='v1', expected_ss_schema=schema)
    with pytest.raises(ValueError, match='schema mismatch'):
        dataset[0]
    np.savez(tmp_path / 'ss.npz', z=np.ones((8, 2, 2, 2), np.float32), schema=np.asarray(schema))
    assert dataset[0]['x_0'].shape == (8, 2, 2, 2)


@pytest.mark.parametrize('groups,cap', [(2, 4), (4, 2), (8, 1)])
def test_pbr_token_balancing_enforces_dense_image_asset_cap(groups, cap):
    from pixal3d.datasets.deformnext import _bounded_load_groups
    # Unbounded balancing puts seven tiny assets together opposite one large
    # asset; NAF memory depends on image count, not sparse token count.
    result = _bounded_load_groups([32768, 1, 1, 1, 1, 1, 1, 1], groups)
    assert max(map(len, result)) == cap
    assert sorted(index for group in result for index in group) == list(range(8))


def test_pbr_collation_caps_real_image_packs(tmp_path):
    manifest, _ = _fixture(tmp_path)
    dataset = DeformNeXtPBR1024(str(manifest), image_size=8, transport_arm='v1')
    items = [dataset[0] for _ in range(8)]
    items[0] = dict(items[0], coords=items[0]['coords'].repeat(8, 1),
                    pbr_feats=items[0]['pbr_feats'].repeat(8, 1), shape_feats=items[0]['shape_feats'].repeat(8, 1))
    packs = dataset.collate_fn(items, split_size=2)
    assert [pack['cond'].shape[0] for pack in packs] == [4, 4]


@pytest.mark.parametrize('stale_count', [1, 9])
def test_versioned_pbr_filter_uses_actual_reencoded_support(tmp_path, stale_count):
    manifest, row = _fixture(tmp_path)
    row.update(latent_geometry_schema='deformnext-m1-render-world-v4', pbr_tokens=stale_count)
    manifest.write_text(json.dumps(row) + '\n')
    retained = DeformNeXtPBR1024(str(manifest), transport_arm='v1', max_tokens=4)
    assert len(retained) == 1
    assert retained.loads == [2]
    excluded = DeformNeXtPBR1024(str(manifest), transport_arm='v1', max_tokens=1)
    assert len(excluded) == 0
    assert excluded.filtered_over_max_tokens == 1


def test_r9_loader_keeps_raw_context_separate_and_refuses_legacy_matches(tmp_path):
    from pixal3d.modules.r9_matching import MATCHER_CONTRACT, ATTENTION_CONTRACT
    manifest, _ = _fixture(tmp_path)
    source = tmp_path / 'transport.npz'
    with np.load(source) as cache: arrays = {k: cache[k] for k in cache.files}
    arrays['target_keys'] = np.zeros((1024, 2048), np.float32)
    arrays['source_keys'] = np.ones((3, 5, 2048), np.float32)
    np.savez(source, **arrays)
    ds = DeformNeXtSS64(str(manifest), transport_arm='c2', transport_implementation='r9-native-v5')
    item = ds[0]
    assert item['transport_target_keys'].shape == (1024, 1024)
    assert item['transport_target_context'].shape == (1024, 2048)
    assert item['transport_target_valid'].shape == (1024,)
    assert 'transport_deformation' not in item
    matches = tmp_path / 'new-matches'; matches.mkdir()
    np.savez(matches / 'sample-a.npz', schema=np.asarray('deformnext-match-v1'))
    ds = DeformNeXtSS64(str(manifest), transport_arm='b3', transport_implementation='r9-native-v5',
                       match_root=str(matches), expected_q2c_sha256='correct')
    with pytest.raises(ValueError, match='incompatible R9'):
        ds[0]
    np.savez(matches / 'sample-a.npz', schema=np.asarray('deformnext-match-r9-v5'),
             matcher_contract=np.asarray(json.dumps(MATCHER_CONTRACT)), aggregation=np.asarray('global-categorical-v1'),
             proposal_policy=np.asarray('GLOBAL64_VALID_OR_NATIVE'),
             attention_contract=np.asarray(json.dumps(ATTENTION_CONTRACT)),
             q2c_sha256=np.asarray('wrong'))
    with pytest.raises(ValueError, match='checkpoint hash'):
        ds[0]


def test_b1_raw_cache_needs_no_q2c_but_remains_bound_to_its_sample(tmp_path):
    from pixal3d.modules.r9_matching import RAW_MATCH_SCHEMA, RAW_MATCH_CONTRACT
    manifest,_=_fixture(tmp_path);root=tmp_path/'raw';root.mkdir()
    arrays=dict(schema=np.asarray(RAW_MATCH_SCHEMA),raw_matching_contract=np.asarray(json.dumps(RAW_MATCH_CONTRACT)),
        sample_id=np.asarray('sample-a'),split=np.asarray('train'),semantic_indices=np.zeros((1024,1),np.uint16))
    np.savez(root/'sample-a.npz',**arrays)
    ds=DeformNeXtSS64(str(manifest),transport_arm='b1',transport_implementation='r9-native-v5',match_root=str(root))
    sample=ds[0]
    assert sample['transport_semantic_indices'].shape==(1024,1)
    assert 'transport_matching_logits' not in sample
    arrays['sample_id']=np.asarray('wrong-sample');np.savez(root/'sample-a.npz',**arrays)
    with pytest.raises(ValueError,match='RAW Top-1'):
        ds[0]

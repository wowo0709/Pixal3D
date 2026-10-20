"""PBR replay transport must preserve native sampling and post-generation GT IO."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deformnext_retargeting.cascade_remote_results import CascadeRemoteResults, array_identities
from experiments.deformnext_retargeting.immutable_result_store import decode_npz, encode_npz
from experiments.deformnext_retargeting import pbr_remote_results as module


class MemoryStore:
    remote_root = '/assigned/results'

    def __init__(self):
        self.objects, self.read_keys = {}, []

    def put_bytes(self, key, payload):
        if key in self.objects and self.objects[key] != payload:
            raise ValueError('immutable collision')
        self.objects[key] = payload
        return dict(key=key, remote_root=self.remote_root, bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(), actual_remote_readback_verified=True)

    def get_bytes(self, record):
        self.read_keys.append(record['key'])
        payload = self.objects[record['key']]
        if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError('downloaded object byte identity differs')
        return payload

    def put_npz(self, key, arrays, *, compressed):
        return dict(self.put_bytes(key, encode_npz(arrays, compressed=compressed)), npz_compressed=compressed)

    def get_npz(self, record):
        return decode_npz(self.get_bytes(record))


def fixture(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'replay'
    source.mkdir()
    output.mkdir()
    store = MemoryStore()
    io = CascadeRemoteResults(store, output=source, prefix='sample/source', stage='generation',
                              input_manifests={}, source_contract={})
    prediction = dict(vertices=np.asfortranarray(np.arange(12, dtype=np.float32).reshape(4, 3)),
                      faces=np.asarray([[0, 1, 2]], dtype=np.int32),
                      attrs=np.asarray([[.1, .2, .3]], dtype=np.float16))
    io.write_npz(source / 'prediction.npz', **prediction)
    io.write_npz(source / 'geometry-evaluation.npz', target_world=prediction['vertices'])
    io.finalize({'sample_id': 'sample'})
    manifest = source / 'remote-npz.json'
    replay = module.PbrRemoteResults(store, output=output, prefix='sample/replay',
                                     input_manifests={str(source): hashlib.sha256(manifest.read_bytes()).hexdigest()},
                                     source_contract={'reverse_AST_exact': True})
    return source, output, store, prediction, replay


def test_actual_native_source_reverses_exact_and_keeps_sampler_noise_and_decoder_calls():
    source = Path(module.__file__).with_name('replay_pbr_on_generated_shape.py').read_text()
    tree, contract = module.remote_pbr_tree(source)
    assert contract['reverse_AST_exact']
    compile(tree, '<native-PBR-remote>', 'exec')
    def important_calls(tree):
        return [ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr in {
                    'sample_tex_slat', 'decode_shape_slat', 'decode_tex_slat', 'manual_seed',
                    'get_rng_state', 'get_proj_cond_shape', 'load_state_dict'}]
    assert important_calls(tree) == important_calls(ast.parse(source))


def test_native_archive_enumeration_and_copy_keeps_geometry_bytes_and_local_disk_empty(tmp_path):
    source, output, store, expected, io = fixture(tmp_path)
    first = io.byte_sha(source / 'prediction.npz')
    with io.read_npz(source / 'prediction.npz') as archive:
        assert archive.files == list(expected)
        # The native replay's exact expression, including its array copies.
        original = {name: archive[name].copy() for name in archive.files}
    prediction = {name: value.copy() for name, value in original.items()}
    prediction['attrs'] = prediction['attrs'] + np.asarray(1, dtype=prediction['attrs'].dtype)
    io.write_npz(output / 'prediction.npz', **prediction)
    io.copy_npz(source / 'geometry-evaluation.npz', output / 'geometry-evaluation.npz')
    assert io.byte_sha(source / 'prediction.npz') == first
    copied = io.byte_sha(output / 'geometry-evaluation.npz')
    result = io.finalize({'metrics': {'CD': .123}, 'original_prediction_SHA256': first})
    assert result['metrics'] == {'CD': .123}
    assert copied == io.writer.records['geometry-evaluation.npz']['record']['sha256']
    assert store.objects['sample/source/geometry-evaluation.npz'] == store.objects['sample/replay/geometry-evaluation.npz']
    assert array_identities(store.get_npz(io.writer.records['prediction.npz']['record'])) == array_identities(prediction)
    assert not list(tmp_path.rglob('*.npz'))
    assert store.read_keys.count('sample/source/prediction.npz') == 3


def test_GT_evaluation_archive_is_not_read_before_prediction_is_saved(tmp_path):
    source, output, store, _, io = fixture(tmp_path)
    with pytest.raises(ValueError, match='only after prediction is saved'):
        io.copy_npz(source / 'geometry-evaluation.npz', output / 'geometry-evaluation.npz')
    assert store.read_keys == []


def test_two_source_hashes_are_actual_downloads_and_reject_changed_blob(tmp_path):
    source, _, store, _, io = fixture(tmp_path)
    io.byte_sha(source / 'prediction.npz')
    store.objects['sample/source/prediction.npz'] += b'changed'
    with pytest.raises(ValueError, match='byte identity differs'):
        io.byte_sha(source / 'prediction.npz')


def test_no_GT_geometry_array_reencoding_during_copy(tmp_path):
    source, output, store, expected, io = fixture(tmp_path)
    io.write_npz(output / 'prediction.npz', **expected)
    source_blob = store.objects['sample/source/geometry-evaluation.npz']
    io.copy_npz(source / 'geometry-evaluation.npz', output / 'geometry-evaluation.npz')
    assert store.objects['sample/replay/geometry-evaluation.npz'] is source_blob
    assert io.writer.records['geometry-evaluation.npz']['arrays'] == io.reader.inputs[source.resolve()]['manifest']['archives']['geometry-evaluation.npz']['arrays']


def test_native_copy_site_change_is_rejected():
    source = Path(module.__file__).with_name('replay_pbr_on_generated_shape.py').read_text()
    source = source.replace('shutil.copyfile(', 'shutil.copy2(')
    with pytest.raises(ValueError, match='archive IO sites changed'):
        module.remote_pbr_tree(source)


def test_raw_output_manifest_cannot_finalize_without_native_hash_audits(tmp_path):
    source, output, _, expected, io = fixture(tmp_path)
    with io.read_npz(source / 'prediction.npz'):
        pass
    io.write_npz(output / 'prediction.npz', **expected)
    io.copy_npz(source / 'geometry-evaluation.npz', output / 'geometry-evaluation.npz')
    with pytest.raises(ValueError, match='two actual unchanged source'):
        io.finalize({'sample_id': 'sample'})

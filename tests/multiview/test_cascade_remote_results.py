"""Remote cascade IO must preserve native calculations and raw array identities."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.deformnext_retargeting import cascade_remote_results as module
from experiments.deformnext_retargeting.immutable_result_store import decode_npz, encode_npz


class MemoryStore:
    remote_root = '/assigned/results'

    def __init__(self):
        self.objects = {}

    def put_bytes(self, key, payload):
        if key in self.objects and self.objects[key] != payload:
            raise ValueError('immutable collision')
        self.objects[key] = payload
        return dict(key=key, remote_root=self.remote_root, bytes=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(), actual_remote_readback_verified=True)

    def put_npz(self, key, arrays, *, compressed):
        return self.put_bytes(key, encode_npz(arrays, compressed=compressed))

    def get_npz(self, record):
        payload = self.objects[record['key']]
        assert hashlib.sha256(payload).hexdigest() == record['sha256']
        return decode_npz(payload)


@pytest.mark.parametrize('stage', list(module.STAGES))
def test_actual_frozen_stages_change_only_declared_io_and_reverse_exact(stage):
    source = Path(module.__file__).with_name(module.STAGES[stage][0] + '.py').read_text()
    tree, contract = module.remote_io_tree(source, stage=stage)
    compile(tree, '<native-remote-test>', 'exec')
    assert contract['reverse_AST_exact']
    # In particular, native model generation and rendering call expressions
    # must be unchanged, not merely approximately equivalent on fake inputs.
    def calculations(tree):
        return [ast.dump(node) for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and (node.func.value.id, node.func.attr) in {
                    ('pipeline', 'run'), ('renderer', 'render'), ('mesh', 'simplify')}]
    assert calculations(tree) == calculations(ast.parse(source))


def test_unexpected_native_archive_io_change_rejected():
    source = Path(module.__file__).with_name('cascade_from_ss_artifact.py').read_text()
    source = source.replace("output/'prediction.npz'", "output/'different-output.npz'")
    with pytest.raises(ValueError, match='explicit IO sites changed'):
        module.remote_io_tree(source, stage='generation')


def generation(tmp_path, store):
    output = tmp_path / 'generation'
    output.mkdir()
    io = module.CascadeRemoteResults(store, output=output, prefix='model/sample/generation',
                                     stage='generation', input_manifests={},
                                     source_contract={'reverse_AST_exact': True}, compressed=True)
    arrays = dict(vertices=np.asfortranarray(np.arange(12, dtype=np.float32).reshape(4, 3)),
                  faces=np.asarray([[0, 1, 2]], dtype=np.int32), value=np.asarray(.5, dtype=np.float16))
    io.write_npz(output / 'prediction.npz', **arrays)
    io.write_npz(output / 'geometry-evaluation.npz', target_world=arrays['vertices'])
    result = io.finalize({'sample_id': 'sample', 'metrics': {'CD': .123}})
    return output, arrays, result


def test_generation_to_render_preserves_every_array_without_local_raw_files(tmp_path):
    store = MemoryStore()
    output, expected, result = generation(tmp_path, store)
    manifest = output / module.MANIFEST_NAME
    assert list(output.iterdir()) == [manifest]
    assert result['metrics'] == {'CD': .123}
    io = module.CascadeRemoteResults(store, output=tmp_path / 'render', prefix='model/sample/render',
                                     stage='render', input_manifests={str(output): hashlib.sha256(manifest.read_bytes()).hexdigest()},
                                     source_contract={'reverse_AST_exact': True})
    with io.read_npz(output / 'prediction.npz') as values:
        assert module.array_identities(values) == module.array_identities(expected)
        for name in values:
            np.testing.assert_array_equal(values[name], expected[name])
    with io.read_npz(output / 'geometry-evaluation.npz') as values:
        np.testing.assert_array_equal(values['target_world'], expected['vertices'])
    (tmp_path / 'render').mkdir()
    io.write_npz(tmp_path / 'render/render-geometry.npz', vertices=expected['vertices'])
    final = io.finalize({'per_view': [{'camera_id': 8, 'PSNR': 19.5}]})
    assert final['per_view'] == [{'camera_id': 8, 'PSNR': 19.5}]
    assert len(final['remote_raw_archives']['actual_inputs']) == 2
    assert not list(tmp_path.rglob('*.npz'))


def test_input_manifest_change_is_terminal(tmp_path):
    store = MemoryStore()
    output, _, _ = generation(tmp_path, store)
    with pytest.raises(ValueError, match='manifest byte identity differs'):
        module.CascadeRemoteResults(store, output=tmp_path / 'render', prefix='model/sample/render',
                                    stage='render', input_manifests={str(output): '0' * 64}, source_contract={})


def test_corrupt_array_identity_is_not_hidden_by_successful_blob_download(tmp_path):
    store = MemoryStore()
    output, _, _ = generation(tmp_path, store)
    path = output / module.MANIFEST_NAME
    payload = json.loads(path.read_text())
    payload['archives']['prediction.npz']['arrays']['vertices']['values_C_SHA256'] = '0' * 64
    path.write_text(json.dumps(payload))
    io = module.CascadeRemoteResults(store, output=tmp_path / 'render', prefix='model/sample/render',
                                     stage='render', input_manifests={str(output): hashlib.sha256(path.read_bytes()).hexdigest()},
                                     source_contract={})
    with pytest.raises(ValueError, match='dtype/shape/layout/values differ'):
        with io.read_npz(output / 'prediction.npz'):
            pass


def test_wrong_or_missing_native_outputs_are_not_accepted(tmp_path):
    store = MemoryStore()
    io = module.CascadeRemoteResults(store, output=tmp_path, prefix='model/sample/generation',
                                     stage='generation', input_manifests={}, source_contract={})
    with pytest.raises(ValueError, match='declared owned raw output'):
        io.write_npz(tmp_path / 'unrelated.npz', value=np.asarray(1))
    with pytest.raises(ValueError, match='incomplete'):
        io.finalize({'sample_id': 'sample'})


def test_shared_material_requires_a_remote_geometry_read(tmp_path):
    io = module.CascadeRemoteResults(MemoryStore(), output=tmp_path, prefix='model/sample/shared',
                                     stage='shared-material', input_manifests={}, source_contract={})
    with pytest.raises(ValueError, match='input archive set is incomplete'):
        io.finalize({'per_view': []})

"""Native arrays and immutable remote artifacts must retain exact identities."""
import hashlib
import io
import json
import subprocess
import zipfile

import numpy as np
import pytest

from experiments.deformnext_retargeting.immutable_result_store import (
    ImmutableResultStore, decode_npz, encode_npz, relative_key,
)


@pytest.mark.parametrize('compressed', [True, False])
def test_numpy_roundtrip_keeps_dtype_shape_values_and_order(compressed):
    arrays = dict(vertices=np.arange(12, dtype=np.float32).reshape(4, 3),
                  faces=np.asarray([[0, 1, 2]], dtype=np.int32),
                  half=np.asarray([.5, -1.25], dtype=np.float16),
                  sample_id=np.asarray('example'), scalar=np.asarray(1024),
                  anchor_rotation=np.asfortranarray(np.eye(3, dtype=np.float32)))
    decoded = decode_npz(encode_npz(arrays, compressed=compressed))
    assert list(decoded) == list(arrays)
    for name, value in arrays.items():
        assert decoded[name].dtype == value.dtype and decoded[name].shape == value.shape
        assert decoded[name].tobytes() == value.tobytes()
        assert decoded[name].flags.c_contiguous == value.flags.c_contiguous
        assert decoded[name].flags.f_contiguous == value.flags.f_contiguous


@pytest.mark.parametrize('key', ['', '/absolute', '../parent', 'a/../b', 'a//b', './a', 'a/', 'bad\x00name'])
def test_unowned_or_ambiguous_paths_rejected(key):
    with pytest.raises(ValueError):
        relative_key(key)


def test_object_arrays_are_not_silently_pickled():
    with pytest.raises(ValueError, match='object/pickle'):
        encode_npz(dict(value=np.asarray([object()], dtype=object)))


def test_duplicate_array_entries_are_not_silently_dropped():
    member = io.BytesIO();np.save(member, np.arange(4, dtype=np.int32))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('coords.npy', member.getvalue())
        with pytest.warns(UserWarning, match='Duplicate name'):
            stream.writestr('coords.npy', member.getvalue())
    with pytest.raises(ValueError, match='duplicate'):
        decode_npz(archive.getvalue())


def test_non_array_archive_entries_are_not_accepted_as_native_results():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('metadata.json', '{}')
    with pytest.raises(ValueError, match='only named NumPy arrays'):
        decode_npz(archive.getvalue())


def test_upload_verifies_actual_receipt_and_keeps_paths_out_of_shell(monkeypatch):
    calls = []

    def fake_run(argv, *, input, capture_output, check):
        header, payload = input.split(b'\n', 1);header = json.loads(header)
        calls.append((argv, header))
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(dict(
            status='PASS_ACTUAL_IMMUTABLE_REMOTE_BLOB', bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(), actual_remote_readback_verified=True)).encode())

    monkeypatch.setattr(subprocess, 'run', fake_run)
    store = ImmutableResultStore(['ssh', 'assigned-host'], '/owned/path with spaces')
    key = 'case/$(literal);quote\'file.npz'
    result = store.put_bytes(key, b'actual bytes')
    assert calls[0][0][:2] == ['ssh', 'assigned-host']
    assert key not in calls[0][0][-1] and calls[0][1]['key'] == key
    assert result['key'] == key and result['remote_root'] == '/owned/path with spaces'


def test_corrupt_download_rejected_without_writing_local_files(monkeypatch):
    monkeypatch.setattr(subprocess, 'run', lambda argv, **kw:
                        subprocess.CompletedProcess(argv, 0, stdout=b'corrupt'))
    store = ImmutableResultStore(['ssh', 'assigned-host'], '/owned')
    with pytest.raises(ValueError, match='byte identity differs'):
        store.get_bytes(dict(key='case/prediction.npz', remote_root='/owned', bytes=3,
                             sha256=hashlib.sha256(b'raw').hexdigest()))

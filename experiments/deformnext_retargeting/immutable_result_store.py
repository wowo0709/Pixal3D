"""Immutable remote NumPy results without accumulating raw meshes on local disk.

Callers provide their assigned SSH connection and an owned output directory.
Put verifies actual remote readback bytes; get verifies bytes again locally.
This is an artifact transport layer, not a model/data transformation or an
automatic fallback inside existing training and inference entrypoints.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import PurePosixPath
import shlex
import subprocess
import zipfile

import numpy as np


def relative_key(value):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('explicit nonempty relative object key required')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {'', '.', '..'} for part in value.split('/')):
        raise ValueError('normalized relative object key required')
    return value


def encode_npz(arrays, *, compressed=True):
    if not isinstance(compressed, bool):
        raise ValueError('explicit boolean NumPy archive compression policy required')
    if not arrays or any(not isinstance(name, str) or not name for name in arrays):
        raise ValueError('explicit named arrays required')
    values = {name: np.asarray(value) for name, value in arrays.items()}
    if any(value.dtype.hasobject for value in values.values()):
        raise ValueError('object/pickle arrays are not native result arrays')
    buffer = io.BytesIO()
    writer = np.savez_compressed if compressed else np.savez
    writer(buffer, **values)
    return buffer.getvalue()


def decode_npz(payload):
    # Read each verified ZIP member once. NumPy's many small reads through a
    # ZipExtFile can be costly for large in-memory ZIP_STORED native meshes.
    result = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for entry in archive.infolist():
            if (not entry.filename.endswith('.npy') or entry.is_dir()
                    or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError('a native NPZ must contain only named NumPy arrays')
            name = entry.filename[:-4]
            if not name or name in result:
                raise ValueError('duplicate or empty native array name')
            value = np.load(io.BytesIO(archive.read(entry)), allow_pickle=False)
            if not isinstance(value, np.ndarray) or value.dtype.hasobject:
                raise ValueError('native numeric/string array required')
            result[name] = value.copy(order='K')
    if not result:
        raise ValueError('nonempty native array archive required')
    return result


# Header fields are received as JSON on stdin; paths never become shell code.
_PUT = r'''
from pathlib import Path
import hashlib,json,os,sys,uuid
header=json.loads(sys.stdin.buffer.readline())
root=Path(sys.argv[1]).resolve();key=header['key'];parts=key.split('/')
assert key and not key.startswith('/') and all(p not in ('','.','..') for p in parts)
assert isinstance(header['bytes'],int) and header['bytes']>=0
root.mkdir(parents=True,exist_ok=True);target=root.joinpath(*parts)
target.parent.mkdir(parents=True,exist_ok=True)
assert target.resolve().is_relative_to(root)
temporary=target.with_name('.'+target.name+'.'+uuid.uuid4().hex+'.upload')
digest=hashlib.sha256();count=0
with temporary.open('xb') as stream:
    while True:
        block=sys.stdin.buffer.read(8*1024*1024)
        if not block:break
        count+=len(block);digest.update(block);stream.write(block)
    stream.flush();os.fsync(stream.fileno())
assert count==header['bytes'] and digest.hexdigest()==header['sha256'],'upload byte identity differs'
reused=False
try:
    os.link(temporary,target)
except FileExistsError:
    reused=True
actual=hashlib.sha256();actual_count=0
with target.open('rb') as stream:
    while True:
        block=stream.read(8*1024*1024)
        if not block:break
        actual_count+=len(block);actual.update(block)
assert actual_count==count and actual.hexdigest()==digest.hexdigest(),'immutable target differs; preserved upload'
temporary.unlink()
print(json.dumps(dict(status='PASS_ACTUAL_IMMUTABLE_REMOTE_BLOB',remote_path=str(target),bytes=actual_count,sha256=actual.hexdigest(),existing_identical_object_reused=reused,actual_remote_readback_verified=True)))
'''

_GET = r'''
from pathlib import Path
import json,sys
header=json.loads(sys.stdin.buffer.readline());root=Path(sys.argv[1]).resolve();key=header['key'];parts=key.split('/')
assert key and not key.startswith('/') and all(p not in ('','.','..') for p in parts)
target=root.joinpath(*parts);assert target.resolve().is_relative_to(root)
with target.open('rb') as stream:
    while True:
        block=stream.read(8*1024*1024)
        if not block:break
        sys.stdout.buffer.write(block)
'''


class ImmutableResultStore:
    """Use explicit authorized SSH argv and an absolute owned remote root."""

    def __init__(self, ssh_argv, remote_root):
        if (not isinstance(ssh_argv, (list, tuple)) or not ssh_argv
                or any(not isinstance(arg, str) for arg in ssh_argv)
                or not isinstance(remote_root, str) or '\x00' in remote_root
                or not PurePosixPath(remote_root).is_absolute()
                or '..' in PurePosixPath(remote_root).parts):
            raise ValueError('explicit SSH argv and absolute remote directory required')
        self.ssh_argv = list(ssh_argv)
        self.remote_root = remote_root

    def _command(self, script):
        return [*self.ssh_argv, 'python3 -c '+shlex.quote(script)+' '+shlex.quote(self.remote_root)]

    def put_bytes(self, key, payload):
        key = relative_key(key)
        if not isinstance(payload, bytes):
            raise TypeError('immutable bytes required')
        digest = hashlib.sha256(payload).hexdigest()
        header = dict(key=key, bytes=len(payload), sha256=digest)
        reply = subprocess.run(self._command(_PUT), input=(json.dumps(header)+'\n').encode()+payload,
                               capture_output=True, check=True)
        record = json.loads(reply.stdout)
        if (record['status'] != 'PASS_ACTUAL_IMMUTABLE_REMOTE_BLOB'
                or record['bytes'] != len(payload) or record['sha256'] != digest
                or not record['actual_remote_readback_verified']):
            raise ValueError('actual remote blob verification differs')
        return dict(record, key=key, remote_root=self.remote_root)

    def get_bytes(self, record):
        key = relative_key(record['key'])
        if record['remote_root'] != self.remote_root:
            raise ValueError('remote root differs from the explicit object binding')
        reply = subprocess.run(self._command(_GET), input=(json.dumps(dict(key=key))+'\n').encode(),
                               capture_output=True, check=True)
        payload = reply.stdout
        if len(payload) != record['bytes'] or hashlib.sha256(payload).hexdigest() != record['sha256']:
            raise ValueError('downloaded object byte identity differs')
        return payload

    def put_npz(self, key, arrays, *, compressed=True):
        return dict(self.put_bytes(key, encode_npz(arrays, compressed=compressed)),
                    npz_compressed=compressed)

    def get_npz(self, record):
        return decode_npz(self.get_bytes(record))

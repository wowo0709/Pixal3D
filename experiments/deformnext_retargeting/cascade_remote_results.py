"""Explicit remote raw-mesh IO for separately versioned native cascade runs.

Only declared archive IO and its metadata pointer are replaced. The complete
remaining native AST must reverse exactly before execution. Render pixels stay
local for the existing metric readers; large raw meshes live in the assigned
immutable result store. Existing entrypoints and queued plans are untouched.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

import numpy as np

from .immutable_result_store import ImmutableResultStore, relative_key


STAGES = {
    'generation': ('cascade_from_ss_artifact',
                   {'prediction.npz', 'geometry-evaluation.npz'}, set(), 'generation.json'),
    'zero-generation': ('cascade_zero_shot_from_ss_artifact',
                        {'prediction.npz', 'geometry-evaluation.npz'}, set(), 'generation.json'),
    'render': ('render_cascade_artifact_v2', {'render-geometry.npz'},
               {'prediction.npz', 'geometry-evaluation.npz'}, 'render-evaluation.json'),
    'shared-material': ('score_cascade_shared_material', set(),
                        {'geometry-evaluation.npz'}, 'render-evaluation.json'),
}
MANIFEST_NAME = 'remote-npz.json'


def array_identities(arrays):
    return {name: dict(dtype=value.dtype.str, shape=list(value.shape),
                       C_contiguous=bool(value.flags.c_contiguous),
                       F_contiguous=bool(value.flags.f_contiguous),
                       values_C_SHA256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
            for name, value in arrays.items()}


class CascadeRemoteResults:
    """One native stage, one owned output prefix, and hash-bound input manifests."""

    def __init__(self, store, *, output, prefix, stage, input_manifests,
                 source_contract, compressed=True):
        if stage not in STAGES or not isinstance(compressed, bool):
            raise ValueError('explicit supported stage and compression policy required')
        self.store, self.output = store, Path(output).resolve()
        self.prefix, self.stage = relative_key(prefix), stage
        self.compressed, self.source_contract = compressed, source_contract
        self.records, self.inputs, self.reads = {}, {}, {}
        for directory, expected_sha in input_manifests.items():
            directory = Path(directory).resolve()
            payload = (directory / MANIFEST_NAME).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected_sha:
                raise ValueError('input remote archive manifest byte identity differs')
            manifest = json.loads(payload)
            if (manifest['schema'] != 'native-cascade-remote-npz-v1'
                    or manifest['remote_root'] != self.store.remote_root):
                raise ValueError('input archive store/schema differs')
            relative_key(manifest['prefix'])
            self.inputs[directory] = dict(manifest=manifest, SHA256=expected_sha)

    def _output_name(self, path):
        path = Path(path).resolve()
        if path.parent != self.output or path.name not in STAGES[self.stage][1]:
            raise ValueError('only the declared owned raw output archives may be remote')
        return path.name

    def location(self, path):
        name = self._output_name(path)
        return str(PurePosixPath(self.store.remote_root) / self.prefix / name)

    def write_npz(self, path, **arrays):
        name = self._output_name(path)
        if name in self.records:
            raise ValueError('a native archive must be written once per stage')
        arrays = {key: np.asarray(value) for key, value in arrays.items()}
        record = self.store.put_npz(f'{self.prefix}/{name}', arrays, compressed=self.compressed)
        self.records[name] = dict(record=record, arrays=array_identities(arrays))

    @contextmanager
    def read_npz(self, path):
        path = Path(path).resolve()
        if path.parent not in self.inputs or path.name not in STAGES[self.stage][2]:
            raise ValueError('explicit hash-bound native input archive required')
        binding = self.inputs[path.parent]
        manifest = binding['manifest']
        item = manifest['archives'][path.name]
        if item['record']['key'] != f'{manifest["prefix"]}/{path.name}':
            raise ValueError('input archive key differs from its exact stage prefix')
        arrays = self.store.get_npz(item['record'])
        if array_identities(arrays) != item['arrays']:
            raise ValueError('decoded native array dtype/shape/layout/values differ')
        self.reads[str(path)] = dict(manifest_SHA256=binding['SHA256'], record=item['record'])
        yield arrays

    def finalize(self, result):
        if set(self.records) != STAGES[self.stage][1]:
            raise ValueError('native raw output archive set is incomplete')
        if {Path(path).name for path in self.reads} != STAGES[self.stage][2]:
            raise ValueError('native raw input archive set is incomplete')
        summary = dict(schema='native-cascade-remote-npz-v1', stage=self.stage,
                       remote_root=self.store.remote_root, prefix=self.prefix,
                       archives=self.records, actual_inputs=self.reads,
                       source_contract=self.source_contract,
                       local_raw_npz_written=False, original_numeric_AST_reversed_exact=True)
        if self.records:
            payload = (json.dumps(summary, indent=2) + '\n').encode()
            with (self.output / MANIFEST_NAME).open('xb') as stream:
                stream.write(payload)
            summary = dict(manifest=str(self.output / MANIFEST_NAME),
                           manifest_SHA256=hashlib.sha256(payload).hexdigest(),
                           actual_remote_manifest=self.store.put_bytes(f'{self.prefix}/{MANIFEST_NAME}', payload),
                           **summary)
        return dict(result, remote_raw_archives=summary)


def _function(node, owner, name):
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == owner and node.attr == name)


def _filename(node):
    if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
        return node.right.value
    return None


def remote_io_tree(source, *, stage):
    """Change only explicit native result IO; prove every other AST node exact."""
    if stage not in STAGES:
        raise ValueError('unsupported native stage')
    original = ast.parse(source)
    _, writes, reads, result_name = STAGES[stage]
    counts = dict(write=0, read=0, location=0, finalize=0)

    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            old = copy.deepcopy(node)
            name = _filename(node.args[0]) if node.args else None
            method = None
            if _function(node.func, 'np', 'savez_compressed') and name in writes:
                method = 'write_npz'
            elif _function(node.func, 'np', 'load') and name in reads:
                method = 'read_npz'
            elif (stage in ('generation', 'zero-generation') and isinstance(node.func, ast.Name)
                  and node.func.id == 'str' and name == 'prediction.npz'):
                method = 'location'
            if method:
                key = {'write_npz': 'write', 'read_npz': 'read', 'location': 'location'}[method]
                counts[key] += 1
                node.func = ast.Attribute(ast.Name('_remote_results', ast.Load()), method, ast.Load())
                node._native_original = old
                return node
            return self.generic_visit(node)

        def visit_Expr(self, node):
            node = self.generic_visit(node)
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'write_text' and _filename(call.func.value) == result_name):
                counts['finalize'] += 1
                added = ast.parse('result = _remote_results.finalize(result)').body[0]
                added._remote_injected = True
                return [added, node]
            return node

    tree = ast.fix_missing_locations(Rewrite().visit(copy.deepcopy(original)))
    expected = dict(write=len(writes), read=len(reads),
                    location=int(stage in ('generation', 'zero-generation')), finalize=1)
    if counts != expected:
        raise ValueError(f'native explicit IO sites changed: expected {expected}, got {counts}')

    class Reverse(ast.NodeTransformer):
        def visit(self, node):
            if getattr(node, '_remote_injected', False):
                return None
            if hasattr(node, '_native_original'):
                return copy.deepcopy(node._native_original)
            return super().visit(node)

    reversed_tree = Reverse().visit(copy.deepcopy(tree))
    if ast.dump(reversed_tree) != ast.dump(original):
        raise ValueError('native numerical/source AST differs outside declared result IO')
    return tree, dict(stage=stage, sites=counts, reverse_AST_exact=True,
                      source_SHA256=hashlib.sha256(source.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.plan.read_text())
    stage = binding['stage']
    module = STAGES[stage][0]
    source_path = Path(__file__).with_name(module + '.py')
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != binding['native_source_SHA256']:
        raise ValueError('native entrypoint source byte identity differs')
    native_path = Path(binding['native_plan'])
    payload = native_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding['native_plan_SHA256']:
        raise ValueError('native plan byte identity differs')
    native = json.loads(payload)
    tree, contract = remote_io_tree(source.decode(), stage=stage)
    store = ImmutableResultStore(binding['store']['ssh_argv'], binding['store']['remote_root'])
    io = CascadeRemoteResults(store, output=native['output'], prefix=binding['output_prefix'],
                              stage=stage, input_manifests=binding['input_manifests'],
                              source_contract=contract, compressed=binding['compressed'])
    namespace = dict(__name__=f'{__package__}._remote_{module}', __package__=__package__,
                     __file__=str(source_path), _remote_results=io)
    exec(compile(tree, str(source_path), 'exec'), namespace)
    previous = sys.argv
    try:
        sys.argv = [str(source_path), '--plan', str(native_path)]
        namespace['main']()
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()

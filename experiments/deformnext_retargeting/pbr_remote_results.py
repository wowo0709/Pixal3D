"""Bounded remote archives for the unchanged generated-shape PBR replay.

The sampler, conditioner's RGB inputs, noise reset and raw geometry checks stay
identical. Only declared archive IO, its byte digests and result pointer move
to the explicit immutable remote store. No original source file is modified.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import sys

from .cascade_remote_results import CascadeRemoteResults, _filename, _function
from .immutable_result_store import ImmutableResultStore


class NativeArrayArchive(dict):
    """The native replay enumerates NpzFile.files before copying its arrays."""

    @property
    def files(self):
        return list(self)


class PbrRemoteResults:
    def __init__(self, store, *, output, prefix, input_manifests, source_contract,
                 compressed=True):
        common = dict(output=output, prefix=prefix, input_manifests=input_manifests,
                      source_contract=source_contract, compressed=compressed)
        self.writer = CascadeRemoteResults(store, stage='generation', **common)
        self.reader = CascadeRemoteResults(store, stage='render', **common)
        self.hash_reads = {}

    def _source(self, path):
        path = Path(path).resolve()
        if (path.parent not in self.reader.inputs
                or path.name not in {'prediction.npz', 'geometry-evaluation.npz'}):
            raise ValueError('explicit hash-bound generated-shape input required')
        binding = self.reader.inputs[path.parent]
        item = binding['manifest']['archives'][path.name]
        if item['record']['key'] != f'{binding["manifest"]["prefix"]}/{path.name}':
            raise ValueError('source key differs from the bound generation prefix')
        return path, binding, item

    @contextmanager
    def read_npz(self, path):
        with self.reader.read_npz(path) as arrays:
            yield NativeArrayArchive(arrays)

    def write_npz(self, path, **arrays):
        if Path(path).name != 'prediction.npz':
            raise ValueError('PBR may write only the new prediction; geometry is copied unchanged')
        self.writer.write_npz(path, **arrays)

    def location(self, path):
        return self.writer.location(path)

    def byte_sha(self, path):
        path = Path(path).resolve()
        if path.parent == self.writer.output:
            item = self.writer.records[path.name]
        else:
            _, _, item = self._source(path)
        # The native replay hashes before and after sampling. Fetch/verify the
        # actual bytes at both sites instead of trusting the recorded checksum.
        payload = self.writer.store.get_bytes(item['record'])
        digest = hashlib.sha256(payload).hexdigest()
        self.hash_reads.setdefault(str(path), []).append(digest)
        return digest

    def copy_npz(self, source, target):
        if 'prediction.npz' not in self.writer.records:
            raise ValueError('GT evaluation archive must be copied only after prediction is saved')
        source, binding, item = self._source(source)
        name = self.writer._output_name(target)
        if source.name != 'geometry-evaluation.npz' or name != source.name:
            raise ValueError('only unchanged post-generation geometry may be copied')
        if name in self.writer.records:
            raise ValueError('raw geometric evaluation must be carried forward once')
        payload = self.writer.store.get_bytes(item['record'])
        record = self.writer.store.put_bytes(f'{self.writer.prefix}/{name}', payload)
        record['npz_compressed'] = item['record']['npz_compressed']
        self.writer.records[name] = dict(record=record, arrays=copy.deepcopy(item['arrays']))
        self.reader.reads[str(source)] = dict(manifest_SHA256=binding['SHA256'], record=item['record'])
        return Path(target)

    def finalize(self, report):
        if {Path(path).name for path in self.reader.reads} != {'prediction.npz', 'geometry-evaluation.npz'}:
            raise ValueError('both native generated-shape input archives must be audited')
        source_hashes = [values for path, values in self.hash_reads.items()
                         if Path(path).name == 'prediction.npz' and Path(path).parent != self.writer.output]
        if len(source_hashes) != 1 or len(source_hashes[0]) != 2 or len(set(source_hashes[0])) != 1:
            raise ValueError('two actual unchanged source-prediction byte checks required')
        geometry_hashes = self.hash_reads.get(str(self.writer.output / 'geometry-evaluation.npz'), [])
        if len(geometry_hashes) != 1:
            raise ValueError('actual copied geometry output byte check required')
        result = self.writer.finalize(report)
        return dict(result, native_archive_operation='PBR replay on generated shape',
                    actual_remote_PBR_inputs=self.reader.reads,
                    actual_remote_byte_checks=self.hash_reads,
                    GT_evaluation_archive_copied_byte_exact_only_after_prediction=True)


def remote_pbr_tree(source):
    original = ast.parse(source)
    counts = dict(write=0, read=0, copy=0, sha=0, location=0, finalize=0)

    class Rewrite(ast.NodeTransformer):
        def visit_Call(self, node):
            before = copy.deepcopy(node)
            name = _filename(node.args[0]) if node.args else None
            method = None
            if _function(node.func, 'np', 'savez_compressed') and name == 'prediction.npz':
                method, key = 'write_npz', 'write'
            elif _function(node.func, 'np', 'load') and name == 'prediction.npz':
                method, key = 'read_npz', 'read'
            elif (_function(node.func, 'shutil', 'copyfile') and name == 'geometry-evaluation.npz'
                  and len(node.args) == 2 and _filename(node.args[1]) == name):
                method, key = 'copy_npz', 'copy'
            elif (isinstance(node.func, ast.Name) and node.func.id == 'sha'
                  and name in {'prediction.npz', 'geometry-evaluation.npz'}):
                method, key = 'byte_sha', 'sha'
            elif isinstance(node.func, ast.Name) and node.func.id == 'str' and name == 'prediction.npz':
                method, key = 'location', 'location'
            if method:
                counts[key] += 1
                node.func = ast.Attribute(ast.Name('_remote_results', ast.Load()), method, ast.Load())
                node._native_original = before
                return node
            return self.generic_visit(node)

        def visit_Expr(self, node):
            node = self.generic_visit(node)
            call = node.value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == 'write_text' and _filename(call.func.value) == 'generation.json'):
                added = ast.parse('report = _remote_results.finalize(report)').body[0]
                added._remote_injected = True
                counts['finalize'] += 1
                return [added, node]
            return node

    tree = ast.fix_missing_locations(Rewrite().visit(copy.deepcopy(original)))
    expected = dict(write=1, read=1, copy=1, sha=3, location=1, finalize=1)
    if counts != expected:
        raise ValueError(f'native PBR archive IO sites changed: expected {expected}, got {counts}')

    class Reverse(ast.NodeTransformer):
        def visit(self, node):
            if getattr(node, '_remote_injected', False):
                return None
            if hasattr(node, '_native_original'):
                return copy.deepcopy(node._native_original)
            return super().visit(node)

    if ast.dump(Reverse().visit(copy.deepcopy(tree))) != ast.dump(original):
        raise ValueError('PBR model/sampler/RNG/source changed outside explicit archive IO')
    return tree, dict(stage='PBR-replay', sites=counts, reverse_AST_exact=True,
                      source_SHA256=hashlib.sha256(source.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    binding = json.loads(args.plan.read_text())
    source_path = Path(__file__).with_name('replay_pbr_on_generated_shape.py')
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != binding['native_source_SHA256']:
        raise ValueError('native PBR source bytes differ')
    native_path = Path(binding['native_plan'])
    payload = native_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != binding['native_plan_SHA256']:
        raise ValueError('native PBR plan bytes differ')
    native = json.loads(payload)
    tree, contract = remote_pbr_tree(source.decode())
    store = ImmutableResultStore(binding['store']['ssh_argv'], binding['store']['remote_root'])
    io = PbrRemoteResults(store, output=native['output'], prefix=binding['output_prefix'],
                          input_manifests=binding['input_manifests'], source_contract=contract,
                          compressed=binding['compressed'])
    namespace = dict(__name__=f'{__package__}._remote_native_PBR_replay', __package__=__package__,
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

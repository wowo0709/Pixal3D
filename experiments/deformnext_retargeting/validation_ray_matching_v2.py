"""Match the native256 query partition, including the final partial chunk.

BF16 GEMM can change its accumulation with a different row count. The native
VAL replay diagnosis found a maximum0.00928 probability difference for a
109-row chunk split into64+45, with every input tensor equal. Replaying the
actual109 rows together was exact. Keep this contract instead of accepting
an unexplained tolerance or changing any frozen reconstruction experiment.
"""
import numpy as np
import torch

from pixal3d.modules.ray_prior_anchored_readout import correspondence_probabilities


def evaluate_native_partition(head, observations, rows, *, verify_native_reference=False):
    head.eval()
    counts = {kind: dict(mass_sum=0., hit1=0, proposal64_hits=0) for kind in ['exact', 'near1']}
    count, per_asset = 0, []
    with torch.no_grad():
        for row in rows:
            x, y = observations.load(row)
            with np.load(observations.root/row['inputs_relative_path'], allow_pickle=False) as archive:
                native_reference = torch.from_numpy(archive['reference_cold_probabilities'].copy())
            parts = []
            for begin in range(0, len(x['native_query']), 256):
                take = slice(begin, begin+256)
                ids = x['candidate_indices'][take].astype(np.int64)
                args = [torch.from_numpy(value).cuda() for value in [x['native_query'][take].astype(np.float32),
                    x['context_query'][take].astype(np.float32), x['source_key_bank'][ids].astype(np.float32),
                    x['source_value_bank'][ids].astype(np.float32), x['candidate_valid'][take],
                    x['scalar_prior_logits'][take].astype(np.float32)]]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    parts.append(correspondence_probabilities(head, *args).cpu())
            actual = torch.cat(parts) if parts else torch.empty(0, 4, 64)
            if verify_native_reference and not torch.equal(actual, native_reference):
                raise ValueError('actual cold-head probability replay must match native256 partition exactly')
            probability = actual.mean(1)
            relation = torch.from_numpy(y['relation'])
            metrics = {}
            for kind in counts:
                label = torch.from_numpy(y[kind])
                metrics[kind] = dict(mass_sum=float((probability[relation]*label).sum()),
                    hit1=int(label.gather(1, probability[relation].argmax(-1)[:, None]).sum()),
                    proposal64_hits=int(label.any(-1).sum()))
                for key, value in metrics[kind].items():
                    counts[kind][key] += value
            count += len(relation)
            per_asset.append(dict(sample_id=row['sample_id'], source_cluster_id=row['source_cluster_id'],
                observable_relations=len(relation), metrics=metrics))
    return dict(assets=len(rows), observable_relations=count, per_asset=per_asset,
        **{kind: dict(mean_probability_mass=value['mass_sum']/count, top1_coverage=value['hit1']/count,
            proposal64_coverage=value['proposal64_hits']/count) for kind, value in counts.items()})

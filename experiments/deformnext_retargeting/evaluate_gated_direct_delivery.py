"""Pinned deterministic physical-NoDQ evaluation with its trained gated consumer."""
import json
from pathlib import Path
import sys


def main():
    from pixal3d.trainers.flow_matching.flow_matching import ImageConditionedProjFlowMatchingCFGTrainer as selected
    from . import evaluate_direct_no_query as native
    from .gated_direct_delivery_runtime import GatedDirectDeliveryMixin, POLICY
    previous = selected.__dict__.get('_inject_deform_transport')
    selected._inject_deform_transport = GatedDirectDeliveryMixin._inject_deform_transport
    try:
        native.main()
    finally:
        if previous is None:
            delattr(selected, '_inject_deform_transport')
        else:
            selected._inject_deform_transport = previous
    out = Path(sys.argv[sys.argv.index('--output')+1])
    path = out/'evaluation.json'; record = json.loads(path.read_text())
    events = []
    for eventfile in out.rglob('gated-delivery-forward-events.rank0.jsonl'):
        events.extend(json.loads(line) for line in eventfile.read_text().splitlines() if line.strip())
    expected = len(record['per_asset'])//record['generation_batch_size']
    if (len(events) != expected or any(e['policy'] != POLICY for e in events)
            or any(e['gradient_enabled'] for e in events)
            or record['optimizer_updates_during_evaluation'] != 0):
        raise ValueError('actual frozen gated consumer must execute every evaluation batch')
    record['gated_direct_delivery'] = dict(direct_delivery_policy=POLICY,
        new_delivery_parameters=0, original_R_input_contains_attention_F=True,
        original_semantic_carrier_preserved=True, actual_forward_events=events,
        optimizer_updates=0, physically_absent_deformation_channels=True)
    path.write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()

"""Versioned carrier diagnostic with explicit removal of unused transport.

The frozen mechanical proof still sees the complete transport tensors.
Only its two carrier-only sampling panels remove those tensors, because
their conditioner is intentionally disabled. Native model/trainer/data
sources and all numerical training recipes are unchanged.
"""
import argparse
import json
from pathlib import Path
import sys


class CarrierOnlyDataset:
    def __init__(self, dataset, allowed_transport_keys):
        self.dataset = dataset
        self.allowed_transport_keys = frozenset('transport_' + key for key in allowed_transport_keys)
        self.removed = set()

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __getitem__(self, index):
        item = self.dataset[index]
        transport = {key for key in item if key.startswith('transport_')}
        if transport - self.allowed_transport_keys:
            raise ValueError('unrecognized carrier diagnostic transport fields')
        self.removed.update(transport)
        return {key: value for key, value in item.items() if key not in transport}


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--readout', choices=['top1', 'weighted', 'attention', 'spatial'], required=True)
    local, remaining = parser.parse_known_args()
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import ImageConditionedProjMixin
    from . import quick_metrics
    if local.readout == 'weighted':
        from . import probe_target_first_observed_weighted as proof
    elif local.readout == 'spatial':
        from . import probe_target_first_observed_spatial as proof
    else:
        from . import probe_target_first_observed_query as proof
    original = quick_metrics._ss_metrics
    events = []

    def carrier_metrics(trainer, dataset, indices, config):
        if trainer.deform_conditioner_config is not None:
            raise ValueError('transport isolation is exclusive to disabled-conditioner carrier diagnostics')
        proxy = CarrierOnlyDataset(dataset, ImageConditionedProjMixin._DEFORM_TRANSPORT_KEYS)
        result = original(trainer, proxy, indices, config)
        events.append(dict(assets=len(indices),removed_fields=sorted(proxy.removed),
                           edited_RGB_camera_and_GT_latent_objects_unchanged=True,
                           conditioner_disabled=True))
        return result

    previous = sys.argv
    quick_metrics._ss_metrics = carrier_metrics
    try:
        sys.argv = [str(Path(__file__)), *remaining]
        proof.main()
    finally:
        quick_metrics._ss_metrics = original
        sys.argv = previous
    if len(events) != 2 or any(event['assets'] != 32 or not event['removed_fields'] for event in events):
        raise ValueError('both real paired carrier32 panels must execute with transport removed')
    output = Path(remaining[remaining.index('--output') + 1])
    (output / 'carrier-transport-isolation.json').write_text(json.dumps(dict(
        status='PASS_ACTUAL_CARRIER_PANELS_WITH_UNUSED_TRANSPORT_REMOVED',readout=local.readout,
        events=events,mechanical_proof_transport_retained=True,
        native_training_or_sampling_sources_modified=False,optimizer_updates=0),indent=2)+'\n')


if __name__ == '__main__':
    main()

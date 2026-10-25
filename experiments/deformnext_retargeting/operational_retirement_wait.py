"""Run the frozen raw-retirement entry with strict process-inspection waits."""
import argparse
import json
import os
from pathlib import Path

from .cascade_view_roles import file_sha256 as sha
from .operational_process_access import process_access_waits
from . import retire_selected_final_raw as frozen


def verify_operational_sources(context):
    expected = context['operational_source_SHA256']
    own = Path(__file__).resolve()
    required = [own, own.with_name('operational_process_access.py'),
                own.with_name('operational_campaign_entry.py')]
    if set(expected) != {str(p) for p in required}:
        raise ValueError('explicit separate operational-adapter source binding required')
    for path, digest in expected.items():
        if sha(path) != digest:
            raise ValueError('operational-adapter source changed: ' + path)


def execute(path):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU-only operational retirement required')
    path = Path(path)
    context = json.loads(path.read_text())
    verify_operational_sources(context)
    events = Path(context['process_access_event_log'])
    owned = Path(context['operational_campaign_root'])
    if not events.is_absolute() or not events.is_relative_to(owned) or events.resolve() != events:
        raise ValueError('owned operational process-access journal required')
    def emit(event):
        with events.open('a') as stream:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
    with process_access_waits(emit=emit):
        result = frozen.execute(path)
    verify_operational_sources(context)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(execute(args.plan), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

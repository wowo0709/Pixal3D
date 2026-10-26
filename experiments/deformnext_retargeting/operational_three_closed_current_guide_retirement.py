"""Use the verified three-producer manifest for completed current guide offload.

The existing retirement procedure owns all file, reader, NAS, and pixel checks.
This separate process changes only its manifest reader for the v636 successor.
"""

from pathlib import Path

from . import operational_closed_current_guide_retirement as original
from . import operational_multi_campaign_reuse as two_source
from . import operational_three_campaign_reuse as three_source
from .selected_pipeline_model_contract import require


def execute(path):
    require(original.load_manifest is two_source.load_manifest,
        'original completed-current guide procedure changed its manifest reader')
    policy_path = Path(path)
    require(policy_path.is_file(), 'explicit closed guide retirement policy required')
    original.load_manifest = three_source.load_manifest
    try:
        return original.execute(policy_path)
    finally:
        original.load_manifest = two_source.load_manifest


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    print(json.dumps(execute(parser.parse_args().plan)), flush=True)


if __name__ == '__main__':
    main()

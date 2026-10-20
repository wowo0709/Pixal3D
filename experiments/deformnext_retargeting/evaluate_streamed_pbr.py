"""Run an existing read-only PBR evaluator with frozen projection streaming."""
import argparse
from pathlib import Path
import runpy
import sys

from .frozen_projection_streaming import install_frozen_projection_streaming


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--delegate', choices=['direct', 'native', 'outside'], required=True)
    local, remaining = parser.parse_known_args()
    output = Path(remaining[remaining.index('--output')+1])
    install_frozen_projection_streaming(max_assets=1,
                                       audit_directory=output/'projection-streaming-audit')
    module = dict(direct='evaluate_direct_pbr', native='evaluate_pbr_streaming',
                  outside='evaluate_outside_proposal')[local.delegate]
    previous = sys.argv
    try:
        sys.argv = [module, *remaining]
        runpy.run_module('experiments.deformnext_retargeting.'+module, run_name='__main__')
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()

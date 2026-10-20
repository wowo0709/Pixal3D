"""PBR native continuation with bounded frozen projection intermediates."""
from pathlib import Path

from . import train_pbr_joint as native_entry
from .frozen_projection_streaming import install_frozen_projection_streaming

_native_worker = native_entry.worker


def worker(local_rank, config):
    install_frozen_projection_streaming(max_assets=1,
        audit_directory=Path(config.output_dir)/'projection-streaming-audit')
    _native_worker(local_rank, config)


def main():
    original_worker = native_entry.worker
    native_entry.worker = worker
    try:
        native_entry.main()
    finally:
        native_entry.worker = original_worker


if __name__ == '__main__':
    main()

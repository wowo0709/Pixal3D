from __future__ import annotations

from .config import PipelineConfig
from .orchestrator import PipelineServices
from .runtime import (
    ArtifactValidationError,
    SafeRegistryStore,
    read_gate_report,
    read_parallelism_report,
)


SOURCE_ORDER = (
    "ABO",
    "HSSD",
    "3D-FUTURE",
    "ObjaverseXL_sketchfab",
    "ObjaverseXL_github",
)


class FullProductionRunner:
    def __init__(self, config: PipelineConfig, services: PipelineServices):
        self.config = config
        self.services = services

    def _registry(self):
        if set(self.config.sources) != set(SOURCE_ORDER):
            raise ArtifactValidationError(
                "full production source set does not match the approved order"
            )
        read_gate_report(self.config, "smoke")
        read_gate_report(self.config, "pilot")
        read_parallelism_report(self.config)
        return SafeRegistryStore(
            self.config.paths.data2_root / "control/assets.parquet",
            self.config,
        ).load()

    def plan(self) -> tuple[str, ...]:
        registry = self._registry()
        lines = []
        for source in SOURCE_ORDER:
            shards = tuple(
                sorted(
                    registry.loc[
                        registry["owner_source"] == source,
                        "shard_id",
                    ].unique()
                )
            )
            if not shards:
                raise ArtifactValidationError(
                    f"full production registry has no shard for source: {source}"
                )
            for shard in shards:
                batches = self.services.plan(
                    "production", source, shard, None, freeze=False
                )
                for batch in batches:
                    try:
                        batch_id, remainder = batch.split(": ", 1)
                        count_text, unit = remainder.split()
                        count = int(count_text)
                    except (AttributeError, TypeError, ValueError) as error:
                        raise ArtifactValidationError(
                            f"invalid production batch plan: {batch!r}"
                        ) from error
                    if unit != "assets" or count <= 0:
                        raise ArtifactValidationError(
                            f"invalid production batch plan: {batch!r}"
                        )
                    chunk_max = self.config.parallelism.chunk_assets
                    chunks = (count + chunk_max - 1) // chunk_max
                    lines.append(
                        f"{source}/{shard}/{batch_id}: {count} assets, "
                        f"{chunks} chunks (max {chunk_max})"
                    )
        return tuple(lines)

    def run(self) -> None:
        registry = self._registry()
        for source in SOURCE_ORDER:
            shards = tuple(
                sorted(
                    registry.loc[
                        registry["owner_source"] == source,
                        "shard_id",
                    ].unique()
                )
            )
            if not shards:
                raise ArtifactValidationError(
                    f"full production registry has no shard for source: {source}"
                )
            for shard in shards:
                self.services.run("production", source, shard, None)
                self.services.audit("production", source, shard)

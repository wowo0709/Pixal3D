from __future__ import annotations

from .config import PipelineConfig
from .orchestrator import PipelineServices
from .runtime import ArtifactValidationError, SafeRegistryStore, read_gate_report


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

    def run(self) -> None:
        if set(self.config.sources) != set(SOURCE_ORDER):
            raise ArtifactValidationError(
                "full production source set does not match the approved order"
            )
        read_gate_report(self.config, "smoke")
        read_gate_report(self.config, "pilot")
        registry = SafeRegistryStore(
            self.config.paths.data2_root / "control/assets.parquet",
            self.config,
        ).load()
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

import pandas as pd

from data_toolkit.pipeline.registry import (
    RegistryStore,
    assign_shards,
    camera_seed,
    canonicalize_sources,
    split_for_sha,
    write_compat_metadata,
)


def test_seed_and_split_are_deterministic():
    sha = "01234567" + "a" * 56
    assert camera_seed(sha, "pixal3d-mv-camera-v1") == camera_seed(
        sha, "pixal3d-mv-camera-v1"
    )
    assert split_for_sha(sha) == "train"
    assert split_for_sha("00000000" + "b" * 56) == "validation"


def test_global_deduplication():
    shared = "a" * 64
    frames = {
        "ABO": pd.DataFrame([{"sha256": shared, "file_identifier": "a.glb"}]),
        "HSSD": pd.DataFrame([{"sha256": shared, "file_identifier": "h.glb"}]),
    }
    result = canonicalize_sources(
        frames, "pixal3d-mv-camera-v1", ("ABO", "HSSD")
    )
    assert len(result) == 1
    assert result.iloc[0]["owner_source"] == "ABO"
    assert result.iloc[0]["duplicate_sources"] == '["HSSD"]'


def test_stable_shards_and_atomic_roundtrip(tmp_path):
    frame = pd.DataFrame(
        {
            "sha256": [f"{i:064x}" for i in range(6)],
            "owner_source": ["ABO"] * 6,
        }
    )
    first = assign_shards(frame, 2).set_index("sha256")["shard_id"].to_dict()
    second = (
        assign_shards(frame.sample(frac=1, random_state=7), 2)
        .set_index("sha256")["shard_id"]
        .to_dict()
    )
    assert first == second
    store = RegistryStore(tmp_path / "assets.parquet")
    store.save(frame)
    assert store.load().to_dict("records") == frame.to_dict("records")
    assert not (tmp_path / "assets.parquet.tmp").exists()


def test_compat_metadata_contains_only_owned_rows(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "sha256": "a" * 64,
                "owner_source": "ABO",
                "file_identifier": "a.glb",
            },
            {
                "sha256": "b" * 64,
                "owner_source": "HSSD",
                "file_identifier": "b.glb",
            },
        ]
    )
    path = tmp_path / "ABO" / "metadata.csv"
    write_compat_metadata(frame, "ABO", path)
    assert pd.read_csv(path)["sha256"].tolist() == ["a" * 64]
    assert not path.with_suffix(".csv.tmp").exists()

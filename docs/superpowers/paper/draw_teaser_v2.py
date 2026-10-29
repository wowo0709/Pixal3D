"""Draw an image-dominant, compact GEAR3D teaser from scored TEST arrays."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image


INK = "#243244"
MUTED = "#65758A"
BLUE = "#327BB2"
PURPLE = "#7658A6"
TEAL = "#148A80"
LIGHT = "#DDE4E9"


def label(ax, x, y, text, *, size=7.3, color=INK, weight="normal", ha="left"):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va="center", zorder=10)


def tile(ax, x, y, w, h, image, *, edge=LIGHT, lw=.7):
    ax.imshow(image, extent=[x, x + w, y, y + h], interpolation="lanczos", zorder=2)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=edge,
                           linewidth=lw, zorder=3))


def load_input(path: Path):
    with Image.open(path) as image:
        rgba = np.asarray(image.convert("RGBA").resize(
            (512, 512), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    return rgba[..., :3] * rgba[..., 3:4]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-audit", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    campaign = json.loads(args.campaign_audit.read_text())
    if (campaign.get("status") != "PASS_ACTUAL_SELECTED_FINAL_CASE_CAMPAIGN"
            or args.sample_id not in campaign.get("actual_cases", {})):
        raise ValueError("completed selected TEST case required")
    entries = campaign["actual_cases"][args.sample_id]
    selected_root = Path(entries["SELECTED"]["audit"]["path"]).parent
    case = json.loads((selected_root / "case-plan.json").read_text())
    generation = case["phases"][0]["plan"]
    input_paths = [Path(path) for path in generation["RGB_paths"]]
    if len(input_paths) != 4:
        raise ValueError("one edited target plus three original source views required")
    inputs = [load_input(path) for path in input_paths]

    predictions, references, hashes = {}, {}, {}
    for alias in ("V1", "SELECTED"):
        root = Path(entries[alias]["audit"]["path"]).parent / "shared"
        for camera in (8, 9):
            path = root / f"camera{camera:02d}.npz"
            with np.load(path) as arrays:
                prediction = np.transpose(arrays["prediction"], (1, 2, 0)).clip(0, 1)
                reference = np.transpose(arrays["GT_shared_renderer"], (1, 2, 0)).clip(0, 1)
            predictions[alias, camera] = prediction
            if camera in references and not np.array_equal(references[camera], reference):
                raise ValueError("shared GT pixels differ across methods")
            references[camera] = reference
            hashes[f"{alias}/camera{camera:02d}"] = hashlib.sha256(path.read_bytes()).hexdigest()

    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.transparent": False})
    fig, ax = plt.subplots(figsize=(7.05, 3.06))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 14.1); ax.set_ylim(0, 6.12); ax.set_aspect("equal"); ax.axis("off")

    label(ax, .28, 5.78, "Geometry-conditioned appearance retargeting", size=9.4, weight="bold")
    ax.plot([.28, 13.86], [5.52, 5.52], color=LIGHT, lw=.75)

    # Image-only input group.
    label(ax, 2.03, 5.16, "Inputs", size=7.7, weight="bold", ha="center")
    tile(ax, .32, 2.42, 2.00, 2.00, inputs[0], edge=BLUE, lw=.9)
    label(ax, 1.32, 2.14, "edited target", size=6.6, color=BLUE, weight="bold", ha="center")
    for index, x in enumerate((2.54, 3.42, 4.30)):
        tile(ax, x, 3.10, .76, .76, inputs[index + 1], edge=PURPLE, lw=.75)
    label(ax, 3.68, 2.73, "original source views", size=6.3, color=PURPLE,
          weight="bold", ha="center")
    ax.add_patch(FancyArrowPatch((4.95, 2.58), (5.58, 2.58), arrowstyle="-|>",
                                 mutation_scale=10, linewidth=1.6, color=TEAL))
    label(ax, 5.27, 2.86, "retarget", size=6.2, color=TEAL, weight="bold", ha="center")

    columns = [("GT", 5.78, INK), ("Vanilla FT", 8.47, MUTED),
               ("GEAR3D", 11.16, TEAL)]
    for name, x, color in columns:
        label(ax, x + 1.18, 5.16, name, size=7.5, color=color,
              weight="bold" if name == "GEAR3D" else "normal", ha="center")
    for row, camera in enumerate((8, 9)):
        y = 2.94 - row * 2.39
        label(ax, 5.62, y + .86, "target" if camera == 8 else "source",
              size=6.3, color=BLUE if camera == 8 else PURPLE, ha="right")
        images = [references[camera], predictions["V1", camera], predictions["SELECTED", camera]]
        for (_, x, color), image in zip(columns, images):
            tile(ax, x, y, 2.36, 1.82, image, edge=color if x == 11.16 else LIGHT,
                 lw=.9 if x == 11.16 else .65)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "paper-teaser-v2.pdf", facecolor="white")
    fig.savefig(args.output_dir / "paper-teaser-v2.png", dpi=280, facecolor="white")
    plt.close(fig)
    provenance = {
        "schema": "gear3d-image-dominant-teaser-v2",
        "sample_id": args.sample_id,
        "scored_fullframe_arrays": True,
        "prediction_SHA256": hashes,
        "input_RGB_SHA256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in input_paths},
        "note": "Illustrative held-out TEST case; no metric claim is made by the teaser.",
    }
    (args.output_dir / "paper-teaser-v2-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

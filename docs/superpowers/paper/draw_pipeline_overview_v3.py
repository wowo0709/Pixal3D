"""Draw the compact, image-grounded GEAR3D overview used by the paper.

The overview deliberately shows only the three paper-level stages.  Matching
and retargeting internals are reserved for the one-column detail figures.
Input pixels are SHA-verified through the archived teaser provenance; the
output is read from an archived scored TEST render.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np

from draw_pipeline_overview_v2 import load_input


INK = "#243244"
MUTED = "#65758A"
BLUE = "#327BB2"
PURPLE = "#7658A6"
TEAL = "#148A80"
AMBER = "#C68234"
GRID = "#B7C8CF"
LIGHT = "#E4E9ED"
PALE_BLUE = "#EDF5FA"
PALE_PURPLE = "#F4F0F8"


def label(ax, x, y, s, *, size=7.4, color=INK, weight="normal", ha="left"):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                   ha=ha, va="center", zorder=20)


def arrow(ax, a, b, *, color=INK, lw=1.25, head=8, curve=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=head,
        linewidth=lw, color=color, connectionstyle=f"arc3,rad={curve}",
        shrinkA=0, shrinkB=0, zorder=15))


def image_tile(ax, x, y, w, h, image, *, edge=LIGHT, lw=.7):
    ax.imshow(image, extent=[x, x+w, y, y+h], interpolation="lanczos", zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=edge,
                           linewidth=lw, zorder=5))


def read_prediction(path: Path):
    with np.load(path) as bank:
        rgb = np.transpose(bank["prediction"], (1, 2, 0)).clip(0, 1)
        alpha = bank["prediction_alpha"].clip(0, 1)
    yy, xx = np.where(alpha > .02)
    if len(xx) == 0:
        raise ValueError("empty prediction alpha")
    side = max(xx.max()-xx.min()+1, yy.max()-yy.min()+1)
    side = int(min(alpha.shape[0], np.ceil(side * 1.26)))
    cx = (xx.min()+xx.max()) / 2
    cy = (yy.min()+yy.max()) / 2
    left = int(np.clip(round(cx-side/2), 0, alpha.shape[1]-side))
    top = int(np.clip(round(cy-side/2), 0, alpha.shape[0]-side))
    crop = rgb[top:top+side, left:left+side]
    return crop, {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "crop_xywh": [left, top, side, side],
    }


def draw_volume(ax, x, y, w=1.68, h=1.62):
    depth = np.array([.42, .30])
    front = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    back = front + depth
    for poly in (front, back):
        ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=GRID,
                             linewidth=.72, zorder=6))
    for i in range(4):
        ax.plot([front[i,0], back[i,0]], [front[i,1], back[i,1]],
                color=GRID, lw=.72, zorder=6)
    for f in (.25, .5, .75):
        ax.plot([x+w*f]*2, [y, y+h], color=LIGHT, lw=.55, zorder=4)
        ax.plot([x, x+w], [y+h*f]*2, color=LIGHT, lw=.55, zorder=4)
    # One source-camera ray, and all voxel sites that share its evidence.
    p0 = np.array([x-.66, y+.48])
    p1 = np.array([x+w+.68, y+h-.20])
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=PURPLE, lw=1.65, zorder=9)
    for f in (.34, .47, .60, .73):
        p = p0*(1-f)+p1*f
        ax.add_patch(Circle(tuple(p), .064, facecolor=PURPLE,
                            edgecolor="white", lw=.45, zorder=10))
    return p0, p1


def draw_cascade(ax):
    y = 2.68
    xs = [9.73, 10.84, 11.95]
    names = [("SS64", TEAL), ("Shape", "#708497"), ("PBR", AMBER)]
    for x, (name, color) in zip(xs, names):
        ax.add_patch(Circle((x, y), .31, facecolor="white", edgecolor=color,
                            linewidth=1.4, zorder=7))
        # A small sparse-grid glyph gives the cascade a 3D rather than box-flow look.
        for dx, dy, s in ((-.10,-.06,.08),(0,.04,.10),(.10,-.01,.07)):
            ax.add_patch(Rectangle((x+dx-s/2, y+dy-s/2), s, s,
                                   facecolor=color, alpha=.78, edgecolor="none", zorder=8))
        label(ax, x, 2.17, name, size=7.0, color=color, weight="bold", ha="center")
    arrow(ax, (10.06,y), (10.50,y), color=INK, lw=1.05)
    arrow(ax, (11.17,y), (11.61,y), color=INK, lw=1.05)
    return (12.29, y)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-provenance", type=Path, required=True)
    parser.add_argument("--selected-render", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs, input_record = load_input(args.input_provenance)
    output, output_record = read_prediction(args.selected_render)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family":"DejaVu Sans", "pdf.fonttype":42,
                         "ps.fonttype":42, "savefig.transparent":False})
    fig, ax = plt.subplots(figsize=(7.05, 2.48))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 14.1); ax.set_ylim(0, 4.96); ax.set_aspect("equal"); ax.axis("off")

    # Three restrained section labels; no detailed module boxes in this figure.
    label(ax, .22, 4.63, "(a) Inputs", size=8.0, weight="bold")
    label(ax, 4.45, 4.63, "(b) GEAR3D evidence transfer", size=8.0, weight="bold")
    label(ax, 10.02, 4.63, "(c) Pixal3D generation", size=8.0, weight="bold")
    ax.plot([.18, 13.92], [4.36, 4.36], color=LIGHT, lw=.75)

    # Target is visually primary; the three original views are a compact source strip.
    image_tile(ax, .24, 2.08, 1.78, 1.78, inputs[0], edge=BLUE, lw=.85)
    label(ax, 1.13, 1.79, "edited target", size=7.0, color=BLUE, weight="bold", ha="center")
    for j, x in enumerate((2.17, 2.96, 3.75)):
        image_tile(ax, x, 2.52, .70, .70, inputs[j+1], edge=PURPLE, lw=.70)
    label(ax, 3.10, 2.24, "original source views", size=6.2, color=PURPLE,
          weight="bold", ha="center")
    label(ax, .24, .83, "target state", size=6.8, color=BLUE, weight="bold")
    ax.plot([.25, 3.89], [.57, .57], color=BLUE, lw=1.35)
    arrow(ax, (3.89,.57), (4.37,.57), color=BLUE, lw=1.35)

    # Visual correspondence: a target patch queries source candidates, which form one
    # posterior shared by the sites of a projected source ray.
    ax.add_patch(Rectangle((4.53, 3.00), .54, .54, facecolor=PALE_BLUE,
                           edgecolor=BLUE, lw=.75, zorder=3))
    for r in range(3):
        for c in range(3):
            ax.add_patch(Rectangle((4.56+c*.17,3.03+r*.17),.13,.13,
                facecolor=BLUE if (r,c)==(1,1) else "#D5E6F0", edgecolor="white", lw=.3, zorder=4))
    label(ax, 4.80, 2.72, "target patch", size=6.2, color=BLUE, ha="center")
    arrow(ax, (5.17,3.27), (5.62,3.27), color=PURPLE, lw=1.15)
    for i, (x, a) in enumerate(((5.77,.95),(6.13,.72),(6.49,.50))):
        ax.add_patch(Circle((x,3.27), .12, facecolor=PURPLE, alpha=a,
                            edgecolor="white", lw=.4, zorder=7))
    label(ax, 6.13, 2.72, "source Top-$K$", size=6.2, color=PURPLE, ha="center")
    arrow(ax, (6.73,3.27), (7.03,3.27), color=PURPLE, lw=1.15)
    draw_volume(ax, 7.10, 1.70)
    label(ax, 7.94, 1.33, "one posterior per source ray", size=6.65,
          color=PURPLE, weight="bold", ha="center")
    label(ax, 7.94, .88, "retargeted correction", size=6.65,
          color=TEAL, weight="bold", ha="center")
    arrow(ax, (8.80,.96), (9.30,1.55), color=TEAL, lw=1.3)

    # The native target path and the source correction meet only at SS; the frozen
    # shape stages and the appearance stage remain visually simple here.
    label(ax, 10.80, 3.63, "Pixal3D cascade", size=7.2, weight="bold", ha="center")
    label(ax, 10.80, 3.29, "SS / frozen Shape / PBR", size=6.55, color=MUTED, ha="center")
    arrow(ax, (9.24,2.68), (9.39,2.68), color=TEAL, lw=1.25)
    end = draw_cascade(ax)
    arrow(ax, end, (12.55,2.68), color=INK, lw=1.15)
    image_tile(ax, 12.70, 1.68, 1.24, 1.92, output, edge=TEAL, lw=.9)
    label(ax, 13.32, 1.36, "target geometry", size=6.0, color=BLUE, ha="center")
    label(ax, 13.28, 1.08, "+ source appearance", size=5.0, color=PURPLE, ha="center")

    fig.savefig(args.output_dir/"paper-overall-v3.pdf", facecolor="white")
    fig.savefig(args.output_dir/"paper-overall-v3.png", dpi=280, facecolor="white")
    plt.close(fig)

    record = {
        "schema": "gear3d-overall-figure-v3",
        "input": input_record,
        "selected_render": output_record,
        "note": "Image-grounded overview; correspondence glyphs are schematic.",
    }
    (args.output_dir/"paper-overall-v3-provenance.json").write_text(
        json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()

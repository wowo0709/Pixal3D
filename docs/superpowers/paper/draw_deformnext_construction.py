"""Compose a vector pipeline and verified, previously selected dataset renders.

No rendering, model inference, recoloring, or new outcome-based selection.
Inputs are original PNGs with recorded hashes, from a gallery or explicit mirror.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import re
from pathlib import Path

import pymupdf as fitz
import numpy as np
from PIL import Image


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--gallery-html", type=Path, required=True)
    parser.add_argument("--membership-proof", type=Path, required=True)
    parser.add_argument("--image-root", type=Path,
                        help="Explicit mirror root containing original /file4/... PNGs")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection_json.read_text())
    membership = json.loads(args.membership_proof.read_text())
    assert membership["status"] == "PASS" and membership["final_snapshot_count"] == 10165
    admitted = {row["sample_id"] for row in membership["selected_rows"]}
    rows = [selection["selected_rows"][i] for i in (0, 1, 3, 6)]
    assert len({row["source_uid"] for row in rows}) == 4
    chosen = []
    wanted = set()
    for row in rows:
        assert row["sample_id"] in admitted
        images = []
        for state in ("M0", "M1"):
            images.append(next(im for im in row["images"]
                               if im["server_path"].endswith(f"/cycles/view_08_{state}.png")))
        for state in ("M0", "M1"):
            images.append(next(im for im in row["images"]
                               if im["server_path"].endswith(f"_{state}/view_14.png")))
        chosen.append({"category": row["category"], "sample_id": row["sample_id"],
                       "source_uid": row["source_uid"], "images": images})
        wanted.update(im["sha256"] for im in images)
    payloads = {}
    if args.image_root is not None:
        for row in chosen:
            for im in row["images"]:
                path = args.image_root / im["server_path"].lstrip("/")
                payload = path.read_bytes()
                if sha(payload) != im["sha256"]:
                    raise ValueError(f"Archived original PNG hash differs: {path}")
                payloads[im["sha256"]] = payload
    for match in re.finditer(r"data:image/[^;]+;base64,([A-Za-z0-9+/=\r\n]+)",
                             args.gallery_html.read_text()):
        payload = base64.b64decode(match.group(1), validate=False)
        digest = sha(payload)
        if digest in wanted:
            payloads[digest] = payload
    if set(payloads) != wanted:
        raise ValueError(f"Historical PNG hashes missing: {sorted(wanted - set(payloads))}")

    doc = fitz.open()
    page = doc.new_page(width=760, height=377)
    dark = (0.10, 0.15, 0.21)
    green = (0.02, 0.39, 0.31)
    boxes = [
        ("PartNeXt pair", "Canonical M0 + donor", "Semantic roles"),
        ("Donor prior", "8 RGB / LoRA 800", "Reused across sources"),
        ("Deformation", "GeoStyle + assembly", "300 production iters"),
        ("Final candidates", "PA / optional JN-L", "120 refinement iters"),
        ("Final-pair QC", "RGB / Shape / SS", "Distinct valid variants"),
    ]
    for i, lines in enumerate(boxes):
        x = 8 + i * 151
        rect = fitz.Rect(x, 7, x + 139, 79)
        page.draw_rect(rect, color=(0.65, 0.73, 0.80), fill=(0.95, 0.97, 0.98), width=0.7)
        for j, line in enumerate(lines):
            r = fitz.Rect(x + 3, 12 + 20 * j, x + 136, 38 + 20 * j)
            used = page.insert_textbox(r, line, fontsize=12.3 if j == 0 else 10.7,
                                       fontname="hebo" if j == 0 else "helv", color=dark,
                                       align=fitz.TEXT_ALIGN_CENTER)
            assert used >= 0, line
        if i < 4:
            page.draw_line((x + 140, 43), (x + 150, 43), color=dark, width=1)
            page.draw_line((x + 147, 40), (x + 150, 43), color=dark, width=1)
            page.draw_line((x + 147, 46), (x + 150, 43), color=dark, width=1)
    for i, row in enumerate(chosen):
        x = 8 + (i % 2) * 380
        y = 97 + (i // 2) * 139
        page.insert_text((x, y + 12), row["category"], fontname="hebo", fontsize=13, color=dark)
        clips = []
        for start in (0, 2):
            masks = []
            for image in row["images"][start:start + 2]:
                array = np.asarray(Image.open(io.BytesIO(payloads[image["sha256"]])).convert("RGBA"))
                if array[..., 3].min() < 255:
                    mask = array[..., 3] > 8
                else:
                    background = np.median(np.array([array[0, 0, :3], array[0, -1, :3],
                                                    array[-1, 0, :3], array[-1, -1, :3]]), axis=0)
                    mask = np.max(np.abs(array[..., :3].astype(float) - background), axis=-1) > 8
                masks.append(mask)
            union = masks[0] | masks[1]
            yy, xx = np.where(union)
            pad = max(12, int(0.10 * max(xx.max() - xx.min(), yy.max() - yy.min())))
            clip = fitz.Rect(max(0, xx.min() - pad), max(0, yy.min() - pad),
                             min(union.shape[1], xx.max() + pad + 1),
                             min(union.shape[0], yy.max() + pad + 1))
            clips.extend([clip, clip])
        row["paired_display_clips_pixels"] = [list(c) for c in clips]
        for j, (label, image) in enumerate(zip(("RGB M0", "RGB M1", "Clay M0", "Clay M1"), row["images"])):
            left = x + j * 90
            page.insert_text((left + 2, y + 29), label, fontsize=10.5,
                             color=green if j % 2 else dark)
            rect = fitz.Rect(left, y + 35, left + 86, y + 121)
            source = fitz.open(stream=payloads[image["sha256"]], filetype="png")
            image_pdf = fitz.open(stream=source.convert_to_pdf(), filetype="pdf")
            pixel_width, pixel_height = Image.open(io.BytesIO(payloads[image["sha256"]])).size
            sx = image_pdf[0].rect.width / pixel_width
            sy = image_pdf[0].rect.height / pixel_height
            c = clips[j]
            pdf_clip = fitz.Rect(c.x0 * sx, c.y0 * sy, c.x1 * sx, c.y1 * sy)
            page.show_pdf_page(rect, image_pdf, 0, clip=pdf_clip, keep_proportion=True)
            image_pdf.close()
            source.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf = args.output_dir / "deformnext-construction.pdf"
    doc.save(pdf, garbage=4, deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(args.output_dir / "deformnext-construction.png")
    provenance = {
        "selection": "Four categories from the existing large-edit illustration selection; not a random quality sample",
        "new_renders": 0,
        "paired_framing": "Archived pixels; identical foreground-union crop for M0/M1 at each camera; no independent normalization or recoloring",
        "selection_sha256": sha(args.selection_json.read_bytes()),
        "membership_proof_sha256": sha(args.membership_proof.read_bytes()),
        "gallery_sha256": sha(args.gallery_html.read_bytes()),
        "all_original_png_hashes_verified": True,
        "figure_sha256": sha(pdf.read_bytes()),
        "examples": chosen,
    }
    (args.output_dir / "deformnext-construction-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Verified {len(wanted)} archived images; saved {pdf}")


if __name__ == "__main__":
    main()

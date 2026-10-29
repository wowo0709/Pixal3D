# GEAR3D CVPR 2027 source

This directory contains the review-version manuscript and supplementary
material for **Geometry-Conditioned Appearance Retargeting for Image-to-3D
Generation**.

Build the paper with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`main.tex` includes the supplementary material after the references. Its
pages use `S1, S2, ...` numbering so their PDF anchors do not collide with the
main-paper pages.

The quantitative and ablation values are kept in `tables/`. The four PDFs in
`figures/` are the only figure assets required by the manuscript. Generated
LaTeX artifacts are intentionally ignored and are not part of the clean
Overleaf archive.

Before submission, replace the anonymized placeholder author block in
`main.tex` with the required submission metadata.

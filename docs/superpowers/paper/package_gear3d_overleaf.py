"""Create a clean, deterministic Overleaf source archive for GEAR3D."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT_FILES = {
    "README.md",
    "main.tex",
    "main.bib",
    "preamble.tex",
    "cvpr.sty",
    "ieeenat_fullname.bst",
}
SOURCE_SUFFIXES = {".tex", ".bib", ".bst", ".sty", ".pdf", ".png", ".json"}
FIGURE_FILES = {
    "figures/paper-matching-v2.pdf",
    "figures/paper-overall-v3.pdf",
    "figures/paper-retargeting-v2.pdf",
    "figures/paper-teaser-v2.pdf",
}
IGNORED_NAMES = {
    "main.pdf",
    "main.aux",
    "main.bbl",
    "main.blg",
    "main.brf",
    "main.fdb_latexmk",
    "main.fls",
    "main.log",
    "main.out",
    "missfont.log",
}
IGNORED_RELATIVE_PATHS = {
    "sec/2_formatting.tex",
    "sec/3_finalcopy.tex",
}


def selected_files(source: Path) -> list[Path]:
    files: list[Path] = []
    for path in source.rglob("*"):
        if not path.is_file() or path.name in IGNORED_NAMES:
            continue
        relative = path.relative_to(source)
        if relative.as_posix() in IGNORED_RELATIVE_PATHS:
            continue
        if len(relative.parts) == 1:
            if path.name in ROOT_FILES:
                files.append(path)
        elif relative.parts[0] in {"sec", "tables"}:
            if path.suffix.lower() in SOURCE_SUFFIXES:
                files.append(path)
        elif relative.as_posix() in FIGURE_FILES:
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(source).as_posix())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not (source / "main.tex").is_file():
        raise FileNotFoundError("Overleaf source must contain main.tex")
    files = selected_files(source)
    required = {source / name for name in ROOT_FILES}
    missing = sorted(str(path) for path in required if path not in files)
    if missing:
        raise FileNotFoundError("missing required sources: " + ", ".join(missing))
    if output.exists() and not args.force:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = (2026, 10, 29, 0, 0, 0)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(source).as_posix()
            info = zipfile.ZipInfo(relative, timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    print(f"PASS_OVERLEAF_SOURCE_ARCHIVE files={len(files)} output={output}")


if __name__ == "__main__":
    main()

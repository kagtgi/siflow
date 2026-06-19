"""Zip import / export for the notebook handoff (download -> upload -> continue).

Each notebook part writes its artifacts under one base dir, zips the relevant
pieces, and auto-downloads them in the browser. The next part uploads that zip
(and any others it needs) and extracts them back under the same base dir. This
makes the parts self-contained on plain Colab -- no Google Drive required.
"""
from __future__ import annotations

import io
import os
import zipfile
from typing import List, Optional


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def export_zip(base: str, out_name: str, include: Optional[List[str]] = None) -> str:
    """Zip selected sub-paths of ``base`` into ``out_name`` (stored under /content
    on Colab, else next to ``base``). ``include`` is a list of paths relative to
    ``base`` (dirs or files); default zips everything under ``base``.
    Returns the zip path."""
    base = os.path.abspath(base)
    out_dir = "/content" if os.path.isdir("/content") else os.path.dirname(base) or "."
    out = out_name if os.path.isabs(out_name) else os.path.join(out_dir, out_name)
    roots = [os.path.join(base, p) for p in (include or ["."])]
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for root in roots:
            if os.path.isfile(root):
                z.write(root, os.path.relpath(root, base))
                n += 1
            elif os.path.isdir(root):
                for dp, _, fs in os.walk(root):
                    for f in fs:
                        fp = os.path.join(dp, f)
                        z.write(fp, os.path.relpath(fp, base))
                        n += 1
    print(f"[io] zipped {n} files -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    return out


def download_file(path: str) -> None:
    """Trigger a browser download on Colab (no-op elsewhere)."""
    if in_colab():
        from google.colab import files  # type: ignore

        files.download(path)
    else:
        print(f"[io] not on Colab; file is at {path}")


def export_and_download(base: str, out_name: str, include: Optional[List[str]] = None) -> str:
    """Zip ``include`` from ``base`` and download it. Call at the end of a notebook."""
    p = export_zip(base, out_name, include)
    download_file(p)
    return p


def import_zips(dest: str) -> List[str]:
    """Open a file picker (Colab), upload one or more ``.zip``s (e.g. previous
    parts' outputs), and extract them under ``dest``. Non-zip files are dropped in
    as-is. Returns the list of uploaded names. No-op (with a hint) off Colab."""
    os.makedirs(dest, exist_ok=True)
    if not in_colab():
        print(f"[io] not on Colab; place prior outputs under {dest} and continue")
        return []
    from google.colab import files  # type: ignore

    uploaded = files.upload()
    names = []
    for name, data in uploaded.items():
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                z.extractall(dest)
        else:
            with open(os.path.join(dest, name), "wb") as f:
                f.write(data)
        names.append(name)
    print(f"[io] imported {names} -> {dest}")
    return names

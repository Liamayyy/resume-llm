#!/usr/bin/env python3
"""
Convert all .tex files in build/latex/ → build/docx/ using pypandoc
-------------------------------------------------------------------
Requires:  pandoc  &  pypandoc  (pip install pypandoc pandocfilters)
"""
import os
from pathlib import Path
import pypandoc
import subprocess
import shutil

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "build" / "latex"
DST = ROOT / "build" / "docx"
DST.mkdir(parents=True, exist_ok=True)

def ensure_pandoc():
    """If pypandoc can't find pandoc, download the standalone binary."""
    try:
        pypandoc.get_pandoc_version()
    except OSError:
        print("⚠️  pandoc not found – downloading (~30 MB)…")
        pypandoc.download_pandoc()

def main():
    ensure_pandoc()
    for tex in SRC.glob("*.tex"):
        out_docx = DST / (tex.stem + ".docx")
        print(f"🔄 {tex.name} → {out_docx.name}")
        try:
            pypandoc.convert_file(
                str(tex), to="docx", format="latex", outputfile=str(out_docx)
            )
        except RuntimeError as e:
            print(f"   Pandoc failed: {e}")
    print("✅ Conversion complete.")

if __name__ == "__main__":
    main()

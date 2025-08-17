#!/usr/bin/env python3
"""
pdf2png_batch.py
Convert every PDF in a given folder to PNG images (one PNG per page)
while preserving high quality.

Usage
-----
python pdf2png_batch.py input_folder  [--out output_folder]  [--dpi 300]

Dependencies
------------
pip install pdf2image pillow tqdm
Poppler must be installed and its binaries added to PATH.
- Windows:  https://blog.alivate.com.au/poppler-windows/
- macOS:    brew install poppler
- Linux:    sudo apt-get install poppler-utils
"""

import argparse
import os
from pathlib import Path
from pdf2image import convert_from_path
from tqdm import tqdm

def pdf_to_png(pdf_path: Path, out_dir: Path, dpi: int = 300) -> None:
    """Convert one PDF into PNG files (one per page)."""
    pages = convert_from_path(pdf_path, dpi=dpi)
    stem = pdf_path.stem
    for idx, page in enumerate(pages, start=1):
        out_file = out_dir / f"{stem}_page{idx}.png"
        page.save(out_file, "PNG")

def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-convert PDFs to PNGs.")
    parser.add_argument("input_folder", help="Folder containing PDF files")
    parser.add_argument(
        "--out", "-o", default=None,
        help="Output folder (default: <input_folder>_png)"
    )
    parser.add_argument(
        "--dpi", "-d", type=int, default=300,
        help="Rendering DPI for quality (default: 300)"
    )
    args = parser.parse_args()

    in_dir = Path(args.input_folder).expanduser().resolve()
    if not in_dir.is_dir():
        parser.error(f"Input folder {in_dir} does not exist or is not a directory.")

    out_dir = Path(args.out) if args.out else in_dir.with_name(f"{in_dir.name}_png")
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted([f for f in in_dir.iterdir() if f.suffix.lower() == ".pdf"])
    if not pdf_files:
        print("No PDF files found in the input folder.")
        return

    print(f"Converting {len(pdf_files)} PDFs from {in_dir} → {out_dir} (dpi={args.dpi})")
    for pdf in tqdm(pdf_files, unit="pdf"):
        pdf_to_png(pdf, out_dir, dpi=args.dpi)

    print("Done.")

if __name__ == "__main__":
    main()

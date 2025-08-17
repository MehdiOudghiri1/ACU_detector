#!/usr/bin/env python3
"""
main_v2.py (silent + JSON output)

- Runs all extractors (NO white band overlay).
- Builds JSON payload in-memory.
- Shows the annotated image window.
- Prints ONLY the JSON payload and saves it to output.json.
"""

import os, sys, json, random, argparse
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pdfplumber

from find_numbers_v3 import BottomExtractor, SectionLinesExtractor, DimensionExtractor
from find_numbers import SCALE, DPI
from components_extractor import ComponentObjectsExtractor

# -------- helpers --------
def parse_number(txt: str):
    if txt is None:
        return None
    t = txt.strip().replace(",", "")
    if t == "":
        return None
    try:
        return float(t)
    except ValueError:
        return None

def swap_numeric_text_to_number(txt: str):
    if txt is None:
        return None
    s = txt.strip().replace(",", "")
    if s == "":
        return None
    flipped = s[::-1]
    try:
        return float(flipped)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None

def swap_all_dimensions(dims_dict):
    out = {}
    for name, tup in dims_dict.items():
        txt = tup[4] if isinstance(tup, (list, tuple)) and len(tup) >= 5 else None
        out[name] = swap_numeric_text_to_number(txt)
    return out

def _dedup_sorted(xs, min_delta=2):
    xs = sorted(int(x) for x in xs)
    out = []
    for x in xs:
        if not out or abs(x - out[-1]) >= min_delta:
            out.append(x)
    return out

def _spans_from_section_lines_px(section_lines_px):
    xs = _dedup_sorted([x for (x, _y0, _y1) in section_lines_px])
    return [(xs[i], xs[i+1]) for i in range(len(xs)-1)] if len(xs) >= 2 else []

def _px_span_to_points(span_px):
    x0_px, x1_px = span_px
    return (x0_px / SCALE, x1_px / SCALE)

def _bottom_center_x_pt(bottom_bbox_pt):
    x0_pt, _t, x1_pt, _b = bottom_bbox_pt
    return 0.5 * (x0_pt + x1_pt)

def build_json_payload(pdf_name, bottom_pairs, section_lines_px, dims_dict, section_report):
    bottom_vals = []
    for bbox_pt, txt in bottom_pairs:
        v = parse_number(txt)
        if v is not None:
            bottom_vals.append(v)
    unit_length = float(sum(bottom_vals)) if bottom_vals else 0.0

    spans_px = _spans_from_section_lines_px(section_lines_px)
    section_qty = len(spans_px)
    dims_swapped = swap_all_dimensions(dims_dict)

    spans_pt = [_px_span_to_points(s) for s in spans_px]
    bottoms_centers = []
    for bbox_pt, txt in bottom_pairs:
        cx_pt = _bottom_center_x_pt(bbox_pt)
        val = parse_number(txt)
        if val is not None:
            bottoms_centers.append((cx_pt, val))

    per_section_lengths = []
    for idx, (x0_pt, x1_pt) in enumerate(spans_pt, start=1):
        length_sum = sum(val for cx_pt, val in bottoms_centers if x0_pt <= cx_pt < x1_pt)
        per_section_lengths.append((idx, length_sum))

    section_length_items = []
    report_by_idx = {s["index"]: s for s in (section_report or [])}
    for idx, length_val in per_section_lengths:
        sec = report_by_idx.get(idx - 1)
        if sec and (sec.get("has_ec_fan_array") or sec.get("has_vertical_array")):
            item = {
                "Index": idx,
                "Length": float(length_val),
                "Label": "EC Fans",
                "ECM": {
                    "Mounting location": "n/a",
                    "Backdraft dampers": "Yes" if sec.get("has_backdraft_damper") else "No",
                    "Vertically mounted": "Yes" if sec.get("has_vertical_array") else "No",
                },
            }
        else:
            item = {
                "Index": idx,
                "Length": float(length_val),
                "Label": "Access",
            }
        section_length_items.append(item)

    return {
        "Unit Tag": os.path.splitext(pdf_name)[0],
        "Unit Properties": {
            "Indoor/Outdoor": "Unknown",
            "Unit size": {
                "Unit Length": unit_length,
                "Width (with base)":    dims_swapped.get("base"),
                "Height (base only)":   dims_swapped.get("base_height"),
                "Cabinet height":       dims_swapped.get("cabinet_height"),
                "Cabinet width":        dims_swapped.get("cabinet_width"),
                "Section quantity":     section_qty,
                "Section length":       section_length_items,
            },
        },
    }

def draw_full_height_verticals(img_obj, page, section_lines_px):
    for x_px, _y0, _y1 in section_lines_px:
        x0_pt = x_px / SCALE
        x1_pt = (x_px + 1) / SCALE
        img_obj.draw_rect((x0_pt, 0, x1_pt, page.height), stroke="red", stroke_width=2)

# -------- pipeline --------
comp_extractor = ComponentObjectsExtractor(weights=None, conf=0.25, debug=False, debug_plot=True)
EXTRACTORS = [
    BottomExtractor(),
    SectionLinesExtractor(),
    DimensionExtractor(),
    comp_extractor,
]

def render_page_and_make_json(page):
    img = page.to_image(resolution=DPI)
    results = {}
    for ext in EXTRACTORS:
        data = ext.extract(page)
        ext.plot(img, data)
        results[ext.__class__.__name__] = data

    bottom_pairs     = results.get("BottomExtractor", [])
    section_lines_px = results.get("SectionLinesExtractor", [])
    dims_dict        = results.get("DimensionExtractor", {})

    draw_full_height_verticals(img, page, section_lines_px)

    payload = build_json_payload(
        pdf_name=os.path.basename(page.pdf.stream.name),
        bottom_pairs=bottom_pairs,
        section_lines_px=section_lines_px,
        dims_dict=dims_dict,
        section_report=comp_extractor.section_report,
    )

    return img, payload, results

# -------- CLI --------
def main():
    parser = argparse.ArgumentParser(description="Run extractors, output JSON, save to file.")
    parser.add_argument("folder", help="Folder containing PDF files")
    parser.add_argument("--file", dest="specific_file", metavar="FILE.pdf",
                        help="If set, only this PDF (inside the folder) will be processed")
    args = parser.parse_args()

    all_pdfs = [f for f in os.listdir(args.folder) if f.lower().endswith(".pdf")]
    if not all_pdfs:
        raise SystemExit(1)

    if args.specific_file:
        if args.specific_file not in all_pdfs:
            raise SystemExit(1)
        picked = args.specific_file
    else:
        picked = random.choice(all_pdfs)

    path = os.path.join(args.folder, picked)
    with pdfplumber.open(path) as pdf:
        page = random.choice(pdf.pages)
        img, payload, _ = render_page_and_make_json(page)

    # Print JSON to console
    print(json.dumps(payload, indent=2))

    # Save JSON to output.json
    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Show annotated image
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img.annotated)
    ax.axis("off")
    plt.show()

if __name__ == "__main__":
    main()

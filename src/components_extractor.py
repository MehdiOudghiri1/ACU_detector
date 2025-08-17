# components_extractor.py
from __future__ import annotations
import glob, math
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    from find_numbers_v3 import DataExtractor, SectionLinesExtractor, WhiteBandExtractor
except ImportError:
    from abc import ABC, abstractmethod
    class DataExtractor(ABC):
        @abstractmethod
        def extract(self, page): ...
        @abstractmethod
        def plot(self, img, data): ...
    class SectionLinesExtractor:
        def extract(self, page):
            return []
    class WhiteBandExtractor:
        def extract(self, page):
            return None

from find_numbers import DPI, SCALE

try:
    from ultralytics import YOLO
except ImportError as e:
    raise SystemExit("Ultralytics not installed. `pip install ultralytics`") from e


# ------------------------ config ------------------------

TARGET_CLASSES = [
    "Control pannel",
    "EC fan array",
    "Vertical array",
    "backdraft dumper",
    "wirless pannel",
]

CONF_EC_VERTICAL = 0.68   # EC fan array & Vertical array
CONF_CONTROL     = 0.50   # Control pannel (wired)
CONF_WIRELESS    = 0.30   # wirless pannel

ARRAY_NAMES_LOWER      = {"ec fan array", "vertical array"}
CONTROL_WIRED_LOWER    = "control pannel"
CONTROL_WIRELESS_LOWER = "wirless pannel"
BACKDRAFT_LOWER        = "backdraft dumper"

SECTION_PALETTE = [
    (0,114,178),(213,94,0),(0,158,115),(204,121,167),
    (86,180,233),(230,159,0),(240,228,66),(0,0,0),
]
NEUTRAL_COLOR = (160,160,160)


# ------------------------ helpers ------------------------

def _auto_latest_weights(root: Path) -> str:
    pts = sorted(
        glob.glob(str(root / "runs" / "**" / "weights" / "*.pt"), recursive=True),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    if not pts:
        raise FileNotFoundError("No .pt weights found under runs/**/weights/")
    return pts[0]


def _dedup_sorted(xs: List[int], min_delta: int = 2) -> List[int]:
    xs = sorted(xs)
    out = []
    for x in xs:
        if not out or abs(x - out[-1]) >= min_delta:
            out.append(x)
    return out


def _clamp_box(b, W, H):
    x1, y1, x2, y2 = b
    return [
        int(max(0, min(W-1, x1))),
        int(max(0, min(H-1, y1))),
        int(max(0, min(W-1, x2))),
        int(max(0, min(H-1, y2))),
    ]


def _center(b):
    x1, y1, x2, y2 = b
    return (0.5*(x1+x2), 0.5*(y1+y2))


def _dist(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1])


def _white_band_y_pixels_auto(page, wb, H: int) -> Tuple[int,int] | None:
    """
    Convert white band quad to pixel Y (top,bottom), auto-detecting units.
    If values look like PDF points (≈ page.height), convert via SCALE; else assume pixels.
    """
    if not wb:
        return None
    try:
        _x0, t_raw, _x1, b_raw = wb[:4]
        if max(t_raw, b_raw) <= page.height * 1.5:
            y_top = int(max(0, min(H, t_raw * SCALE)))
            y_bot = int(max(0, min(H, b_raw * SCALE)))
        else:
            y_top = int(max(0, min(H, t_raw)))
            y_bot = int(max(0, min(H, b_raw)))
        if y_top > y_bot:
            y_top, y_bot = y_bot, y_top
        return (y_top, y_bot)
    except Exception:
        return None


# -------------------- extractor class --------------------

class ComponentObjectsExtractor(DataExtractor):
    """
    Returns: { class_name: [[x1,y1,x2,y2], ...], ... } (pixel coords)

    Side outputs (no printing):
      - self.section_report: per-section flags (EC/Vertical/Backdraft + optional control info)
      - self._plot_groups: per-section color groups for plotting
    """

    def __init__(self, weights: str | None = None, conf: float = 0.35,
                 project_root: str | Path | None = None,
                 debug: bool = False, debug_plot: bool = False):
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]
        self.weights = weights or _auto_latest_weights(self.project_root)
        self.model = YOLO(self.weights)
        self.conf = conf
        self.model_names: Dict[int,str] = {int(k): v for k,v in self.model.names.items()}
        self._want = {c.lower(): c for c in TARGET_CLASSES}

        self.section_report: List[Dict[str,Any]] = []
        self._plot_groups: Dict[Any,Any] = {}
        self.debug = False
        self.debug_plot = bool(debug_plot)

    # Optional horizontal line drawer (only if debug_plot=True; still no prints)
    def _draw_hline(self, img, y_px, color, width=1):
        if not self.debug_plot:
            return
        img.draw_line(((0, y_px / SCALE), (img.page.width, y_px / SCALE)),
                      stroke=color, stroke_width=width)

    def extract(self, page) -> Dict[str, List[List[int]]]:
        # Rasterize
        img_obj = page.to_image(resolution=DPI)
        pil_img = img_obj.original.convert("RGB")
        arr = np.array(pil_img)
        W, H = pil_img.size

        # Sections
        xs_raw = SectionLinesExtractor().extract(page)  # [(x_px,y0,y1),...]
        xs = _dedup_sorted([int(x) for (x,_y0,_y1) in xs_raw])
        spans: List[Tuple[int,int]] = [(xs[i], xs[i+1]) for i in range(len(xs)-1)] if len(xs) >= 2 else []

        # White band (pixels)
        whiteband = WhiteBandExtractor().extract(page)
        wb_y = _white_band_y_pixels_auto(page, whiteband, H)
        y_top_band = wb_y[0] if wb_y else None

        # Safety margin to avoid “touching band” leakage
        band_margin = max(4, int(round(0.004 * H)))  # ~0.4% of height (>=4 px)
        cutoff = (y_top_band - band_margin) if y_top_band is not None else None

        # YOLO
        _ = self.model.predict(np.zeros((32,32,3), dtype=np.uint8), imgsz=32, conf=0.01, verbose=False)
        res = self.model.predict(arr, imgsz=960, conf=min(self.conf, 0.59), verbose=False)[0]

        out: Dict[str,List[List[int]]] = {name: [] for name in TARGET_CLASSES}
        arrays: List[Dict[str,Any]] = []
        wired_panels: List[Dict[str,Any]] = []
        wireless_panels: List[Dict[str,Any]] = []
        backdrafts: List[Dict[str,Any]] = []

        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for (x1,y1,x2,y2), cidx, cf in zip(xyxy, clss, confs):
                name = self.model_names.get(int(cidx), str(cidx))
                name_lower = name.lower()
                if name_lower not in self._want:
                    continue

                # thresholds
                if name_lower in ARRAY_NAMES_LOWER:
                    if cf < CONF_EC_VERTICAL: 
                        continue
                elif name_lower == CONTROL_WIRED_LOWER:
                    if cf < CONF_CONTROL: 
                        continue
                elif name_lower == CONTROL_WIRELESS_LOWER:
                    if cf < CONF_WIRELESS: 
                        continue
                else:
                    # backdraft or others → use default object conf
                    if cf < self.conf: 
                        continue

                box = _clamp_box((x1,y1,x2,y2), W, H)
                canonical = self._want[name_lower]
                out[canonical].append(box)

                if name_lower in ARRAY_NAMES_LOWER:
                    arrays.append({"label": name_lower, "bbox": box, "conf": float(cf)})
                elif name_lower == CONTROL_WIRED_LOWER:
                    wired_panels.append({"label": name_lower, "bbox": box, "conf": float(cf)})
                elif name_lower == CONTROL_WIRELESS_LOWER:
                    wireless_panels.append({"label": name_lower, "bbox": box, "conf": float(cf)})
                elif name_lower == BACKDRAFT_LOWER:
                    backdrafts.append({"label": name_lower, "bbox": box, "conf": float(cf)})

        # Wired panels allowed (strictly above band with margin)
        wired_above: List[Dict[str,Any]] = []
        if cutoff is not None:
            for p in wired_panels:
                _cx, cy = _center(p["bbox"])
                if cy <= cutoff:
                    wired_above.append(p)

        # helpers
        def arrays_in_section(x0,x1):
            here=[]
            for a in arrays:
                cx,_ = _center(a["bbox"])
                if x0 <= cx < x1:
                    here.append(a)
            return here

        def backdrafts_in_section(x0,x1):
            here=[]
            for b in backdrafts:
                cx,_ = _center(b["bbox"])
                if x0 <= cx < x1:
                    here.append(b)
            return here

        def section_center(x0,x1):
            return (0.5*(x0+x1), 0.5*H)

        # build sections
        groups: Dict[Any,Any] = {}
        sections: List[Dict[str,Any]] = []

        for i,(x0,x1) in enumerate(spans):
            color = SECTION_PALETTE[i % len(SECTION_PALETTE)]
            arr_here = arrays_in_section(x0,x1)
            bkd_here = backdrafts_in_section(x0,x1)

            # ABOVE-BAND arrays: ENTIRE bbox above band top - margin
            arr_above: List[Dict[str,Any]] = []
            if cutoff is not None:
                for a in arr_here:
                    _x1a,y1a,_x2a,y2a = a["bbox"]
                    if y2a <= cutoff:
                        arr_above.append(a)

            # Envelope from ABOVE-BAND arrays ONLY
            env_top = min((b["bbox"][1] for b in arr_above), default=None)  # min y1
            env_bot = max((b["bbox"][3] for b in arr_above), default=None)  # max y2

            # mark if this section has backdrafts AND has arrays above band
            has_backdraft_damper = (len(arr_above) >= 1 and len(bkd_here) >= 1)

            sec = {
                "index": i, "x0": int(x0), "x1": int(x1),
                "center": [int(0.5*(x0+x1)), int(0.5*H)],
                "arrays_all": arr_here,
                "arrays_above": arr_above,
                "backdrafts": bkd_here,
                "env_top": env_top,
                "env_bot": env_bot,
                "has_backdraft_damper": has_backdraft_damper,
                "assigned": None,
            }
            sections.append(sec)
            groups[i] = {"color": color, "arrays": [a["bbox"] for a in arr_here], "control": None}

        # Eligible sections must have >=1 array above band
        eligible = {s["index"] for s in sections if len(s["arrays_above"]) >= 1}

        # 1) Assign wireless first (nearest section center by Euclidean)
        for w in wireless_panels:
            wc = _center(w["bbox"])
            best, best_d = None, float("inf")
            for s in sections:
                if s["index"] not in eligible or s["assigned"] is not None: 
                    continue
                d = _dist(wc, s["center"])
                if d < best_d: 
                    best_d, best = d, s
            if best:
                best["assigned"] = {
                    "type": "wireless",
                    "bbox": [int(v) for v in w["bbox"]],
                    "center": [int(wc[0]), int(wc[1])],
                    "conf": float(w.get("conf",0.0)),
                    "label": "remote mounted control pannel",
                }
                groups[best["index"]]["control"] = best["assigned"]["bbox"]

        # 2) Assign wired above-band (nearest section center)
        for p in wired_above:
            pc = _center(p["bbox"])
            best, best_d = None, float("inf")
            for s in sections:
                if s["index"] not in eligible or s["assigned"] is not None: 
                    continue
                d = _dist(pc, s["center"])
                if d < best_d: 
                    best_d, best = d, s
            if best:
                best["assigned"] = {
                    "type": "wired",
                    "bbox": [int(v) for v in p["bbox"]],
                    "center": [int(pc[0]), int(pc[1])],
                    "conf": float(p.get("conf",0.0)),
                    "label": None,
                }
                groups[best["index"]]["control"] = best["assigned"]["bbox"]

        # Singleton rule: exactly one array above band + no control ⇒ drop arrays
        for s in sections:
            if s["assigned"] is None and len(s["arrays_above"]) == 1:
                s["arrays_above"].clear()
                s["env_top"] = None
                s["env_bot"] = None
                s["has_backdraft_damper"] = False

        # Mounting decision for WIRED using the envelope (env_top/env_bot) of ABOVE-BAND arrays
        section_report: List[Dict[str,Any]] = []
        for s in sections:
            a = s["assigned"]
            label = None
            if a and a["type"] == "wired":
                env_top, env_bot = s["env_top"], s["env_bot"]
                if env_top is None or env_bot is None:
                    label = "end mounted control pannel"
                else:
                    _ccx, ccy = a["center"]
                    if ccy < env_top:   
                        label = "left mounted control pannel"
                    elif ccy > env_bot: 
                        label = "right mounted control pannel"
                    else:               
                        label = "end mounted control pannel"
                a["label"] = label

            section_report.append({
                "index": s["index"], "x0": s["x0"], "x1": s["x1"],
                "has_ec_fan_array": any(x["label"]=="ec fan array" for x in s["arrays_above"]),
                "has_vertical_array": any(x["label"]=="vertical array" for x in s["arrays_above"]),
                "has_backdraft_damper": bool(s.get("has_backdraft_damper", False)),
                "closest_control": (None if a is None else {
                    "label": a["label"], "type": a["type"], "center": a["center"],
                    "bbox": a["bbox"], "conf": a["conf"],
                }),
                "_debug": {
                    "white_band_top": y_top_band,
                    "band_margin": band_margin,
                    "cutoff": cutoff,
                    "arrays_all_boxes": [x["bbox"] for x in s["arrays_all"]],
                    "arrays_above_boxes": [x["bbox"] for x in s["arrays_above"]],
                    "backdraft_boxes": [x["bbox"] for x in s.get("backdrafts", [])],
                    "env_top": s["env_top"], "env_bot": s["env_bot"],
                }
            })

        # neutrals (arrays outside spans drawn gray; unassigned panels not drawn)
        neutrals_arrays = []
        for a in arrays:
            cx,_ = _center(a["bbox"])
            if not any(s["x0"] <= cx < s["x1"] for s in sections):
                neutrals_arrays.append(a["bbox"])
        groups["_neutral"] = {"arrays": neutrals_arrays, "controls": []}

        self.section_report = section_report
        self._plot_groups = groups
        return out

    def plot(self, img, data: Dict[str, List[List[int]]]) -> None:
        # per-section colored groups (silent)
        for idx, g in self._plot_groups.items():
            if idx == "_neutral":
                continue
            color = g["color"]
            # arrays
            for x1,y1,x2,y2 in g["arrays"]:
                img.draw_rect((x1/SCALE, y1/SCALE, x2/SCALE, y2/SCALE), stroke=color, stroke_width=2)
            # assigned control
            if g["control"] is not None:
                x1,y1,x2,y2 = g["control"]
                img.draw_rect((x1/SCALE, y1/SCALE, x2/SCALE, y2/SCALE), stroke=color, stroke_width=3)

            # optional envelope lines (no printing)
            if self.debug_plot and isinstance(idx, int):
                sec = next((s for s in self.section_report if s["index"] == idx), None)
                if sec:
                    env_top = sec["_debug"].get("env_top")
                    env_bot = sec["_debug"].get("env_bot")
                    if env_top is not None: self._draw_hline(img, int(env_top), color, 1)
                    if env_bot is not None: self._draw_hline(img, int(env_bot), color, 1)

        # neutrals
        if "_neutral" in self._plot_groups:
            for x1,y1,x2,y2 in self._plot_groups["_neutral"]["arrays"]:
                img.draw_rect((x1/SCALE, y1/SCALE, x2/SCALE, y2/SCALE), stroke=NEUTRAL_COLOR, stroke_width=1)

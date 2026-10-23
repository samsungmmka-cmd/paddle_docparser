"""Document parser: decides the cheapest good route for every page.

Routes
------
PDF page with a real text layer   -> read text/tables from the PDF (no OCR)
   + large embedded images          -> OCR just those image regions
   + borderless tables              -> structure model on just those regions
Scanned page or image             -> orientation fix -> layout check
   + table/formula/seal found       -> PP-StructureV3 (only the needed modules)
   + plain text                     -> fast OCR, then re-read weak lines
                                       with the stronger (handwriting) model

Charts are reported as image blocks (with any OCR'd labels as text). Chart data
extraction (PP-Chart2Table) is deliberately not used: it is very slow and can hang.
"""
import threading
import time

import numpy as np
from PIL import Image

from .confidence import UNITS, cell_confidence, summarize, wmean
from .config import Settings
from .loaders import PdfFile, detect_file_type, load_image_pages, parse_page_range
from .utils import (clean_rows, html_table_to_rows, move_block, overlap, reading_order,
                    scale_bbox, shift_bbox, to_bbox, to_plain_text)

# Layout label -> block type in our output
BLOCK_TYPES = {
    "doc_title": "title", "paragraph_title": "heading",
    "figure_title": "caption", "table_title": "caption", "chart_title": "caption",
    "header": "header", "footer": "footer", "footnote": "footnote", "number": "page_number",
    "table": "table", "seal": "seal",
    "formula": "formula", "display_formula": "formula", "inline_formula": "formula",
    "image": "image", "figure": "image", "header_image": "image", "footer_image": "image",
    "chart": "image",  # no chart data extraction: a chart is reported as an image region
}
IMAGE_LABELS = {"image", "figure", "header_image", "footer_image", "chart"}
# Layout label -> PP-StructureV3 module to switch on
STRUCTURE_LABELS = {"table": "tables", "formula": "formulas",
                    "display_formula": "formulas", "seal": "seals"}
# Orientation label -> number of 90° counter-clockwise turns to make the page upright
ROTATE_TURNS = {0: 0, 90: 1, 180: 2, 270: 3}


class DocumentParser:
    def __init__(self, settings: Settings | None = None, models=None):
        self.s = settings or Settings.from_env()
        if models is None:
            from .models import PaddleModels
            models = PaddleModels(self.s)
        self.models = models
        self._lock = threading.Lock()  # Paddle predictors are not thread-safe

    # ================= public API =================

    def parse(self, data: bytes, filename: str | None = None, pages: str | None = None) -> dict:
        start = time.perf_counter()
        file_type = detect_file_type(data)
        with self._lock:
            if file_type == "pdf":
                page_count, results = self._parse_pdf(data, pages)
            else:
                images = load_image_pages(data)
                page_count = len(images)
                results = [self._scanned_page(images[i], i + 1)
                           for i in self._select(pages, page_count)]
        confidence = summarize(results, self.s.handwriting_threshold)
        return {
            "filename": filename,
            "file_type": file_type,
            "page_count": page_count,
            "confidence": confidence,
            "text": to_plain_text(results),
            "pages": results,
            "elapsed_ms": _ms(start),
        }

    # ================= PDF =================

    def _parse_pdf(self, data, pages):
        results = []
        with PdfFile(data) as pdf:
            page_count = pdf.page_count
            for i in self._select(pages, page_count):
                t0 = time.perf_counter()
                layer = pdf.text_layer(i)
                if self._has_usable_text(layer):
                    page = self._native_page(pdf, i, layer)
                else:
                    page = self._scanned_page(pdf.render(i, self._page_scale(layer)), i + 1)
                    page["source"] = "ocr"
                page["elapsed_ms"] = _ms(t0)
                results.append(page)
        return page_count, results

    def _has_usable_text(self, layer) -> bool:
        if layer["char_count"] < self.s.min_text_layer_chars:
            return False
        # A full-page image with text on top is a scan with a (often poor) OCR layer
        return layer["image_coverage"] < self.s.scanned_image_coverage

    def _page_scale(self, layer) -> float:
        """PDF points -> pixels. Lowered for oversized pages so a render never
        exceeds max_render_pixels (a tiny PDF can declare a huge page)."""
        area_pt = max(layer["width"] * layer["height"], 1.0)
        return min(self.s.render_dpi / 72, (self.s.max_render_pixels / area_pt) ** 0.5)

    def _native_page(self, pdf, index, layer) -> dict:
        """Digital PDF page: trust the text layer, OCR only what it cannot give us."""
        scale = self._page_scale(layer)  # PDF points -> pixels of the rendered page
        routes = ["text_layer"]
        blocks = []

        tables = [t for t in layer["tables"] if t["rows"]]
        for t in tables:
            rows = clean_rows(t["rows"])
            units = [(cell, 1.0, False) for row in rows for cell in row]  # text layer = exact
            blocks.append({"type": "table", "bbox": scale_bbox(t["bbox"], scale), "rows": rows,
                           "cell_confidence": [[1.0] * len(row) for row in rows],
                           "confidence": wmean(units), "source": "text_layer", UNITS: units})
        if tables:
            routes.append("pdf_tables")

        # Regions handled by OCR / structure (filled below); their text-layer lines are dropped
        replaced = [t["bbox"] for t in tables]
        page_img = None

        # 1) Large embedded images (scanned inserts, photos of receipts)
        page_area = layer["width"] * layer["height"]
        big_images = [b for b in layer["images"]
                      if (b[2] - b[0]) * (b[3] - b[1]) >= self.s.embedded_image_min_area * page_area]
        if big_images:
            page_img = pdf.render(index, scale)
            routes.append("embedded_image_ocr")
            for box in big_images:
                blocks += self._ocr_region(page_img, scale_bbox(box, scale))
                replaced.append(box)

        # 2) Borderless tables drawn as vector graphics: ask the layout model
        if self.s.enable_tables:
            if page_img is None:
                page_img = pdf.render(index, scale)
            for region in self._layout(page_img):
                if region["label"] != "table":
                    continue
                box_px = region["bbox"]
                box_pt = scale_bbox(box_px, 1 / scale)
                if any(overlap(box_pt, r) > 0.5 for r in replaced):
                    continue  # already covered by a PDF table or an embedded image
                found = self._structure_region(page_img, box_px, {"tables"})
                if found:
                    blocks += found
                    replaced.append(box_pt)
                    routes.append("layout_table")

        # 3) Remaining text-layer lines, grouped into paragraphs
        lines = [{"text": ln["text"], "bbox": scale_bbox(ln["bbox"], scale), "confidence": 1.0}
                 for ln in layer["lines"]
                 if not any(overlap(ln["bbox"], r) > 0.5 for r in replaced)]
        blocks += [_text_block(group, "text", "text_layer") for group in _paragraphs(lines)]

        blocks.sort(key=reading_order)
        return {
            "page": index + 1,
            "width": round(layer["width"] * scale), "height": round(layer["height"] * scale),
            "source": "text_layer", "routes": _unique(routes),
            "orientation": 0, "handwriting_likely": False, "blocks": blocks,
        }

    # ================= scanned pages / images =================

    def _scanned_page(self, img, page_number) -> dict:
        t0 = time.perf_counter()
        blocks, info = self._process_image(img)
        return {
            "page": page_number, "width": info["width"], "height": info["height"],
            "source": "ocr", "routes": info["routes"], "orientation": info["orientation"],
            "handwriting_likely": info["handwriting"], "blocks": blocks,
            "elapsed_ms": _ms(t0),
        }

    def _process_image(self, img):
        """Route one image. Returned boxes are in the upright image at its original size."""
        img, factor = _limit_size(img, self.s.max_image_side)

        angle = 0
        if self.s.enable_orientation:
            angle = self.models.detect_orientation(img)
            if angle:
                img = np.ascontiguousarray(np.rot90(img, ROTATE_TURNS.get(angle, 0)))

        layout = self._layout(img)
        needed = {STRUCTURE_LABELS[r["label"]] for r in layout if r["label"] in STRUCTURE_LABELS}
        needed = {m for m in needed if getattr(self.s, f"enable_{m}")}
        routes = ["orientation"] if angle else []

        if needed:
            blocks = self._structure(img, needed)
            routes.append("structure:" + "+".join(sorted(needed)))
            handwriting = False
        else:
            blocks, handwriting = self._text_ocr(img, layout)
            routes.append("ocr")
            if handwriting:
                routes.append("handwriting")

        if factor != 1.0:
            blocks = [move_block(b, lambda bb: scale_bbox(bb, 1 / factor)) for b in blocks]
        h, w = img.shape[:2]
        info = {"width": round(w / factor), "height": round(h / factor), "orientation": angle,
                "routes": routes, "handwriting": handwriting}
        return blocks, info

    def _layout(self, img) -> list:
        return [r for r in self.models.detect_layout(img) if r["score"] >= self.s.layout_threshold]

    def _text_ocr(self, img, layout):
        """Fast OCR, then re-read only the low-confidence lines with the stronger model."""
        lines = self.models.run_ocr(img)
        weak = [i for i, ln in enumerate(lines) if ln["confidence"] < self.s.handwriting_threshold]

        crops, idx = [], []
        h, w = img.shape[:2]
        for i in weak:
            x1, y1, x2, y2 = lines[i]["bbox"]
            x1, y1 = max(0, int(x1) - 2), max(0, int(y1) - 2)
            x2, y2 = min(w, int(x2) + 2), min(h, int(y2) + 2)
            if x2 - x1 >= 4 and y2 - y1 >= 4:
                crops.append(img[y1:y2, x1:x2])
                idx.append(i)
        if crops:
            for i, (text, score) in zip(idx, self.models.recognize(crops)):
                if score > lines[i]["confidence"]:
                    lines[i].update(text=text, confidence=round(score, 4), escalated=True)

        handwriting = bool(lines) and len(weak) / len(lines) >= self.s.handwriting_page_ratio
        return _group_by_layout(lines, layout), handwriting

    def _structure(self, img, needed: set) -> list:
        raw = self.models.run_structure(img, **{m: True for m in needed})
        return structure_to_blocks(raw)

    # ---------- region helpers (used for native PDF pages) ----------

    def _crop(self, img, box):
        h, w = img.shape[:2]
        x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
        x2, y2 = min(w, int(box[2])), min(h, int(box[3]))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None, 0, 0
        return np.ascontiguousarray(img[y1:y2, x1:x2]), x1, y1

    def _ocr_region(self, page_img, box) -> list:
        crop, dx, dy = self._crop(page_img, box)
        if crop is None:
            return []
        blocks, info = self._process_image(crop)
        if info["orientation"]:
            # Rotated insert: coordinates are in its own frame, so report the whole region
            text = "\n".join(b.get("text") or "" for b in blocks).strip()
            units = [u for b in blocks for u in b.get(UNITS, [])]
            return [{"type": "image", "bbox": list(box), "text": text or None, "confidence": wmean(units),
                     "source": "embedded_image_ocr", "orientation": info["orientation"], UNITS: units}]
        out = []
        for b in blocks:
            b["source"] = "embedded_image_ocr"
            out.append(move_block(b, lambda bb: shift_bbox(bb, dx, dy)))
        return out

    def _structure_region(self, page_img, box, needed) -> list:
        crop, dx, dy = self._crop(page_img, box)
        if crop is None:
            return []
        blocks = [b for b in self._structure(crop, needed) if b["type"] == "table"]
        return [move_block(b, lambda bb: shift_bbox(bb, dx, dy)) for b in blocks]

    def _select(self, pages, total):
        return parse_page_range(pages, total, max_pages=self.s.max_pages)


# ================= output helpers =================

def structure_to_blocks(raw: dict) -> list:
    """Convert a PP-StructureV3 result into our block format, with confidence scores.

    The structure blocks carry text but no scores, so every OCR line found on the
    page (`overall_ocr_res`) is given to the block it sits in, and the block is
    scored from those lines.
    """
    ocr = raw.get("overall_ocr_res") or {}
    boxes = ocr.get("rec_boxes")
    if boxes is None or len(boxes) == 0:
        boxes = ocr.get("rec_polys", [])
    lines = [{"text": t, "confidence": round(float(s), 4), "bbox": [round(v, 1) for v in to_bbox(b)]}
             for t, s, b in zip(ocr.get("rec_texts", []), ocr.get("rec_scores", []), boxes)]

    blocks = []
    table_html = [t.get("pred_html", "") for t in raw.get("table_res_list") or []]
    for item in raw.get("parsing_res_list") or []:
        label = item.get("block_label") or item.get("label") or "text"
        content = item.get("block_content") or item.get("content") or ""
        kind = BLOCK_TYPES.get(label, "text")
        block = {"type": kind, "label": label,
                 "bbox": [round(v, 1) for v in to_bbox(item.get("block_bbox") or item.get("bbox"))],
                 "source": "structure"}
        if kind == "table":
            fallback = table_html.pop(0) if table_html else ""
            html = content if "<table" in content.lower() else fallback
            block.update(html=html or None, rows=html_table_to_rows(html))
        elif kind == "formula":
            block.update(latex=content)
        else:
            block["text"] = content or None
        blocks.append(block)

    if not blocks:  # no layout result: fall back to the plain OCR lines
        blocks = [_text_block(g, "text", "structure") for g in _paragraphs(lines)]
        blocks.sort(key=reading_order)
        return blocks

    owned = [[] for _ in blocks]
    for ln in lines:
        k = _owner(ln["bbox"], blocks)
        if k is not None:
            owned[k].append(ln)
    for block, own in zip(blocks, owned):
        _score_block(block, own)
    blocks.sort(key=reading_order)
    return blocks


def _owner(line_box, blocks):
    """Index of the block a line belongs to: text/table blocks first, images last."""
    for wanted_image in (False, True):
        best, best_score = None, 0.0
        for k, b in enumerate(blocks):
            if (b["type"] == "image") != wanted_image:
                continue
            score = overlap(line_box, b["bbox"])
            if score > best_score:
                best, best_score = k, score
        if best_score >= 0.5:
            return best
    return None


def _score_block(block: dict, lines: list):
    """Set confidence (and cell scores for tables) from the OCR lines a block owns."""
    lines = sorted(lines, key=lambda l: (round(l["bbox"][1] / 10), l["bbox"][0]))
    block[UNITS] = [(l["text"], l["confidence"], True) for l in lines]
    block["confidence"] = wmean(block[UNITS])
    block["lines"] = lines
    if block["type"] == "table":
        block["cell_confidence"] = cell_confidence(block["rows"], lines)
    elif block["type"] == "image" and not block.get("text") and lines:
        block["text"] = "\n".join(l["text"] for l in lines)


def _group_by_layout(lines: list, layout: list) -> list:
    """Put OCR lines into the text regions found by the layout model."""
    regions = [r for r in layout if r["label"] not in IMAGE_LABELS and r["label"] not in STRUCTURE_LABELS]
    images = [r for r in layout if r["label"] in IMAGE_LABELS]
    buckets = [[] for _ in regions]
    image_buckets = [[] for _ in images]
    loose = []
    for ln in lines:
        best = _best_region(ln["bbox"], regions)
        if best is not None:
            buckets[best].append(ln)
            continue
        best = _best_region(ln["bbox"], images)  # e.g. labels printed on a chart
        if best is not None:
            image_buckets[best].append(ln)
        else:
            loose.append(ln)

    blocks = []
    for region, group in zip(regions, buckets):
        if group:
            block = _text_block(group, BLOCK_TYPES.get(region["label"], "text"), "ocr")
            block["label"] = region["label"]
            block["bbox"] = [round(v, 1) for v in region["bbox"]]
            blocks.append(block)
    blocks += [_text_block(g, "text", "ocr") for g in _paragraphs(loose)]
    for region, group in zip(images, image_buckets):
        block = {"type": "image", "label": region["label"],
                 "bbox": [round(v, 1) for v in region["bbox"]], "text": None, "source": "ocr"}
        _score_block(block, group)  # text + confidence of any text inside the image, else None
        blocks.append(block)
    blocks.sort(key=reading_order)
    return blocks


def _best_region(box, regions):
    scores = [overlap(box, r["bbox"]) for r in regions]
    best = max(range(len(regions)), key=lambda k: scores[k], default=None)
    return best if best is not None and scores[best] >= 0.5 else None


def _paragraphs(lines: list) -> list:
    """Group lines that sit directly under each other into paragraphs."""
    groups = []
    for ln in sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0])):
        x1, y1, x2, y2 = ln["bbox"]
        height = max(1.0, y2 - y1)
        for g in groups:
            px1, _, px2, py2 = g[-1]["bbox"]
            if 0 <= y1 - py2 < 0.8 * height and min(x2, px2) > max(x1, px1):
                g.append(ln)
                break
        else:
            groups.append([ln])
    return groups


def _text_block(lines: list, kind: str, source: str) -> dict:
    lines = sorted(lines, key=lambda l: (round(l["bbox"][1] / 10), l["bbox"][0]))
    boxes = [l["bbox"] for l in lines]
    units = [(l["text"], l.get("confidence"), source != "text_layer") for l in lines]
    return {
        "type": kind,
        "bbox": [round(min(b[0] for b in boxes), 1), round(min(b[1] for b in boxes), 1),
                 round(max(b[2] for b in boxes), 1), round(max(b[3] for b in boxes), 1)],
        "text": "\n".join(l["text"] for l in lines),
        "confidence": wmean(units),
        "lines": lines,
        "source": source,
        UNITS: units,
    }


def _limit_size(img, max_side):
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img, 1.0
    factor = max_side / max(h, w)
    small = Image.fromarray(img[:, :, ::-1]).resize((round(w * factor), round(h * factor)), Image.LANCZOS)
    return np.ascontiguousarray(np.asarray(small)[:, :, ::-1]), factor


def _unique(items):
    return list(dict.fromkeys(items))


def _ms(start):
    return round((time.perf_counter() - start) * 1000, 1)

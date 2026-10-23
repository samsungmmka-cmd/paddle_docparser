"""Confidence scores for blocks, table cells, pages and the whole document.

Every block keeps a private list of "units" while the page is parsed: the pieces
of text it owns and their score, as (text, score, is_ocr). Text read from a PDF's
text layer is exact, so it scores 1.0 and is_ocr is False.

All averages are weighted by the number of characters, so one doubtful
character counts less than a long, clearly read line.
"""

UNITS = "_units"  # internal key, removed before the result is returned


def wmean(units) -> float | None:
    """Character-weighted mean of (text, score, ...) items; None when there is no text."""
    total = weight = 0.0
    for unit in units:
        text, score = unit[0], unit[1]
        n = len((text or "").strip())
        if n and score is not None:
            total += n * score
            weight += n
    return round(total / weight, 4) if weight else None


def cell_confidence(rows: list, lines: list) -> list:
    """Score each table cell by matching it to the OCR lines found inside the table.

    The table structure model returns cell text but no scores, so each cell is
    matched to the OCR line(s) that make up its text. A cell that is empty, or
    whose text cannot be matched to any line, gets None.
    """
    def norm(s):
        return "".join((s or "").split()).lower()

    pool = [(norm(ln["text"]), ln) for ln in lines]
    used = set()
    scores = [[None] * len(row) for row in rows]
    cells = [(r, c, norm(text)) for r, row in enumerate(rows) for c, text in enumerate(row) if norm(text)]

    # Pass 1: a line whose text equals the cell text (most cells are one line)
    for r, c, text in cells:
        for i, (line_text, ln) in enumerate(pool):
            if i not in used and line_text == text:
                used.add(i)
                scores[r][c] = round(ln["confidence"], 4)
                break

    # Pass 2: cells made of several lines (wrapped text): unused lines that are part of the cell
    for r, c, text in cells:
        if scores[r][c] is not None:
            continue
        parts, length = [], 0
        for i, (line_text, ln) in enumerate(pool):
            if i not in used and line_text and line_text in text and length + len(line_text) <= len(text):
                parts.append(i)
                length += len(line_text)
        if parts and length >= 0.5 * len(text):
            used.update(parts)
            scores[r][c] = wmean([(pool[i][1]["text"], pool[i][1]["confidence"]) for i in parts])
    return scores


def summarize(pages: list, low_threshold: float) -> dict:
    """Add `confidence` to each page, drop the internal units, return the document summary."""
    all_units = []
    for i, page in enumerate(pages):
        units = [u for b in page["blocks"] for u in b.pop(UNITS, [])]
        all_units += units
        # Rebuild the page so `confidence` sits just before `blocks`
        blocks, rest = page.pop("blocks"), page.pop("elapsed_ms", None)
        page["confidence"] = wmean(units)
        page["blocks"] = blocks
        if rest is not None:
            page["elapsed_ms"] = rest
        pages[i] = page

    ocr_units = [u for u in all_units if u[2]]
    overall, ocr = wmean(all_units), wmean(ocr_units)
    return {
        "overall": round(overall, 3) if overall is not None else None,
        "ocr": round(ocr, 3) if ocr is not None else None,
        "low_confidence_lines": sum(1 for u in ocr_units if (u[0] or "").strip() and u[1] < low_threshold),
        "handwriting_pages": [p["page"] for p in pages if p.get("handwriting_likely")],
    }

"""Small helpers: bounding boxes, table parsing, markdown output."""
from html.parser import HTMLParser


# ---------- bounding boxes: [x1, y1, x2, y2] ----------

def to_bbox(value) -> list:
    """Accept [x1,y1,x2,y2] or a polygon [[x,y], ...] and return [x1,y1,x2,y2]."""
    if value is None or len(value) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    if isinstance(value[0], (list, tuple)):
        xs = [p[0] for p in value]
        ys = [p[1] for p in value]
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    return [float(v) for v in value[:4]]


def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlap(a, b) -> float:
    """Fraction of box `a` that lies inside box `b`."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = area([x1, y1, x2, y2]) if x2 > x1 and y2 > y1 else 0.0
    a_area = area(a)
    return inter / a_area if a_area else 0.0


def scale_bbox(b, factor: float) -> list:
    return [round(v * factor, 1) for v in b]


def shift_bbox(b, dx: float, dy: float) -> list:
    return [round(b[0] + dx, 1), round(b[1] + dy, 1), round(b[2] + dx, 1), round(b[3] + dy, 1)]


def move_block(block: dict, fn) -> dict:
    """Apply a bbox transform to a block and its lines."""
    block["bbox"] = fn(block["bbox"])
    for line in block.get("lines") or []:
        line["bbox"] = fn(line["bbox"])
    return block


def reading_order(block: dict):
    # Group into ~15px bands top-to-bottom, then left-to-right
    return (round(block["bbox"][1] / 15), block["bbox"][0])


# ---------- tables ----------

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows, self._row, self._cell, self._colspan = [], None, None, 1

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []
            self._colspan = int(dict(attrs).get("colspan", 1) or 1)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._row.extend([""] * (self._colspan - 1))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def html_table_to_rows(html: str) -> list:
    parser = _TableParser()
    parser.feed(html or "")
    return parser.rows


def clean_rows(rows) -> list:
    return [[(c or "").replace("\n", " ").strip() for c in row] for row in rows or []]


# ---------- markdown ----------

def rows_to_markdown(rows) -> str:
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    esc = lambda c: str(c).replace("|", "\\|")
    lines = ["| " + " | ".join(esc(c) for c in rows[0]) + " |",
             "| " + " | ".join("---" for _ in range(width)) + " |"]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def to_plain_text(pages: list) -> str:
    """All text of the document in reading order, one string.
    Pages start with '=== Page N ==='; table rows are written as '| a | b |'."""
    out = []
    for page in pages:
        chunks = []
        for b in page["blocks"]:
            kind, text = b["type"], (b.get("text") or "").strip()
            if kind == "table" and b.get("rows"):
                chunks.append("\n".join("| " + " | ".join(row) + " |" for row in b["rows"]))
            elif kind == "image":
                chunks.append("[image]" + (f"\n{text}" if text else ""))
            elif kind == "formula":
                latex = (b.get("latex") or text).strip()
                if latex:
                    chunks.append(latex)
            elif text:
                chunks.append(text)
        out.append(f"=== Page {page['page']} ===\n" + "\n\n".join(chunks))
    return "\n\n".join(out)


def to_markdown(result: dict) -> str:
    out = []
    for page in result["pages"]:
        out.append(f"<!-- page {page['page']} -->")
        for b in page["blocks"]:
            kind, text = b["type"], (b.get("text") or "").strip()
            if kind == "title":
                out.append(f"# {text}")
            elif kind == "heading":
                out.append(f"## {text}")
            elif kind == "table" and b.get("rows"):
                out.append(rows_to_markdown(b["rows"]))
            elif kind == "formula":
                out.append(f"$$\n{b.get('latex') or text}\n$$")
            elif kind == "image":
                out.append("*[image]*" + (f"\n\n{text}" if text else ""))
            elif text:
                out.append(text)
    return "\n\n".join(out) + "\n"

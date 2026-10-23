"""Confidence scores and the full `text` property."""
import io
import json

import pytest
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from docparse.confidence import UNITS, cell_confidence, wmean


def _all_blocks(result):
    return [b for p in result["pages"] for b in p["blocks"]]


EXACT_TEXT = "INVOICE 2026-0042 Acme Trading LLC, Abu Dhabi, United Arab Emirates"


def _two_page_pdf():
    """Page 1: digital text. Page 2: one full-page image (scanned)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.drawString(72, 800, EXACT_TEXT)  # over 50 chars, so the page counts as digital
    c.showPage()
    img = io.BytesIO()
    Image.new("RGB", (1240, 1754), "white").save(img, "PNG")
    c.drawImage(ImageReader(io.BytesIO(img.getvalue())), 0, 0, *A4)
    c.showPage()
    c.save()
    return buf.getvalue()


# ---------- helpers ----------

def test_wmean_is_character_weighted():
    # 10 chars at 1.0 and 1 char at 0.0 -> 10/11, not the plain mean 0.5
    assert wmean([("abcdefghij", 1.0), ("x", 0.0)]) == round(10 / 11, 4)
    assert wmean([("", 0.3), ("  ", 0.1)]) is None
    assert wmean([]) is None


def test_cell_confidence_matches_cells_to_lines():
    rows = [["Item", "Qty"], ["Ink cartridge", "1"], ["Paper A4 sheets", ""]]
    lines = [{"text": "Item", "confidence": 0.99}, {"text": "Qty", "confidence": 0.98},
             {"text": "Ink cartridge", "confidence": 0.94}, {"text": "1", "confidence": 0.73},
             {"text": "Paper A4", "confidence": 0.90}, {"text": "sheets", "confidence": 0.80}]  # wrapped cell
    scores = cell_confidence(rows, lines)
    assert scores[0] == [0.99, 0.98]
    assert scores[1] == [0.94, 0.73]           # "1" not stolen by another cell
    assert scores[2][0] == round((8 * 0.90 + 6 * 0.80) / 14, 4)  # two lines, character-weighted
    assert scores[2][1] is None                 # empty cell


def test_cell_without_matching_line_is_null():
    assert cell_confidence([["Total"]], [{"text": "Grand", "confidence": 0.9}]) == [[None]]


# ---------- document output ----------

def test_document_confidence_matches_hand_calculation(parser, fake):
    fake.lines = [
        {"text": "DELIVERY NOTE", "confidence": 0.99, "bbox": [10, 10, 300, 40]},      # 13 chars
        {"text": "Call Ahmed", "confidence": 0.60, "bbox": [10, 60, 300, 90]},          # 10 chars, weak
    ]
    fake.recognized = ("Call Ahmed", 0.70)  # re-read improves it, still below 0.80
    result = parser.parse(_two_page_pdf())

    ocr = (13 * 0.99 + 10 * 0.70) / 23
    exact_chars = len(EXACT_TEXT)
    overall = (exact_chars * 1.0 + 13 * 0.99 + 10 * 0.70) / (exact_chars + 23)
    assert result["confidence"] == {
        "overall": round(overall, 3),
        "ocr": round(ocr, 3),
        "low_confidence_lines": 1,
        "handwriting_pages": [2],   # 2 of 2 lines were weak on first read
    }
    assert [p["confidence"] for p in result["pages"]] == [1.0, round(ocr, 4)]


def test_digital_only_document_has_no_ocr_score(parser, native_pdf):
    conf = parser.parse(native_pdf)["confidence"]
    assert conf == {"overall": 1.0, "ocr": None, "low_confidence_lines": 0, "handwriting_pages": []}


def test_every_block_has_confidence_including_tables(parser, native_pdf):
    result = parser.parse(native_pdf)
    for block in _all_blocks(result):
        assert "confidence" in block
    table = next(b for b in _all_blocks(result) if b["type"] == "table")
    assert table["confidence"] == 1.0
    assert [len(r) for r in table["cell_confidence"]] == [len(r) for r in table["rows"]]


def test_structure_table_gets_scores_from_ocr_lines(parser, fake, png_bytes):
    fake.layout = [{"label": "table", "score": 0.9, "bbox": [0, 0, 600, 400]}]
    fake.structure = {
        "parsing_res_list": [
            {"block_label": "paragraph_title", "block_bbox": [10, 10, 400, 40], "block_content": "DELIVERY NOTE"},
            {"block_label": "table", "block_bbox": [10, 60, 500, 300],
             "block_content": "<table><tr><td>Item</td><td>Qty</td></tr><tr><td>Ink cartridge</td><td>1</td></tr></table>"},
            {"block_label": "image", "block_bbox": [520, 10, 590, 60], "block_content": ""},
        ],
        "overall_ocr_res": {
            "rec_texts": ["DELIVERY NOTE", "Item", "Qty", "Ink cartridge", "1"],
            "rec_scores": [0.9872, 0.9935, 0.991, 0.9415, 0.7342],
            "rec_boxes": [[12, 12, 300, 38], [20, 70, 100, 100], [300, 70, 360, 100],
                          [20, 120, 260, 150], [300, 120, 320, 150]],
        },
    }
    result = parser.parse(png_bytes)
    blocks = {b["type"]: b for b in _all_blocks(result)}

    assert blocks["heading"]["confidence"] == 0.9872
    table = blocks["table"]
    assert table["cell_confidence"] == [[0.9935, 0.991], [0.9415, 0.7342]]
    assert table["confidence"] == wmean([("Item", 0.9935), ("Qty", 0.991), ("Ink cartridge", 0.9415), ("1", 0.7342)])
    assert blocks["image"]["confidence"] is None and blocks["image"]["text"] is None
    assert result["confidence"]["low_confidence_lines"] == 1     # the "1" cell
    assert "_units" not in json.dumps(result)                    # internal data never leaks


def test_full_text_property(parser, fake, native_pdf):
    text = parser.parse(native_pdf)["text"]
    assert text.startswith("=== Page 1 ===\nINVOICE 2026-0042")
    assert "| Item | Qty | Price |\n| Paper | 2 | 10.00 |\n| Ink | 1 | 25.00 |" in text


def test_full_text_covers_every_page_in_order(parser):
    text = parser.parse(_two_page_pdf())["text"]
    assert text.index("=== Page 1 ===") < text.index("INVOICE") < text.index("=== Page 2 ===") < text.index("TOTAL 42.00")


def test_blank_page_is_safe(parser, fake):
    fake.lines = []
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, "PNG")
    result = parser.parse(buf.getvalue())
    assert result["pages"][0]["confidence"] is None
    assert result["confidence"] == {"overall": None, "ocr": None, "low_confidence_lines": 0, "handwriting_pages": []}
    assert result["text"] == "=== Page 1 ===\n"


@pytest.mark.parametrize("fixture", ["native_pdf", "scanned_pdf", "png_bytes"])
def test_output_is_strict_json(parser, request, fixture):
    result = parser.parse(request.getfixturevalue(fixture))
    s = json.dumps(result, allow_nan=False, ensure_ascii=False)  # raises on NaN/Infinity/numpy types
    assert json.loads(s)["confidence"]["overall"] is not None
    assert UNITS not in s


def test_output_key_order(parser, native_pdf):
    result = parser.parse(native_pdf)
    assert list(result) == ["filename", "file_type", "page_count", "confidence", "text", "pages", "elapsed_ms"]
    assert list(result["pages"][0])[-3:] == ["confidence", "blocks", "elapsed_ms"]

"""Routing behaviour: each page should take the cheapest route that handles it."""
import io

import pytest
from PIL import Image

from docparse.loaders import (InvalidPageRangeError, TooManyPagesError, UnsupportedFileError,
                              detect_file_type, parse_page_range)


def test_detects_file_types(png_bytes, native_pdf):
    assert detect_file_type(png_bytes) == "png"
    assert detect_file_type(native_pdf) == "pdf"
    with pytest.raises(UnsupportedFileError):
        detect_file_type(b"PK\x03\x04 not a document")


def test_page_range():
    assert parse_page_range("1-3,5", 10) == [0, 1, 2, 4]
    assert parse_page_range(None, 3) == [0, 1, 2]
    assert parse_page_range("9-12", 10) == [8, 9]
    assert parse_page_range("50-60", 10) == []


def test_huge_page_range_is_clamped_before_expanding():
    # H1: used to take seconds and gigabytes; now clamped to the document size first
    import time
    t = time.perf_counter()
    assert parse_page_range("1-10000000000", 5, max_pages=100) == [0, 1, 2, 3, 4]
    assert time.perf_counter() - t < 0.05


def test_page_limit_is_100_by_default():
    from docparse import Settings
    assert Settings().max_pages == 100
    assert len(parse_page_range(None, 100, max_pages=100)) == 100
    with pytest.raises(TooManyPagesError, match="limit is 100"):
        parse_page_range(None, 101, max_pages=100)
    with pytest.raises(TooManyPagesError):
        parse_page_range("1-60,100-150", 500, max_pages=100)
    assert len(parse_page_range("1-100", 500, max_pages=100)) == 100  # a range within a big file is fine


@pytest.mark.parametrize("bad", ["abc", "1-x", "--", "3-a,5"])
def test_invalid_page_range(bad):
    with pytest.raises(InvalidPageRangeError):
        parse_page_range(bad, 10)


def test_oversized_pdf_page_render_is_capped(parser):
    # H2: a 1 KB PDF declaring a 14400 x 14400 pt page used to render at 4.8 GB
    import io as _io
    from reportlab.pdfgen import canvas
    buf = _io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(14400, 14400))
    c.drawString(10, 10, "x")
    c.save()
    page = parser.parse(buf.getvalue())["pages"][0]
    assert page["width"] * page["height"] <= parser.s.max_render_pixels * 1.01
    assert page["width"] == pytest.approx(5000, abs=2)


def test_normal_pdf_page_keeps_full_resolution(parser, native_pdf):
    page = parser.parse(native_pdf)["pages"][0]
    assert (page["width"], page["height"]) == (1654, 2339)  # A4 at 200 DPI, unchanged


def test_native_pdf_uses_text_layer_without_ocr(parser, fake, native_pdf):
    result = parser.parse(native_pdf, "invoice.pdf")
    page = result["pages"][0]

    assert page["source"] == "text_layer"
    assert "ocr" not in fake.calls                 # no OCR on a digital page
    texts = " ".join(b.get("text") or "" for b in page["blocks"])
    assert "INVOICE 2026-0042" in texts
    table = next(b for b in page["blocks"] if b["type"] == "table")
    assert table["rows"][0] == ["Item", "Qty", "Price"]
    assert ["Ink", "1", "25.00"] in table["rows"]
    assert "Paper" not in texts                     # table text not duplicated as paragraphs


def test_scanned_pdf_goes_to_ocr(parser, fake, scanned_pdf):
    page = parser.parse(scanned_pdf)["pages"][0]
    assert page["source"] == "ocr"
    assert "ocr" in fake.calls
    assert page["blocks"][0]["text"] == "TOTAL 42.00"


def test_plain_text_image_skips_structure_model(parser, fake, png_bytes):
    page = parser.parse(png_bytes)["pages"][0]
    assert page["routes"] == ["ocr"]
    assert not any(c.startswith("structure") for c in fake.calls)
    assert not any(c.startswith("recognize") for c in fake.calls)  # confident line: no re-read


def test_table_on_image_runs_only_table_module(parser, fake, png_bytes):
    fake.layout = [{"label": "table", "score": 0.9, "bbox": [5, 5, 300, 200]}]
    page = parser.parse(png_bytes)["pages"][0]

    assert "structure:tables" in fake.calls        # charts/formulas not switched on
    table = page["blocks"][0]
    assert table["type"] == "table"
    assert table["rows"] == [["Item", "Price"], ["Tea", "3"]]


def test_chart_is_an_image_and_never_reaches_the_structure_model(parser, fake, png_bytes):
    # Chart2Table removed: a chart-only page takes the fast OCR route
    fake.layout = [{"label": "chart", "score": 0.9, "bbox": [0, 0, 400, 300]}]
    page = parser.parse(png_bytes)["pages"][0]
    assert not any(c.startswith("structure") for c in fake.calls)
    chart = next(b for b in page["blocks"] if b.get("label") == "chart")
    assert chart["type"] == "image"
    # labels printed on the chart are still OCR'd and belong to the chart block
    assert chart["text"] == "TOTAL 42.00" and chart["confidence"] == 0.97


def test_chart_next_to_table_runs_tables_only(parser, fake, png_bytes):
    fake.layout = [{"label": "chart", "score": 0.9, "bbox": [0, 0, 300, 200]},
                   {"label": "table", "score": 0.9, "bbox": [0, 250, 300, 390]}]
    fake.structure = {"parsing_res_list": [
        {"block_label": "chart", "block_bbox": [0, 0, 300, 200], "block_content": ""},
        {"block_label": "table", "block_bbox": [0, 250, 300, 390],
         "block_content": "<table><tr><td>a</td></tr></table>"}]}
    page = parser.parse(png_bytes)["pages"][0]
    assert "structure:tables" in fake.calls
    assert [b["type"] for b in page["blocks"]] == ["image", "table"]


def test_chart_recognition_is_never_enabled_in_paddle_models():
    import inspect
    from docparse import models
    src = inspect.getsource(models)
    assert "use_chart_recognition=False" in src
    assert "use_chart_recognition=True" not in src and "enable_charts" not in src


def test_low_confidence_lines_are_reread(parser, fake, png_bytes):
    fake.lines = [
        {"text": "Dear Sam", "confidence": 0.55, "bbox": [10, 10, 200, 40]},
        {"text": "see you", "confidence": 0.60, "bbox": [10, 50, 200, 80]},
        {"text": "Printed footer", "confidence": 0.98, "bbox": [10, 300, 300, 330]},
    ]
    page = parser.parse(png_bytes)["pages"][0]

    assert "recognize:2" in fake.calls             # only the 2 weak lines were cropped
    assert page["handwriting_likely"] is True
    assert "handwriting" in page["routes"]
    lines = [l for b in page["blocks"] for l in b["lines"]]
    assert sum(1 for l in lines if l.get("escalated")) == 2


def test_rotated_page_is_turned_upright(parser, fake):
    buf = io.BytesIO()
    Image.new("RGB", (600, 300), "white").save(buf, "PNG")
    fake.angle = 90
    page = parser.parse(buf.getvalue())["pages"][0]
    assert page["orientation"] == 90
    assert (page["width"], page["height"]) == (300, 600)  # reported in the upright frame


def test_large_images_are_downscaled_and_boxes_mapped_back(fake):
    from docparse import DocumentParser, Settings
    parser = DocumentParser(Settings(max_image_side=1000), models=fake)
    buf = io.BytesIO()
    Image.new("RGB", (4000, 2000), "white").save(buf, "PNG")
    page = parser.parse(buf.getvalue())["pages"][0]
    assert (page["width"], page["height"]) == (4000, 2000)
    assert page["blocks"][0]["lines"][0]["bbox"] == [40.0, 40.0, 800.0, 120.0]  # x4 scale

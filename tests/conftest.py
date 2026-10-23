"""Shared test fixtures. A fake model stands in for PaddleOCR so tests run in seconds."""
import io

import pytest
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle

from docparse import DocumentParser, Settings


class FakeModels:
    """Mimics PaddleModels. Set `layout`, `lines`, `structure` per test."""

    def __init__(self):
        self.layout = []
        self.lines = [{"text": "TOTAL 42.00", "confidence": 0.97, "bbox": [10, 10, 200, 30]}]
        self.recognized = ("re-read text", 0.95)
        self.structure = {"parsing_res_list": [
            {"block_label": "table", "block_bbox": [5, 5, 300, 200],
             "block_content": "<table><tr><td>Item</td><td>Price</td></tr><tr><td>Tea</td><td>3</td></tr></table>"},
        ]}
        self.angle = 0
        self.calls = []

    def detect_orientation(self, img):
        self.calls.append("orientation")
        return self.angle

    def detect_layout(self, img):
        self.calls.append("layout")
        return [dict(r) for r in self.layout]

    def run_ocr(self, img):
        self.calls.append("ocr")
        return [dict(l) for l in self.lines]

    def recognize(self, crops):
        self.calls.append(f"recognize:{len(crops)}")
        return [self.recognized for _ in crops]

    def run_structure(self, img, **flags):
        self.calls.append("structure:" + ",".join(sorted(k for k, v in flags.items() if v)))
        return self.structure


@pytest.fixture
def fake():
    return FakeModels()


API_KEY = "test-key-0123456789abcdef"


@pytest.fixture
def parser(fake):
    return DocumentParser(Settings(api_key=API_KEY), models=fake)


def _png(size=(600, 400), text="Hello"):
    img = Image.new("RGB", size, "white")
    ImageDraw.Draw(img).text((20, 20), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return _png()


@pytest.fixture
def native_pdf():
    """Digital PDF: a paragraph plus a ruled table."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    y = 800
    for line in ["INVOICE 2026-0042", "Acme Trading LLC, Abu Dhabi",
                 "Thank you for your business. Payment due in 30 days."]:
        c.drawString(72, y, line)
        y -= 16
    t = Table([["Item", "Qty", "Price"], ["Paper", "2", "10.00"], ["Ink", "1", "25.00"]])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    t.wrapOn(c, 400, 200)
    t.drawOn(c, 72, 600)
    c.save()
    return buf.getvalue()


@pytest.fixture
def scanned_pdf():
    """A PDF page that is just one big image (like a scanner output)."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.drawImage(ImageReader(io.BytesIO(_png((1240, 1754), "scanned receipt"))), 0, 0, *A4)
    c.save()
    return buf.getvalue()

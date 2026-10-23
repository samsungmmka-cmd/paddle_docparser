"""Image loading: phone photos (MPO), multi-page TIFF, and damaged files."""
import io

import pytest
from PIL import Image

from conftest import API_KEY
from docparse.loaders import InvalidDocumentError, load_image_pages


def _mpo(break_second_picture=False):
    """A phone-style JPEG: main photo + a hidden second picture (MPO format)."""
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(
        buf, "MPO", save_all=True, append_images=[Image.new("RGB", (160, 120), "gray")])
    data = bytearray(buf.getvalue())
    if break_second_picture:
        offset = Image.open(io.BytesIO(bytes(data)))._MpoImageFile__mpoffsets[1]
        data[offset:offset + 4] = b"\x00\x00\x00\x00"
    return bytes(data)


def test_phone_mpo_photo_is_one_page():
    pages = load_image_pages(_mpo())
    assert len(pages) == 1 and pages[0].shape[:2] == (600, 800)


def test_phone_photo_with_damaged_hidden_picture_still_works():
    # Used to fail with "SyntaxError: not a JPEG file"
    pages = load_image_pages(_mpo(break_second_picture=True))
    assert len(pages) == 1 and pages[0].shape[:2] == (600, 800)


def test_multi_page_tiff_still_returns_every_page():
    buf = io.BytesIO()
    frames = [Image.new("RGB", (100, 100), c) for c in ("white", "gray", "black")]
    frames[0].save(buf, "TIFF", save_all=True, append_images=frames[1:])
    assert len(load_image_pages(buf.getvalue())) == 3


def test_parse_phone_photo_end_to_end(parser):
    result = parser.parse(_mpo(break_second_picture=True), "IMG_2026.jpeg")
    assert result["file_type"] == "jpeg" and result["page_count"] == 1


@pytest.mark.parametrize("data", [
    b"\xff\xd8\xff\xe0" + b"\x00" * 200,              # JPEG header, garbage body
    b"\x89PNG\r\n\x1a\n" + b"0" * 50,                 # truncated PNG
])
def test_damaged_image_gives_clear_error(data):
    with pytest.raises(InvalidDocumentError, match="could not be read"):
        load_image_pages(data)


@pytest.mark.parametrize("name,data", [
    ("bad.jpeg", b"\xff\xd8\xff\xe0" + b"\x00" * 200),
    ("bad.png", b"\x89PNG\r\n\x1a\n" + b"0" * 50),
    ("bad.pdf", b"%PDF-1.7\nnot really a pdf"),
])
def test_damaged_files_return_400_not_500(parser, name, data):
    from fastapi.testclient import TestClient
    from docparse.api import create_app
    with TestClient(create_app(parser, warmup=False), raise_server_exceptions=False) as client:
        r = client.post("/v1/parse", headers={"X-API-Key": API_KEY}, files={"file": (name, data)})
    assert r.status_code == 400
    assert "could not be read" in r.json()["detail"]

"""Read input files: detect type, load image pages, read PDF text layer, render PDF pages."""
import io

import numpy as np
import pdfplumber
import pypdfium2 as pdfium
from PIL import Image, ImageOps, ImageSequence

Image.MAX_IMAGE_PIXELS = 200_000_000  # guard against decompression bombs


class UnsupportedFileError(ValueError):
    pass


class InvalidPageRangeError(ValueError):
    pass


class TooManyPagesError(ValueError):
    pass


class InvalidDocumentError(ValueError):
    """The file looks like a supported type but is damaged or cannot be decoded."""


def detect_file_type(data: bytes) -> str:
    """Detect by content (magic bytes), not by file extension."""
    head = data[:16]
    if b"%PDF-" in data[:1024]:
        return "pdf"
    if head.startswith(b"\x89PNG"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head.startswith(b"BM"):
        return "bmp"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head.startswith(b"GIF8"):
        return "gif"
    raise UnsupportedFileError("Unsupported file type. Use PDF, PNG, JPEG, TIFF, BMP, WEBP or GIF.")


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """PaddleOCR expects BGR numpy arrays (OpenCV order)."""
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])


def load_image_pages(data: bytes) -> list:
    """Return the image's pages as BGR arrays. Phone photos are rotated per their EXIF tag.

    Only TIFF files can have several pages. Other formats use the first picture only:
    phone cameras often save "JPEG" photos as MPO, which hide extra pictures (a preview
    or depth map) after the main photo. Those are not pages, and they are often damaged.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.format != "TIFF":
                return [pil_to_bgr(ImageOps.exif_transpose(im))]
            return [pil_to_bgr(ImageOps.exif_transpose(frame)) for frame in ImageSequence.Iterator(im)]
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        # OSError covers truncated files and Pillow's UnidentifiedImageError
        raise InvalidDocumentError(f"The image could not be read; it may be damaged ({e}).") from None


class PdfFile:
    """Opens a PDF once with pdfplumber (text/tables) and pdfium (fast rendering)."""

    def __init__(self, data: bytes):
        try:
            self._plumber = pdfplumber.open(io.BytesIO(data))
            self._pdfium = pdfium.PdfDocument(data)
            len(self._pdfium)  # fails here for a broken page tree
        except Exception as e:  # pdfium and pdfminer raise many different error types
            raise InvalidDocumentError(f"The PDF could not be read; it may be damaged or encrypted ({e}).") from None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._plumber.close()
        self._pdfium.close()

    @property
    def page_count(self) -> int:
        return len(self._pdfium)

    def text_layer(self, index: int) -> dict:
        """Text lines, tables and image positions from the PDF itself (units: points)."""
        page = self._plumber.pages[index]
        w, h = float(page.width), float(page.height)

        lines = [{"text": ln["text"], "bbox": [ln["x0"], ln["top"], ln["x1"], ln["bottom"]]}
                 for ln in page.extract_text_lines(strip=True) if ln["text"].strip()]
        tables = [{"bbox": list(t.bbox), "rows": t.extract()} for t in page.find_tables()]

        images = []
        for im in page.images:
            box = [max(0, im["x0"]), max(0, im["top"]), min(w, im["x1"]), min(h, im["bottom"])]
            if box[2] > box[0] and box[3] > box[1]:
                images.append(box)

        page_area = w * h or 1
        biggest = max(((b[2] - b[0]) * (b[3] - b[1]) for b in images), default=0)
        return {
            "width": w, "height": h,
            "lines": lines, "tables": tables, "images": images,
            "char_count": sum(len(ln["text"]) for ln in lines),
            "image_coverage": biggest / page_area,
        }

    def render(self, index: int, scale: float) -> np.ndarray:
        """Render a page. `scale` = pixels per PDF point (the parser caps it for huge pages)."""
        page = self._pdfium[index]
        try:
            return pil_to_bgr(page.render(scale=scale).to_pil())
        finally:
            page.close()


def parse_page_range(spec: str | None, total: int, max_pages: int | None = None) -> list:
    """'1-3,5' -> [0, 1, 2, 4] (zero-based). None/empty -> all pages.

    Every range is clamped to 1..total BEFORE it is expanded, so a request like
    '1-10000000000' costs nothing. Raises TooManyPagesError when more than
    `max_pages` pages would be processed, and InvalidPageRangeError for bad input.
    """
    def too_many(count):
        return TooManyPagesError(
            f"{count} pages requested; the limit is {max_pages} per request. "
            f"Use the 'pages' option to pick a range, e.g. pages=1-{max_pages}.")

    if not spec or not spec.strip():
        if max_pages is not None and total > max_pages:
            raise too_many(total)
        return list(range(total))

    pages = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
            else:
                a = b = int(part)
        except ValueError:
            raise InvalidPageRangeError(f"Invalid page range '{spec}'. Use a form like 1-3,5.") from None
        a, b = max(1, a), min(total, b)
        if a > b:
            continue  # outside the document
        pages.update(range(a, b + 1))  # at most `total` items, thanks to the clamp
        if max_pages is not None and len(pages) > max_pages:
            raise too_many(len(pages))
    return sorted(p - 1 for p in pages)

"""Settings, read from environment variables prefixed with DOCPARSE_.

Example: DOCPARSE_DEVICE=gpu:0  DOCPARSE_CPU_THREADS=8
"""
import os
from dataclasses import dataclass, field, fields


@dataclass
class Settings:
    # Hardware
    device: str = "cpu"                 # "cpu" or "gpu:0"
    cpu_threads: int = min(8, os.cpu_count() or 4)
    enable_mkldnn: bool = True          # Intel CPU acceleration
    enable_hpi: bool = False            # high-performance inference (TensorRT/ONNX)

    # Models (mobile = fast on CPU; switch to *_server_* on GPU)
    det_model: str = "PP-OCRv5_mobile_det"
    rec_model: str = "PP-OCRv5_mobile_rec"
    handwriting_rec_model: str = "PP-OCRv5_server_rec"
    router_layout_model: str = "PP-DocLayout-S"
    structure_layout_model: str = "PP-DocLayout-M"
    orientation_model: str = "PP-LCNet_x1_0_doc_ori"

    # Optional features (only *run* when the router detects them on a page)
    enable_orientation: bool = True
    enable_tables: bool = True
    enable_formulas: bool = True
    enable_seals: bool = False

    # Tuning
    render_dpi: int = 200               # PDF page -> image resolution
    max_image_side: int = 2400          # downscale bigger images first
    det_limit_side_len: int = 1600
    layout_threshold: float = 0.5
    handwriting_threshold: float = 0.80  # lines below this score get re-read
    handwriting_page_ratio: float = 0.30 # share of weak lines => "handwritten"
    min_text_layer_chars: int = 50       # fewer chars => treat page as scanned
    scanned_image_coverage: float = 0.85 # page that is one big image => scanned
    embedded_image_min_area: float = 0.10  # OCR images covering >=10% of a page
    max_pages: int = 100                 # pages processed per request
    max_render_pixels: int = 25_000_000  # cap for one rendered PDF page (~5000 x 5000 px)
    max_upload_mb: int = 50

    # Security: clients must send this in the X-API-Key header (required by the API)
    api_key: str = field(default="", repr=False)

    @classmethod
    def from_env(cls) -> "Settings":
        values = {}
        for f in fields(cls):
            raw = os.getenv(f"DOCPARSE_{f.name.upper()}")
            if raw is None:
                continue
            default = getattr(cls, f.name, None)
            if isinstance(default, bool):
                values[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                values[f.name] = int(raw)
            elif isinstance(default, float):
                values[f.name] = float(raw)
            else:
                values[f.name] = raw
        return cls(**values)

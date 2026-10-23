"""PaddleOCR models, loaded lazily (once) and wrapped to return plain Python data.

Every model is created the first time it is needed and then reused, so a
request never pays the model-loading cost twice. Paddle predictors are not
thread-safe; the parser serialises calls with a lock.
"""
import threading

import numpy as np

from .config import Settings
from .utils import to_bbox


def _first(results):
    for r in results:
        return r.json["res"]
    return {}


class PaddleModels:
    def __init__(self, settings: Settings):
        self.s = settings
        self._cache = {}
        self._lock = threading.Lock()

    # ---------- lazy loading ----------

    def _common(self) -> dict:
        kw = {"device": self.s.device, "enable_hpi": self.s.enable_hpi}
        if self.s.device.startswith("cpu"):
            kw.update(enable_mkldnn=self.s.enable_mkldnn, cpu_threads=self.s.cpu_threads)
        return kw

    def _get(self, name, factory):
        if name not in self._cache:
            with self._lock:
                if name not in self._cache:
                    self._cache[name] = factory()
        return self._cache[name]

    @property
    def orientation(self):
        from paddleocr import DocImgOrientationClassification
        return self._get("orientation", lambda: DocImgOrientationClassification(
            model_name=self.s.orientation_model, **self._common()))

    @property
    def layout(self):
        from paddleocr import LayoutDetection
        return self._get("layout", lambda: LayoutDetection(
            model_name=self.s.router_layout_model, **self._common()))

    @property
    def ocr(self):
        from paddleocr import PaddleOCR
        return self._get("ocr", lambda: PaddleOCR(
            text_detection_model_name=self.s.det_model,
            text_recognition_model_name=self.s.rec_model,
            use_doc_orientation_classify=False,   # done once, up front, by the parser
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_limit_side_len=self.s.det_limit_side_len,
            text_det_limit_type="max",
            **self._common()))

    @property
    def handwriting(self):
        from paddleocr import TextRecognition
        return self._get("handwriting", lambda: TextRecognition(
            model_name=self.s.handwriting_rec_model, **self._common()))

    @property
    def structure(self):
        from paddleocr import PPStructureV3
        return self._get("structure", lambda: PPStructureV3(
            layout_detection_model_name=self.s.structure_layout_model,
            text_detection_model_name=self.s.det_model,
            text_recognition_model_name=self.s.rec_model,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_table_recognition=self.s.enable_tables,
            use_chart_recognition=False,   # PP-Chart2Table is never loaded (slow, can hang)
            use_formula_recognition=self.s.enable_formulas,
            use_seal_recognition=self.s.enable_seals,
            text_det_limit_side_len=self.s.det_limit_side_len,
            text_det_limit_type="max",
            **self._common()))

    def warmup(self):
        """Load every model and run a tiny image through the fast path."""
        blank = np.full((320, 320, 3), 255, dtype=np.uint8)
        if self.s.enable_orientation:
            self.detect_orientation(blank)
        self.detect_layout(blank)
        self.run_ocr(blank)
        self.handwriting  # noqa: B018 - load only
        self.structure    # noqa: B018 - load only

    # ---------- inference (plain data in, plain data out) ----------

    def detect_orientation(self, img) -> int:
        res = _first(self.orientation.predict(img))
        labels = res.get("label_names") or ["0"]
        return int(labels[0])

    def detect_layout(self, img) -> list:
        res = _first(self.layout.predict(img))
        return [{"label": b["label"], "score": float(b["score"]), "bbox": to_bbox(b["coordinate"])}
                for b in res.get("boxes", [])]

    def run_ocr(self, img) -> list:
        res = _first(self.ocr.predict(img))
        boxes = res.get("rec_boxes")
        if boxes is None or len(boxes) == 0:
            boxes = res.get("rec_polys", [])
        return [{"text": t, "confidence": round(float(s), 4), "bbox": to_bbox(b)}
                for t, s, b in zip(res.get("rec_texts", []), res.get("rec_scores", []), boxes)]

    def recognize(self, crops: list) -> list:
        """Re-read cropped text lines with the stronger model. Returns [(text, score)]."""
        out = []
        for r in self.handwriting.predict(crops, batch_size=8):
            j = r.json["res"]
            out.append((j.get("rec_text", ""), float(j.get("rec_score", 0.0))))
        return out

    def run_structure(self, img, tables=False, formulas=False, seals=False) -> dict:
        return _first(self.structure.predict(
            img,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            use_table_recognition=tables and self.s.enable_tables,
            use_chart_recognition=False,
            use_formula_recognition=formulas and self.s.enable_formulas,
            use_seal_recognition=seals and self.s.enable_seals,
        ))

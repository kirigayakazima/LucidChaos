"""
Vision engine powered by RapidOCR and OpenCV.
Provides dynamic text recognition, semantic clicking, and template matching.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from rapidocr_onnxruntime import RapidOCR

class VisionEngine:
    def __init__(self):
        # Initialize RapidOCR with default lightweight onnx models
        self.ocr = RapidOCR()

    def ocr_image(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run OCR on image.
        Returns a list of dicts:
        [
            {
                "text": "连结结晶",
                "score": 0.98,
                "box": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                "center_client": (cx, cy),
                "center_rel": (rx, ry)
            },
            ...
        ]
        """
        if img_bgr is None or img_bgr.size == 0:
            return []

        h, w = img_bgr.shape[:2]
        result, _ = self.ocr(img_bgr)
        if not result:
            return []

        items = []
        for dt_box, text, score in result:
            box = np.array(dt_box, dtype=np.int32)
            cx = int(np.mean(box[:, 0]))
            cy = int(np.mean(box[:, 1]))
            rx = cx / max(1, w)
            ry = cy / max(1, h)

            items.append({
                "text": text.strip(),
                "score": float(score),
                "box": dt_box,
                "center_client": (cx, cy),
                "center_rel": (rx, ry)
            })
        return items

    def find_text(self, img_bgr: np.ndarray, keyword: str, min_score: float = 0.6) -> Optional[Dict[str, Any]]:
        """Find the best matching text item containing the keyword."""
        items = self.ocr_image(img_bgr)
        for item in items:
            if item["score"] >= min_score and keyword in item["text"]:
                return item
        return None

    def find_template(self, img_bgr: np.ndarray, tpl_bgr: np.ndarray, threshold: float = 0.8) -> Optional[Tuple[int, int, float]]:
        """
        Template matching using normalized cross-correlation.
        Returns (center_x, center_y, max_val) or None.
        """
        if img_bgr is None or tpl_bgr is None:
            return None
        res = cv2.matchTemplate(img_bgr, tpl_bgr, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            th, tw = tpl_bgr.shape[:2]
            cx = max_loc[0] + tw // 2
            cy = max_loc[1] + th // 2
            return cx, cy, float(max_val)
        return None

    @staticmethod
    def calc_image_hash(img_bgr: np.ndarray) -> str:
        """Calculate a fast 64-bit dHash for screen stagnation/freeze detection."""
        if img_bgr is None or img_bgr.size == 0:
            return ""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        return hex(int("".join(["1" if b else "0" for b in diff.flatten()]), 2))

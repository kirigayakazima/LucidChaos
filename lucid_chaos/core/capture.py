"""
High-performance screen capture module using mss.
Captures the client area of the target game window.
"""

import numpy as np
import mss
from typing import Optional
from .window import GameWindow

class ScreenCapture:
    def __init__(self, window: GameWindow):
        self.window = window
        self.sct = mss.mss()

    def capture(self) -> Optional[np.ndarray]:
        """
        Capture the current client area of the game window.
        Returns a BGR numpy image (OpenCV compatible) or None if window is invalid.
        """
        if not self.window.is_valid:
            if not self.window.find_window():
                return None

        sx, sy, w, h = self.window.get_client_rect()
        if w <= 0 or h <= 0:
            return None

        monitor = {"top": sy, "left": sx, "width": w, "height": h}
        try:
            # mss grab returns BGRA
            shot = self.sct.grab(monitor)
            img = np.array(shot, dtype=np.uint8)
            # Drop Alpha channel to get standard BGR
            return img[:, :, :3]
        except Exception:
            # 捕获 GDI / BitBlt 临时抖动，重建 mss 实例并返回 None 由主循环重试
            try:
                self.sct = mss.mss()
            except Exception:
                pass
            return None

    def capture_roi_rel(self, rx: float, ry: float, rw: float, rh: float) -> Optional[np.ndarray]:
        """
        Capture a Region of Interest (ROI) defined by relative bounding box (0.0 ~ 1.0).
        """
        full_img = self.capture()
        if full_img is None:
            return None
        h, w = full_img.shape[:2]
        x1 = max(0, int(round(rx * w)))
        y1 = max(0, int(round(ry * h)))
        x2 = min(w, int(round((rx + rw) * w)))
        y2 = min(h, int(round((ry + rh) * h)))
        return full_img[y1:y2, x1:x2]

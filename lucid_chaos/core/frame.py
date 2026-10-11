"""
帧上下文（FrameContext）—— 一帧只做一次观测，所有判据共用同一份结果。

改造前的问题：同一帧里 observer 跑一次 OCR，路由器再判断"要不要自己再跑一次"，
各 handler 又各自拿模板、各自 resize；模板缓存有 7 份、缩放没有缓存。
本类把「图像 + OCR 结果 + 模板库 + 亮度归一化」收拢成一个对象：

    frame = FrameContext.from_image(img, ocr=vision)   # 一帧一次 OCR
    state = observer.observe_frame(frame)              # observer 复用 ctx.targets
    res   = router.classify_from_frame(frame)          # 路由器复用同一份
    hit   = frame.best_match(ROGUE_NODE_TEMPLATES, fx=0.22 * frame.scale)

**本类只做收拢，不改变任何判据与阈值**，因此不改变现有行为。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .template_bank import BANK, MatchResult, TemplateBank, normalize_brightness
from .vision_service import VisionTarget


class FrameContext:
    """一帧的「观测结果包」。"""

    def __init__(self, img_bgr: Optional[np.ndarray] = None,
                 targets: Optional[Sequence[VisionTarget]] = None,
                 bank: Optional[TemplateBank] = None):
        self.img = img_bgr
        if img_bgr is not None and getattr(img_bgr, "size", 0):
            self.h, self.w = int(img_bgr.shape[0]), int(img_bgr.shape[1])
        else:
            self.h, self.w = 0, 0
        self.scale = self.h / 576.0 if self.h else 0.0
        self.bank = bank or BANK
        self.targets: List[VisionTarget] = list(targets or [])
        self.ocr_done = bool(targets)
        self._norm: Optional[np.ndarray] = None
        self._elements: Optional[Dict[str, VisionTarget]] = None

    # ---------------- 构造 ----------------
    @classmethod
    def from_image(cls, img_bgr: np.ndarray, ocr=None,
                   targets: Optional[Sequence[VisionTarget]] = None,
                   bank: Optional[TemplateBank] = None) -> "FrameContext":
        ctx = cls(img_bgr, targets=targets, bank=bank)
        if ocr is not None and not ctx.targets:
            ctx.observe(ocr)
        return ctx

    # ---------------- 观测 ----------------
    def observe(self, vision) -> "FrameContext":
        """跑一次 OCR（一帧只跑一次）。有现成 targets 时不重复跑。"""
        if not self.ocr_done:
            self.targets = vision.locate_all_texts(self.img, min_score=0.5)
            self.ocr_done = True
            self._elements = None
        return self

    @property
    def valid(self) -> bool:
        return self.img is not None and getattr(self.img, "size", 0) > 0

    @property
    def corpus(self) -> str:
        return " ".join(t.text for t in self.targets)

    @property
    def elements(self) -> Dict[str, VisionTarget]:
        """按 text 去重的 dict（兼容旧代码）。注意重名元素会被覆盖——
        需要"有几个同类元素"时请用 `targets`（不去重）。"""
        if self._elements is None:
            self._elements = {t.text: t for t in self.targets}
        return self._elements

    # ---------------- 亮度 ----------------
    def normalized(self) -> Optional[np.ndarray]:
        """亮度归一化副本（懒加载，一帧只算一次）。用于压暗/过渡帧的模板匹配。"""
        if self._norm is None and self.valid:
            self._norm = normalize_brightness(self.img)
        return self._norm

    # ---------------- 图像操作 ----------------
    def crop_rel(self, rx1: float, ry1: float, rx2: float, ry2: float) -> Optional[np.ndarray]:
        """按相对坐标裁剪（越界自动夹紧）。"""
        if not self.valid:
            return None
        x1, y1 = max(0, int(rx1 * self.w)), max(0, int(ry1 * self.h))
        x2, y2 = min(self.w, int(rx2 * self.w)), min(self.h, int(ry2 * self.h))
        if x2 <= x1 or y2 <= y1:
            return None
        return self.img[y1:y2, x1:x2]

    def to_px(self, rx: float, ry: float) -> Tuple[int, int]:
        return int(round(rx * self.w)), int(round(ry * self.h))

    # ---------------- 模板匹配（委托给共用模板库） ----------------
    def match(self, name, fx: float, fy: Optional[float] = None,
              region: Optional[Sequence[float]] = None, normalize: bool = False) -> Optional[MatchResult]:
        return self.bank.match(self.img, name, fx, fy=fy, region=region, normalize=normalize)

    def best_match(self, names: Sequence, fx: float, fy: Optional[float] = None,
                   region: Optional[Sequence[float]] = None, normalize: bool = False) -> Optional[MatchResult]:
        return self.bank.best_match(self.img, names, fx, fy=fy, region=region, normalize=normalize)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"<FrameContext {self.w}x{self.h} texts={len(self.targets)} ocr_done={self.ocr_done}>"

"""
Phase 4: Unified Vision Service with Fallback Chain
Implements OCR, multi-scale template matching, and normalized coordinate fallbacks.
"""

import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import numpy as np
import cv2
from rapidocr_onnxruntime import RapidOCR

@dataclass
class VisionTarget:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h in client coordinates
    center: Tuple[int, int]          # cx, cy in client coordinates
    center_rel: Tuple[float, float]  # rx, ry in relative coordinates (0.0 ~ 1.0)
    source: str                      # 'ocr', 'template', 'normalized'

from pathlib import Path
from collections import OrderedDict

class VisionService:
    def __init__(self, template_dir: Optional[str] = None):
        self._ocr_engine = RapidOCR()
        # OCR 结果缓存（性能）：改造前 `self.ocr` 是裸 RapidOCR，没有任何缓存，
        # 而各个 handler 会对**同一帧**反复调 locate_text（每次都是一次完整 OCR，约 1.4s/次）：
        #   handle_stage_map: 進入 + 进入                       -> 2 次
        #   handle_settlement: 確認/确认/確定/完成/關閉/关闭     -> 最多 6 次
        # 实测选关那一帧 dispatch 花了 6.1s，大头就是这里的重复 OCR。
        # 这里做一层"按图像内容"的小 LRU（最近 3 帧），对调用方完全透明：
        # `self.ocr(img)` 仍然可调用，并且**同一张图第二次起直接返回缓存**。
        self._ocr_cache: "OrderedDict[tuple, object]" = OrderedDict()
        self.ocr = self._ocr_cached
        if template_dir is None:
            template_dir = str(Path(__file__).resolve().parent.parent / "templates")
        self.template_dir = template_dir
        self.templates: Dict[str, np.ndarray] = {}
        if self.template_dir and os.path.exists(self.template_dir):
            self._load_templates()

    # ---------------- OCR（带内容缓存） ----------------
    @staticmethod
    def _img_key(img: np.ndarray) -> tuple:
        """按图像内容做键：id + 尺寸 + 三个采样点（便宜且足够区分不同画面）。"""
        try:
            h, w = img.shape[:2]
            return (id(img), h, w,
                    img[0, 0].tobytes(),
                    img[h // 2, w // 2].tobytes(),
                    img[h // 4, (w * 3) // 4].tobytes())
        except Exception:  # noqa: BLE001
            return (id(img),)

    def _ocr_cached(self, img_bgr: np.ndarray, *args, **kwargs):
        """与裸 RapidOCR 等价，但同一张图只真正跑一次 OCR。"""
        if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            return self._ocr_engine(img_bgr, *args, **kwargs) if img_bgr is not None else (None, None)
        key = self._img_key(img_bgr)
        hit = self._ocr_cache.get(key)
        if hit is not None:
            self._ocr_cache.move_to_end(key)
            self.ocr_hits = getattr(self, "ocr_hits", 0) + 1
            return hit
        out = self._ocr_engine(img_bgr, *args, **kwargs)
        self._ocr_cache[key] = out
        while len(self._ocr_cache) > 3:
            self._ocr_cache.popitem(last=False)
        return out

    def _load_templates(self):
        # 统一走全项目共用的模板库（语义不变：仍然只索引顶层 png，键仍是文件名去扩展名）
        from .template_bank import BANK
        for fname in os.listdir(self.template_dir):
            if fname.lower().endswith(('.png', '.jpg', '.bmp')):
                tname = os.path.splitext(fname)[0]
                tpl = BANK.get(fname)
                if tpl is not None:
                    self.templates[tname] = tpl

    @staticmethod
    def _text_similarity(s1: str, s2: str) -> float:
        """中文模糊相似度。实现已抽到 `core/text_match.py`（场景判定的"文字锁定"也用它）。"""
        from .text_match import text_similarity
        return text_similarity(s1, s2)

    @staticmethod
    def _is_overlay_hud_text(text: str, rx: float = 0.0, ry: float = 0.0) -> bool:
        """
        判断文本是否为 ScreenOverlay 自身的调试/路由徽章（防 OCR 自激死循环）。
        """
        if not text:
            return False
        t = text.strip()
        if "路由" in t or "【路由" in t:
            return True
        if ry <= 0.08 and rx <= 0.35:
            if "[" in t or "【" in t or "分" in t or "%" in t or "％" in t or "AI" in t or "BOT" in t:
                return True
        return False

    def locate_text(self, img_bgr: np.ndarray, keyword: str, min_score: float = 0.6) -> Optional[VisionTarget]:
        """OCR-based text localization with fuzzy traditional/simplified support."""
        if img_bgr is None or img_bgr.size == 0:
            return None
        h, w = img_bgr.shape[:2]
        res, _ = self.ocr(img_bgr)
        if not res:
            return None

        best_target = None
        best_score = 0.0

        for dt_box, text, score in res:
            score = float(score)
            sim = self._text_similarity(keyword, text)
            if sim >= 0.7 and score >= min_score and (score * sim) > best_score:
                box = np.array(dt_box, dtype=np.int32)
                x1 = int(np.min(box[:, 0]))
                y1 = int(np.min(box[:, 1]))
                x2 = int(np.max(box[:, 0]))
                y2 = int(np.max(box[:, 1]))
                bw = x2 - x1
                bh = y2 - y1
                cx = int(x1 + bw / 2)
                cy = int(y1 + bh / 2)
                rx = cx / max(1, w)
                ry = cy / max(1, h)

                if self._is_overlay_hud_text(text, rx, ry):
                    continue

                best_target = VisionTarget(
                    text=text,
                    confidence=score,
                    bbox=(x1, y1, bw, bh),
                    center=(cx, cy),
                    center_rel=(rx, ry),
                    source="ocr"
                )
                best_score = score * sim

        return best_target

    def pick_text(self, targets: List[VisionTarget], keyword: str, min_score: float = 0.4,
                  region: Optional[Tuple[float, float, float, float]] = None,
                  min_sim: float = 0.7) -> Optional[VisionTarget]:
        """
        在**已经 OCR 好的结果**里挑最匹配的一个（不重新跑 OCR），可选限定相对区域。

        为什么需要"区域"（真机教训）：关卡详情抽屉里有一句正文
        `和阿黛海特一起進入夢中的世界` 含「進入」，`locate_text('進入')` 会命中这句正文
        (rx 0.742, ry 0.335)，而真正的【進入】按钮在右下 (rx 0.80, ry 0.92) ——
        位置校验失败 -> 跳过"点進入"分支 -> 反复点节点 -> 抽屉反复打开，**永远进不去**。
        加上区域限定后，"按钮"必须出现在按钮该在的地方。
        """
        best, best_key = None, 0.0
        for t in (targets or []):
            if region is not None:
                rx, ry = t.center_rel
                if not (region[0] <= rx <= region[2] and region[1] <= ry <= region[3]):
                    continue
            sim = self._text_similarity(keyword, t.text)
            if sim < min_sim or t.confidence < min_score:
                continue
            # 「按钮名要短」偏好：我们要找的是按钮，而正文里也会出现同样的词。
            # 实测正文『和阿黛海特一起進入夢中的世界』(11 字) vs 按钮『進入』(2 字)，
            # 光靠区域有时分不开（同一区域里两者都在）——所以对"比关键字长得多"的文本降权。
            extra = max(0, len(t.text) - len(keyword))
            brevity = 1.0 / (1.0 + 0.15 * extra)
            key = t.confidence * sim * brevity
            if key > best_key:
                best, best_key = t, key
        return best

    def pick_texts(self, targets: List[VisionTarget], keywords, min_score: float = 0.4,
                   region: Optional[Tuple[float, float, float, float]] = None) -> Optional[VisionTarget]:
        """按顺序尝试多个关键字（等价于原来 `locate_text(a) or locate_text(b) or ...`，但只 OCR 一次）。"""
        for kw in keywords:
            got = self.pick_text(targets, kw, min_score=min_score, region=region)
            if got is not None:
                return got
        return None

    def locate_all_texts(self, img_bgr: np.ndarray, min_score: float = 0.5) -> List[VisionTarget]:
        """Scan all text elements on screen."""
        if img_bgr is None or img_bgr.size == 0:
            return []
        h, w = img_bgr.shape[:2]
        res, _ = self.ocr(img_bgr)
        if not res:
            return []

        results = []
        for dt_box, text, score in res:
            score = float(score)
            if score >= min_score:
                box = np.array(dt_box, dtype=np.int32)
                x1 = int(np.min(box[:, 0]))
                y1 = int(np.min(box[:, 1]))
                x2 = int(np.max(box[:, 0]))
                y2 = int(np.max(box[:, 1]))
                bw = x2 - x1
                bh = y2 - y1
                cx = int(x1 + bw / 2)
                cy = int(y1 + bh / 2)
                rx = cx / max(1, w)
                ry = cy / max(1, h)

                t_str = text.strip()
                if self._is_overlay_hud_text(t_str, rx, ry):
                    continue

                results.append(VisionTarget(
                    text=t_str,
                    confidence=score,
                    bbox=(x1, y1, bw, bh),
                    center=(cx, cy),
                    center_rel=(rx, ry),
                    source="ocr"
                ))
        return results

    def locate_template(self, img_bgr: np.ndarray, template_name: str, threshold: float = 0.8) -> Optional[VisionTarget]:
        """Template matching scaled dynamically relative to 1080p."""
        if template_name not in self.templates or img_bgr is None:
            return None

        tpl = self.templates[template_name]
        h_curr, w_curr = img_bgr.shape[:2]

        # Dynamic scale based on current client height vs reference 1080p
        scale = h_curr / 1080.0
        new_w = max(1, int(round(tpl.shape[1] * scale)))
        new_h = max(1, int(round(tpl.shape[0] * scale)))
        scaled_tpl = cv2.resize(tpl, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if scaled_tpl.shape[0] > h_curr or scaled_tpl.shape[1] > w_curr:
            return None

        res = cv2.matchTemplate(img_bgr, scaled_tpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            cx = max_loc[0] + new_w // 2
            cy = max_loc[1] + new_h // 2
            return VisionTarget(
                text=template_name,
                confidence=float(max_val),
                bbox=(max_loc[0], max_loc[1], new_w, new_h),
                center=(cx, cy),
                center_rel=(cx / w_curr, cy / h_curr),
                source="template"
            )
        return None

    def locate(self, img_bgr: np.ndarray, keyword: Optional[str] = None, template_name: Optional[str] = None,
               fallback_rel: Optional[Tuple[float, float]] = None) -> Optional[VisionTarget]:
        """
        Hierarchical Fallback Localization:
        OCR -> Template -> Normalized Coordinate -> None
        """
        # 1. Try OCR
        if keyword:
            target = self.locate_text(img_bgr, keyword)
            if target:
                return target

        # 2. Try Template
        if template_name and template_name in self.templates:
            target = self.locate_template(img_bgr, template_name)
            if target:
                return target

        # 3. Try Normalized Fallback Coordinate
        if fallback_rel and img_bgr is not None:
            h, w = img_bgr.shape[:2]
            rx, ry = fallback_rel
            cx = int(round(rx * w))
            cy = int(round(ry * h))
            return VisionTarget(
                text=keyword or template_name or "fallback",
                confidence=0.5,
                bbox=(cx - 10, cy - 10, 20, 20),
                center=(cx, cy),
                center_rel=(rx, ry),
                source="normalized"
            )

        return None

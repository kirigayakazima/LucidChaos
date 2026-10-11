"""
统一的模板库（TemplateBank）—— 全项目唯一的模板读取 / 缩放缓存。

为什么要有它：改造前项目里有 **7 份**各自独立的模板缓存
（`VisionService.templates`、`SceneEvaluators._tpl_cache`、`NodeDecisionEngine` 的 3 个 class 变量、
`StoryPusher` 的 3 个、`PathPlanner._templates_cache`、`StageMapDetector._tpl_cache`、
以及 `battle_manager` 里一个**完全没有缓存、每次读盘**的加载器），
而且**没有任何缩放缓存**——`scan_mini_map` 每帧要对 5 个模板 × 18 个格子做 90 次 resize。

本模块只做三件事，且**不改变任何匹配语义**：
  1. `get(name)`      —— 按相对路径/文件名/裸名读模板，带缓存（BGR，IMREAD_COLOR，与旧行为逐位一致）
  2. `scaled(...)`    —— 带缓存的缩放（内部仍用 `cv2.resize(tpl, (0,0), fx=, fy=)`，保证与旧代码同结果）
  3. `match/best_match` —— 便捷封装（可选 region / 可选亮度归一化，默认关闭以保持行为不变）

亮度归一化（`normalize_brightness`）单独提供、**默认不参与匹配**：
游戏有明暗过渡蒙版时模板分会整体下滑（实测选路图同一模板 0.731 → 0.632），
需要时显式开启（`match(..., normalize=True)`），避免静默改变既有判据。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

PathLike = Union[str, Path]


class MatchResult(NamedTuple):
    """一次模板匹配的结果。坐标是**整图客户区像素**坐标。"""
    name: str
    score: float
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]      # x, y, w, h


def normalize_brightness(img_bgr: np.ndarray) -> np.ndarray:
    """
    亮度归一化：在 LAB 的 L 通道上做 CLAHE，把"整体被压暗"的影响拉平，色相/饱和度不动。
    用途：模板匹配遇到明暗过渡帧时（分数整体下滑）先归一化再匹配。
    """
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)


def _crop(img_bgr: np.ndarray, region: Optional[Sequence[float]]):
    """region = (rx1, ry1, rx2, ry2) 相对坐标；返回 (crop, x0, y0)。"""
    if region is None:
        return img_bgr, 0, 0
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(region[0] * w))
    y1 = max(0, int(region[1] * h))
    x2 = min(w, int(region[2] * w))
    y2 = min(h, int(region[3] * h))
    if x2 <= x1 or y2 <= y1:
        return None, 0, 0
    return img_bgr[y1:y2, x1:x2], x1, y1


class TemplateBank:
    def __init__(self, template_dir: Optional[PathLike] = None):
        self.dir = Path(template_dir) if template_dir else TEMPLATE_DIR
        self._raw: Dict[str, Optional[np.ndarray]] = {}
        self._scaled: Dict[Tuple[str, float, float], np.ndarray] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.scale_hits = 0
        self.scale_misses = 0

    # ---------------- 读取 ----------------
    def resolve(self, name: PathLike) -> Optional[Path]:
        """支持 'skip_btn'、'skip_btn.png'、'rogue/rogue_sword.png'、Path(...) 四种写法。"""
        p = Path(str(name))
        if p.suffix == "":
            p = p.with_suffix(".png")
        cand = self.dir / p
        return cand if cand.exists() else None

    def get(self, name: PathLike) -> Optional[np.ndarray]:
        key = str(name)
        with self._lock:
            if key in self._raw:
                self.hits += 1
                return self._raw[key]
            self.misses += 1
        path = self.resolve(name)
        arr: Optional[np.ndarray] = None
        if path is not None:
            try:
                with open(str(path), "rb") as f:
                    arr = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            except Exception:  # noqa: BLE001
                arr = None
        with self._lock:
            self._raw[key] = arr
        return arr

    def scaled(self, name: PathLike, fx: float, fy: Optional[float] = None) -> Optional[np.ndarray]:
        """带缓存的缩放。内部仍走 `cv2.resize(tpl, (0,0), fx=, fy=)`，与旧代码逐位一致。"""
        tpl = self.get(name)
        if tpl is None:
            return None
        fy = fx if fy is None else fy
        key = (str(name), round(float(fx), 6), round(float(fy), 6))
        with self._lock:
            got = self._scaled.get(key)
            if got is not None:
                self.scale_hits += 1
                return got
            self.scale_misses += 1
        out = cv2.resize(tpl, (0, 0), fx=fx, fy=fy)
        with self._lock:
            self._scaled[key] = out
        return out

    def scaled_to(self, name: PathLike, w: int, h: int) -> Optional[np.ndarray]:
        """带缓存的「显式尺寸」缩放，等价于 `cv2.resize(tpl, (w, h))`（截断尺寸的旧代码用它）。"""
        tpl = self.get(name)
        if tpl is None:
            return None
        w, h = max(1, int(w)), max(1, int(h))
        if (w, h) == (tpl.shape[1], tpl.shape[0]):
            return tpl
        key = (str(name), w, h)
        with self._lock:
            got = self._scaled.get(key)
            if got is not None:
                self.scale_hits += 1
                return got
            self.scale_misses += 1
        out = cv2.resize(tpl, (w, h))
        with self._lock:
            self._scaled[key] = out
        return out

    # ---------------- 匹配 ----------------
    def match(self, img_bgr: np.ndarray, name: PathLike, fx: float, fy: Optional[float] = None,
              region: Optional[Sequence[float]] = None, method: int = cv2.TM_CCOEFF_NORMED,
              normalize: bool = False) -> Optional[MatchResult]:
        tpl = self.scaled(name, fx, fy)
        if tpl is None or img_bgr is None or img_bgr.size == 0:
            return None
        src = normalize_brightness(img_bgr) if normalize else img_bgr
        crop, x0, y0 = _crop(src, region)
        if crop is None or crop.shape[0] < tpl.shape[0] or crop.shape[1] < tpl.shape[1]:
            return None
        res = cv2.matchTemplate(crop, tpl, method)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        tw, th = tpl.shape[1], tpl.shape[0]
        return MatchResult(str(name), float(max_val),
                           (x0 + max_loc[0] + tw // 2, y0 + max_loc[1] + th // 2),
                           (x0 + max_loc[0], y0 + max_loc[1], tw, th))

    def best_match(self, img_bgr: np.ndarray, names: Sequence[PathLike], fx: float,
                   fy: Optional[float] = None, region: Optional[Sequence[float]] = None,
                   method: int = cv2.TM_CCOEFF_NORMED,
                   normalize: bool = False) -> Optional[MatchResult]:
        best: Optional[MatchResult] = None
        for n in names:
            r = self.match(img_bgr, n, fx, fy, region=region, method=method, normalize=normalize)
            if r is not None and (best is None or r.score > best.score):
                best = r
        return best

    # ---------------- 诊断 ----------------
    def stats(self) -> Dict[str, int]:
        return {"raw_cache": len(self._raw), "scale_cache": len(self._scaled),
                "hits": self.hits, "misses": self.misses,
                "scale_hits": self.scale_hits, "scale_misses": self.scale_misses}


# 全项目共用的单例
BANK = TemplateBank()


def list_templates() -> List[str]:
    """列出 templates 目录下所有模板（相对路径），便于清点/测试。"""
    if not TEMPLATE_DIR.exists():
        return []
    return sorted(p.relative_to(TEMPLATE_DIR).as_posix() for p in TEMPLATE_DIR.rglob("*.png"))

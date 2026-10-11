"""
自动取证（FixtureCollector）—— 脚本自己攒"形状特征"的素材。

动机：形状/布局特征必须用**真实截图**标定，而让主人每次手动截图太费事，
且很多界面（战斗、肉鸽细节）主人手上没有现成图。所以让脚本自己存：

  - **卡住取证**：同一画面连续停留 ≥ `stuck_threshold` 轮 → 存一张（同一幕只存一次）
  - **界面清单取证**：本场运行里第一次出现某个 `(屏幕, 路由)` 组合 → 存一张

产物：
    lucid_chaos/fixtures_auto/<时间戳>_<screen>_<route>_stuck<N>.png
    lucid_chaos/fixtures_auto/<同名>.txt     ← 决策/路由/置信度/stuck 轮数（便于回溯为什么存它）

上限 `max_files`（默认 16，约 45MB），超出删最旧。
不影响任何判定与动作：**只读图像 + 写文件**，失败一律静默。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional, Set, Tuple

from ..logic.action_runner import StuckDetector

logger = logging.getLogger("FixtureCollector")


class FixtureCollector:
    def __init__(self, out_dir, max_files: int = 16, stuck_threshold: int = 4,
                 enabled: bool = True, logger_=None):
        self.out_dir = Path(out_dir)
        self.max_files = max(1, int(max_files))
        self.stuck_threshold = max(1, int(stuck_threshold))
        self.enabled = bool(enabled)
        self.logger = logger_ or logger
        self.detector = StuckDetector()
        self._seen_pairs: Set[Tuple[str, str]] = set()
        self._stuck_saved_for: Optional[bytes] = None
        self.saved = 0
        self.skipped = 0

    # ---------------- 内部 ----------------
    def _prune(self) -> None:
        try:
            files = sorted(self.out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
        except Exception:  # noqa: BLE001
            return
        for p in files[: max(0, len(files) - self.max_files)]:
            try:
                p.unlink()
                p.with_suffix(".txt").unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    def _save(self, img, tag: str, note: str) -> Optional[Path]:
        try:
            import cv2
            self.out_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            base = f"{stamp}_{tag}"
            path = self.out_dir / f"{base}.png"
            ok, buf = cv2.imencode(".png", img)
            if not ok:
                return None
            buf.tofile(str(path))
            path.with_suffix(".txt").write_text(note, encoding="utf-8")
            self.saved += 1
            self._prune()
            self.logger.info(f"【自动取证】已保存 {path.name}（{note.splitlines()[0] if note else ''}）")
            return path
        except Exception as e:  # noqa: BLE001
            self.logger.debug(f"【自动取证】保存失败（忽略）: {e}")
            return None

    @staticmethod
    def _safe(s: str) -> str:
        return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))[:24]

    # ---------------- 主入口 ----------------
    def tick(self, img, *, decision: str = "", route: Any = None, screen: str = "",
             stuck: Optional[int] = None, extra: str = "") -> Optional[Path]:
        """每帧调用一次。返回被保存的文件路径（没存则 None）。"""
        if not self.enabled or img is None or getattr(img, "size", 0) == 0:
            return None

        if stuck is None:
            same, _stuck = self.detector.update(img)
        else:
            _stuck = int(stuck)
            same = _stuck > 0

        route_name = getattr(route, "value", route) or "-"
        note_head = f"screen={screen} route={route_name} decision={decision} stuck={_stuck}"
        if extra:
            note_head += f" | {extra}"

        # ① 卡住取证：同一幕只存一次（用当前指纹做去重）
        if _stuck >= self.stuck_threshold:
            sig = self.detector.sig
            key = sig.tobytes() if sig is not None else None
            if key is not None and key != self._stuck_saved_for:
                self._stuck_saved_for = key
                return self._save(img, f"{self._safe(screen)}_{self._safe(route_name)}_stuck{_stuck}",
                                  note_head + "\n（卡住取证：同一画面连续停留，动作疑似无效）")
            self.skipped += 1
            return None

        # ② 界面清单取证：本场第一次见到这个 (screen, route) 组合
        pair = (str(screen), str(route_name))
        if pair not in self._seen_pairs:
            self._seen_pairs.add(pair)
            return self._save(img, f"{self._safe(screen)}_{self._safe(route_name)}",
                              note_head + "\n（界面清单取证：本场首次出现该组合）")
        self.skipped += 1
        return None

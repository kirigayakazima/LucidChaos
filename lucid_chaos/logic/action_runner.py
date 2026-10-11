"""
统一的「执行框架」：动作 → 效果验证（卡死检测）→ 升级 / 回退。

改造前这套机制散在 4 个地方各写各的：
  - `handle_event` 的搜索气泡升级阶梯（`_EVT_LADDER` + `_event_scene_same`）
  - `handle_epiphany_selection` 的选牌轮换
  - `handle_equip_selection` / `_click_assign_row_then_confirm` 的领奖轮换
  - `handle_event` 的事件框轮换
措辞不同、计数各自维护，还共用同一个 `_evt_stuck`。

本模块把它抽成两个小部件：

  - `StuckDetector` —— 画面指纹卡死检测（64x36 灰度，容差 4.0）。
    语义（与原来完全一致）：和上一帧"同一幕"则卡住轮数 +1，画面内容一变立刻归零。
    「卡住轮数」实际含义 = **这个动作已经试过几次**。
  - `ActionLadder`  —— 按卡住轮数在阶梯里选一级（规则同原来：`min(stuck, n-1)`，封顶不乱试），
    并执行该级动作。

设计约束：**不改变任何既有行为**。所以
  - 索引规则、封顶规则与原代码逐字一致；
  - 日志措辞不由本模块决定，调用方通过 `on_pick` 回调自己写（保证站点措辞不变）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger("ActionRunner")

# 与改造前一致：64x36 灰度指纹，容差 4.0
# （实测同一幕的雾/呼吸动画漂移约 1.2~1.7，4.0 有足够裕量）
SIG_W = 64
SIG_H = 36
SAME_SCENE_TOL = 4.0


class StuckDetector:
    """画面指纹卡死检测。"""

    def __init__(self, tol: float = SAME_SCENE_TOL, sig_w: int = SIG_W, sig_h: int = SIG_H):
        self.tol = tol
        self.sig_w = sig_w
        self.sig_h = sig_h
        self._sig: Optional[np.ndarray] = None
        self._stuck = 0
        self._frames = 0

    # ---------------- 指纹 ----------------
    def signature(self, img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if img is None or getattr(img, "size", 0) == 0:
            return None
        try:
            small = cv2.resize(img, (self.sig_w, self.sig_h), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        except Exception:  # noqa: BLE001
            return None

    # ---------------- 更新 ----------------
    def update(self, img: Optional[np.ndarray]) -> Tuple[bool, int]:
        """
        返回 `(是否与上一帧同幕, 卡住轮数)`，并更新内部状态。

        注意语义与改造前一致：**同一个画面**连续出现时轮数递增；
        画面内容一变（哪怕只是翻了一页）立刻归零。
        """
        sig = self.signature(img)
        if sig is None:
            return False, self._stuck
        same = False
        if self._sig is not None and getattr(self._sig, "shape", None) == sig.shape:
            same = float(np.abs(sig - self._sig).mean()) < self.tol
        self._sig = sig
        self._frames += 1
        self._stuck = (self._stuck + 1) if same else 0
        return same, self._stuck

    @property
    def stuck(self) -> int:
        return self._stuck

    @property
    def sig(self) -> Optional[np.ndarray]:
        """当前帧的指纹（诊断/对照用）。"""
        return self._sig

    @property
    def frames(self) -> int:
        return self._frames

    def reset(self) -> None:
        self._sig = None
        self._stuck = 0


@dataclass
class Attempt:
    """阶梯里的一级：一个标签 + 一个动作。"""
    label: str
    run: Callable[[], None]


class ActionLadder:
    """
    动作阶梯：同一幕连续卡住就逐级升级，画面一变自动回到第 0 级。

    `ctx` 需要有 `.input`（InputController，测试里可打桩）。
    """

    def __init__(self, ctx, steps: Sequence[Attempt], tag: str = "动作",
                 detector: Optional[StuckDetector] = None,
                 logger_: Optional[logging.Logger] = None):
        self.ctx = ctx
        self.steps: List[Attempt] = list(steps)
        self.tag = tag
        self.detector = detector or StuckDetector()
        self.logger = logger_ or logger

    # ---------------- 选级 ----------------
    @staticmethod
    def index_for(stuck: int, n_steps: int) -> int:
        """与改造前逐字一致：`min(stuck, n-1)`，封顶在最后一级不再乱试。"""
        if n_steps <= 0:
            return 0
        return max(0, min(int(stuck), n_steps - 1))

    @property
    def n(self) -> int:
        return len(self.steps)

    def attempt_for(self, stuck: int) -> Tuple[int, Attempt]:
        idx = self.index_for(stuck, len(self.steps))
        return idx, self.steps[idx]

    # ---------------- 执行 ----------------
    def run(self, idx: int) -> str:
        """执行第 idx 级（越界自动夹紧），返回该级标签。"""
        idx = max(0, min(int(idx), len(self.steps) - 1))
        step = self.steps[idx]
        step.run()
        return step.label

    def escalate(self, stuck: int, on_pick: Optional[Callable[[int, str], None]] = None) -> Tuple[int, str]:
        """
        按卡住轮数选一级并执行。

        `on_pick(idx, label)` 会在执行前被调用 —— 调用方用它写**自己站点措辞**的日志
        （本框架不掺和措辞，避免改掉既有日志）。
        """
        idx, step = self.attempt_for(stuck)
        if on_pick is not None:
            on_pick(idx, step.label)
        step.run()
        return idx, step.label

    # ---------------- 便捷：观察 + 升级 ----------------
    def observe_and_escalate(self, img, on_pick: Optional[Callable[[int, str], None]] = None) -> Tuple[bool, int, str]:
        """先更新卡死检测，再按结果升级。返回 `(是否同幕, idx, label)`。"""
        same, stuck = self.detector.update(img)
        idx, label = self.escalate(stuck, on_pick=on_pick)
        return same, idx, label

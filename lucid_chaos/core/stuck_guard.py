"""
全局卡死守卫（GlobalStuckGuard）—— 跨 handler 地发现"原地打转"并逐级自救。

主人反馈（2026/10/6）："工具能跑全流程，但很容易断；断了只能手动 Ctrl+C，因为有时死循环。"

现状缺口：系统**其实知道**自己在打转——自动取证就存下了 `_stuck4` … `_stuck18`（连续 18 轮同一画面），
但这个信号只用来存图，没有任何人处理它。各 handler 自己那套 `ActionLadder` 只管自己那一步，
一旦"点 A 又回到 A"（例如抽屉反复被节点点击重新打开），没人拦。

本守卫做三件事：
  1. **发现**：同一画面指纹连续停留 >= `warn` 轮，或者**同一个动作标签连续重复 >=3 次**（强循环信号）
  2. **升级**：按级别做越来越强的自救（清防抖 -> 发 ESC -> 点返回/关闭 -> 兜底）
  3. **兜底**：到 `stop` 轮仍无变化 -> **安全停止**并打印"卡住报告"（含取证图路径），
     主人不用再 Ctrl+C，而且知道该把哪张图发给开发者

设计原则：
  - 每一级**只做一次**（不会疯狂发 ESC），升级后有冷却
  - 画面一旦变化立刻整体归零（真的推进了就完全放手）
  - 计时/计数都是纯状态，便于单测
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..logic.action_runner import StuckDetector

logger = logging.getLogger("StuckGuard")


LEVEL_NAMES = {
    0: "正常",
    1: "重置判定状态（清防抖 + 清各 handler 阶梯）",
    2: "发送 ESC 退回上一层",
    3: "ESC + 点击右上角关闭/返回区域",
    4: "安全停止并输出卡住报告",
}


@dataclass
class StuckDecision:
    level: int
    reason: str
    stuck_rounds: int
    route: str = ""
    action: str = ""
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def description(self) -> str:
        return LEVEL_NAMES.get(self.level, f"级别{self.level}")


class GlobalStuckGuard:
    def __init__(self, *, warn: int = 4, esc: int = 6, close: int = 9, stop: int = 14,
                 repeat_action_limit: int = 3, enabled: bool = True, logger_=None,
                 slow_routes: tuple = ("剧情对白演出",)):
        self.warn = max(1, int(warn))
        self.esc = max(1, int(esc))
        self.close = max(1, int(close))
        self.stop = max(0, int(stop))                 # 0 = 不自动停止
        self.repeat_action_limit = max(2, int(repeat_action_limit))
        self.enabled = bool(enabled)
        self.logger = logger_ or logger
        # 「慢场景」白名单：这些界面里"画面长时间不变"是**正常**的（过场动画/演出），
        # 允许自救（清粘连、ESC），但**绝不自动停止**脚本。
        self.slow_routes = tuple(slow_routes or ())

        self.detector = StuckDetector()
        self.same_rounds = 0                 # 本画面连续停留轮数（含本轮，从 1 开始）
        self._last_action = ""
        self._repeat_action = 0              # 同一动作连续重复次数（与画面是否变化无关）
        self._episode_done: set = set()      # 本回合已经用过的级别
        self._last_level = 0

    # 兼容旧名字（日志/测试里用过 stuck_rounds）
    @property
    def stuck_rounds(self) -> int:
        return self.same_rounds

    # ---------------- 内部 ----------------
    def _reset_episode(self, *, clear_repeat: bool = True) -> None:
        self.same_rounds = 0
        if clear_repeat:
            self._repeat_action = 0
        self._episode_done.clear()
        self._last_level = 0

    def _decide(self, effective: int, route: str, action: str, reason: str) -> Optional[StuckDecision]:
        """根据"有效停留轮数"决定该升到哪一级（每级只做一次）。"""
        if not self.enabled:
            return None
        target = 0
        if self.stop and effective >= self.stop:
            target = 4
        elif effective >= self.close:
            target = 3
        elif effective >= self.esc:
            target = 2
        elif effective >= self.warn:
            target = 1
        if target == 4 and route in self.slow_routes:
            # 过场/对白里画面不变是正常的 -> 自救到级别 3 为止，**不自动停止**
            target = 3
            reason = f"{reason}（慢场景[{route}]，不自动停止）"
        if target == 0 or target in self._episode_done:
            return None
        # 允许一次跨多级（例如卡很久后直接跳到停止级）
        for lvl in range(1, target + 1):
            self._episode_done.add(lvl)
        self._last_level = target
        return StuckDecision(level=target, reason=reason, stuck_rounds=effective,
                             route=route, action=action)

    # ---------------- 主入口 ----------------
    def update(self, img, action: str = "", route: str = "") -> Optional[StuckDecision]:
        """
        每轮调用一次。返回需要执行的升级动作（None = 不用管）。
        `action` 是本轮实际做了什么。

        判定用的"有效轮数" = max(同画面连续轮数, 同动作连续次数+1)：
          - 画面**完全没变** -> 同画面轮数增长
          - 画面在**微动**（动画/抖动，指纹判"变了"）但**动作一直是同一个** -> 靠动作重复次数兜住
        只有"画面变了 **且** 动作也变了"才算真的推进，此时完全放手。
        """
        if not self.enabled or img is None or getattr(img, "size", 0) == 0:
            return None

        same, _stuck = self.detector.update(img)

        prev_action = self._last_action
        repeating = bool(action) and action == prev_action
        if repeating:
            self._repeat_action += 1
        else:
            self._repeat_action = 0
        self._last_action = action

        if not same and not repeating:
            # 画面变了 + 动作也变了 => 真的推进了，完全放手
            if self._last_level:
                self.logger.info(f"【卡死守卫】画面与动作都已变化，解除守卫（上一级别 {self._last_level}）")
            self._reset_episode()
            return None

        if same:
            self.same_rounds += 1

        effective = self.same_rounds
        reason = ""
        if self.same_rounds >= self.warn:
            reason = f"同一画面连续 {self.same_rounds} 轮无变化"
        if self._repeat_action >= self.repeat_action_limit:
            eff_rep = self._repeat_action + 1
            if eff_rep >= effective:
                effective = eff_rep
                reason = f"同一动作连续重复 {eff_rep} 次（画面在微动）：[{action}]"
        if not reason or effective < self.warn:
            return None
        return self._decide(effective, route, action, reason)

"""
Phase 5: Screen State and GameState Definitions
Defines structured enum and dataclass for screen understanding.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import time
from ..core.vision_service import VisionTarget

class ScreenState(Enum):
    UNKNOWN = auto()
    OPENING_REROLL = auto()      # 开局物主代理人 / 选择卡牌装备
    MAP_NAVIGATION = auto()      # 迷宫地图路线规划
    BATTLE = auto()              # 战斗中 (回合 / EP / 技能卡牌)
    SHOP = auto()                # 德朗商店 (购买 / 删卡)
    CAMP = auto()                # 安全区营地 (休息 / 训练)
    EVENT = auto()               # 问号神秘事件
    EQUIP_SELECTION = auto()     # 获得装备 / 战斗员装备选择
    EPIPHANY_SELECTION = auto()  # 灵光一闪卡牌效果选择
    VICTORY_SETTLEMENT = auto()  # 战斗胜利 / 结算确认
    ESCAPE_CONFIRM = auto()      # 逃脱 / 离开迷宫确认弹窗
    DIALOG = auto()              # 剧情对话 / NPC 交互
    DISCONNECTED = auto()        # 断线重连提示窗口

@dataclass
class GameState:
    screen: ScreenState
    confidence: float
    raw_texts: List[str] = field(default_factory=list)
    elements: Dict[str, VisionTarget] = field(default_factory=dict)
    # OCR 命中项的**完整列表**（elements 是按 text 去重的 dict，重名项会被覆盖：
    # 例如「卡牌獎勵→選戰鬥員」界面三个一模一样的「等級」标签只会剩最后一个，
    # 靠 texts 数行数就会数错。需要「有几行/几个同类元素」时必须用 targets）
    targets: List[VisionTarget] = field(default_factory=list)
    # 本帧的 FrameContext（图像 + OCR + 共用模板库 + 亮度归一化），由 orchestrator 挂上
    frame: Any = None
    # router 的单一场景判定结论（SceneType + 完整 SceneResult，含各候选分数=证据）
    route: Any = None
    scene_result: Any = None
    # 本帧最终决策说明：`route:局内互动事件` / `safety:bottom_center_button` / `handler:...`
    decision: str = ""
    timestamp: float = field(default_factory=time.time)

    def has_element(self, name: str) -> bool:
        return name in self.elements

    def get_element(self, name: str) -> Optional[VisionTarget]:
        return self.elements.get(name)

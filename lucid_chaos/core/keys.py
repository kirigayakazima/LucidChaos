"""
按键表（来自游戏「快捷鍵設定」页 + 界面上自带的按键提示）。

为什么单独成文件：主人 2026/10/6 确认"键盘更准"，而且这游戏**把快捷键直接画在按钮/卡牌上**：
  - 战斗：每张手牌角上有 1/2/3/4（= 使用N號卡牌）；F1/F2/F3 = 自我意識技能 / N號互動；
          右下 E = 結束回合；右上 R = 自動戰鬥、Q = 快捷選單
  - 战后结算：左 `F1 跳過`、右 `ENTER 獲得`
  - 灵光一闪选牌：三张牌上分别标 1/2/3

所以"用键盘"不只是更快，而是**能直接照抄游戏自己的说明**，比猜坐标可靠得多。

⚠️ 同一个键在不同界面含义不同（数字键/F 键在非战斗界面是"任務/出擊/模擬/活動/補給…"），
   因此调用方必须**先由路由层确认场景**再发键（见 router 的场景判定）。
"""

from __future__ import annotations

from typing import Dict, Optional

import win32con

# 名字 -> 虚拟键码
KEYS: Dict[str, int] = {
    # 通用
    "ENTER": win32con.VK_RETURN,
    "SPACE": win32con.VK_SPACE,
    "ESC": win32con.VK_ESCAPE,
    "TAB": win32con.VK_TAB,
    "CTRL": win32con.VK_CONTROL,
    "SHIFT": win32con.VK_SHIFT,
    "CAPS": win32con.VK_CAPITAL,
    "UP": win32con.VK_UP,
    "DOWN": win32con.VK_DOWN,
    "LEFT": win32con.VK_LEFT,
    "RIGHT": win32con.VK_RIGHT,
    "END": win32con.VK_END,
    # 字母
    "A": 0x41, "B": 0x42, "C": 0x43, "D": 0x44, "E": 0x45, "F": 0x46, "G": 0x47,
    "H": 0x48, "I": 0x49, "J": 0x4A, "K": 0x4B, "L": 0x4C, "M": 0x4D, "N": 0x4E,
    "O": 0x4F, "P": 0x50, "Q": 0x51, "R": 0x52, "S": 0x53, "T": 0x54, "U": 0x55,
    "V": 0x56, "W": 0x57, "X": 0x58, "Y": 0x59, "Z": 0x5A,
    # 数字（0 在战斗里表示"第10张卡牌"）
    "0": 0x30, "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34,
    "5": 0x35, "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39,
    # 功能键
    "F1": win32con.VK_F1, "F2": win32con.VK_F2, "F3": win32con.VK_F3, "F4": win32con.VK_F4,
    "F5": win32con.VK_F5, "F6": win32con.VK_F6, "F7": win32con.VK_F7, "F8": win32con.VK_F8,
    "F9": win32con.VK_F9,
}

# 「界面上的按键提示」用的别名（OCR 可能读到这些写法）
ALIASES: Dict[str, str] = {
    "回车": "ENTER", "确认": "ENTER", "確認": "ENTER", "ENTER": "ENTER",
    "空格": "SPACE", "对话": "SPACE", "對話": "SPACE", "SPACE": "SPACE",
    "取消": "ESC", "返回": "ESC", "ESC": "ESC",
    "跳过": "Z", "跳過": "Z", "Z": "Z",
    "倍速": "CTRL", "切换倍速": "CTRL", "切換倍速": "CTRL", "CTRL": "CTRL",
    "结束回合": "E", "結束回合": "E", "E": "E",
    "自动战斗": "R", "自動戰鬥": "R", "R": "R",
    "快捷菜单": "Q", "快捷選單": "Q", "Q": "Q",
    "隐藏ui": "U", "隱藏UI": "U", "U": "U",
    "地图": "M", "地圖": "M", "M": "M",
    "查看内容": "CAPS", "查看內容": "CAPS", "CAPS": "CAPS",
    "祈愿": "K", "祈願": "K", "K": "K",
    "选择卡牌使用目标": "TAB", "選擇卡牌使用目標": "TAB", "TAB": "TAB",
}


def resolve(name) -> Optional[int]:
    """把按键名（或已是 vk 的整数）解析成虚拟键码；未知返回 None。"""
    if isinstance(name, int):
        return name
    if not name:
        return None
    key = str(name).strip().upper()
    if key in KEYS:
        return KEYS[key]
    alias = ALIASES.get(str(name).strip()) or ALIASES.get(key)
    if alias:
        return KEYS.get(alias)
    return None


# -------------------------------------------------------------
# 语义动作 -> 按键（本游戏实测/官方表；调用方仍须先确认场景）
# -------------------------------------------------------------
ACTION_KEYS: Dict[str, str] = {
    "advance_dialogue": "SPACE",     # 對話
    "skip_cutscene": "Z",            # 跳過
    "toggle_speed": "CTRL",          # 切換倍速（切换型！）
    "confirm": "ENTER",              # 確認
    "cancel": "ESC",                 # 取消/返回
    "end_turn": "E",                 # 結束回合
    "auto_battle": "R",              # 開啟/關閉自動戰鬥（切换型！）
    "quick_menu": "Q",               # 快捷選單
    "hide_ui": "U",                  # 隱藏UI
    "map": "M",                      # 地圖
    "next_page": "TAB",              # 選擇卡牌使用目標 / 翻页
    "wish": "K",                     # 祈願
    "deck": "A",                     # 抽牌堆
    "discard": "S",                  # 墳墓
    "all_cards": "D",                # 所有卡牌
}


def action_key(action: str) -> Optional[int]:
    """把语义动作（如 'end_turn'）解析成虚拟键码。"""
    return resolve(ACTION_KEYS.get(action, ""))


def card_key(index_1based: int) -> Optional[int]:
    """
    第 N 张手牌对应的按键：1..9 -> '1'..'9'，10 -> '0'（游戏里 0 = 使用10號卡牌）。
    """
    if index_1based < 1:
        return None
    if index_1based == 10:
        return KEYS["0"]
    if index_1based > 10:
        return None
    return KEYS[str(index_1based)]

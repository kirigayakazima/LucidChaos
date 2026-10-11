"""
按键通路探针 —— 30 秒验证「PostMessage 键盘消息能不能被游戏接受」。

    python -m lucid_chaos.tools.key_probe SPACE
    python -m lucid_chaos.tools.key_probe Z
    python -m lucid_chaos.tools.key_probe CTRL
    python -m lucid_chaos.tools.key_probe F1
    python -m lucid_chaos.tools.key_probe --list

它做的事：找到游戏窗口 -> 倒计时 3 秒（你切回游戏画面看效果）-> **只投递一次按键消息**
（不解锁前台、不动真实键盘、不动光标）-> 打印结果提示。

为什么要它：游戏支持键盘（官方快捷键表），而"投递消息"这条路**不占主人前台**。
如果这一步有效，那么"跳过/倍速/推进对白/结束回合/按编号出牌/F1-F3 交互"全都可以用按键做，
比鼠标点击稳得多（尤其是场景里的世界空间交互，鼠标点不动）。

⚠️ 风险提示（每个键在原游戏里都有含义，请对着画面确认再按）：
    SPACE  對話（推进对白）      Z  跳過（会跳过剧情！）      CTRL 切換倍速（是"切换"，会来回切）
    ENTER  確認                  ESC 取消/返回                E    結束回合
    1~9,0  使用第N張卡牌（非战斗界面另有含义：任務/出擊/模擬…）
    F1/F2/F3 自我意識技能 / N號互動（很可能就是场景里的交互键）
"""

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import win32con

from lucid_chaos.core.window import GameWindow
from lucid_chaos.core.input import InputController

# 名字 -> (虚拟键码, 说明, 风险)
KEYS = {
    "SPACE": (win32con.VK_SPACE, "對話：推进对白", "低"),
    "ENTER": (win32con.VK_RETURN, "確認", "低"),
    "ESC": (win32con.VK_ESCAPE, "取消 / 返回上一层", "低"),
    "Z": (0x5A, "跳過：跳过剧情", "中（会跳剧情）"),
    "CTRL": (win32con.VK_CONTROL, "切換倍速（1X/2X 来回切）", "中（切换型）"),
    "E": (0x45, "結束回合", "低（战斗内）"),
    "Q": (0x51, "快捷選單", "中"),
    "TAB": (win32con.VK_TAB, "選擇卡牌使用目標", "中"),
    "R": (0x52, "開啟/關閉自動戰鬥", "中（切换型）"),
    "M": (0x4D, "地圖", "中"),
    "U": (0x55, "隱藏UI", "低"),
    "F1": (win32con.VK_F1, "1號自我意識技能 / 1號互動", "中（非战斗界面是'活動'）"),
    "F2": (win32con.VK_F2, "2號自我意識技能 / 2號互動", "中（非战斗界面是'補給'）"),
    "F3": (win32con.VK_F3, "3號自我意識技能 / 3號互動", "中（非战斗界面是'日常任務'）"),
    "1": (0x31, "使用1號卡牌 / 選擇1號", "中（非战斗界面是'任務（故事）'）"),
    "2": (0x32, "使用2號卡牌 / 選擇2號", "中（非战斗界面是'出擊'）"),
    "3": (0x33, "使用3號卡牌 / 選擇3號", "中（非战斗界面是'模擬'）"),
}


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help", "--list"):
        print(__doc__)
        print("可用按键：")
        for k, (vk, desc, risk) in KEYS.items():
            print(f"   {k:<6} vk={vk:<4} {desc}  风险={risk}")
        return 0

    name = argv[1].upper()
    if name not in KEYS:
        print(f"未知按键 {name!r}；用 --list 看可用列表")
        return 2
    vk, desc, risk = KEYS[name]

    win = GameWindow()
    if not win.find_window():
        print("没找到游戏窗口《卡厄思梦境》，请先开着游戏")
        return 3
    print(f"游戏窗口 OK: hwnd={win.hwnd}  client={win.get_client_rect()}")
    print(f"将要投递按键: {name} (vk={vk}) —— {desc}  风险={risk}")
    print("注意：只投递消息，**不会激活游戏窗口、不动你的键盘和光标**")
    for i in (3, 2, 1):
        print(f"   {i}...（切到游戏画面看效果）")
        time.sleep(1.0)

    ctrl = InputController(win)
    ctrl.send_key(vk, delay_after=0.2, hold_ms=60, label=f"探针 {name}")
    print(f"已投递一次 {name}。请看游戏有没有反应：")
    print("   有反应 -> PostMessage 键盘通路有效，我们可以大量改用按键（跳过/倍速/出牌/交互…）")
    print("   没反应 -> 该游戏键盘走的是 Raw Input 或要求前台，按键只能在你聚焦游戏时用")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

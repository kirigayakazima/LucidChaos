"""
全局卡死守卫单测：

    python -m lucid_chaos.tests.test_stuck_guard

背景（主人反馈）："工具能跑全流程，但很容易断；断了只能手动 Ctrl+C，因为有时死循环。"
守卫要做的事：发现原地打转 -> 逐级自救 -> 实在不行**安全停止**并给卡住报告。

验证：
  1. 画面一直不变：第 4 轮报警(级别1) -> 6 轮 ESC(2) -> 9 轮强退(3) -> 14 轮停止(4)，且**每级只触发一次**
  2. 画面一变立刻归零（真的推进了就放手）
  3. 同一动作连续重复 3 次（即使画面有细微变化）也要报警
  4. stop=0 时永不自动停止
  5. 关闭开关后什么都不做
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import cv2
import numpy as np

from lucid_chaos.core.stuck_guard import GlobalStuckGuard

FIXTURES = _HERE.parent / "fixtures"
FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


STUCK = imread(FIXTURES / "stage_map_drawer_text_trap.png")
MOVED = imread(FIXTURES / "stage_map_chapter1.png")

print("=== 1) 画面一直不变：级别序列必须是 1,2,3,4 且每级只一次 ===")
g = GlobalStuckGuard(warn=4, esc=6, close=9, stop=14)
levels = []
for i in range(18):
    d = g.update(STUCK, action="click_client(1122,776) 【外层选关】可推进关卡", route="外层选关")
    if d:
        levels.append((i + 1, d.level, d.reason[:24]))
print("   触发序列(轮次, 级别):", [(r, l) for r, l, _ in levels])
check("级别序列 = [1,2,3,4]", [l for _, l, _ in levels] == [1, 2, 3, 4], f"{levels}")
check("第 4 轮报警(级别1)", levels and levels[0] == (4, 1, levels[0][2]), f"{levels[0] if levels else None}")
check("第 6 轮 ESC(级别2)", len(levels) > 1 and levels[1][0] == 6 and levels[1][1] == 2)
check("第 9 轮强退(级别3)", len(levels) > 2 and levels[2][0] == 9 and levels[2][1] == 3)
check("第 14 轮停止(级别4)", len(levels) > 3 and levels[3][0] == 14 and levels[3][1] == 4)
check("每级只触发一次（不会疯狂发 ESC）", len(levels) == 4, f"{len(levels)} 次")

print("\n=== 2) 画面一变立刻归零 ===")
g = GlobalStuckGuard(warn=4, esc=6, close=9, stop=14)
for _ in range(5):
    g.update(STUCK, action="A", route="X")
check("卡到 5 轮（已报警 1 次）", g.stuck_rounds >= 4, f"stuck={g.stuck_rounds}")
d = g.update(MOVED, action="B", route="X")
check("换画面 -> 当轮不升级", d is None)
check("计数归零", g.stuck_rounds == 0, f"stuck={g.stuck_rounds}")
seq = [g.update(STUCK, action="A", route="X") for _ in range(5)]
check("归零后重新计数（第 4 轮再次报警）",
      [s.level for s in seq if s] == [1], f"{[s.level for s in seq if s]}")

print("\n=== 3) 同一动作连续重复 3 次也要报警（画面只有细微变化）===")
g = GlobalStuckGuard(warn=4, esc=6, close=9, stop=14, repeat_action_limit=3)
noise = STUCK.copy()
hits = []
for i in range(4):
    noise = (STUCK.astype(np.int16) + (i % 2) * 12).clip(0, 255).astype(np.uint8)   # 细微变化 -> 指纹判定"变了"
    d = g.update(noise, action="click_client(1122,776) 【外层选关】可推进关卡", route="外层选关")
    if d:
        hits.append((i + 1, d.level, d.reason))
print("   命中:", hits)
check("重复动作触发报警", any("重复" in r for _, _, r in hits), f"{hits}")

print("\n=== 4) stop=0 -> 永不自动停止 ===")
g = GlobalStuckGuard(warn=4, esc=6, close=9, stop=0)
lv = [d.level for d in (g.update(STUCK, action="A") for _ in range(30)) if d]
check("最高只到级别 3", lv == [1, 2, 3], f"{lv}")

print("\n=== 5) 关闭开关 -> 永远返回 None ===")
g = GlobalStuckGuard(enabled=False)
check("全 None", all(g.update(STUCK, action="A") is None for _ in range(20)))

print("\n=== 6) 空图/异常输入不崩 ===")
g = GlobalStuckGuard()
check("None 不崩", g.update(None) is None)
check("空数组不崩", g.update(np.zeros((0, 0, 3), dtype=np.uint8)) is None)

print("\n=== 7) 慢场景（对白）允许自救但不自动停止 ===")
g = GlobalStuckGuard(warn=4, esc=6, close=9, stop=14)
seq = [(i + 1, d.level) for i in range(30)
       for d in [g.update(STUCK, action="click_rel(0.92,0.85) 【主线剧情】推进台词", route="剧情对白演出")] if d]
print("   对白场景触发:", seq)
check("对白场景只触发 1/2/3 各一次（不会每轮重复 ESC）",
      [l for _, l in seq] == [1, 2, 3], f"{seq}")
g2 = GlobalStuckGuard(warn=4, esc=6, close=9, stop=14, slow_routes=())
seq2 = [(i + 1, d.level) for i in range(30)
        for d in [g2.update(STUCK, action="A", route="剧情对白演出")] if d]
check("把 slow_routes 清空后仍然会自动停止（可配置）", [l for _, l in seq2] == [1, 2, 3, 4], f"{seq2}")

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)

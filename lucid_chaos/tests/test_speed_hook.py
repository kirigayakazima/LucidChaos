"""
「局内常驻 2X 加速」单测（主人要求：局内也要自动激活右上角加速按钮）：

    python -m lucid_chaos.tests.test_speed_hook

验证：
  1. 局内事件图（⏩ 未点亮）-> 真的去点，坐标落在 (0.865~0.895, 0.035~0.068)
  2. 肉鸽选路图（没有这个按钮）-> 不点
  3. 场景白名单：对白/战斗/外层选关/未知 **不**走这道补刀
     （对白由 handle_cutscene_dialog 自己点、战斗 AUTO 由 BattleManager 管；
       重复点同一个开关会把它**关掉**）
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

from lucid_chaos.logic.story_pusher import StoryPusher
from lucid_chaos.router.scene_types import SceneType, should_activate_speed

FIXTURES = _HERE.parent / "fixtures"
FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


class Rec:
    def __init__(self):
        self.calls = []

    def click_area_rel(self, a, b, c, d, **k):
        self.calls.append((round((a + c) / 2, 3), round((b + d) / 2, 3), k.get("label")))

    def click_client(self, cx, cy, **k):
        self.calls.append((int(cx), int(cy), k.get("label")))

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        def _rec(*a, **k):
            self.calls.append((None, None, k.get("label")))
        return _rec


class Ctx:
    vision = None
    overlay = None
    capture = None

    def __init__(self):
        self.input = Rec()


print("=== 1) 局内事件图：应当点加速 ===")
ctx = Ctx()
pusher = StoryPusher(ctx, {})
did = pusher.ensure_speed_2x(imread(FIXTURES / "event_phantom_choice.png"))
print("   调用结果:", did, " 动作:", ctx.input.calls)
check("返回 True（点了）", did is True)
check("确实发出一次点击", len(ctx.input.calls) == 1, f"{ctx.input.calls}")
check("坐标在右上角加速按钮内",
      ctx.input.calls and 0.86 <= ctx.input.calls[0][0] <= 0.90 and 0.03 <= ctx.input.calls[0][1] <= 0.07,
      f"{ctx.input.calls[0] if ctx.input.calls else None}")

print("\n=== 2) 没有该按钮的画面：不点 ===")
for name in ("rogue_map_real.png", "settlement_stage_reward.png", "card_reward_3choose.png"):
    ctx = Ctx()
    p = StoryPusher(ctx, {})
    did = p.ensure_speed_2x(imread(FIXTURES / name))
    check(f"{name} 不点", did is False and ctx.input.calls == [], f"{did} {ctx.input.calls}")

print("\n=== 3) 场景白名单 ===")
check("局内互动事件 -> 走补刀", should_activate_speed(SceneType.EVENT_CHOICE) is True)
check("灵光一闪选牌 -> 走补刀", should_activate_speed(SceneType.EPIPHANY_CARDS) is True)
check("肉鸽选路 -> 走补刀", should_activate_speed(SceneType.ROGUELIKE_MAP) is True)
check("对白 -> **不**走（对白 handler 自己点）", should_activate_speed(SceneType.CUTSCENE_DIALOG) is False)
check("战斗 -> **不**走（AUTO 归 BattleManager）", should_activate_speed(SceneType.BATTLE) is False)
check("外层选关 -> 不走", should_activate_speed(SceneType.STAGE_MAP) is False)
check("未知 -> 不走", should_activate_speed(SceneType.UNKNOWN) is False)
check("None -> 不走", should_activate_speed(None) is False)

print("\n=== 4) 对白图仍由对白 handler 负责（不重复点）===")
ctx = Ctx()
p = StoryPusher(ctx, {})
p.ensure_speed_2x(imread(FIXTURES / "dialogue_lei.png"))
print(f"   对白图上 ensure_speed_2x 点击次数={len(ctx.input.calls)}（该函数本身会点；"
      f"是否需要它由第 3 步的白名单决定）")

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)

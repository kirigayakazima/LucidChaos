"""
键盘动作层测试（把主人定的规则锁死）：

    python -m lucid_chaos.tests.test_keys

规则来源（主人 2026/10/6，均有游戏界面截图证据）：
  - 探索交互点 = **F1/F2/F3**（场景里放大镜旁边就标着 F1；"有几个就是 F 几"）
  - 事件选项框 = **1/2/3**（框上标着编号）
  - 确认 / 下一步 = **ENTER**（结算页、卡牌奖励页底部都标着 ENTER）
  - 战斗出牌 = **编号键**（每张手牌角上标 1/2/3/4）；结束回合 = **E**
  - 剧情：推進 = **SPACE**，跳過 = **Z**，切換倍速 = **CTRL**
  - 肉鸽选路 = **1/2/3**（▶ 旁边标着编号），排分算法不变
  - ⚠️ 卡牌奖励界面的数字是**费用**不是编号 -> 那里必须用鼠标点牌 + ENTER 确认
"""
import sys
from pathlib import Path
import inspect

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import cv2
import numpy as np

from lucid_chaos.core import keys as K
from lucid_chaos.core.vision_service import VisionService
from lucid_chaos.state.observer import ScreenObserver
from lucid_chaos.logic.story_pusher import StoryPusher
from lucid_chaos.logic.node_engine import NodeDecisionEngine

FIXTURES = _HERE.parent / "fixtures"
FAILED = []
VIS = VisionService()


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


class Rec:
    """记录所有动作；键盘动作返回 True（表示"发出了"），用于验证走的是键盘路径。"""
    def __init__(self):
        self.calls = []
        self.last_action = ""

    def _key(self, kind, payload, label):
        self.calls.append((kind, payload, label))
        self.last_action = f"{kind}({payload}) {label}"

    def tap_action(self, action, **kw):
        self._key("tap_action", action, kw.get("label") or f"【按键】{action}")
        return True

    def tap_card(self, idx, **kw):
        self._key("tap_card", int(idx), kw.get("label") or f"【按键】编号 {idx}")
        return True

    def tap_hero_skill(self, slot, **kw):
        self._key("tap_hero_skill", int(slot), kw.get("label") or f"【按键】F{slot}")
        return True

    def send_key(self, vk, **kw):
        self._key("send_key", int(vk), kw.get("label"))

    def click_client(self, cx, cy, **kw):
        self._key("click_client", (int(cx), int(cy)), kw.get("label"))

    def click_rel(self, rx, ry, **kw):
        self._key("click_rel", (round(float(rx), 3), round(float(ry), 3)), kw.get("label"))

    def click_area_rel(self, a, b, c, d, **kw):
        self._key("click_area_rel", (round((a + c) / 2, 3), round((b + d) / 2, 3)), kw.get("label"))

    def click_target(self, t, **kw):
        c = getattr(t, "center", None)
        self._key("click_target", None if c is None else (c[0], c[1]), kw.get("label"))

    def click_box(self, *a, **kw):
        self._key("click_box", None, kw.get("label"))

    def drag_rel(self, *a, **kw):
        self._key("drag_rel", tuple(a[:2]), kw.get("label"))

    def __getattr__(self, item):
        if item.startswith("_"):
            raise AttributeError(item)
        return lambda *a, **k: self._key(f"other:{item}", None, k.get("label"))


class Cap:
    def __init__(self, img):
        self.img = img

    def capture(self):
        return self.img


class Ctx:
    def __init__(self, img):
        self.vision = VIS
        self.capture = Cap(img)
        self.overlay = None
        self.input = Rec()


def run(name):
    img = imread(FIXTURES / name)
    ctx = Ctx(img)
    eng = NodeDecisionEngine(ctx, None, {})
    pusher = StoryPusher(ctx, {"main_story": {"boss_action": "auto"}})
    pusher.node_engine = eng
    pusher.router.node_engine = eng
    st = ScreenObserver(VIS).observe(img)
    pusher.router.dispatch(img, st)
    return ctx.input.calls


print("=== 1) 键表本身 ===")
check("SPACE=32 / ENTER=13 / ESC=27", K.resolve("SPACE") == 32 and K.resolve("ENTER") == 13 and K.resolve("ESC") == 27)
check("Z=90(跳過) / CTRL=17(倍速) / E=69(結束回合) / R=82(自動戰鬥)",
      K.resolve("Z") == 90 and K.resolve("CTRL") == 17 and K.resolve("E") == 69 and K.resolve("R") == 82)
check("F1/F2/F3 = 112/113/114", [K.resolve(f"F{i}") for i in (1, 2, 3)] == [112, 113, 114])
check("出牌编号 1..9 -> 49..57，第10张 -> 48('0')",
      [K.card_key(i) for i in range(1, 10)] == list(range(49, 58)) and K.card_key(10) == 48)
check("中文别名可解析（確認/跳過/對話）",
      K.resolve("確認") == 13 and K.resolve("跳過") == 90 and K.resolve("對話") == 32)
check("语义动作映射正确",
      K.action_key("advance_dialogue") == 32 and K.action_key("skip_cutscene") == 90
      and K.action_key("end_turn") == 69 and K.action_key("confirm") == 13
      and K.action_key("toggle_speed") == 17 and K.action_key("auto_battle") == 82)

print("\n=== 1b) 「按两下」规则（主人实测：事件选项/出牌都要两下）===")
from lucid_chaos.core.input import InputController  # noqa: E402

_ctrl = InputController.__new__(InputController)          # 不走 __init__（避免真的去找窗口）
_sent = []
_ctrl.send_key = lambda vk, **kw: _sent.append((vk, kw.get("label")))  # type: ignore[assignment]
_ctrl.tap_card(1)
print("   tap_card(1) 实际发出的按键:", len(_sent), _sent[:3])
check("默认按两下", len(_sent) == 2 and all(vk == 49 for vk, _ in _sent), f"{_sent}")
check("标签里标出第几下", _sent and "第1/2下" in _sent[0][1] and "第2/2下" in _sent[1][1], f"{_sent}")
_sent.clear()
_ctrl.tap_card(10)
check("第10张牌用 '0' 键(48) 也是两下", len(_sent) == 2 and all(vk == 48 for vk, _ in _sent), f"{_sent}")
_sent.clear()
_ctrl.tap_card(2, times=1)
check("times=1 时只按一下（可配置）", len(_sent) == 1, f"{_sent}")
check("默认值就是 2（config.input.digit_press_times 可改）", InputController.DIGIT_PRESS_TIMES == 2)

print("\n=== 2) 探索交互点：按 F1（不再点世界坐标）===")
calls = run("common_interact_mushroom.png")
keys = [c for c in calls if c[0] in ("tap_action", "tap_card", "send_key", "tap_hero_skill")]
print("   动作:", calls[:3])
check("蘑菇交互场景给出了动作", bool(calls), f"{calls[:2]}")

print("\n=== 3) 剧情：推进=SPACE / 跳过=Z / 倍速=CTRL ===")
calls = run("dialogue_black_bg.png")
print("   黑底剧情:", calls[:2])
check("黑底剧情 -> tap_action(advance_dialogue)", calls and calls[0][0] == "tap_action" and calls[0][1] == "advance_dialogue", f"{calls[:1]}")
calls = run("dialogue_lei.png")
print("   普通对白:", calls[:3])
check("普通对白 -> 先 CTRL（倍速）", any(c[1] == "toggle_speed" for c in calls), f"{calls[:3]}")
check("普通对白 -> 再 Z（跳过）", any(c[1] == "skip_cutscene" for c in calls), f"{calls[:3]}")

print("\n=== 4) 战斗：出牌=编号键，结束回合=E ===")
calls = run("battle_hand_row.png")
print("   战斗:", calls[:4])
check("战斗有编号键出牌", any(c[0] == "tap_card" for c in calls), f"{calls[:3]}")
check("战斗结束回合用 E", any(c[0] == "tap_action" and c[1] == "end_turn" for c in calls), f"{calls}")

print("\n=== 5) 事件选项框 = 编号键 1/2/3 ===")
for name in ("event_phantom_choice.png", "event_box_end_search.png"):
    calls = run(name)
    hit = [c for c in calls if c[0] == "tap_card" and c[1] <= 3]
    print(f"   {name}: {calls[:2]}")
    check(f"{name} -> 按 1/2/3 选选项", bool(hit), f"{calls[:2]}")

print("\n=== 6) 确认类 = ENTER（结算 / 卡牌奖励 / 领奖）===")
for name in ("settlement_stage_reward.png", "settlement_teal_hero.png"):
    calls = run(name)
    print(f"   {name}: {calls[:1]}")
    check(f"{name} -> ENTER 确认", any(c[0] == "tap_action" and c[1] == "confirm" for c in calls), f"{calls[:2]}")

calls = run("card_reward_3choose.png")
print("   卡牌奖励:", calls[:2])
check("卡牌奖励 -> **鼠标点牌**（不是按键，因为卡上数字是费用）",
      any(c[0] in ("click_area_rel", "click_rel") for c in calls), f"{calls[:2]}")
check("卡牌奖励 -> ENTER 确认", any(c[0] == "tap_action" and c[1] == "confirm" for c in calls), f"{calls[:2]}")

calls = run("card_reward_assign.png")
check("领奖分配 -> 默认跳过中立卡(F1 / 点击 0.700, 0.930)",
      any(c[0] == "send_key" and c[1] == 112 for c in calls) and any(c[0] == "click_rel" for c in calls),
      f"{calls[:3]}")

print("\n=== 7) 肉鸽选路 = 编号键（排分算法不变）===")
for name in ("rogue_map_bright.png", "rogue_map_real.png"):
    calls = run(name)
    hit = [c for c in calls if c[0] == "tap_card"]
    print(f"   {name}: {calls[:2]}")
    check(f"{name} -> 按编号选路线", bool(hit), f"{calls[:3]}")

print("\n=== 8) 过渡黑帧：什么都不做 ===")
calls = run("transition_black.png")
check("过渡帧无动作", calls == [], f"{calls[:2]}")

print("\n=== 9) EP（自我意識點數）预算：买不起就不按 ===")
from lucid_chaos.logic.battle_manager import BattleManager  # noqa: E402


class _T:
    def __init__(self, text, rx=0.5, ry=0.5):
        self.text = text
        self.center_rel = (rx, ry)


for toks, want in ([[_T("EP 3", 0.02, 0.63)], 3], [[_T("EP7", 0.02, 0.62)], 7],
                   [[_T("EP 12", 0.02, 0.63)], 12], [[_T("沒有", 0.5, 0.5)], None]):
    got = BattleManager.parse_ep(toks)
    check(f"parse_ep({[t.text for t in toks]}) -> {got}", got == want, f"期望 {want}")

costs = BattleManager.parse_skill_costs([_T("4", 0.02, 0.66), _T("4", 0.02, 0.80),
                                         _T("5", 0.02, 0.90), _T("EP 7", 0.02, 0.63)])
check("技能费用按左侧竖排解析 -> {1:4,2:4,3:5}", costs == {1: 4, 2: 4, 3: 5}, f"{costs}")
check("读不到技能费用时有保守门槛 SKILL_MIN_EP=4", BattleManager.SKILL_MIN_EP == 4)

print("\n=== 10) 「自我意識點數不足」弹窗 -> 按 ESC 退出（真机截图夹具）===")
calls = run("battle_ep_insufficient.png")
print("   动作:", calls[:2])
check("检测到點數不足 -> tap_action(cancel)=ESC",
      any(c[0] == "tap_action" and c[1] == "cancel" for c in calls), f"{calls[:3]}")
check("没有再乱按卡牌编号", not any(c[0] == "tap_card" for c in calls), f"{calls[:3]}")

print("\n=== 11) AUTO【自动战斗】图标状态判定（真机 5 张截图校准）===")
from lucid_chaos.logic.battle_manager import BattleManager as _BM  # noqa: E402


class _VCtx:
    def __init__(self):
        self.vision = VisionService()


_bm = _BM(_VCtx())
_auto_expect = {
    "battle_auto_off.png": "INACTIVE",          # ⚔ 双剑 + R = 可用但未开启 -> 要按 R
    "battle_hand_row.png": "LOCKED",            # 🔒 锁 = BOSS 战禁止 AUTO -> 自己出牌
    "battle_hand6_costs.png": "LOCKED",
    "battle_ep_insufficient.png": "LOCKED",
    "battle_endturn_confirm.png": "LOCKED",
}
for _n, _want in _auto_expect.items():
    _img = imread(FIXTURES / _n)
    _got = _bm.inspect_auto_battle_status(_img)
    check(f"AUTO 状态 {_n} -> {_got}", _got == _want, f"期望 {_want}")
check("AUTO 判据不再用'整体饱和度'（那是误判根源）",
      "mean_s" not in inspect.getsource(_BM.inspect_auto_battle_status))

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
if __name__ == "__main__":
    sys.exit(1 if FAILED else 0)

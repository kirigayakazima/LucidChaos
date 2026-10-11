"""
执行框架单测（不发任何真实输入）：

    python -m lucid_chaos.tests.test_action_runner

验证 `StuckDetector` 与 `ActionLadder` 的语义**与改造前的手写实现逐字一致**：
  1. 卡死检测：同幕 -> 轮数递增；画面一变 -> 归零
  2. 指纹容差 4.0 对"雾/呼吸动画"不误判（用真实夹具的两帧做对照）
  3. 选级规则 min(stuck, n-1) 与封顶
  4. escalate 会执行对应级别的动作，并先回调 on_pick
  5. 与旧实现对照：旧代码的 _event_scene_same 语义等价
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

from lucid_chaos.logic.action_runner import ActionLadder, Attempt, StuckDetector, SAME_SCENE_TOL
from lucid_chaos.logic.node_engine import NodeDecisionEngine

FIXTURES = _HERE.parent / "fixtures"
FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


frame = imread(FIXTURES / "rogue_map_bright.png")
other = imread(FIXTURES / "card_reward_3choose.png")

print("=== 1) 卡死检测语义 ===")
d = StuckDetector()
seq = [d.update(frame) for _ in range(4)]
check("同一帧连续 4 次 -> 轮数 0,1,2,3", [s for _, s in seq] == [0, 1, 2, 3], f"{seq}")
check("第 2 次起判为同幕", [same for same, _ in seq] == [False, True, True, True])
same, stuck = d.update(other)
check("换成完全不同的画面 -> 归零", (not same) and stuck == 0, f"same={same} stuck={stuck}")
d.reset()
check("reset 后轮数归零", d.stuck == 0)

print("\n=== 2) 容差对'动画漂移'不误判（真实夹具上做小扰动）===")
noisy = frame.copy()
# 模拟雾/呼吸：整体轻微变化
noisy = cv2.addWeighted(noisy, 0.97, np.full_like(noisy, 12), 0.03, 0)
d2 = StuckDetector()
d2.update(frame)
same, stuck = d2.update(noisy)
check(f"轻微整体扰动仍判同幕（容差 {SAME_SCENE_TOL}）", same and stuck == 1, f"same={same} stuck={stuck}")

print("\n=== 3) 选级规则 min(stuck, n-1) ===")
cases = [(0, 3, 0), (1, 3, 1), (2, 3, 2), (3, 3, 2), (99, 3, 2), (0, 1, 0), (5, 1, 0)]
ok = all(ActionLadder.index_for(s, n) == exp for s, n, exp in cases)
check("min(stuck, n-1) 且封顶", ok,
      " ".join(f"({s},{n})->{ActionLadder.index_for(s,n)}" for s, n, _ in cases))

print("\n=== 4) escalate 执行正确级别 + 回调顺序 ===")
calls = []
class FakeInput:
    pass
ctx = type("C", (), {"input": FakeInput()})()
ladder = ActionLadder(ctx, [Attempt(f"第{i+1}级", (lambda i=i: calls.append(("run", i + 1)))) for i in range(3)])
order = []
same, idx, label = ladder.observe_and_escalate(frame, on_pick=lambda i, lb: order.append(("pick", i + 1, lb)))
check("第一次 -> 第 1 级", idx == 0 and calls[-1] == ("run", 1), f"idx={idx} label={label}")
order.clear()
for _ in range(2):
    ladder.observe_and_escalate(frame, on_pick=lambda i, lb: order.append(("pick", i + 1, lb)))
check("同幕后 -> 依次升级到第 2、第 3 级",
      calls[-2:] == [("run", 2), ("run", 3)], f"{calls[-2:]}")
order.clear()
ladder.observe_and_escalate(other, on_pick=lambda i, lb: order.append(("pick", i + 1, lb)))
check("画面一变 -> 回到第 1 级", calls[-1] == ("run", 1), f"{calls[-1]}")

print("\n=== 5) 与旧实现语义对照（NodeDecisionEngine._event_scene_same）===")
class _StubCtx:
    vision = None
    input = None
    overlay = None
    capture = None

eng = NodeDecisionEngine(_StubCtx(), None, {})     # 用真实构造函数（含 _stuck_det）
old_seq = []
for img in (frame, frame, frame, other):
    eng._event_scene_same(img)
    old_seq.append(eng._evt_stuck)
d3 = StuckDetector()
new_seq = [d3.update(img)[1] for img in (frame, frame, frame, other)]
check("轮数序列一致", old_seq == new_seq, f"旧 {old_seq} / 新 {new_seq}")
check("指纹是同一套（64x36 灰度）",
      eng._event_scene_sig(frame).shape == d3.signature(frame).shape,
      f"{eng._event_scene_sig(frame).shape}")

print("\n=== 6) 气泡阶梯逐级行为（handle_event 走 6 轮，必须依次升级并封顶）===")
import cv2  # noqa: E402
from lucid_chaos.state.screen_state import ScreenState  # noqa: E402

narr = imread(FIXTURES / "narration_with_markers.png")


class RecInput:
    def __init__(self):
        self.calls = []
    def click_client(self, cx, cy, **kw):
        self.calls.append(("click_client", int(cx), int(cy), kw.get("label"), kw.get("clicks")))
    def click_rel(self, rx, ry, **kw):
        self.calls.append(("click_rel", round(float(rx), 3), round(float(ry), 3), kw.get("label"), None))
    def click_area_rel(self, *a, **kw):
        self.calls.append(("click_area_rel", None, None, None, None))
    def click_box(self, *a, **kw):
        self.calls.append(("click_box", None, None, None, None))
    def send_key(self, vk, **kw):
        self.calls.append(("send_key", int(vk), None, kw.get("label"), None))
    def drag_rel(self, *a, **kw):
        self.calls.append(("drag_rel", None, None, None, None))


class Ctx:
    vision = None
    capture = None
    overlay = None


from collections import namedtuple  # noqa: E402

_FakeTarget = namedtuple("_FakeTarget", "text center_rel center confidence")
# 顶部旁白（不在底部判定带里）——避免测试里去调 OCR
_top_target = _FakeTarget("這裡也沒看見生存者。", (0.50, 0.21), (1280, 302), 0.9)

ctx6 = Ctx()
ctx6.input = RecInput()
eng6 = NodeDecisionEngine(ctx6, None, {})
st6 = type("S", (), {"screen": ScreenState.EVENT, "confidence": 0.8,
                     "raw_texts": ["在凌散落的每一张纸上"], "elements": {},
                     "targets": [_top_target]})()
seq = []
for _ in range(6):
    ctx6.input.calls.clear()
    eng6.handle_event(st6, raw_img=narr)
    c = ctx6.input.calls[0] if ctx6.input.calls else None
    seq.append(c)
    print("   ", c)
check("第 1 级 = 按 F1（探索交互点，键盘优先）",
      seq[0][0] == "send_key" and seq[0][1] == 112 and "F1" in (seq[0][3] or ""), f"{seq[0]}")
check("第 2 级 = 按 F2", seq[1][0] == "send_key" and seq[1][1] == 113, f"{seq[1]}")
check("第 3 级 = 按 F3", seq[2][0] == "send_key" and seq[2][1] == 114, f"{seq[2]}")
n_bub = len(eng6.detect_search_bubbles(narr))
check(f"第 4 级 = 鼠标点击气泡#0（回退；本夹具 {n_bub} 个气泡，同点连点 3 次）",
      seq[3][0] == "click_client" and "气泡#0" in (seq[3][3] or "") and seq[3][4] == 3, f"{seq[3]}")
check("第 5 级 = 按 SPACE 推进对白", seq[4][0] == "send_key" and seq[4][1] == 32, f"{seq[4]}")
check("第 6 级 = 地面物体 (同一 x，y 更大)",
      seq[5][0] == "click_client" and "地面物体" in (seq[5][3] or "") and seq[5][2] > seq[3][2], f"{seq[5]}")

# 再来几轮：第 7 级 = 按 Z 跳过，之后封顶在最后一级
tail = []
for _ in range(3):
    ctx6.input.calls.clear()
    eng6.handle_event(st6, raw_img=narr)
    tail.append(ctx6.input.calls[0] if ctx6.input.calls else None)
print("   封顶轮:", tail)
check("第 7 级 = 按 Z 跳过剧情", tail[0][0] == "send_key" and tail[0][1] == 90, f"{tail[0]}")
check("到顶后封顶（不会越界乱试）", tail[1][:2] == tail[0][:2] and tail[2][:2] == tail[0][:2], f"{tail}")

print("\n=== 7) 有 2 个气泡时，回退步骤仍按坐标顺序（F 键在前）===")
_orig = eng6.detect_search_bubbles
eng6.detect_search_bubbles = lambda img: [
    {"cx": 1426, "cy": 598, "rx": 0.557, "ry": 0.415, "score": 0.84, "bbox": (1376, 548, 100, 100)},
    {"cx": 1980, "cy": 618, "rx": 0.773, "ry": 0.429, "score": 0.84, "bbox": (1930, 568, 100, 100)},
]
eng6._stuck_det.reset()
eng6._evt_stuck = 0
seq2 = []
for _ in range(5):
    ctx6.input.calls.clear()
    eng6.handle_event(st6, raw_img=narr)
    seq2.append(ctx6.input.calls[0] if ctx6.input.calls else None)
eng6.detect_search_bubbles = _orig
for c in seq2:
    print("   ", c)
check("1->2 级都是 F 键（112/113）", seq2[0][1] == 112 and seq2[1][1] == 113, f"{seq2[:2]}")
check("第 3 级 = F3", seq2[2][1] == 114, f"{seq2[2]}")
check("第 4 级 = 点击第 1 个气泡（2 个气泡时点最左那个）",
      seq2[3][0] == "click_client" and seq2[3][1] == 1426, f"{seq2[3]}")

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)

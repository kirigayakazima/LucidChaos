"""
M2 统一场景判定 的验证（不发任何真实输入）：

    python -m lucid_chaos.tests.test_scene_unify

验证：
  1. SceneType <-> ScreenState 映射完整
  2. 默认 authority="observer"：state.screen 与改造前完全一致（行为不变）
  3. authority="router"：state.screen 改成路由映射值，且修掉已知的 3 处矛盾
  4. 分歧会被记录（【判定分歧】日志）
  5. dispatch_frame(frame) 与 dispatch(img) 的结论/动作**完全一致**（管道替换安全）
  6. StoryPusher.step 复用 state.frame，不再重复截图
"""
import logging
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

from lucid_chaos.core.frame import FrameContext
from lucid_chaos.core.vision_service import VisionService
from lucid_chaos.state.observer import ScreenObserver
from lucid_chaos.state.screen_state import ScreenState
from lucid_chaos.router.router import SceneRouter
from lucid_chaos.router.scene_types import SceneType, SCENE_TO_SCREEN, SCREEN_TO_SCENE
from lucid_chaos.logic.node_engine import NodeDecisionEngine
from lucid_chaos.logic.path_planner import PathPlanner
from lucid_chaos.logic.story_pusher import StoryPusher

FIXTURES = _HERE.parent / "fixtures"
FAILED = []

# 已知的 observer 判错、router 判对的三张图
KNOWN_CONTRADICTIONS = {
    "card_reward_assign.png": (ScreenState.VICTORY_SETTLEMENT, SceneType.EQUIP_ASSIGN),
    "rogue_map_bright.png": (ScreenState.EVENT, SceneType.ROGUELIKE_MAP),
    "rogue_map_dim.png": (ScreenState.EVENT, SceneType.ROGUELIKE_MAP),
}


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


class RecInput:
    def __init__(self):
        self.calls = []
    def click_client(self, cx, cy, **k): self.calls.append(("click_client", int(cx), int(cy), k.get("label")))
    def click_rel(self, rx, ry, **k): self.calls.append(("click_rel", round(float(rx), 3), round(float(ry), 3), k.get("label")))
    def click_area_rel(self, a, b, c, d, **k): self.calls.append(("click_area", round((a+c)/2, 3), round((b+d)/2, 3), k.get("label")))
    def click_box(self, *a, **k): self.calls.append(("click_box", None, None, None))
    def send_key(self, *a, **k): self.calls.append(("send_key", a[:1], k.get("label")))
    # handler 现在会用到的键盘动作方法（缺了会让测试在收尾阶段崩掉）
    def tap_action(self, action, **k): self.calls.append(("tap_action", action, k.get("label"))); return True
    def tap_card(self, idx, **k): self.calls.append(("tap_card", idx, k.get("label"))); return True
    def tap_hero_skill(self, slot, **k): self.calls.append(("tap_hero_skill", slot, k.get("label"))); return True
    def drag_rel(self, *a, **k): self.calls.append(("drag_rel", None, None, None))


class FakeCapture:
    """只统计 capture() 被调用几次，不真的截图。"""
    def __init__(self, calls):
        self._calls = calls

    def capture(self):
        self._calls.append(1)
        return None


class Stub:
    def __init__(self, vision, capture_calls=None):
        self.vision = vision
        self.overlay = None
        self.input = RecInput()
        self._capture_calls = capture_calls if capture_calls is not None else []
        self.capture = FakeCapture(self._capture_calls)


def build(vision, authority, capture_calls=None):
    ctx = Stub(vision, capture_calls)
    eng = NodeDecisionEngine(ctx, None, {})
    eng._evt_sig = None
    eng._evt_stuck = 0
    router = SceneRouter(ctx, {"classification": {"authority": authority}},
                         node_engine=eng, path_planner=PathPlanner())
    return ctx, eng, router


vision = VisionService()
obs = ScreenObserver(vision)

print("=== 1) 映射完整性 ===")
missing = [s.name for s in SceneType if s not in SCENE_TO_SCREEN]
check("每个 SceneType 都有 ScreenState 映射", not missing, f"缺: {missing}")
check("反查表可用", len(SCREEN_TO_SCENE) >= 8, f"{len(SCREEN_TO_SCENE)} 项")

print("\n=== 2/3) 统一判定入口 + 两种 authority ===")
divergence_count = 0
for f in sorted(FIXTURES.glob("*.png")):
    img = imread(f)
    _obs0 = obs.observe(img)
    base_screen = _obs0.screen                            # 改造前 state.screen 就是这个
    base_conf = float(getattr(_obs0, "confidence", 0.0) or 0.0)

    fr = FrameContext.from_image(img, ocr=vision)
    ctx0, eng0, router0 = build(vision, "observer")
    res = router0.classify_frame(fr)               # 唯一判定入口（不受安全检查短路影响）
    check(f"{f.name} classify_frame 有结论", res is not None and res.scene is not None,
          f"{res.scene.value} {res.confidence:.0f}分")

    # --- 默认 authority=observer：只有"路由分明显更高(>观察者+5)"时才被裁决改掉 ---
    # 真机教训（主人 2026/10/6）：观察者会把肉鸽选路/选关判成 EVENT，
    # 而路由能判对(MAP_NAVIGATION)，此时再无条件采信观察者就会跑错 handler。
    # 所以现在的契约是"**谁分高听谁的**"：分高时才改，否则保持观察者结论。
    st = obs.observe(img)
    ctx, eng, router = build(vision, "observer")
    router.dispatch_frame(fr, st)
    _obs_pct = base_conf * 100.0 if base_conf <= 1.0 else base_conf
    _router_wins = res.confidence > _obs_pct + 5.0 and SCENE_TO_SCREEN.get(res.scene) is not None
    if _router_wins:
        check(f"[observer] {f.name} 路由分更高 -> state.screen 跟随路由",
              st.screen == SCENE_TO_SCREEN.get(res.scene),
              f"{base_screen.name} -> {st.screen.name} (route={res.scene.value} {res.confidence:.0f} vs obs {_obs_pct:.0f})")
    else:
        check(f"[observer] {f.name} state.screen 未被改动", st.screen == base_screen,
              f"{base_screen.name} -> {st.screen.name} (route={res.scene.value} {res.confidence:.0f} vs obs {_obs_pct:.0f})")
    check(f"[observer] {f.name} 有决策记录", bool(st.decision), f"decision={st.decision!r}")

    # --- authority=router：state.screen 应等于映射后的路由 ---
    st2 = obs.observe(img)
    ctx2, eng2, router2 = build(vision, "router")
    router2._reconcile_screen(st2, res)
    mapped = SCENE_TO_SCREEN.get(res.scene)
    check(f"[router] {f.name} state.screen == 映射(route)", st2.screen == mapped,
          f"{base_screen.name} -> {st2.screen.name} (route={res.scene.value})")

    if base_screen != mapped:
        divergence_count += 1
        exp = KNOWN_CONTRADICTIONS.get(f.name)
        if exp:
            check(f"已知矛盾 {f.name}", (base_screen, res.scene) == exp,
                  f"observer={base_screen.name} router={res.scene.value}")

print(f"\n   8 张图中 observer 与 router 分歧 {divergence_count} 张")

print("\n=== 3b) authority=router 模式下完整 dispatch 冒烟（不能抛异常） ===")
for f in sorted(FIXTURES.glob("*.png")):
    img = imread(f)
    ctx3, eng3, router3 = build(vision, "router")
    st3 = obs.observe_frame(FrameContext.from_image(img, ocr=vision))
    try:
        router3.dispatch_frame(st3.frame, st3)
        ok, detail = True, f"decision={st3.decision} screen={st3.screen.name}"
    except Exception as e:  # noqa: BLE001
        ok, detail = False, repr(e)
    check(f"[router 冒烟] {f.name}", ok, detail)

print("\n=== 4) 分歧会被记录（【判定分歧】日志） ===")
records = []


class Catch(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())


lg = logging.getLogger("SceneRouter")
lg.addHandler(Catch())
lg.setLevel(logging.INFO)
for fname in ("card_reward_assign.png", "card_reward_3choose.png"):
    img = imread(FIXTURES / fname)
    st = obs.observe(img)
    ctx, eng, router = build(vision, "observer")
    fr = FrameContext.from_image(img, ocr=vision)
    router.dispatch_frame(fr, st)
div_lines = [m for m in records if "判定分歧" in m]
print("   捕获到分歧日志:", div_lines if div_lines else "(无)")
check("矛盾图会打【判定分歧】", any("card_reward_assign" in m or "VICTORY_SETTLEMENT" in m for m in div_lines))
check("一致图不该打分歧", not any("EPIPHANY" in m and "未知" in m for m in div_lines))

print("\n=== 5) dispatch_frame 与 dispatch 等价 ===")
for f in sorted(FIXTURES.glob("*.png")):
    img = imread(f)
    ctx_a, eng_a, router_a = build(vision, "observer")
    st_a = obs.observe(img)
    router_a.dispatch(img, st_a)
    act_a = list(ctx_a.input.calls)

    ctx_b, eng_b, router_b = build(vision, "observer")
    fr = FrameContext.from_image(img, ocr=vision)
    st_b = obs.observe_frame(fr)
    router_b.dispatch_frame(fr, st_b)
    act_b = list(ctx_b.input.calls)

    same = (st_a.route == st_b.route) and (act_a == act_b)
    check(f"等价 {f.name}", same,
          f"route {getattr(st_a.route,'value',None)}/{getattr(st_b.route,'value',None)}"
          + ("" if act_a == act_b else f"  动作不同 {act_a} vs {act_b}"))

print("\n=== 6) StoryPusher.step 复用 state.frame（不再重复截图） ===")
img = imread(FIXTURES / "card_reward_3choose.png")
calls = []
ctx, eng, router = build(vision, "observer", capture_calls=calls)
sp = StoryPusher(ctx, {})
sp.router = router
st = obs.observe_frame(FrameContext.from_image(img, ocr=vision))
st.frame = FrameContext.from_image(img, ocr=vision)
sp.step(st)
check("有 frame 时不调用 capture()", len(calls) == 0, f"capture 调用 {len(calls)} 次")
st_no = obs.observe(img)
sp.step(st_no)
check("没有 frame 时回退到 capture()", len(calls) == 1, f"capture 调用 {len(calls)} 次")

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)

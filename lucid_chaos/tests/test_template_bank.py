"""
模板库 / 帧上下文 的单元测试（不发任何真实输入）。

    python -m lucid_chaos.tests.test_template_bank

验证三件事：
  1. 7 处旧加载器现在拿到的是**同一个对象**（真的统一成一份缓存了）
  2. 缩放缓存确实命中（同一 fx 只 resize 一次）
  3. 亮度归一化对「压暗帧」的模板匹配确实有帮助（证据，供以后显式启用）
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

import cv2
import numpy as np

from lucid_chaos.core.template_bank import BANK, list_templates, normalize_brightness
from lucid_chaos.core.frame import FrameContext
from lucid_chaos.core.vision_service import VisionService
from lucid_chaos.router.evaluators import SceneEvaluators
from lucid_chaos.logic.path_planner import PathPlanner
from lucid_chaos.logic.node_engine import NodeDecisionEngine
from lucid_chaos.logic.story_pusher import StoryPusher
from lucid_chaos.core.stage_detector import StageMapDetector

FIXTURES = _HERE.parent / "fixtures"
FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


print("=== 1) 旧加载器全部指向同一个对象 ===")
check("SceneEvaluators._get_template('skip_btn.png') is BANK.get(...)",
      SceneEvaluators._get_template("skip_btn.png") is BANK.get("skip_btn.png"))
check("NodeDecisionEngine._get_search_bubble_template() is BANK.get(...)",
      NodeDecisionEngine._get_search_bubble_template() is BANK.get("explore_search_bubble.png"))
check("NodeDecisionEngine._get_close_x_template() is BANK.get(...)",
      NodeDecisionEngine._get_close_x_template() is BANK.get("btn_close_x.png"))
check("NodeDecisionEngine._get_recommend_badge_template() is BANK.get('rogue/...')",
      NodeDecisionEngine._get_recommend_badge_template() is BANK.get("rogue/recommend_badge.png"))
check("StoryPusher._get_skip_template() is BANK.get(...)",
      StoryPusher._get_skip_template() is BANK.get("skip_btn.png"))
check("StoryPusher._get_arrow_template() is BANK.get(...)",
      StoryPusher._get_arrow_template() is BANK.get("next_chapter_arrow.png"))
check("PathPlanner.get_template('SWORD') is BANK.get('rogue/rogue_sword.png')",
      PathPlanner.get_template("SWORD") is BANK.get("rogue/rogue_sword.png"))
check("PathPlanner.get_template('cyan_arrow_2k') is BANK.get('rogue_arrow_cyan.png')",
      PathPlanner.get_template("cyan_arrow_2k") is BANK.get("rogue_arrow_cyan.png"))
check("StageMapDetector._get_template('skip_btn.png') is BANK.get(...)",
      StageMapDetector._get_template("skip_btn.png") is BANK.get("skip_btn.png"))
_vs = VisionService()
check("VisionService.templates['skip_btn'] is BANK.get('skip_btn.png')",
      _vs.templates.get("skip_btn") is BANK.get("skip_btn.png"))

print("\n=== 2) 缩放缓存 ===")
a = BANK.scaled("rogue/rogue_sword.png", 0.22 * 2.5)
b = BANK.scaled("rogue/rogue_sword.png", 0.22 * 2.5)
check("scaled() 同一 fx 返回同一对象", a is b, f"shape={None if a is None else a.shape}")
c = BANK.scaled_to("explore_search_bubble.png", 99, 108)
d = BANK.scaled_to("explore_search_bubble.png", 99, 108)
check("scaled_to() 同尺寸返回同一对象", c is d, f"shape={None if c is None else c.shape}")
before = dict(BANK.stats())
for _ in range(50):
    BANK.scaled("rogue/rogue_sword.png", 0.22 * 2.5)
after = dict(BANK.stats())
check("50 次重复缩放新增 0 次 resize",
      after["scale_misses"] == before["scale_misses"],
      f"scale_misses {before['scale_misses']} -> {after['scale_misses']}")

print("\n=== 3) 模板清点 ===")
tpls = list_templates()
check("模板数量 == 22", len(tpls) == 22, f"实际 {len(tpls)}: {tpls}")

print("\n=== 4) 亮度归一化对压暗帧的帮助（证据）===")
NAMES = ("rogue/rogue_sword.png", "rogue/rogue_question.png", "rogue/rogue_fire.png", "rogue/rogue_fire_shop.png")
for fname in ("rogue_map_bright.png", "rogue_map_dim.png"):
    p = FIXTURES / fname
    if not p.exists():
        continue
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    scale = img.shape[0] / 576.0
    region = (0.01, 0.62, 0.42, 0.98)
    raw = BANK.best_match(img, NAMES, 0.22 * scale, region=region, normalize=False)
    nor = BANK.best_match(img, NAMES, 0.22 * scale, region=region, normalize=True)
    print(f"   {fname}: 原始最佳 {raw.score:.3f} ({Path(raw.name).name})  ->  归一化后 {nor.score:.3f} ({Path(nor.name).name})")
    check(f"{fname} 归一化后分数不下降", nor.score >= raw.score - 1e-9)

print("\n=== 5) FrameContext 一帧一次 OCR ===")


class CountingVision:
    def __init__(self):
        self.calls = 0
        self._v = VisionService()

    def locate_all_texts(self, img, min_score=0.5):
        self.calls += 1
        return self._v.locate_all_texts(img, min_score=min_score)


cv_ = CountingVision()
p = FIXTURES / "card_reward_3choose.png"
img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
frame = FrameContext.from_image(img, ocr=cv_)
_ = frame.observe(cv_)
_ = frame.corpus
_ = frame.elements
check("构造 + 再次 observe 只跑一次 OCR", cv_.calls == 1, f"OCR 调用 {cv_.calls} 次")
check("frame.scale == h/576", abs(frame.scale - img.shape[0] / 576.0) < 1e-9, f"{frame.scale:.4f}")
n1 = frame.normalized()
n2 = frame.normalized()
check("normalized() 只算一次（缓存）", n1 is n2)

print("\n=== 6) 一次真实 dispatch 的缓存命中统计 ===")
t0 = time.perf_counter()
from lucid_chaos.state.observer import ScreenObserver
from lucid_chaos.router.router import SceneRouter


class RecInput:
    """记录/忽略动作的桩。必须覆盖 handler 会用到的**全部**输入方法：
    曾经因为缺少 tap_action 让本测试在收尾阶段崩掉（而摘要行已打印，看着像通过）。"""

    def __init__(self):
        self.calls = []
        self.last_action = ""

    def click_client(self, *a, **k): self.calls.append(("click_client", a, k.get("label")))
    def click_rel(self, *a, **k): self.calls.append(("click_rel", a, k.get("label")))
    def click_area_rel(self, *a, **k): self.calls.append(("click_area_rel", a, k.get("label")))
    def click_box(self, *a, **k): self.calls.append(("click_box", a, k.get("label")))
    def click_target(self, *a, **k): self.calls.append(("click_target", a, k.get("label")))
    def drag_rel(self, *a, **k): self.calls.append(("drag_rel", a, k.get("label")))
    def drag_client(self, *a, **k): self.calls.append(("drag_client", a, k.get("label")))
    def send_key(self, *a, **k): self.calls.append(("send_key", a, k.get("label")))
    def tap_action(self, action, **k): self.calls.append(("tap_action", action, k.get("label"))); return True
    def tap_card(self, idx, **k): self.calls.append(("tap_card", idx, k.get("label"))); return True
    def tap_hero_skill(self, slot, **k): self.calls.append(("tap_hero_skill", slot, k.get("label"))); return True


class Stub:
    def __init__(self):
        self.vision = _vs
        self.capture = None
        self.overlay = None
        self.input = RecInput()


for fname in ("rogue_map_dim.png", "card_reward_assign.png", "event_box_survival_signal.png"):
    p = FIXTURES / fname
    img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
    ctx = Stub()
    eng = NodeDecisionEngine(ctx, None, {})
    eng._evt_sig = None
    eng._evt_stuck = 0
    router = SceneRouter(ctx, {}, node_engine=eng, path_planner=PathPlanner())
    obs = ScreenObserver(_vs)
    stt = obs.observe(img)
    t1 = time.perf_counter()
    router.dispatch(img, stt)
    print(f"   {fname}: dispatch 耗时 {(time.perf_counter()-t1)*1000:.0f} ms")
print(f"   合计 {(time.perf_counter()-t0)*1000:.0f} ms")
print(f"   模板库统计: {BANK.stats()}")

print("\n=== 7) observer 拆分等价性：observe(img) 与 observe_frame(FrameContext) 必须完全一致 ===")
_obs = ScreenObserver(_vs)
for f in sorted(FIXTURES.glob("*.png")):
    im = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
    st_a = _obs.observe(im)
    fr = FrameContext.from_image(im, ocr=_vs)
    st_b = _obs.observe_frame(fr)
    same = (st_a.screen == st_b.screen) and (st_a.raw_texts == st_b.raw_texts) \
        and ([t.text for t in st_a.targets] == [t.text for t in st_b.targets])
    check(f"等价 {f.name}", same,
          f"{st_a.screen.name}/{st_b.screen.name} 文本{len(st_a.raw_texts)}/{len(st_b.raw_texts)}")
    check(f"frame 已挂到 state {f.name}", getattr(st_b, "frame", None) is fr)

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)

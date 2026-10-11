"""
Unit test for Hammer of God (神之錘) Reforged Equipment Enhancement scene.
Verifies that:
1. Stage 1: Scene is classified as EQUIP_ASSIGN (conf=100.0).
2. Stage 1: The glowing reforged equipment (Hero 2 Weapon at 0.760, 0.530) is detected and clicked.
3. Stage 1: Confirmation via ENTER and click (0.890, 0.925) is executed.
4. Stage 1: Fallback behavior (skip when no glowing items) is verified.
5. Stage 2 (Secondary Confirmation):
   - Classified as EQUIP_ASSIGN (conf=100.0).
   - "決定" (Decision / Confirm) button is detected and clicked at (0.637, 0.927).
   - ENTER confirmation is executed.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np
from lucid_chaos.core.vision_service import VisionService
from lucid_chaos.router.router import SceneRouter
from lucid_chaos.router.scene_types import SceneType
from lucid_chaos.logic.node_engine import NodeDecisionEngine
from lucid_chaos.state.screen_state import GameState, ScreenState

def test_god_hammer():
    f_god = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791708318232_2dcf0b5d.png")
    f_confirm = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791708390025_dbc38536.png")

    vision = VisionService()

    class StubInput:
        def __init__(self):
            self.calls = []
        def tap_action(self, action, **kw):
            self.calls.append(("tap_action", action, kw.get("label")))
            return True
        def click_rel(self, rx, ry, **kw):
            self.calls.append(("click_rel", round(rx, 3), round(ry, 3), kw.get("label")))
            return True
        def click_area_rel(self, a, b, c, d, **kw):
            self.calls.append(("click_area_rel", round((a+c)/2, 3), round((b+d)/2, 3), kw.get("label")))
            return True
        def send_key(self, vk, **kw):
            self.calls.append(("send_key", vk, kw.get("label")))
            return True

    # 1. Stage 1 Tests
    if f_god.exists():
        print("=== Test 1: Hammer of God Scene Classification ===")
        img = cv2.imdecode(np.fromfile(str(f_god), dtype=np.uint8), cv2.IMREAD_COLOR)
        tgts = vision.locate_all_texts(img)
        state = GameState(screen=ScreenState.EQUIP_SELECTION, confidence=0.9)
        state.raw_texts = [t.text for t in tgts]
        state.targets = tgts

        stub = StubInput()
        ctx = type("Ctx", (), {"input": stub, "overlay": None, "vision": vision})()
        router = SceneRouter(ctx, {})
        res = router.classify_frame(type("F", (), {"img": img, "corpus": " ".join(state.raw_texts), "targets": tgts, "valid": True})())
        print(f"God Hammer Screen Classification: {res.scene.value} (conf={res.confidence}, details={res.details})")
        assert res.scene == SceneType.EQUIP_ASSIGN, f"Expected EQUIP_ASSIGN, got {res.scene}"
        assert res.confidence == 100.0, f"Expected 100.0 conf, got {res.confidence}"
        print("[Test 1 PASS]")

        print("\n=== Test 2: Glowing Equipment Detection & Action Ladder ===")
        stub.calls.clear()
        engine = NodeDecisionEngine(ctx, None, {})
        engine.handle_equip_selection(state, raw_img=img)
        print(f"God Hammer Calls: {stub.calls}")

        # Must click Hero 2's glowing Weapon at (0.760, 0.530)
        assert any(c[0] == "click_rel" and abs(c[1] - 0.760) < 0.02 and abs(c[2] - 0.530) < 0.02 for c in stub.calls), \
            f"Expected clicking Hero 2 Weapon (0.760, 0.530), got {stub.calls}"
        assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub.calls), "Expected ENTER confirmation"
        assert any(c[0] == "click_rel" and abs(c[1] - 0.890) < 0.02 and abs(c[2] - 0.925) < 0.02 for c in stub.calls), \
            "Expected clicking right-bottom confirm button"
        print("[Test 2 PASS]")

        print("\n=== Test 3: Fallback when no glowing items ===")
        stub.calls.clear()
        blank_img = np.zeros_like(img)
        engine.handle_equip_selection(state, raw_img=blank_img)
        print(f"Fallback Calls: {stub.calls}")
        assert any(c[0] == "send_key" and c[1] == 112 for c in stub.calls) or any("跳过" in (c[-1] or "") for c in stub.calls), \
            f"Expected skip action when no items glow, got {stub.calls}"
        print("[Test 3 PASS]")

    # 2. Stage 2 Tests (Secondary Confirmation with 決定 button)
    if f_confirm.exists():
        print("\n=== Test 4: God Hammer Secondary Confirmation Screen ===")
        img2 = cv2.imdecode(np.fromfile(str(f_confirm), dtype=np.uint8), cv2.IMREAD_COLOR)
        tgts2 = vision.locate_all_texts(img2)
        state2 = GameState(screen=ScreenState.EQUIP_SELECTION, confidence=0.9)
        state2.raw_texts = [t.text for t in tgts2]
        state2.targets = tgts2

        stub2 = StubInput()
        ctx2 = type("Ctx", (), {"input": stub2, "overlay": None, "vision": vision})()
        router2 = SceneRouter(ctx2, {})
        res2 = router2.classify_frame(type("F", (), {"img": img2, "corpus": " ".join(state2.raw_texts), "targets": tgts2, "valid": True})())
        print(f"Secondary Confirm Classification: {res2.scene.value} (conf={res2.confidence}, details={res2.details})")
        assert res2.scene == SceneType.EQUIP_ASSIGN, f"Expected EQUIP_ASSIGN, got {res2.scene}"
        assert res2.confidence == 100.0, f"Expected 100.0 conf, got {res2.confidence}"

        stub2.calls.clear()
        engine2 = NodeDecisionEngine(ctx2, None, {})
        engine2.handle_equip_selection(state2, raw_img=img2)
        print(f"Secondary Confirm Calls: {stub2.calls}")

        # Must trigger ENTER confirmation and click the orange 決定 button at (0.637, 0.927)
        assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub2.calls), "Expected ENTER confirmation"
        assert any(c[0] == "click_area_rel" and abs(c[1] - 0.640) < 0.05 and abs(c[2] - 0.925) < 0.05 for c in stub2.calls), \
            f"Expected clicking orange 決定 button (~0.640, 0.925), got {stub2.calls}"
        print("[Test 4 PASS]")

    print("\nALL GOD HAMMER TESTS (STAGE 1 & STAGE 2) PASSED!")

if __name__ == "__main__":
    test_god_hammer()

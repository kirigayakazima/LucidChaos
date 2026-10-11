"""
Unit test for Trigger Epiphany Card Selection screen.
Verifies that:
1. Screen is correctly classified as EPIPHANY_CARDS (conf=100.0 or high).
2. Action ladder handles:
   - stuck_cnt == 0: click card 1 (0.355, 0.310), send SPACE (32), ENTER / confirm.
   - stuck_cnt == 1: click card 2 (0.355, 0.715), send SPACE (32), ENTER / confirm.
   - stuck_cnt >= 2: drag scroll downwards to browse further cards, click card, send SPACE, confirm.
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

def test_epiphany_trigger_card():
    f_user = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791706971495_206c7c9e.png")

    vision = VisionService()

    class StubInput:
        def __init__(self):
            self.calls = []
        def tap_action(self, action, **kw):
            self.calls.append(("tap_action", action, kw.get("label")))
            return True
        def tap_card(self, idx, **kw):
            self.calls.append(("tap_card", idx, kw.get("label")))
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
        def drag_rel(self, s_rx, s_ry, e_rx, e_ry, **kw):
            self.calls.append(("drag_rel", round(s_rx, 3), round(s_ry, 3), round(e_rx, 3), round(e_ry, 3), kw.get("label")))
            return True

    if not f_user.exists():
        print(f"Skipping test: {f_user} does not exist.")
        return

    print("=== Test 1: Classification of Trigger Epiphany Card Selection Screen ===")
    img = cv2.imdecode(np.fromfile(str(f_user), dtype=np.uint8), cv2.IMREAD_COLOR)
    tgts = vision.locate_all_texts(img)
    state = GameState(screen=ScreenState.EPIPHANY_SELECTION, confidence=0.9)
    state.raw_texts = [t.text for t in tgts]
    state.targets = tgts

    stub = StubInput()
    ctx = type("Ctx", (), {"input": stub, "overlay": None, "vision": vision})()
    router = SceneRouter(ctx, {})
    res = router.classify_frame(type("F", (), {"img": img, "corpus": " ".join(state.raw_texts), "targets": tgts, "valid": True})())
    print(f"Scene Classification: {res.scene.value} (conf={res.confidence}, details={res.details})")
    assert res.scene == SceneType.EPIPHANY_CARDS, f"Expected EPIPHANY_CARDS, got {res.scene}"
    assert res.confidence >= 90.0, f"Expected high confidence, got {res.confidence}"
    print("[Classification PASS]")

    print("\n=== Test 2: Decision Ladder (stuck_cnt = 0) ===")
    stub.calls.clear()
    engine = NodeDecisionEngine(ctx, None, {})
    engine._evt_stuck = 0
    engine.handle_epiphany_selection(state, raw_img=img)
    print(f"Calls (stuck=0): {stub.calls}")
    assert any(c[0] == "click_rel" and abs(c[1] - 0.355) < 0.01 and abs(c[2] - 0.310) < 0.01 for c in stub.calls), "Expected click on Card 1 (0.355, 0.310)"
    assert any(c[0] == "send_key" and c[1] == 32 for c in stub.calls), "Expected SPACE key press (32)"
    assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub.calls), "Expected confirm ENTER"
    print("[stuck_cnt = 0 PASS]")

    print("\n=== Test 3: Decision Ladder (stuck_cnt = 1) ===")
    stub.calls.clear()
    engine._evt_stuck = 1
    engine.handle_epiphany_selection(state, raw_img=img)
    print(f"Calls (stuck=1): {stub.calls}")
    assert any(c[0] == "click_rel" and abs(c[1] - 0.355) < 0.01 and abs(c[2] - 0.715) < 0.01 for c in stub.calls), "Expected click on Card 2 (0.355, 0.715)"
    assert any(c[0] == "send_key" and c[1] == 32 for c in stub.calls), "Expected SPACE key press (32)"
    assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub.calls), "Expected confirm ENTER"
    print("[stuck_cnt = 1 PASS]")

    print("\n=== Test 4: Decision Ladder (stuck_cnt = 2: drag scroll) ===")
    stub.calls.clear()
    engine._evt_stuck = 2
    engine.handle_epiphany_selection(state, raw_img=img)
    print(f"Calls (stuck=2): {stub.calls}")
    assert any(c[0] == "drag_rel" for c in stub.calls), "Expected drag_rel scrolling operation"
    assert any(c[0] == "send_key" and c[1] == 32 for c in stub.calls), "Expected SPACE key press (32)"
    assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub.calls), "Expected confirm ENTER"
    print("[stuck_cnt = 2 PASS]")

    print("\nALL EPIPHANY TRIGGER CARD TESTS PASSED!")

if __name__ == "__main__":
    test_epiphany_trigger_card()

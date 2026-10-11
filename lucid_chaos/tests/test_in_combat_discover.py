"""
Unit test for in-combat card discovery / generation selection modal.
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
from lucid_chaos.state.screen_state import GameState, ScreenState

def test_combat_card_discover():
    ws_dir = _HERE.parent.parent
    user_img_path = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791702494094_2f4759ef.png")
    stuck_img_path = _HERE.parent.parent / "fixtures_auto" / "20261011-150812_VICTORY_SETTLEMENT_战斗中_stuck4.png"

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
        def tap_card(self, idx, **kw):
            self.calls.append(("tap_card", idx, kw.get("label")))
            return True

    for p in [user_img_path, stuck_img_path]:
        if not p.exists():
            print(f"File {p} does not exist, skipping")
            continue
        print(f"\nTesting image: {p.name}")
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        tgts = vision.locate_all_texts(img)
        state = GameState(screen=ScreenState.BATTLE, confidence=0.9)
        state.raw_texts = [t.text for t in tgts]
        state.targets = tgts

        stub_in = StubInput()
        ctx_stub = type("Ctx", (), {"input": stub_in, "overlay": None, "vision": vision})()
        router = SceneRouter(ctx_stub, {"mode": "system_zero"})

        ok = router.dispatch(img, state)
        print(f"Dispatch ok: {ok}")
        print(f"Decision: {state.decision}")
        print(f"Calls: {stub_in.calls}")
        assert state.decision == "safety:combat_card_discover", f"Expected safety:combat_card_discover, got {state.decision}"
        assert any("ENTER" in (c[-1] or "") for c in stub_in.calls), "Should confirm with ENTER"
        assert any(c[0] == "click_rel" and 0.08 <= c[1] <= 0.16 for c in stub_in.calls), "Should click Card 1 around x=0.12"

    print("\nALL IN-COMBAT DISCOVERY TESTS PASSED!")

if __name__ == "__main__":
    test_combat_card_discover()

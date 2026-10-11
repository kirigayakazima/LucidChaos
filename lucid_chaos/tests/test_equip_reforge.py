"""
Unit test for Equipment Reforging (重鑄裝備) scene.
Verifies that:
1. Scene is classified as EQUIP_ASSIGN (conf=100.0).
2. Reforge option on the left is selected.
3. An equipped item on the right (e.g. Hero 2's weapon) is clicked to activate assignment.
4. ENTER / confirm is executed.
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

def test_equip_reforge():
    f_reforge = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791705833722_fc13e4f1.png")

    if not f_reforge.exists():
        print(f"Skipping test: {f_reforge} not found")
        return

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

    print("=== Testing Equipment Reforge (重鑄裝備) ===")
    img = cv2.imdecode(np.fromfile(str(f_reforge), dtype=np.uint8), cv2.IMREAD_COLOR)
    tgts = vision.locate_all_texts(img)
    state = GameState(screen=ScreenState.EQUIP_SELECTION, confidence=0.9)
    state.raw_texts = [t.text for t in tgts]
    state.targets = tgts

    stub = StubInput()
    ctx = type("Ctx", (), {"input": stub, "overlay": None, "vision": vision})()
    router = SceneRouter(ctx, {})
    res = router.classify_frame(type("F", (), {"img": img, "corpus": " ".join(state.raw_texts), "targets": tgts, "valid": True})())
    print(f"Reforge Screen Classification: {res.scene.value} (conf={res.confidence})")
    assert res.scene == SceneType.EQUIP_ASSIGN, f"Expected EQUIP_ASSIGN, got {res.scene}"

    engine = NodeDecisionEngine(ctx, None, {})
    engine.handle_equip_selection(state, raw_img=img)
    print(f"Reforge Calls: {stub.calls}")

    # Check that option was clicked
    assert any("选择重铸效果" in (c[-1] or "") for c in stub.calls), "Expected selecting reforge option"
    # Check that an equipped item was clicked (Hero 2 Weapon at 0.760, 0.530)
    assert any("武器" in (c[-1] or "") or "装备" in (c[-1] or "") for c in stub.calls), "Expected selecting equipped item"
    # Check confirmation
    assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub.calls), "Expected ENTER confirmation"
    print("[Test PASS] Reforge equipment handling successfully verified!")

if __name__ == "__main__":
    test_equip_reforge()

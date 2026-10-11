"""
Unit test for 2-choice equipment picking and smart slot-based combatant assignment.
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

def test_equip_choice_and_assign():
    f1 = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791703480948_6c0314a3.png")
    f2 = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791703527484_e1159f7e.png")

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

    # 1. Test Image 1: 2-choice equipment screen
    if f1.exists():
        print("=== Test 1: Image 1 (2-choice equipment picking) ===")
        img1 = cv2.imdecode(np.fromfile(str(f1), dtype=np.uint8), cv2.IMREAD_COLOR)
        tgts1 = vision.locate_all_texts(img1)
        state1 = GameState(screen=ScreenState.EQUIP_SELECTION, confidence=0.9)
        state1.raw_texts = [t.text for t in tgts1]
        state1.targets = tgts1

        stub_in1 = StubInput()
        ctx_stub1 = type("Ctx", (), {"input": stub_in1, "overlay": None, "vision": vision})()
        router1 = SceneRouter(ctx_stub1, {})
        res1 = router1.classify_frame(type("F", (), {"img": img1, "corpus": " ".join(state1.raw_texts), "targets": tgts1, "valid": True})())
        print(f"Image 1 Scene Classification: {res1.scene.value} (conf={res1.confidence})")
        assert res1.scene == SceneType.EQUIP_ASSIGN, f"Expected EQUIP_ASSIGN, got {res1.scene}"

        engine1 = NodeDecisionEngine(ctx_stub1, None, {})
        engine1.handle_equip_selection(state1, raw_img=img1)
        print(f"Image 1 Calls: {stub_in1.calls}")
        assert any("选中第1件装备" in (c[-1] or "") for c in stub_in1.calls), "Should click 1st equipment card"
        assert any("ENTER" in (c[-1] or "") or "確認" in (c[-1] or "") for c in stub_in1.calls), "Should confirm equipment choice"
        print("[Test 1 PASS]")

    # 2. Test Image 2: Combatant assignment with glowing slot empty check
    if f2.exists():
        print("\n=== Test 2: Image 2 (Combatant assignment with slot check) ===")
        img2 = cv2.imdecode(np.fromfile(str(f2), dtype=np.uint8), cv2.IMREAD_COLOR)
        tgts2 = vision.locate_all_texts(img2)
        state2 = GameState(screen=ScreenState.EQUIP_SELECTION, confidence=0.9)
        state2.raw_texts = [t.text for t in tgts2]
        state2.targets = tgts2

        stub_in2 = StubInput()
        ctx_stub2 = type("Ctx", (), {"input": stub_in2, "overlay": None, "vision": vision})()
        router2 = SceneRouter(ctx_stub2, {})
        res2 = router2.classify_frame(type("F", (), {"img": img2, "corpus": " ".join(state2.raw_texts), "targets": tgts2, "valid": True})())
        print(f"Image 2 Scene Classification: {res2.scene.value} (conf={res2.confidence})")
        assert res2.scene == SceneType.EQUIP_ASSIGN, f"Expected EQUIP_ASSIGN, got {res2.scene}"

        engine2 = NodeDecisionEngine(ctx_stub2, None, {})
        engine2.handle_equip_selection(state2, raw_img=img2)
        print(f"Image 2 Calls: {stub_in2.calls}")
        # In Image 2, Hero 1 has slot 3, Hero 2 has slot 3, only Hero 3 has slot 3 empty!
        assert any("选择第3战斗员" in (c[-1] or "") for c in stub_in2.calls), f"Must select Hero 3 because Hero 1 & 2 already have accessory equipped! Got: {stub_in2.calls}"
        assert any("ENTER" in (c[-1] or "") or "確認" in (c[-1] or "") for c in stub_in2.calls), "Should confirm assignment"
        print("[Test 2 PASS]")

    print("\nALL EQUIPMENT CHOICE & ASSIGNMENT TESTS PASSED!")

if __name__ == "__main__":
    test_equip_choice_and_assign()

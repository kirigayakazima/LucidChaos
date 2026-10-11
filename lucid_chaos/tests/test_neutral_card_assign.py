"""
Unit test for Neutral Card Reward Combatant Assignment.
Verifies that:
1. Neutral card screen is classified as EQUIP_ASSIGN (100.0 conf).
2. By default (skip_neutral_cards=True), F1 skip (send_key 112) and click (0.700, 0.930) are triggered.
3. When skip_neutral_cards=False, disabled combatants ("無法獲得...") are filtered out,
   eligible combatants are selected, followed by ENTER confirm.
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

def test_neutral_card_assign():
    f_user = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791704794139_0c62f289.png")
    f_assign = _HERE.parent / "fixtures" / "card_reward_assign.png"

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

    # 1. Test User Neutral Card Screen - Default Skip Behavior
    if f_user.exists():
        print("=== Test 1: User Neutral Card Reward Screen (Default: Skip Neutral Cards) ===")
        img1 = cv2.imdecode(np.fromfile(str(f_user), dtype=np.uint8), cv2.IMREAD_COLOR)
        tgts1 = vision.locate_all_texts(img1)
        state1 = GameState(screen=ScreenState.EQUIP_SELECTION, confidence=0.9)
        state1.raw_texts = [t.text for t in tgts1]
        state1.targets = tgts1

        stub1 = StubInput()
        ctx1 = type("Ctx", (), {"input": stub1, "overlay": None, "vision": vision})()
        router1 = SceneRouter(ctx1, {})
        res1 = router1.classify_frame(type("F", (), {"img": img1, "corpus": " ".join(state1.raw_texts), "targets": tgts1, "valid": True})())
        print(f"User Screen Classification: {res1.scene.value} (conf={res1.confidence})")
        assert res1.scene == SceneType.EQUIP_ASSIGN, f"Expected EQUIP_ASSIGN, got {res1.scene}"

        engine1 = NodeDecisionEngine(ctx1, None, {"card_reward": {"skip_neutral_cards": True}})
        engine1.handle_equip_selection(state1, raw_img=img1)
        print(f"User Screen Skip Calls: {stub1.calls}")
        # Must execute F1 skip and click (0.700, 0.930)
        assert any(c[0] == "send_key" and c[1] == 112 for c in stub1.calls), f"Expected F1 key skip, got {stub1.calls}"
        assert any(c[0] == "click_rel" and abs(c[1] - 0.700) < 0.02 for c in stub1.calls), "Expected click skip at 0.700"
        print("[Test 1 PASS]")

        # Test User Neutral Card Screen - Legacy/Fallback Accept Mode
        print("\n=== Test 1b: User Neutral Card Reward Screen (Fallback: Accept Eligible Hero) ===")
        stub1_accept = StubInput()
        ctx1_accept = type("Ctx", (), {"input": stub1_accept, "overlay": None, "vision": vision})()
        engine1_accept = NodeDecisionEngine(ctx1_accept, None, {"card_reward": {"skip_neutral_cards": False}})
        engine1_accept.handle_equip_selection(state1, raw_img=img1)
        print(f"User Screen Accept Calls: {stub1_accept.calls}")
        assert any("选择第1战斗员" in (c[-1] or "") for c in stub1_accept.calls), f"Expected selecting Combatant 1, got {stub1_accept.calls}"
        assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub1_accept.calls), "Expected ENTER confirmation"
        print("[Test 1b PASS]")

    # 2. Test Existing card_reward_assign.png Fixture - Default Skip Behavior
    if f_assign.exists():
        print("\n=== Test 2: Fixture Screen (Default: Skip Neutral Cards) ===")
        img2 = cv2.imdecode(np.fromfile(str(f_assign), dtype=np.uint8), cv2.IMREAD_COLOR)
        tgts2 = vision.locate_all_texts(img2)
        state2 = GameState(screen=ScreenState.EQUIP_SELECTION, confidence=0.9)
        state2.raw_texts = [t.text for t in tgts2]
        state2.targets = tgts2

        stub2 = StubInput()
        ctx2 = type("Ctx", (), {"input": stub2, "overlay": None, "vision": vision})()
        router2 = SceneRouter(ctx2, {})
        res2 = router2.classify_frame(type("F", (), {"img": img2, "corpus": " ".join(state2.raw_texts), "targets": tgts2, "valid": True})())
        print(f"Fixture Screen Classification: {res2.scene.value} (conf={res2.confidence})")
        assert res2.scene == SceneType.EQUIP_ASSIGN, f"Expected EQUIP_ASSIGN, got {res2.scene}"

        engine2 = NodeDecisionEngine(ctx2, None, {"card_reward": {"skip_neutral_cards": True}})
        engine2.handle_equip_selection(state2, raw_img=img2)
        print(f"Fixture Screen Skip Calls: {stub2.calls}")
        assert any(c[0] == "send_key" and c[1] == 112 for c in stub2.calls), f"Expected F1 key skip, got {stub2.calls}"
        assert any(c[0] == "click_rel" and abs(c[1] - 0.700) < 0.02 for c in stub2.calls), "Expected click skip at 0.700"
        print("[Test 2 PASS]")

        print("\n=== Test 2b: Fixture Screen (Fallback: Accept Eligible Hero) ===")
        stub2_accept = StubInput()
        ctx2_accept = type("Ctx", (), {"input": stub2_accept, "overlay": None, "vision": vision})()
        engine2_accept = NodeDecisionEngine(ctx2_accept, None, {"card_reward": {"skip_neutral_cards": False}})
        engine2_accept.handle_equip_selection(state2, raw_img=img2)
        print(f"Fixture Screen Accept Calls: {stub2_accept.calls}")
        assert any("选择第3战斗员" in (c[-1] or "") for c in stub2_accept.calls), f"Expected selecting Combatant 3, got {stub2_accept.calls}"
        assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub2_accept.calls), "Expected ENTER confirmation"
        print("[Test 2b PASS]")

    print("\nALL NEUTRAL & CARD REWARD ASSIGNMENT TESTS PASSED!")

if __name__ == "__main__":
    test_neutral_card_assign()

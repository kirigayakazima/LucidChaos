"""
Unit test for Event Choice Screen containing "Reforge" Reward description.
Verifies that:
1. The screen is NOT falsely locked to SceneType.EQUIP_ASSIGN by text lock rules.
2. The screen is classified as SceneType.EVENT_CHOICE.
3. NodeDecisionEngine.handle_event identifies the choice bubbles and selects Option 2 (Key 2 / click)
   instead of falsely delegating to handle_equip_selection.
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
from lucid_chaos.router.scene_types import SceneType, text_locked_scene
from lucid_chaos.logic.node_engine import NodeDecisionEngine
from lucid_chaos.state.screen_state import GameState, ScreenState


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
        self.calls.append(("click_area_rel", round((a + c) / 2, 3), round((b + d) / 2, 3), kw.get("label")))
        return True

    def send_key(self, vk, **kw):
        self.calls.append(("send_key", vk, kw.get("label")))
        return True

    def tap_card(self, idx, **kw):
        self.calls.append(("tap_card", idx, kw.get("label")))
        return True


def test_event_choice_with_reforge():
    user_img_path = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791721615966_12f48916.png")
    assert user_img_path.exists(), f"Image not found: {user_img_path}"

    vision = VisionService()
    img = cv2.imdecode(np.fromfile(str(user_img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    targets = vision.locate_all_texts(img)
    texts = [t.text for t in targets]
    corpus = " ".join(texts)

    print("=== Test 1: L1 Text Lock Rule on Event with Reforge Description ===")
    locked = text_locked_scene(corpus)
    print(f"Text lock result: {locked}")
    assert locked is None or locked[0] != SceneType.EQUIP_ASSIGN, (
        f"Event screen with reforge reward must NOT be locked to EQUIP_ASSIGN! got {locked}"
    )
    print("[Test 1 PASS]")

    print("=== Test 2: Dynamic Pool Classification ===")
    stub_in = StubInput()
    ctx = type("Ctx", (), {"input": stub_in, "overlay": None, "vision": vision, "capture": None})()
    engine = NodeDecisionEngine(ctx, None, {})
    router = SceneRouter(ctx, {"mode": "system_zero"})
    router.node_engine = engine

    res = router.classifier.classify(img, corpus, targets)
    print(f"Classification result: {res.scene.value} (conf={res.confidence})")
    assert res.scene == SceneType.EVENT_CHOICE, f"Expected EVENT_CHOICE, got {res.scene}"
    assert res.confidence >= 70.0, f"Confidence too low: {res.confidence}"
    print("[Test 2 PASS]")

    print("=== Test 3: Router Dispatch and Event Choice Selection ===")
    state = GameState(screen=ScreenState.UNKNOWN, confidence=0.0)
    state.raw_texts = texts
    state.targets = targets

    ok = router.dispatch(img, state)
    assert ok is True, "Dispatch failed"
    print(f"Executed calls: {stub_in.calls}")

    # Must choose option 2 (tap_card 2 or click near 0.58, 0.82)
    has_card_2 = any(c[0] == "tap_card" and c[1] == 2 for c in stub_in.calls)
    has_click_opt2 = any(c[0] == "click_rel" and abs(c[1] - 0.58) < 0.10 and abs(c[2] - 0.82) < 0.10 for c in stub_in.calls)
    assert has_card_2 or has_click_opt2, f"Expected option 2 to be selected, got: {stub_in.calls}"

    # Must NOT execute any equip selection logic (like clicking heroes at 0.70, 0.51)
    has_hero_equip_click = any(c[0] == "click_area_rel" and "选择第" in str(c[3]) for c in stub_in.calls)
    assert not has_hero_equip_click, "Must not execute equip selection on event screen!"
    print("[Test 3 PASS]")

    print("\nALL EVENT REFORGE TESTS PASSED!")


if __name__ == "__main__":
    test_event_choice_with_reforge()

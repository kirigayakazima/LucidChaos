"""
Unit test for Treasure Chest Vault (寶物盒庫 / 寶箱房間) scene.
Verifies that:
1. detect_treasure_chests detects all 3 chests in the user's real screenshot.
2. SceneRouter.dispatch prioritizes clicking the treasure chests (open_treasure_chest)
   and DOES NOT prematurely tap ENTER on the "繼續進行" button.
3. When no chests remain (all opened), SceneRouter.dispatch proceeds via ENTER.
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
from lucid_chaos.core.features import detect_treasure_chests
from lucid_chaos.router.router import SceneRouter
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


def test_treasure_chests():
    user_img_path = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791722923635_c24d6e4c.jpg")
    assert user_img_path.exists(), f"Image not found: {user_img_path}"

    vision = VisionService()
    img = cv2.imdecode(np.fromfile(str(user_img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    targets = vision.locate_all_texts(img)
    texts = [t.text for t in targets]

    print("=== Test 1: Feature Detection on Treasure Chests ===")
    chests = detect_treasure_chests(img)
    print(f"Detected {len(chests)} chests:")
    for i, (rx, ry, sc) in enumerate(chests):
        print(f"  Chest #{i+1}: pos=({rx:.3f}, {ry:.3f}), score={sc:.3f}")
    assert len(chests) == 3, f"Expected 3 chests, found {len(chests)}"
    print("[Test 1 PASS]")

    print("=== Test 2: Router Dispatch Prioritizes Opening Chest Over Proceed Button ===")
    stub_in = StubInput()
    ctx = type("Ctx", (), {"input": stub_in, "overlay": None, "vision": vision, "capture": None})()
    router = SceneRouter(ctx, {"mode": "system_zero"})

    state = GameState(screen=ScreenState.UNKNOWN, confidence=0.0)
    state.raw_texts = texts
    state.targets = targets

    ok = router.dispatch(img, state)
    assert ok is True, "Dispatch failed"
    print(f"Executed calls: {stub_in.calls}, decision={state.decision}")

    # Must be safety:open_treasure_chest
    assert state.decision == "safety:open_treasure_chest", f"Expected open_treasure_chest, got {state.decision}"
    # Must click one of the chests
    has_chest_click = any(c[0] == "click_rel" and "开启宝箱" in str(c[3]) for c in stub_in.calls)
    assert has_chest_click, f"Expected click on chest! calls={stub_in.calls}"

    # Must NEVER tap confirm / ENTER proceed button when chests exist
    has_premature_proceed = any(c[0] == "tap_action" and c[1] == "confirm" for c in stub_in.calls)
    assert not has_premature_proceed, f"Prematurely tapped proceed button! calls={stub_in.calls}"
    print("[Test 2 PASS]")

    print("=== Test 3: Router Proceeds When All Chests Are Opened ===")
    stub_in_3 = StubInput()
    ctx_3 = type("Ctx", (), {"input": stub_in_3, "overlay": None, "vision": vision, "capture": None})()
    router_3 = SceneRouter(ctx_3, {"mode": "system_zero"})

    # 模拟宝箱开完后的画面（遮盖宝箱区域）
    img_opened = img.copy()
    h, w = img.shape[:2]
    img_opened[int(0.42 * h):int(0.76 * h), int(0.38 * w):int(0.92 * w)] = 0

    state_3 = GameState(screen=ScreenState.UNKNOWN, confidence=0.0)
    state_3.raw_texts = texts
    state_3.targets = targets

    ok_3 = router_3.dispatch(img_opened, state_3)
    assert ok_3 is True
    print(f"Executed calls when opened: {stub_in_3.calls}, decision={state_3.decision}")
    assert state_3.decision == "safety:proceed_button", f"Expected proceed_button, got {state_3.decision}"
    has_proceed_enter = any(c[0] == "tap_action" and c[1] == "confirm" and "繼續進行" in str(c[2]) for c in stub_in_3.calls)
    assert has_proceed_enter, f"Expected ENTER proceed! calls={stub_in_3.calls}"
    print("[Test 3 PASS]")

    print("\nALL TREASURE CHEST TESTS PASSED!")


if __name__ == "__main__":
    test_treasure_chests()

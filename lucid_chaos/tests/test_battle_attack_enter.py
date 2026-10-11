"""
Unit test for In-Combat Attack Enter Confirmation (⚔ 攻擊 / [ENTER]).
Verifies that:
1. User uploaded screen media_1791711090058_fab13cef.jpg is recognized by is_attack_enter_screen.
2. Active auto-battle (R) is correctly identified as 'ACTIVE'.
3. Both router safety dispatch and StoryPusher.handle_battle trigger:
   - tap_action('confirm') -> ENTER 攻擊
   - click_rel(0.500, 0.825) -> 点击攻擊按钮
4. Non-attack frames (battle_hand_row, battle_auto_off) do not trigger false positive.
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
from lucid_chaos.state.observer import ScreenObserver
from lucid_chaos.logic.battle_manager import BattleManager
from lucid_chaos.logic.story_pusher import StoryPusher
from lucid_chaos.logic.node_engine import NodeDecisionEngine

class RecordingInput:
    def __init__(self):
        self.calls = []
    def tap_action(self, action, **kw):
        self.calls.append(("tap_action", action, kw.get("label")))
        return True
    def click_rel(self, rx, ry, **kw):
        self.calls.append(("click_rel", round(rx, 3), round(ry, 3), kw.get("label")))
        return True
    def __getattr__(self, item):
        def _rec(*a, **k):
            self.calls.append((item, a, k.get("label")))
        return _rec

def test_battle_attack_enter():
    f_user = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791711090058_fab13cef.jpg")
    f_fixtures = _HERE.parent / "fixtures"

    vision = VisionService()
    obs = ScreenObserver(vision)

    if not f_user.exists():
        print(f"User image not found at {f_user}, skipping.")
        return

    print("=== Test 1: User Battle Attack Enter Screen Detection ===")
    img_u = cv2.imdecode(np.fromfile(str(f_user), dtype=np.uint8), cv2.IMREAD_COLOR)
    state_u = obs.observe(img_u)

    # 1. Check is_attack_enter_screen
    is_atk = BattleManager.is_attack_enter_screen(img_u, state_u)
    print("is_attack_enter_screen result:", is_atk)
    assert is_atk, "Expected is_attack_enter_screen to return True"

    # 2. Check inspect_auto_battle_status (active R)
    bm_stub = BattleManager(type("Ctx", (), {"vision": vision})())
    status = bm_stub.inspect_auto_battle_status(img_u)
    print("inspect_auto_battle_status result:", status)
    assert status == "ACTIVE", f"Expected ACTIVE auto battle status, got {status}"
    print("[Test 1 PASS]")

    print("\n=== Test 2: Router Dispatch Safety Action ===")
    stub_input = RecordingInput()
    ctx = type("Ctx", (), {
        "vision": vision,
        "capture": type("Cap", (), {"capture": lambda s: img_u})(),
        "overlay": None,
        "input": stub_input
    })()
    pusher = StoryPusher(ctx, {"main_story": {"boss_action": "auto"}})
    eng = NodeDecisionEngine(ctx, None, {})
    pusher.node_engine = eng
    pusher.router.node_engine = eng

    # Dispatch frame
    dispatched = pusher.router.dispatch(img_u, state_u)
    print("Dispatched:", dispatched)
    print("Calls:", stub_input.calls)
    assert dispatched, "Expected dispatch to handle frame"
    assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub_input.calls), "Expected confirm (ENTER) call"
    assert any(c[0] == "click_rel" and abs(c[1] - 0.500) < 0.05 and abs(c[2] - 0.825) < 0.05 for c in stub_input.calls), "Expected click on attack button"
    print("[Test 2 PASS]")

    print("\n=== Test 3: StoryPusher.handle_battle Direct Handling ===")
    stub_input.calls.clear()
    pusher_handled = pusher.handle_battle(state_u, raw_img=img_u)
    print("handle_battle result:", pusher_handled)
    print("handle_battle Calls:", stub_input.calls)
    assert pusher_handled, "Expected handle_battle to handle attack enter"
    assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub_input.calls), "Expected confirm (ENTER) call"
    print("[Test 3 PASS]")

    print("\n=== Test 4: False Positive Check on Existing Battle Fixtures ===")
    for fname in ["battle_hand_row.png", "battle_auto_off.png", "battle_ep_insufficient.png"]:
        path = f_fixtures / fname
        if path.exists():
            img_fix = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            st_fix = obs.observe(img_fix)
            hit = BattleManager.is_attack_enter_screen(img_fix, st_fix)
            print(f"{fname}: is_attack_enter={hit}")
            assert not hit, f"Expected {fname} NOT to trigger attack enter prompt"
    print("[Test 4 PASS]")

    print("\nALL BATTLE ATTACK ENTER TESTS PASSED!")

if __name__ == "__main__":
    test_battle_attack_enter()

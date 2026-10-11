"""
Unit tests verifying Home Lobby, Account Info, and All-Out Battle routing & safety guards.
Guarantees:
1. Home Lobby is reliably locked to SceneType.HOME_LOBBY, dispatches key 4 / chaos card click,
   and NEVER gets falsely detected as BATTLE or triggers all-out battle safe exit.
2. Account Info popup triggers safety:account_info_exit (clicks < or ESC).
3. All-Out Battle interior triggers safety:all_out_battle_exit (clicks <).
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
from lucid_chaos.state.screen_state import GameState, ScreenState
from lucid_chaos.state.observer import ScreenObserver


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


def read_img(path: Path) -> np.ndarray:
    assert path.exists(), f"Image not found: {path}"
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def test_home_lobby_and_safety():
    fixtures_dir = _HERE.parent / "fixtures"
    vision = VisionService()
    observer = ScreenObserver(vision)

    p_lobby = fixtures_dir / "home_lobby_real.jpg"
    p_account = fixtures_dir / "account_info_dialog.png"
    p_all_out = fixtures_dir / "all_out_battle_interior.png"

    print("=== Testing Home Lobby, Account Info & All-Out Battle ===")

    # -------------------------------------------------------------
    # Case 1: Real Home Lobby Screen
    # -------------------------------------------------------------
    img_lobby = read_img(p_lobby)
    tgts_lobby = vision.locate_all_texts(img_lobby)
    texts_lobby = [t.text for t in tgts_lobby]
    corpus_lobby = " ".join(texts_lobby)

    # 1.1 Observer must NOT classify Home Lobby as BATTLE
    gs_obs = observer.observe(img_lobby)
    print(f"[Case 1.1] Observer on Home Lobby: {gs_obs.screen.name} (confidence={gs_obs.confidence})")
    assert gs_obs.screen != ScreenState.BATTLE, f"Home Lobby should not be classified as BATTLE! got {gs_obs.screen}"

    # 1.2 Text lock must lock to HOME_LOBBY
    locked = text_locked_scene(corpus_lobby)
    print(f"[Case 1.2] Text Lock on Home Lobby: {locked}")
    assert locked is not None, "Home Lobby failed to hit TEXT_LOCK_RULES!"
    assert locked[0] == SceneType.HOME_LOBBY, f"Expected HOME_LOBBY, got {locked[0]}"

    # 1.3 Dispatch must perform press 4 / click chaos card
    stub_in_1 = StubInput()
    ctx_1 = type("Ctx", (), {"input": stub_in_1, "overlay": None, "vision": vision})()
    router_1 = SceneRouter(ctx_1, {"mode": "system_zero"})
    state_lobby = GameState(screen=ScreenState.UNKNOWN, confidence=0.0)
    state_lobby.raw_texts = texts_lobby
    state_lobby.targets = tgts_lobby

    ok_lobby = router_1.dispatch(img_lobby, state_lobby)
    assert ok_lobby is True, "Home Lobby dispatch failed!"
    print(f"[Case 1.3] Home Lobby actions: {stub_in_1.calls}")
    # Must send key 4 (0x34) or click chaos card
    has_key_4 = any(c[0] == "send_key" and c[1] == 0x34 for c in stub_in_1.calls)
    has_chaos_click = any(c[0] == "click_area_rel" and "卡厄思" in str(c[3]) for c in stub_in_1.calls)
    assert has_key_4 or has_chaos_click, f"Home Lobby must enter chaos mode! calls={stub_in_1.calls}"

    # Must NEVER click top-left exit in home lobby (no avatar misclick!)
    for call in stub_in_1.calls:
        if call[0] in ("click_rel", "click_area_rel"):
            rx, ry = call[1], call[2]
            assert not (rx < 0.10 and ry < 0.10), f"Home Lobby clicked top-left avatar area! call={call}"
    print("[PASS] Case 1: Home Lobby verification passed.")

    # -------------------------------------------------------------
    # Case 2: Account Info Dialog Screen (误入账号资讯)
    # -------------------------------------------------------------
    img_account = read_img(p_account)
    tgts_account = vision.locate_all_texts(img_account)
    texts_account = [t.text for t in tgts_account]

    stub_in_2 = StubInput()
    ctx_2 = type("Ctx", (), {"input": stub_in_2, "overlay": None, "vision": vision})()
    router_2 = SceneRouter(ctx_2, {"mode": "system_zero"})
    state_account = GameState(screen=ScreenState.UNKNOWN, confidence=0.0)
    state_account.raw_texts = texts_account
    state_account.targets = tgts_account

    ok_account = router_2.dispatch(img_account, state_account)
    assert ok_account is True, "Account info dispatch failed!"
    print(f"[Case 2] Account Info actions: {stub_in_2.calls}, decision={state_account.decision}")
    assert state_account.decision == "safety:account_info_exit", f"Expected safety:account_info_exit, got {state_account.decision}"
    print("[PASS] Case 2: Account Info safety exit passed.")

    # -------------------------------------------------------------
    # Case 3: All-Out Battle Interior Screen (误入全力战强敌)
    # -------------------------------------------------------------
    img_all_out = read_img(p_all_out)
    tgts_all_out = vision.locate_all_texts(img_all_out)
    texts_all_out = [t.text for t in tgts_all_out]

    stub_in_3 = StubInput()
    ctx_3 = type("Ctx", (), {"input": stub_in_3, "overlay": None, "vision": vision})()
    router_3 = SceneRouter(ctx_3, {"mode": "system_zero"})
    state_all_out = GameState(screen=ScreenState.UNKNOWN, confidence=0.0)
    state_all_out.raw_texts = texts_all_out
    state_all_out.targets = tgts_all_out

    ok_all_out = router_3.dispatch(img_all_out, state_all_out)
    assert ok_all_out is True, "All-out battle interior dispatch failed!"
    print(f"[Case 3] All-Out Battle actions: {stub_in_3.calls}, decision={state_all_out.decision}")
    assert state_all_out.decision == "safety:all_out_battle_exit", f"Expected safety:all_out_battle_exit, got {state_all_out.decision}"
    print("[PASS] Case 3: All-Out Battle safety exit passed.")

    print("\nALL HOME LOBBY & SAFETY TESTS PASSED!")


if __name__ == "__main__":
    test_home_lobby_and_safety()

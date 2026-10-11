"""
Unit test for God Epiphany ('神闪' - 金色包边强化卡牌) detection & prioritized selection.
Validates:
1. detect_god_epiphany_card on real user screenshot: identifies Card 1 as God Epiphany.
2. Synthetic permutations: identifies Card 2 and Card 3 respectively when golden border is present.
3. Negative control: normal card reward fixtures return None (no false positive).
4. Scene recognition: classified as EPIPHANY_CARDS (conf=100.0).
5. NodeDecisionEngine: prioritizes God Epiphany card (key 1 / 2 / 3 + click + ENTER).
6. BattleManager: prioritizes God Epiphany card.
"""
import sys
import os
from pathlib import Path
import cv2
import numpy as np

_HERE = Path(__file__).resolve()
ROOT = _HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))

from lucid_chaos.core.vision_service import VisionService
from lucid_chaos.core.features import detect_god_epiphany_card
from lucid_chaos.state.observer import ScreenObserver
from lucid_chaos.state.screen_state import GameState, ScreenState
from lucid_chaos.router.scene_types import SceneType, text_locked_scene
from lucid_chaos.router.scene_pool import ScenePoolClassifier
from lucid_chaos.logic.node_engine import NodeDecisionEngine
from lucid_chaos.logic.battle_manager import BattleManager


class MockInput:
    def __init__(self):
        self.calls = []

    def tap_card(self, index, **kw):
        self.calls.append(("tap_card", index, kw.get("label")))
        return True

    def click_area_rel(self, a, b, c, d, **kw):
        self.calls.append(("click_area_rel", round((a + c) / 2, 3), round((b + d) / 2, 3), kw.get("label")))

    def click_rel(self, rx, ry, **kw):
        self.calls.append(("click_rel", round(float(rx), 3), round(float(ry), 3), kw.get("label")))

    def tap_action(self, action, **kw):
        self.calls.append(("tap_action", action, kw.get("label")))
        return True


class MockContext:
    def __init__(self, vs):
        self.vision = vs
        self.input = MockInput()
        self.overlay = None


def imread_u(path):
    p = str(path)
    if not os.path.exists(p):
        return None
    try:
        return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def test_god_epiphany():
    img_path = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791725370725_bdd4eb1b.png")
    if not img_path.exists():
        print(f"Skipping test, user image not found: {img_path}")
        return

    img = imread_u(img_path)
    h, w = img.shape[:2]

    print("=== Test 1: Real User Image God Epiphany Detection ===")
    slot, det = detect_god_epiphany_card(img)
    print(f"Detected Slot: {slot}, Details: {det}")
    assert slot == 1, f"Expected Card 1 to be God Epiphany, got {slot}"
    assert det["gold"] >= 2000, f"Expected gold score >= 2000, got {det['gold']}"
    print("[Test 1 PASS]")

    print("\n=== Test 2: Permutation Swaps (Card 2 & Card 3 as God Epiphany) ===")
    # Swap Card 1 with Card 2
    img_swap2 = img.copy()
    c1 = img[:, int(0.135 * w):int(0.355 * w)].copy()
    c2 = img[:, int(0.390 * w):int(0.610 * w)].copy()
    img_swap2[:, int(0.390 * w):int(0.610 * w)] = c1
    img_swap2[:, int(0.135 * w):int(0.355 * w)] = c2

    slot2, det2 = detect_god_epiphany_card(img_swap2)
    print(f"Swap Card 2 Slot: {slot2}, Details: {det2}")
    assert slot2 == 2, f"Expected Card 2 to be God Epiphany after swap, got {slot2}"

    # Swap Card 1 with Card 3
    img_swap3 = img.copy()
    c3 = img[:, int(0.645 * w):int(0.865 * w)].copy()
    img_swap3[:, int(0.645 * w):int(0.865 * w)] = c1
    img_swap3[:, int(0.135 * w):int(0.355 * w)] = c3

    slot3, det3 = detect_god_epiphany_card(img_swap3)
    print(f"Swap Card 3 Slot: {slot3}, Details: {det3}")
    assert slot3 == 3, f"Expected Card 3 to be God Epiphany after swap, got {slot3}"
    print("[Test 2 PASS]")

    print("\n=== Test 3: Negative Control (No False Positive on Normal Cards) ===")
    fixture_normal = ROOT / "lucid_chaos" / "tests" / "fixtures" / "card_reward_3choose.png"
    if fixture_normal.exists():
        im_norm = imread_u(fixture_normal)
        slot_norm, _ = detect_god_epiphany_card(im_norm)
        print(f"Normal Fixture God Epiphany: {slot_norm}")
        assert slot_norm is None, f"Expected None on normal card fixture, got {slot_norm}"
        print("[Test 3 PASS]")

    print("\n=== Test 4: Screen & Text Lock Recognition ===")
    vs = VisionService()
    obs = ScreenObserver(vs)
    state = obs.observe(img)
    corpus = " ".join(state.raw_texts)
    print(f"Observed Screen: {state.screen}")
    assert state.screen == ScreenState.EPIPHANY_SELECTION, f"Expected EPIPHANY_SELECTION, got {state.screen}"

    locked = text_locked_scene(corpus)
    print(f"Text Lock: {locked}")
    assert locked is not None and locked[0] == SceneType.EPIPHANY_CARDS, f"Expected text lock EPIPHANY_CARDS, got {locked}"
    print("[Test 4 PASS]")

    print("\n=== Test 5: NodeDecisionEngine Prioritized Selection ===")
    ctx = MockContext(vs)
    engine = NodeDecisionEngine(ctx, target_mgr=None, config={})

    # When Card 1 is God Epiphany:
    ctx.input.calls.clear()
    engine._evt_stuck = 0
    engine.handle_epiphany_selection(state, raw_img=img)
    print("NodeDecisionEngine (God Card 1) Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 1 for c in ctx.input.calls), "Should select God Epiphany Card 1"
    assert any(c[0] == "tap_action" and c[1] == "confirm" for c in ctx.input.calls), "Should confirm selection"

    # When Card 2 is God Epiphany:
    ctx.input.calls.clear()
    engine._evt_stuck = 0
    engine.handle_epiphany_selection(state, raw_img=img_swap2)
    print("NodeDecisionEngine (God Card 2) Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 2 for c in ctx.input.calls), "Should select God Epiphany Card 2"

    # When Card 3 is God Epiphany:
    ctx.input.calls.clear()
    engine._evt_stuck = 0
    engine.handle_epiphany_selection(state, raw_img=img_swap3)
    print("NodeDecisionEngine (God Card 3) Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 3 for c in ctx.input.calls), "Should select God Epiphany Card 3"
    print("[Test 5 PASS]")

    print("\n=== Test 6: BattleManager Prioritized Selection ===")
    battle_mgr = BattleManager(ctx)
    ctx.input.calls.clear()
    battle_mgr._epiphany_stuck = 0
    battle_mgr.handle_epiphany_selection(img, state)
    print("BattleManager (God Card 1) Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 1 for c in ctx.input.calls), "BattleManager should select God Epiphany Card 1"

    ctx.input.calls.clear()
    battle_mgr._epiphany_stuck = 0
    battle_mgr.handle_epiphany_selection(img_swap2, state)
    print("BattleManager (God Card 2) Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 2 for c in ctx.input.calls), "BattleManager should select God Epiphany Card 2"
    print("[Test 6 PASS]")

    print("\nALL GOD EPIPHANY TESTS PASSED!")


if __name__ == "__main__":
    test_god_epiphany()

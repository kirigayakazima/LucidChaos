"""
Unit test for Epiphany Hero Selection Rotation Ladder.
Validates:
1. Classification of media_1791723756099_ed92a911.png as SceneType.EPIPHANY_HERO.
2. NodeDecisionEngine ladder: stuck=0 -> Hero 1, stuck=1 -> Hero 2, stuck=2 -> Hero 3.
3. BattleManager ladder: stuck=0 -> Hero 1, stuck=1 -> Hero 2, stuck=2 -> Hero 3.
4. Auto-delegation from handle_epiphany_selection to handle_epiphany_hero_selection.
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
from lucid_chaos.state.observer import ScreenObserver
from lucid_chaos.state.screen_state import GameState, ScreenState
from lucid_chaos.router.scene_types import SceneType
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


def test_epiphany_hero_rotation():
    img_path = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791723756099_ed92a911.png")
    if not img_path.exists():
        print(f"Skipping image classification test, file not found: {img_path}")
        img = np.zeros((576, 1024, 3), dtype=np.uint8)
    else:
        img = cv2.imread(str(img_path))

    vs = VisionService()
    obs = ScreenObserver(vs)
    ctx = MockContext(vs)

    print("=== Test 1: Screen & Scene Recognition ===")
    if img_path.exists():
        state = obs.observe(img)
        print(f"Observed Screen: {state.screen}")
        assert state.screen == ScreenState.EPIPHANY_SELECTION, f"Expected EPIPHANY_SELECTION, got {state.screen}"

        classifier = ScenePoolClassifier()
        corpus = " ".join(state.raw_texts)
        res = classifier.classify(img, corpus, state.targets)
        print(f"Classifier Scene: {res.scene.name}, Conf: {res.confidence}")
        assert res.scene == SceneType.EPIPHANY_HERO, f"Expected EPIPHANY_HERO, got {res.scene}"

    print("\n=== Test 2: NodeDecisionEngine Rotation Ladder ===")
    node_engine = NodeDecisionEngine(ctx, target_mgr=None, config={})
    dummy_state = GameState(
        screen=ScreenState.EPIPHANY_SELECTION,
        raw_texts=["請選擇欲獲得【靈光一閃】的戰鬥員", "確認", "ENTER"],
        confidence=0.95
    )

    # Round 0 (stuck = 0): Slot 1
    ctx.input.calls.clear()
    node_engine._evt_stuck = 0
    node_engine._event_scene_same = lambda raw_img: False
    node_engine.handle_epiphany_hero_selection(dummy_state, raw_img=img)
    print("Round 0 Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 1 for c in ctx.input.calls), "Should select Slot 1 on round 0"

    # Round 1 (stuck = 1): Slot 2
    ctx.input.calls.clear()
    node_engine._evt_stuck = 1
    node_engine._event_scene_same = lambda raw_img: True
    node_engine.handle_epiphany_hero_selection(dummy_state, raw_img=img)
    print("Round 1 Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 2 for c in ctx.input.calls), "Should switch to Slot 2 on round 1"

    # Round 2 (stuck = 2): Slot 3
    ctx.input.calls.clear()
    node_engine._evt_stuck = 2
    node_engine._event_scene_same = lambda raw_img: True
    node_engine.handle_epiphany_hero_selection(dummy_state, raw_img=img)
    print("Round 2 Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 3 for c in ctx.input.calls), "Should switch to Slot 3 on round 2"

    print("\n=== Test 3: BattleManager Rotation Ladder ===")
    battle_mgr = BattleManager(ctx)
    battle_mgr.is_attack_enter_screen = lambda raw_img, state: False

    # Round 0 (stuck = 0): Slot 1
    ctx.input.calls.clear()
    battle_mgr._epiphany_stuck = 0
    battle_mgr._epiphany_scene_same = lambda raw_img: False
    battle_mgr.handle_epiphany_selection(img, dummy_state)
    print("BattleManager Round 0 Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 1 for c in ctx.input.calls), "BattleManager should select Slot 1 on round 0"

    # Round 1 (stuck = 1): Slot 2
    ctx.input.calls.clear()
    battle_mgr._epiphany_stuck = 1
    battle_mgr._epiphany_scene_same = lambda raw_img: True
    battle_mgr.handle_epiphany_selection(img, dummy_state)
    print("BattleManager Round 1 Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 2 for c in ctx.input.calls), "BattleManager should switch to Slot 2 on round 1"

    # Round 2 (stuck = 2): Slot 3
    ctx.input.calls.clear()
    battle_mgr._epiphany_stuck = 2
    battle_mgr._epiphany_scene_same = lambda raw_img: True
    battle_mgr.handle_epiphany_selection(img, dummy_state)
    print("BattleManager Round 2 Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 3 for c in ctx.input.calls), "BattleManager should switch to Slot 3 on round 2"

    print("\n=== Test 4: Auto-delegation via handle_epiphany_selection ===")
    ctx.input.calls.clear()
    node_engine._evt_stuck = 1
    node_engine._event_scene_same = lambda raw_img: True
    node_engine.handle_epiphany_selection(dummy_state, raw_img=img)
    print("Delegation Calls:", ctx.input.calls)
    assert any(c[0] == "tap_card" and c[1] == 2 for c in ctx.input.calls), "Auto-delegation should route to hero ladder"

    print("\nALL EPIPHANY HERO ROTATION TESTS PASSED!")


if __name__ == "__main__":
    test_epiphany_hero_rotation()

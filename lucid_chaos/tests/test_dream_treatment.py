"""
Unit test for Dream Mode (夢境模式) and Treatment Center (治療中心).
Verifies:
1. media_1791713114494_e1bdcd8a.png is recognized as SceneType.DREAM_MODE (100%),
   and triggers click on bottom-left 【治療中心】 (0.088, 0.917).
2. media_1791713173059_1edd98fe.png is recognized as SceneType.TREATMENT_CENTER (100%),
   detects the 2 trauma badges on 千鶴 and 賽雷妮爾, clicks the pod, clicks 【消除記憶】 (0.550, 0.925),
   and confirms with ENTER.
3. media_1791713155008_fd3f9126.png is recognized as SceneType.STAGE_MAP (配队界面),
   and does not misroute to VICTORY_SETTLEMENT.
4. Non-dream/treatment fixtures are not falsely triggered.
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
from lucid_chaos.router.router import SceneRouter
from lucid_chaos.router.scene_types import SceneType, text_locked_scene
from lucid_chaos.logic.story_pusher import StoryPusher

class RecordingInput:
    def __init__(self):
        self.calls = []
    def tap_action(self, action, **kw):
        self.calls.append(("tap_action", action, kw.get("label")))
        return True
    def click_rel(self, rx, ry, **kw):
        self.calls.append(("click_rel", round(rx, 3), round(ry, 3), kw.get("label")))
        return True
    def click_area_rel(self, x1, y1, x2, y2, **kw):
        rx = round((x1 + x2) / 2, 3)
        ry = round((y1 + y2) / 2, 3)
        self.calls.append(("click_area_rel", rx, ry, kw.get("label")))
        return True
    def __getattr__(self, item):
        def _rec(*a, **k):
            self.calls.append((item, a, k.get("label")))
        return _rec

def test_dream_mode_and_treatment():
    f_dream = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791713114494_e1bdcd8a.png")
    f_team = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791713155008_fd3f9126.png")
    f_treat = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791713173059_1edd98fe.png")

    vision = VisionService()
    obs = ScreenObserver(vision)

    # -------------------------------------------------------------
    # Test 1: Dream Mode (夢境模式)
    # -------------------------------------------------------------
    print("=== Test 1: Dream Mode Detection & Click Treatment Center ===")
    img_dream = cv2.imdecode(np.fromfile(str(f_dream), dtype=np.uint8), cv2.IMREAD_COLOR)
    st_dream = obs.observe(img_dream)
    corpus_dream = " ".join(st_dream.raw_texts)

    locked = text_locked_scene(corpus_dream)
    print("Dream screen text lock:", locked)
    assert locked is not None and locked[0] == SceneType.DREAM_MODE, f"Expected DREAM_MODE text lock, got {locked}"

    stub_input = RecordingInput()
    ctx = type("Ctx", (), {
        "vision": vision,
        "capture": type("Cap", (), {"capture": lambda s: img_dream})(),
        "overlay": None,
        "input": stub_input
    })()
    pusher = StoryPusher(ctx, {})
    router = SceneRouter(ctx, {}, story_pusher=pusher)

    res = router.classifier.classify(img_dream, corpus_dream, st_dream.targets)
    print("Dream classifier winning scene:", res.scene, "score:", res.confidence)
    assert res.scene == SceneType.DREAM_MODE, f"Expected DREAM_MODE, got {res.scene}"

    dispatched = router.dispatch(img_dream, st_dream)
    print("Dispatched dream mode:", dispatched)
    print("Calls:", stub_input.calls)
    assert dispatched, "Expected dispatch to handle dream mode"
    # Verify clicked on bottom-left 治療中心 (rx around 0.088, ry around 0.917)
    assert any(("click" in c[0] and abs(c[1] - 0.088) < 0.06 and abs(c[2] - 0.917) < 0.06) for c in stub_input.calls), \
        f"Expected click on 治療中心 button, got {stub_input.calls}"
    print("[Test 1 PASS]")

    # -------------------------------------------------------------
    # Test 2: Treatment Center (治療中心)
    # -------------------------------------------------------------
    print("\n=== Test 2: Treatment Center Detection & Memory Wipe Routine ===")
    img_treat = cv2.imdecode(np.fromfile(str(f_treat), dtype=np.uint8), cv2.IMREAD_COLOR)
    st_treat = obs.observe(img_treat)
    corpus_treat = " ".join(st_treat.raw_texts)

    locked_t = text_locked_scene(corpus_treat)
    print("Treatment screen text lock:", locked_t)
    assert locked_t is not None and locked_t[0] == SceneType.TREATMENT_CENTER, f"Expected TREATMENT_CENTER text lock, got {locked_t}"

    stub_input.calls.clear()
    res_t = router.classifier.classify(img_treat, corpus_treat, st_treat.targets)
    print("Treatment classifier winning scene:", res_t.scene, "score:", res_t.confidence)
    assert res_t.scene == SceneType.TREATMENT_CENTER, f"Expected TREATMENT_CENTER, got {res_t.scene}"

    dispatched_t = router.dispatch(img_treat, st_treat)
    print("Dispatched treatment:", dispatched_t)
    print("Calls:", stub_input.calls)
    assert dispatched_t, "Expected dispatch to handle treatment"
    # Should click pod 1 (around rx 0.41, ry 0.36)
    assert any("click" in c[0] and abs(c[1] - 0.41) < 0.08 and abs(c[2] - 0.36) < 0.08 for c in stub_input.calls), \
        f"Expected click on character pod, got {stub_input.calls}"
    # Should click 消除記憶 (around rx 0.55, ry 0.925)
    assert any("click" in c[0] and abs(c[1] - 0.55) < 0.08 and abs(c[2] - 0.925) < 0.06 for c in stub_input.calls), \
        f"Expected click on 消除記憶 button, got {stub_input.calls}"
    # Should confirm with ENTER
    assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub_input.calls), \
        f"Expected ENTER confirmation, got {stub_input.calls}"
    print("[Test 2 PASS]")

    # -------------------------------------------------------------
    # Test 2b: Treatment Center Exit when all characters cured
    # -------------------------------------------------------------
    print("\n=== Test 2b: Treatment Center Exit when cured ===")
    img_cured = img_treat.copy()
    # Mask out the badge row so 0 badges detected
    h_t, w_t = img_cured.shape[:2]
    img_cured[int(0.44 * h_t):int(0.53 * h_t), int(0.25 * w_t):int(0.85 * w_t)] = 0
    stub_input.calls.clear()
    pusher.handle_treatment_center(st_treat, raw_img=img_cured)
    print("Calls when cured:", stub_input.calls)
    assert any("click" in c[0] and c[1] <= 0.08 and c[2] <= 0.08 for c in stub_input.calls), \
        f"Expected click on back button, got {stub_input.calls}"
    print("[Test 2b PASS]")

    # -------------------------------------------------------------
    # Test 3: Team Config Screen (配置隊伍)
    # -------------------------------------------------------------
    print("\n=== Test 3: Team Config Screen Text Lock ===")
    img_team = cv2.imdecode(np.fromfile(str(f_team), dtype=np.uint8), cv2.IMREAD_COLOR)
    st_team = obs.observe(img_team)
    corpus_team = " ".join(st_team.raw_texts)

    locked_team = text_locked_scene(corpus_team)
    print("Team screen text lock:", locked_team)
    assert locked_team is not None and locked_team[0] == SceneType.STAGE_MAP, f"Expected STAGE_MAP text lock, got {locked_team}"

    # Verify settlement loot vetoes team config
    score_settle, det_settle = router.classifier.EVALUATORS if hasattr(router.classifier, "EVALUATORS") else (0, {})
    # Call directly via SceneEvaluators
    from lucid_chaos.router.evaluators import SceneEvaluators
    s_settle, _ = SceneEvaluators.eval_settlement_loot(img_team, corpus_team, st_team.targets)
    print("eval_settlement_loot on team config:", s_settle)
    assert s_settle == 0.0, f"Expected 0.0 settlement score on team config, got {s_settle}"
    # -------------------------------------------------------------
    # Test 4: Treatment Center Modal Secondary Confirmation (真机弹窗 media_1791714616227_d1e35cca.png)
    # -------------------------------------------------------------
    print("\n=== Test 4: Treatment Center Secondary Confirmation Modal ===")
    f_modal = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791714616227_d1e35cca.png")
    img_modal = cv2.imdecode(np.fromfile(str(f_modal), dtype=np.uint8), cv2.IMREAD_COLOR)
    st_modal = obs.observe(img_modal)
    corpus_modal = " ".join(st_modal.raw_texts)

    stub_input.calls.clear()
    handled_modal = pusher.handle_treatment_center(st_modal, raw_img=img_modal)
    print("Handled modal:", handled_modal)
    print("Calls for modal:", stub_input.calls)
    assert handled_modal, "Expected handle_treatment_center to handle confirmation modal"
    # Verify clicked on orange confirmation button (rx ~ 0.66, ry ~ 0.70)
    assert any("click" in c[0] and abs(c[1] - 0.66) < 0.10 and abs(c[2] - 0.70) < 0.06 for c in stub_input.calls), \
        f"Expected click on orange 消除記憶 confirm button, got {stub_input.calls}"
    # Verify sent ENTER
    assert any(c[0] == "tap_action" and c[1] == "confirm" for c in stub_input.calls), \
        f"Expected ENTER confirmation, got {stub_input.calls}"
    # Verify did NOT click back button
    assert not any("click" in c[0] and c[1] < 0.10 and c[2] < 0.10 for c in stub_input.calls), \
        f"Should NOT click back button when modal is open! Calls: {stub_input.calls}"
    # -------------------------------------------------------------
    # Test 5: Anti-Loop Protection against ScreenOverlay Debug Badge
    # -------------------------------------------------------------
    print("\n=== Test 5: Anti-Loop Protection against ScreenOverlay HUD Badge ===")
    f_stuck = Path(r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\20261011-183714_MAP_NAVIGATION_治療中心_stuck5.png")
    if f_stuck.exists():
        img_stuck = cv2.imdecode(np.fromfile(str(f_stuck), dtype=np.uint8), cv2.IMREAD_COLOR)
        st_stuck = obs.observe(img_stuck)
        corpus_stuck = " ".join(st_stuck.raw_texts)
        print("Stuck frame corpus:", corpus_stuck)
        # 1. Verify overlay HUD text is filtered out
        assert "路由" not in corpus_stuck, f"Overlay badge text must be filtered out! Got: {corpus_stuck}"
        # 2. Verify text lock does NOT lock to TREATMENT_CENTER
        locked_stuck = text_locked_scene(corpus_stuck)
        print("Stuck frame locked scene:", locked_stuck)
        assert locked_stuck is None or locked_stuck[0] != SceneType.TREATMENT_CENTER, \
            f"Must NOT lock to TREATMENT_CENTER on lobby/stuck frame! Got: {locked_stuck}"
        # 3. Verify handle_treatment_center safely returns False without spurious exit actions
        stub_input.calls.clear()
        handled_stuck = pusher.handle_treatment_center(st_stuck, raw_img=img_stuck)
        print("Handled stuck:", handled_stuck, "Calls:", stub_input.calls)
        assert not handled_stuck, "handle_treatment_center should return False on non-treatment frame"
        assert len(stub_input.calls) == 0, f"Should NOT perform exit actions on non-treatment frame! Calls: {stub_input.calls}"
        print("[Test 5 PASS]")

    print("\nALL DREAM MODE & TREATMENT CENTER TESTS PASSED!")

if __name__ == "__main__":
    test_dream_mode_and_treatment()

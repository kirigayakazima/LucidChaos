"""
Unit test for voucher refill modal and codex fix.
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
from lucid_chaos.state.screen_state import GameState

def test_codex_and_voucher():
    ws_dir = _HERE.parent.parent
    fixtures_auto = ws_dir / "lucid_chaos" / "fixtures_auto"

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
        def tap_card(self, idx, **kw):
            self.calls.append(("tap_card", idx, kw.get("label")))
            return True

    # 1. Test 142841: Codex Select
    p_codex = fixtures_auto / "20261011-142841_EPIPHANY_SELECTION_灵光一闪选牌.png"
    if p_codex.exists():
        img_codex = cv2.imdecode(np.fromfile(str(p_codex), dtype=np.uint8), cv2.IMREAD_COLOR)
        tgts_codex = vision.locate_all_texts(img_codex)
        state_codex = GameState()
        state_codex.raw_texts = [t.text for t in tgts_codex]
        state_codex.targets = tgts_codex

        stub_in = StubInput()
        ctx_stub = type("Ctx", (), {"input": stub_in, "overlay": None, "vision": vision})()
        router = SceneRouter(ctx_stub, {"mode": "system_zero", "system_zero": {"target_codex": 3}, "archive": {"use_travel_ticket": True}})

        res = router.classifier.classify(img_codex, " ".join(state_codex.raw_texts), tgts_codex)
        print(f"[TEST 1] 142841 Scene Classification: {res.scene.value} (conf={res.confidence}, margin={res.margin})")
        assert res.scene == SceneType.CODEX_SELECT, f"Expected CODEX_SELECT, got {res.scene}"

        router.dispatch(img_codex, state_codex)
        print(f"[TEST 1] Dispatch calls for 142841: {stub_in.calls}")
        assert any("03号法典" in (call[3] or "") or "03" in (call[3] or "") for call in stub_in.calls), "Should select Codex 03"
        assert any("進入" in (call[3] or "") for call in stub_in.calls), "Should click 進入"
        print("[TEST 1] PASS")

    # 2. Test 142934: Ticket Voucher Refill Modal
    p_voucher = fixtures_auto / "20261011-142934_BATTLE_肉鸽选路.png"
    if p_voucher.exists():
        img_voucher = cv2.imdecode(np.fromfile(str(p_voucher), dtype=np.uint8), cv2.IMREAD_COLOR)
        tgts_voucher = vision.locate_all_texts(img_voucher)
        state_voucher = GameState()
        state_voucher.raw_texts = [t.text for t in tgts_voucher]
        state_voucher.targets = tgts_voucher

        stub_in2 = StubInput()
        ctx_stub2 = type("Ctx", (), {"input": stub_in2, "overlay": None, "vision": vision})()
        router2 = SceneRouter(ctx_stub2, {"mode": "system_zero", "archive": {"use_travel_ticket": True}})

        router2.dispatch(img_voucher, state_voucher)
        print(f"[TEST 2] Decision for 142934: {state_voucher.decision}")
        print(f"[TEST 2] Calls for 142934: {stub_in2.calls}")
        assert state_voucher.decision == "safety:voucher_refill_confirm", f"Expected safety:voucher_refill_confirm, got {state_voucher.decision}"
        assert any("ENTER" in (c[2] or "") for c in stub_in2.calls), "Should confirm refill with ENTER"
        print("[TEST 2] PASS")

    print("\nALL CODEX & VOUCHER TESTS PASSED!")

if __name__ == "__main__":
    test_codex_and_voucher()

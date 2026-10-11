"""
Phase 7: RerollManager (开局物主代理人自动刷取)
Handles candidate detection, target matching, quitting on mismatch, and dungeon re-entry.
"""

import time
import logging
from typing import Dict, Any, List, Optional
from ..fsm.engine import FSMContext, FSMStateId, StateNode
from ..state.screen_state import GameState, ScreenState
from .target_manager import TargetManager

logger = logging.getLogger("RerollManager")

class RerollManager:
    def __init__(self, ctx: FSMContext, target_mgr: TargetManager, max_retries: int = 50):
        self.ctx = ctx
        self.target_mgr = target_mgr
        self.max_retries = max_retries
        self.attempts = 0

    def evaluate_candidates(self, state: GameState) -> Dict[str, Any]:
        """
        Evaluate visible items on opening screen.
        Returns decision structure with full logging metrics.
        """
        matched_cards, matched_equips = self.target_mgr.check_matches(state.raw_texts)
        has_hit = bool(matched_cards or matched_equips)

        matched_targets = matched_cards + matched_equips
        decision = "ACCEPT" if has_hit else "RETRY"

        result = {
            "attempt": self.attempts + 1,
            "detected_texts": state.raw_texts,
            "matched_targets": matched_targets,
            "decision": decision,
            "failure_reason": None if has_hit else "No target matched"
        }

        logger.info(f"Reroll #{result['attempt']}: Targets={matched_targets} -> Decision={decision}")
        return result

    def execute_accept(self, item_name: str, state: GameState) -> bool:
        """Click on the matched target and confirm."""
        elem = state.get_element(item_name)
        if elem:
            cx, cy = elem.center
            self.ctx.input.click_client(cx, cy, delay_after=1.0)
            self.target_mgr.mark_acquired(item_name)
            return True
        return False

    def execute_retry(self) -> bool:
        """Exit dungeon and re-enter."""
        self.attempts += 1
        if self.attempts >= self.max_retries:
            logger.warning(f"Reached max reroll retries ({self.max_retries}). Stopping...")
            return False

        logger.info(f"Initiating dungeon exit (Retry {self.attempts}/{self.max_retries})...")
        
        # 1. Click settings / pause button (top right: rx=0.96, ry=0.05)
        self.ctx.input.click_rel(0.96, 0.05, delay_after=0.8)

        # 2. Find and click '退出' / '放弃'
        img = self.ctx.capture.capture()
        quit_target = self.ctx.vision.locate(img, keyword="退出", fallback_rel=(0.5, 0.6))
        if quit_target:
            cx, cy = quit_target.center
            self.ctx.input.click_client(cx, cy, delay_after=1.0)

        # 3. Confirm exit prompt
        time.sleep(1.0)
        confirm_img = self.ctx.capture.capture()
        confirm_target = self.ctx.vision.locate(confirm_img, keyword="确认", fallback_rel=(0.6, 0.6))
        if confirm_target:
            cx, cy = confirm_target.center
            self.ctx.input.click_client(cx, cy, delay_after=2.0)

        # 4. Click Start / Enter Dungeon (bottom right: rx=0.85, ry=0.88)
        time.sleep(2.0)
        self.ctx.input.click_rel(0.85, 0.88, delay_after=2.0)
        return True

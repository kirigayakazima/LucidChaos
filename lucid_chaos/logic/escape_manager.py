"""
Phase 12: EscapeManager
Monitors waypoint / boss milestones and triggers safe dungeon escape or pause.
"""

import time
import logging
from typing import Dict, Any
from ..fsm.engine import FSMContext
from ..state.screen_state import GameState

logger = logging.getLogger("EscapeManager")

class EscapeManager:
    def __init__(self, ctx: FSMContext, config: Dict[str, Any]):
        self.ctx = ctx
        self.config = config
        dungeon_cfg = config.get("dungeon", {})
        self.stop_at = dungeon_cfg.get("stop_at", "first_boss")
        self.stop_action = dungeon_cfg.get("stop_action", "escape")

    def should_escape(self, current_milestone: str) -> bool:
        """Check if reached the configured stop milestone."""
        return current_milestone.lower() == self.stop_at.lower()

    def execute_escape(self) -> bool:
        """Trigger Escape scroll/skill or pause."""
        logger.info(f"Target milestone reached ({self.stop_at}). Action: [{self.stop_action}]")
        if self.stop_action == "pause":
            logger.info("Script paused as requested by config.")
            return True

        # Execute Escape
        logger.info("Executing Escape from dungeon...")
        # 1. Click backpack or escape item icon (top right or bottom bar)
        self.ctx.input.click_rel(0.92, 0.05, delay_after=1.0)

        # 2. OCR find '逃脱' / '离开'
        img = self.ctx.capture.capture()
        escape_btn = self.ctx.vision.locate(img, keyword="逃脱", fallback_rel=(0.5, 0.5))
        if escape_btn:
            cx, cy = escape_btn.center
            self.ctx.input.click_client(cx, cy, delay_after=1.5)

        # 3. Confirm escape prompt
        time.sleep(1.0)
        confirm_btn = self.ctx.vision.locate(self.ctx.capture.capture(), keyword="确认", fallback_rel=(0.6, 0.6))
        if confirm_btn:
            cx, cy = confirm_btn.center
            self.ctx.input.click_client(cx, cy, delay_after=2.0)
        return True

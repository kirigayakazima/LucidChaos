"""
Phase 14: Watchdog and RecoveryManager
Monitors frame stagnation (dHash), state timeouts, and executes tiered recovery levels.
Level 1: Re-capture / Re-OCR
Level 2: ESC key
Level 3: Return to home/current page
Level 4: Reconnect dialog
Level 5: Restart client
Level 6: Emergency STOP
"""

import time
import logging
from typing import Dict, Any, Optional
import win32con
from ..fsm.engine import FSMContext, FSMStateId
from ..state.screen_state import GameState, ScreenState

logger = logging.getLogger("Watchdog")

class Watchdog:
    def __init__(self, ctx: FSMContext, config: Dict[str, Any]):
        self.ctx = ctx
        self.config = config
        self.last_frame_hash = ""
        self.last_state_time = time.time()
        self.last_state_name = ""
        self.stagnant_frames = 0
        self.recovery_count = 0
        self.max_recovery = 3

        # State timeout limits (seconds)
        self.state_timeouts = {
            "BATTLE": 360.0,
            "SHOP": 25.0,
            "CAMP": 20.0,
            "EVENT": 30.0,
            "MAP_NAVIGATION": 20.0,
            "OPENING_REROLL": 45.0
        }

    def check(self, state: GameState) -> bool:
        """
        Check health of current state.
        Returns True if normal, False if anomaly detected and recovery triggered.
        """
        current_time = time.time()
        state_name = state.screen.name

        # 1. Reset timer on state transition
        if state_name != self.last_state_name:
            self.last_state_name = state_name
            self.last_state_time = current_time
            self.stagnant_frames = 0
            return True

        # 2. State Timeout Check
        timeout_limit = self.state_timeouts.get(state_name, 60.0)
        elapsed = current_time - self.last_state_time
        if elapsed > timeout_limit:
            logger.warning(f"State [{state_name}] timed out ({elapsed:.1f}s > {timeout_limit}s)!")
            self._trigger_recovery(level=2, reason=f"State timeout ({state_name})")
            return False

        # 3. Disconnect Detection
        if state.screen == ScreenState.DISCONNECTED:
            logger.warning("Network disconnection dialog detected!")
            self._trigger_recovery(level=4, reason="Disconnected")
            return False

        return True

    def _trigger_recovery(self, level: int, reason: str):
        self.recovery_count += 1
        if self.recovery_count > self.max_recovery:
            logger.critical(f"Exceeded max recovery attempts ({self.max_recovery}). Halting script (Level 6 STOP).")
            self.ctx.running = False
            return

        logger.info(f"Executing Recovery Level {level} (Reason: {reason}, Attempt {self.recovery_count}/{self.max_recovery})...")

        if level == 1:
            # Level 1: Delay and re-observe
            time.sleep(1.0)
        elif level == 2:
            if self.last_state_name == "BATTLE":
                logger.warning("【看门狗自愈】当前处于战斗中，按 E 结束回合尝试脱困（防止按 ESC 退出）...")
                self.ctx.input.tap_action("end_turn", delay_after=1.0, label="【看门狗】战斗E结束回合")
            else:
                # Level 2: Send ESC key to dismiss modal/stuck UI
                self.ctx.input.send_key(win32con.VK_ESCAPE, delay_after=1.0)
        elif level == 3:
            if self.last_state_name == "BATTLE":
                logger.warning("【看门狗自愈】当前处于战斗中，点击屏幕中央并再次按 E...")
                self.ctx.input.click_rel(0.50, 0.50, delay_after=0.5)
                self.ctx.input.tap_action("end_turn", delay_after=1.0)
            else:
                # Level 3: Click back / neutral area
                self.ctx.input.click_rel(0.05, 0.05, delay_after=1.5)
        elif level == 4:
            # Level 4: Click Reconnect
            img = self.ctx.capture.capture()
            btn = self.ctx.vision.locate(img, keyword="重连") or self.ctx.vision.locate(img, keyword="确定")
            if btn:
                self.ctx.input.click_client(*btn.center, delay_after=3.0)
        
        # Reset state timer after recovery action
        self.last_state_time = time.time()

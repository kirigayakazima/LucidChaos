"""
Concrete state implementations for LucidChaos.
"""

import time
import logging
from .fsm import BaseState, BotContext

logger = logging.getLogger("ChaosBot")

class IdleState(BaseState):
    """Initial state: waits for game window and verifies screen."""
    def execute(self):
        if not self.ctx.window.find_window():
            logger.warning("Waiting for '卡厄思梦境' window...")
            time.sleep(2)
            return

        info = self.ctx.window.get_info()
        logger.info(f"Connected to game window: {info['title']} ({info['width']}x{info['height']})")

        # Route to opening reroll or dungeon
        if self.ctx.config.get("reroll", {}).get("enable_reroll", False):
            self.ctx.transition_to(RerollOpeningState(self.ctx))
        else:
            self.ctx.transition_to(DungeonExploreState(self.ctx))


class RerollOpeningState(BaseState):
    """
    Reroll Opening Logic (开局物主代理人刷初始).
    OCR scans the opening cards and equipments. If not matching targets,
    quits dungeon and restarts.
    """
    def execute(self):
        img = self.ctx.capture.capture()
        if img is None:
            time.sleep(0.5)
            return

        ocr_results = self.ctx.vision.ocr_image(img)
        texts = [r["text"] for r in ocr_results]
        logger.debug(f"Visible texts on screen: {texts[:10]}")

        target_cards = self.ctx.config.get("reroll", {}).get("target_cards", [])
        target_equips = self.ctx.config.get("reroll", {}).get("target_equipments", [])
        targets = target_cards + target_equips

        # 1. Check if targets appeared on screen
        matched_item = None
        for item in ocr_results:
            for t in targets:
                if t in item["text"]:
                    matched_item = item
                    break
            if matched_item:
                break

        if matched_item:
            logger.info(f"Target found: [{matched_item['text']}]! Selecting...")
            cx, cy = matched_item["center_client"]
            self.ctx.input.click_client(cx, cy, delay_after=1.0)
            
            # Transition to dungeon exploration
            self.ctx.transition_to(DungeonExploreState(self.ctx))
            return

        # 2. Check if we are currently at the opening selection prompt
        # In opening selection, we either hit target, or need to reroll
        is_opening = any("代理人" in t or "选择" in t or "卡牌" in t or "装备" in t for t in texts)
        if is_opening:
            self.ctx.reroll_count += 1
            logger.info(f"Reroll #{self.ctx.reroll_count}: Target not found. Quitting dungeon to re-enter...")

            # Click top-right or top-left settings/pause (rel coords: x=0.96, y=0.05)
            self.ctx.input.click_rel(0.96, 0.05, delay_after=0.8)

            # OCR scan for '退出' / '放弃' / '离开'
            sub_img = self.ctx.capture.capture()
            quit_btn = self.ctx.vision.find_text(sub_img, "退出") or self.ctx.vision.find_text(sub_img, "放弃")
            if quit_btn:
                self.ctx.input.click_client(*quit_btn["center_client"], delay_after=1.0)
            else:
                # Fallback center confirmation click
                self.ctx.input.click_rel(0.5, 0.6, delay_after=1.0)

            # Re-enter dungeon
            time.sleep(2.0)
            self._reenter_dungeon()
            return

        # 3. If in battle during opening, handle battle
        if any("回合" in t or "EP" in t or "獲得分數" in t or "战斗分数" in t for t in texts):
            self.ctx.transition_to(BattleState(self.ctx))
            return

        time.sleep(0.5)

    def _reenter_dungeon(self):
        """Click to enter dungeon again."""
        logger.info("Clicking to re-enter dungeon...")
        # Common start/enter button (bottom right: x=0.85, y=0.88)
        self.ctx.input.click_rel(0.85, 0.88, delay_after=2.0)


class DungeonExploreState(BaseState):
    """
    Dungeon Path Planning and Room Decision.
    """
    def execute(self):
        img = self.ctx.capture.capture()
        if img is None:
            time.sleep(0.5)
            return

        ocr_results = self.ctx.vision.ocr_image(img)
        texts = [r["text"] for r in ocr_results]

        # Check if in battle
        if any("EP" in t or "獲得分數" in t or "战斗分数" in t for t in texts):
            self.ctx.transition_to(BattleState(self.ctx))
            return

        # Check if in Rest Area (安全区)
        rest_choice = self.ctx.config.get("dungeon", {}).get("rest_area_choice", "training")
        if any("安全区" in t or "休息" in t or "训练" in t for t in texts):
            logger.info(f"Safe Area detected. Preference: {rest_choice}")
            target_keyword = "训练" if rest_choice == "training" else "休息"
            btn = self.ctx.vision.find_text(img, target_keyword)
            if btn:
                self.ctx.input.click_client(*btn["center_client"], delay_after=1.0)
            return

        # Check if in Shop (德朗商店)
        if any("商店" in t or "德朗" in t or "购买" in t for t in texts):
            logger.info("Shop detected. Processing shop...")
            # Leave or buy
            leave_btn = self.ctx.vision.find_text(img, "离开") or self.ctx.vision.find_text(img, "退出")
            if leave_btn:
                self.ctx.input.click_client(*leave_btn["center_client"], delay_after=1.0)
            return

        # Default dungeon advance: click ahead on route (middle area)
        self.ctx.input.click_rel(0.5, 0.45, delay_after=1.0)
        time.sleep(1.0)


class BattleState(BaseState):
    """
    Automated Combat Execution.
    Plays cards and skills when EP is available, handles victory screens.
    """
    def execute(self):
        img = self.ctx.capture.capture()
        if img is None:
            time.sleep(0.5)
            return

        ocr_results = self.ctx.vision.ocr_image(img)
        texts = [r["text"] for r in ocr_results]

        # 1. Victory / Battle End Check
        victory_btn = (self.ctx.vision.find_text(img, "胜利") or 
                       self.ctx.vision.find_text(img, "确定") or 
                       self.ctx.vision.find_text(img, "确认") or
                       self.ctx.vision.find_text(img, "继续"))
        if victory_btn:
            logger.info("Victory or confirmation prompt detected. Clicking...")
            self.ctx.input.click_client(*victory_btn["center_client"], delay_after=1.5)
            self.ctx.transition_to(DungeonExploreState(self.ctx))
            return

        # 2. Check if still in combat
        in_battle = any("EP" in t or "獲得分數" in t or "战斗分数" in t or "連鎖結晶" in t for t in texts)
        if in_battle:
            logger.debug("Executing combat round: selecting cards/skills...")
            # Click bottom card slots (e.g. card 1 at x=0.1, y=0.75; card 2 at x=0.1, y=0.83)
            # Drag or click skill
            self.ctx.input.click_rel(0.1, 0.72, delay_after=0.3)
            self.ctx.input.click_rel(0.1, 0.83, delay_after=0.3)
            # Click bottom center card (x=0.5, y=0.9)
            self.ctx.input.click_rel(0.5, 0.9, delay_after=0.3)
            # Click auto / speedup button if available (top right)
            time.sleep(1.0)
            return

        # If battle ended and no text matched, return to dungeon state
        self.ctx.transition_to(DungeonExploreState(self.ctx))

"""
Finite State Machine (FSM) core and base state definitions for Chaos Bot.
"""

from abc import ABC, abstractmethod
import time
import logging
from typing import Optional, Dict, Any
from ..core import GameWindow, InputController, ScreenCapture, VisionEngine

logger = logging.getLogger("ChaosBot")

class BotContext:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window = GameWindow(config.get("window", {}).get("title", "卡厄思梦境"))
        self.input = InputController(self.window)
        self.capture = ScreenCapture(self.window)
        self.vision = VisionEngine()
        
        # Runtime variables
        self.running = True
        self.current_state: Optional["BaseState"] = None
        self.reroll_count = 0
        self.last_screen_hash = ""
        self.freeze_counter = 0

    def transition_to(self, new_state: "BaseState"):
        if self.current_state:
            self.current_state.exit()
        prev_name = self.current_state.__class__.__name__ if self.current_state else "None"
        next_name = new_state.__class__.__name__
        logger.info(f"State transition: [{prev_name}] -> [{next_name}]")
        self.current_state = new_state
        self.current_state.enter()

    def update(self):
        if self.current_state:
            self.current_state.execute()


class BaseState(ABC):
    def __init__(self, ctx: BotContext):
        self.ctx = ctx

    def enter(self):
        pass

    @abstractmethod
    def execute(self):
        pass

    def exit(self):
        pass

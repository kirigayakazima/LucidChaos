"""
Phase 6: Robust FSM Engine adhering strictly to the execution model:
ENTER -> OBSERVE -> DECIDE -> ACTION -> VERIFY -> TRANSITION
No blind state transitions allowed without verification.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
import time
import logging
from typing import Optional, Dict, Any, List

from ..core.window import GameWindow
from ..core.capture import ScreenCapture
from ..core.input import InputController
from ..core.vision_service import VisionService, VisionTarget
from ..state.observer import ScreenObserver
from ..state.screen_state import ScreenState, GameState
from ..core.overlay import ScreenOverlay

logger = logging.getLogger("FSM")

class FSMStateId(Enum):
    INIT = auto()
    WAIT_GAME_READY = auto()
    ENTER_DUNGEON = auto()

    # Reroll
    REROLL_CHECK = auto()
    REROLL_SELECT = auto()
    REROLL_EXIT = auto()

    # Map & Route
    MAP_DETECT = auto()
    PATH_PLAN = auto()
    MOVE_TO_NODE = auto()

    # Nodes
    NODE_DETECT = auto()
    BATTLE = auto()
    EVENT = auto()
    SHOP = auto()
    CAMP = auto()
    WAYPOINT = auto()
    BOSS = auto()

    # Escape & Target
    TARGET_CHECK = auto()
    ESCAPE = auto()

    # Archive
    ARCHIVE_CHECK = auto()
    ARCHIVE_CLEANUP = auto()

    # System & Watchdog
    RECOVERY = auto()
    PAUSED = auto()
    STOPPED = auto()
    ERROR = auto()

class FSMContext:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.window = GameWindow(config.get("window", {}).get("title", "卡厄思梦境"))
        self.capture = ScreenCapture(self.window)
        self.vision = VisionService()
        self.observer = ScreenObserver(self.vision)

        # Real-time transparent on-screen overlay HUD
        enable_overlay = config.get("overlay", {}).get("enabled", True)
        if enable_overlay:
            self.overlay = ScreenOverlay(
                window_getter=lambda: self.window.get_client_rect() if self.window.is_valid else None
            )
        else:
            self.overlay = None

        self.input = InputController(self.window, overlay=self.overlay)

        self.running = True
        self.current_state: Optional["StateNode"] = None
        self.last_game_state: Optional[GameState] = None

        # Metrics & targets
        self.reroll_count = 0
        self.max_retries = config.get("reroll", {}).get("max_retries", 50)
        self.targets_achieved = False
        self.recovery_attempts = 0
        self.max_recovery_attempts = 3

    def stop(self):
        """Cleanly shutdown FSM and close overlay."""
        self.running = False
        if getattr(self, "overlay", None):
            self.overlay.close()

    def transition_to(self, new_state: "StateNode"):
        prev_name = self.current_state.state_id.name if self.current_state else "START"
        if self.current_state:
            self.current_state.exit()
        
        logger.info(f"FSM Transition: [{prev_name}] -> [{new_state.state_id.name}]")
        self.current_state = new_state
        self.current_state.enter()

    def step(self):
        """Execute one complete cycle of the current state."""
        if not self.current_state:
            return
        
        # 1. OBSERVE
        img = self.capture.capture()
        game_state = self.observer.observe(img)
        self.last_game_state = game_state

        # 2. State Cycle (DECIDE -> ACTION -> VERIFY -> TRANSITION)
        self.current_state.step(game_state)


class StateNode(ABC):
    def __init__(self, state_id: FSMStateId, ctx: FSMContext):
        self.state_id = state_id
        self.ctx = ctx
        self.enter_time = 0.0

    def enter(self):
        self.enter_time = time.time()
        logger.debug(f"Entering [{self.state_id.name}]")

    @abstractmethod
    def step(self, state: GameState):
        """Standard pipeline: DECIDE -> ACTION -> VERIFY -> TRANSITION."""
        pass

    def exit(self):
        logger.debug(f"Exiting [{self.state_id.name}]")

    def verify_action(self, verify_fn, timeout: float = 2.0, interval: float = 0.3) -> bool:
        """Helper to verify that an action caused the expected UI state change."""
        start = time.time()
        while time.time() - start < timeout:
            time.sleep(interval)
            img = self.ctx.capture.capture()
            new_state = self.ctx.observer.observe(img)
            if verify_fn(new_state):
                return True
        return False

"""
Full FSM Orchestrator wiring all phase managers together according to Roadmap v0.1.
Enforces OBSERVE -> DECIDE -> ACTION -> VERIFY -> TRANSITION.
"""

import time
import logging
from typing import Dict, Any

import win32con

from .engine import FSMContext, FSMStateId, StateNode
from ..core.frame import FrameContext
from ..state.screen_state import GameState, ScreenState
from ..logic.target_manager import TargetManager
from ..logic.reroll_manager import RerollManager
from ..logic.path_planner import PathPlanner, MapNode
from ..logic.node_engine import NodeDecisionEngine
from ..logic.battle_manager import BattleManager
from ..logic.escape_manager import EscapeManager
from ..logic.archive_manager import ArchiveManager
from ..logic.watchdog import Watchdog

from ..logic.story_pusher import StoryPusher
from ..router.scene_types import should_activate_speed

logger = logging.getLogger("Orchestrator")

class ChaosOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = config.get("mode", "main_story")
        self.ctx = FSMContext(config)

        # Wire Main Story Pusher
        self.story_pusher = StoryPusher(self.ctx, config)

        # Wire Labyrinth / Phase 7 ~ Phase 14 Managers
        reroll_cfg = config.get("reroll", {})
        self.target_mgr = TargetManager(
            target_cards=reroll_cfg.get("target_cards", []),
            target_equipments=reroll_cfg.get("target_equipments", [])
        )
        self.reroll_mgr = RerollManager(
            self.ctx, self.target_mgr,
            max_retries=reroll_cfg.get("max_retries", 50)
        )
        sz_cfg = config.get("system_zero", {})
        event_only = (self.mode == "system_zero" or sz_cfg.get("event_only", False))
        pref = "event_only" if event_only else config.get("path_planning", {}).get("preference", "non_battle")
        self.path_planner = PathPlanner(
            preference=pref,
            event_only=event_only
        )
        self.node_engine = NodeDecisionEngine(self.ctx, self.target_mgr, config)
        self.story_pusher.node_engine = self.node_engine
        self.battle_mgr = BattleManager(self.ctx)
        self.escape_mgr = EscapeManager(self.ctx, config)
        self.archive_mgr = ArchiveManager(self.ctx, config)
        self.watchdog = Watchdog(self.ctx, config)
        self.router = self.story_pusher.router

        self._last_log_time = 0.0
        self._last_logged_screen = ""

        # 自动取证（默认开，可在 config.debug.collect_fixtures 关掉）
        debug_cfg = config.get("debug", {}) if isinstance(config, dict) else {}
        if debug_cfg.get("collect_fixtures", True):
            from ..core.fixture_collector import FixtureCollector
            from pathlib import Path as _Path
            self.fixture_collector = FixtureCollector(
                _Path(__file__).resolve().parent.parent / "fixtures_auto",
                max_files=int(debug_cfg.get("max_fixtures", 16)),
                stuck_threshold=int(debug_cfg.get("stuck_threshold", 4)),
            )
        else:
            self.fixture_collector = None

        # 全局卡死守卫（默认开，`config.debug.stuck_guard` 可调；stop=0 表示不自动停止）
        guard_cfg = debug_cfg.get("stuck_guard", {}) if isinstance(debug_cfg, dict) else {}
        if guard_cfg.get("enabled", True):
            from ..core.stuck_guard import GlobalStuckGuard
            self.stuck_guard = GlobalStuckGuard(
                warn=int(guard_cfg.get("warn", 4)),
                esc=int(guard_cfg.get("esc", 6)),
                close=int(guard_cfg.get("close", 9)),
                stop=int(guard_cfg.get("stop", 14)),
                repeat_action_limit=int(guard_cfg.get("repeat_action_limit", 3)),
                slow_routes=tuple(guard_cfg.get("slow_routes", ["剧情对白演出"])),
            )
        else:
            self.stuck_guard = None

        # 数字键"选择"类动作按几下（主人实测：事件选项 / 出牌 都要**按两下**）
        try:
            times = int((config.get("input", {}) or {}).get("digit_press_times", 2))
            self.ctx.input.DIGIT_PRESS_TIMES = max(1, times)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"读取 input.digit_press_times 失败（用默认 2）: {e}")

        # Start at INIT
        self.ctx.transition_to(InitState(self))

    def _ensure_speed_2x(self, img, game_state) -> None:
        """
        局内常驻 2X 补刀：右上角 ⏩/AUTO 未点亮就点亮一次。

        - 只在 `should_activate_speed(scene)` 的场景做（对白/战斗自己管，重复点会把开关关掉）
        - 放在场景动作之后，不抢主动作；带冷却，避免连点时把开关反复切换
        """
        if not getattr(self, "story_pusher", None) or img is None:
            return
        if getattr(self, "_speed_cooldown", 0) > 0:
            self._speed_cooldown -= 1
            return
        scene = getattr(game_state, "route", None)
        try:
            if not should_activate_speed(scene):
                return
            if self.story_pusher.ensure_speed_2x(img):
                self._speed_cooldown = 5      # 点过之后给游戏几帧时间去点亮图标
        except Exception as e:  # noqa: BLE001
            logger.debug(f"【常驻加速】检查失败（忽略）: {e}")

    def stop(self):
        """Stop orchestrator and release overlay."""
        self.ctx.stop()

    def step(self):
        """Single loop step with watchdog protection."""
        if not self.ctx.running:
            return

        # Ensure target window is connected
        if not self.ctx.window.is_valid:
            if not self.ctx.window.find_window():
                logger.info("等待游戏窗口《卡厄思梦境》出现...")
                time.sleep(2.0)
                return

        # Capture & Observe
        img = self.ctx.capture.capture()
        if img is None:
            time.sleep(0.5)
            return

        # 一帧只做一次观测：FrameContext 收拢 OCU 结果 + 共用模板库 + 亮度归一化副本，
        # observer 直接复用其中的 targets（不再重复 OCR）。
        frame = FrameContext.from_image(img, ocr=self.ctx.vision)
        game_state = self.ctx.observer.observe_frame(frame)
        game_state.frame = frame
        self.ctx.last_game_state = game_state

        # Periodic logging
        now = time.time()
        screen_name = game_state.screen.name
        if screen_name != self._last_logged_screen or (now - self._last_log_time > 4.0):
            if self.mode == "system_zero":
                mode_tag = "【零号系统】"
            elif self.mode == "main_story":
                mode_tag = "【主线推进】"
            else:
                mode_tag = "【迷宫肉鸽】"
            from ..ui.worker import SCENE_NAMES_CN
            cn_screen = SCENE_NAMES_CN.get(screen_name, screen_name)
            logger.info(f"{mode_tag} 当前处于: [{cn_screen}] (识别度: {game_state.confidence:.0f}%)")
            self._last_logged_screen = screen_name
            self._last_log_time = now

        # Watchdog Health Check
        if not self.watchdog.check(game_state):
            return

        # Execution Routing based on Mode
        if self.mode in ("main_story", "system_zero"):
            self.story_pusher.step(game_state)
        else:
            # FSM State execution for labyrinth
            if self.ctx.current_state:
                self.ctx.current_state.step(game_state)

        # 局内常驻 2X 补刀：右上角 ⏩/AUTO 没点亮就点亮（放在场景动作之后，不抢主动作）
        self._ensure_speed_2x(img, game_state)

        # 自动取证：卡住 / 本场首次出现的界面 -> 存图（给自己攒形状特征素材，不改任何判定）
        if self.fixture_collector is not None:
            self.fixture_collector.tick(
                img,
                decision=getattr(game_state, "decision", ""),
                route=getattr(game_state, "route", None),
                screen=game_state.screen.name,
            )

        # 全局卡死守卫：发现"原地打转"就逐级自救，实在不行**安全停止**（主人不用再 Ctrl+C）
        self._run_stuck_guard(img, game_state)

    # -------------------------------------------------------------
    # 全局卡死守卫
    # -------------------------------------------------------------
    def _run_stuck_guard(self, img, game_state) -> None:
        if self.stuck_guard is None:
            return
        action = getattr(self.ctx.input, "last_action", "") or ""
        route = getattr(getattr(game_state, "route", None), "value", "") or ""
        decision = self.stuck_guard.update(img, action=action, route=route)
        if decision is None:
            return

        logger.warning(
            f"【卡死守卫】{decision.reason} -> 升级到级别 {decision.level}：{decision.description}"
            f"（路由={decision.route or '-'}｜本轮动作={decision.action or '-'}）"
        )

        if decision.level == 1:
            # 清掉"上一帧稳定场景"的粘连 + 各 handler 的卡死阶梯，让下一轮重新判、重新试
            try:
                self.router.classifier.reset_debounce()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"【卡死守卫】重置防抖失败（忽略）: {e}")
            try:
                self.node_engine.reset_stuck_state()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"【卡死守卫】重置阶梯失败（忽略）: {e}")
            logger.warning("【卡死守卫】已清空判定粘连与卡死阶梯，下一次将重新判定场景并重试动作")

        elif decision.level == 2:
            if route == "战斗中" or getattr(game_state, "screen", None) == ScreenState.BATTLE:
                logger.warning("【卡死守卫】当前处于战斗中，不发送 ESC（防止误退），按 E 键结束回合尝试脱困...")
                self.ctx.input.tap_action("end_turn", delay_after=1.0, label="【卡死守卫】战斗按E脱困")
            else:
                logger.warning("【卡死守卫】发送 ESC，尝试退回上一层...")
                self.ctx.input.send_key(win32con.VK_ESCAPE, delay_after=1.0)

        elif decision.level == 3:
            if route == "战斗中" or getattr(game_state, "screen", None) == ScreenState.BATTLE:
                logger.warning("【卡死守卫】当前处于战斗中，不点击退出区域，点击屏幕中央并按 1 号键尝试出牌...")
                self.ctx.input.click_rel(0.50, 0.50, delay_after=0.5, label="【卡死守卫】点击屏幕中央")
                self.ctx.input.tap_card(1, delay_after=1.0, label="【卡死守卫】尝试出1号牌")
            else:
                logger.warning("【卡死守卫】ESC + 点击右上角关闭/返回区域，尝试强退当前层...")
                self.ctx.input.send_key(win32con.VK_ESCAPE, delay_after=0.6)
                self.ctx.input.click_area_rel(0.93, 0.03, 0.965, 0.075, delay_after=1.0,
                                              label="【卡死守卫】关闭/返回")

        elif decision.level == 4:
            self._dump_stuck_report(decision)
            logger.error(
                "【卡死守卫】多轮自救均无效，按配置**安全停止**脚本（避免死循环空转）。"
                "请把上面那张【自动取证】图片发给开发者即可定位。"
            )
            self.ctx.running = False

    def _dump_stuck_report(self, decision) -> None:
        """打印"卡住报告"：路由、动作、最近识别文本、取证图路径——让主人一句话就能求助。"""
        try:
            st = self.getattr_last_state()
            texts = list(getattr(st, "raw_texts", []) or [])[:8]
            shots = []
            if self.fixture_collector is not None:
                from pathlib import Path as _P
                files = sorted(self.fixture_collector.out_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
                shots = [f.name for f in files[-2:]]
            logger.error("==================== 卡住报告 ====================")
            logger.error(f"  路由      : {decision.route or '-'}")
            logger.error(f"  重复动作  : {decision.action or '-'}")
            logger.error(f"  停留轮数  : {decision.stuck_rounds}")
            logger.error(f"  原因      : {decision.reason}")
            logger.error(f"  屏幕文字  : {texts}")
            logger.error(f"  取证图片  : {shots}（见 lucid_chaos/fixtures_auto/）")
            logger.error("=================================================")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"【卡死守卫】输出报告失败（忽略）: {e}")

    def getattr_last_state(self):
        return getattr(self.ctx, "last_game_state", None)


class InitState(StateNode):
    def __init__(self, orchestrator: ChaosOrchestrator):
        super().__init__(FSMStateId.INIT, orchestrator.ctx)
        self.orch = orchestrator

    def step(self, state: GameState):
        if not self.ctx.window.find_window():
            logger.info("等待游戏窗口《卡厄思梦境》出现...")
            time.sleep(2.0)
            return

        win_info = self.ctx.window.get_info()
        logger.info(f"成功锁定游戏窗口: {win_info['title']} ({win_info['width']}x{win_info['height']})")
        if self.ctx.config.get("reroll", {}).get("enable_reroll", False):
            self.ctx.transition_to(RerollState(self.orch))
        else:
            self.ctx.transition_to(DungeonState(self.orch))


class RerollState(StateNode):
    def __init__(self, orchestrator: ChaosOrchestrator):
        super().__init__(FSMStateId.REROLL_CHECK, orchestrator.ctx)
        self.orch = orchestrator

    def step(self, state: GameState):
        if state.screen == ScreenState.BATTLE:
            logger.info("检测到处于战斗中，转入战斗状态...")
            self.ctx.transition_to(BattleState(self.orch))
            return

        if state.screen == ScreenState.OPENING_REROLL or any("代理人" in t or "选择" in t for t in state.raw_texts):
            eval_res = self.orch.reroll_mgr.evaluate_candidates(state)
            if eval_res["decision"] == "ACCEPT":
                target_name = eval_res["matched_targets"][0]
                self.orch.reroll_mgr.execute_accept(target_name, state)
                self.ctx.transition_to(DungeonState(self.orch))
            else:
                ok = self.orch.reroll_mgr.execute_retry()
                if not ok:
                    self.ctx.transition_to(StoppedState(self.orch))
            return

        # Still loading or advancing to opening
        self.ctx.input.click_rel(0.5, 0.5, delay_after=1.0)


class DungeonState(StateNode):
    def __init__(self, orchestrator: ChaosOrchestrator):
        super().__init__(FSMStateId.MAP_DETECT, orchestrator.ctx)
        self.orch = orchestrator

    def step(self, state: GameState):
        raw_img = self.ctx.capture.capture()
        if raw_img is None:
            return
        self.orch.router.dispatch(raw_img, state)


class BattleState(StateNode):
    def __init__(self, orchestrator: ChaosOrchestrator):
        super().__init__(FSMStateId.BATTLE, orchestrator.ctx)
        self.orch = orchestrator

    def step(self, state: GameState):
        if self.orch.battle_mgr.check_battle_ended(state):
            # Check if targets complete or reached milestone
            if self.orch.target_mgr.is_complete():
                logger.info("所有目标已达成！准备逃脱返程...")
                self.ctx.transition_to(EscapeState(self.orch))
            else:
                self.ctx.transition_to(DungeonState(self.orch))
            return

        if state.screen == ScreenState.EPIPHANY_SELECTION or self.orch.battle_mgr.is_epiphany_screen(state=state):
            raw_img = self.ctx.capture.capture()
            self.orch.battle_mgr.handle_epiphany_selection(raw_img=raw_img, state=state)
            return

        if state.screen == ScreenState.BATTLE:
            raw_img = self.ctx.capture.capture()
            self.orch.battle_mgr.play_round(state, raw_img=raw_img)
        else:
            self.ctx.transition_to(DungeonState(self.orch))


class EscapeState(StateNode):
    def __init__(self, orchestrator: ChaosOrchestrator):
        super().__init__(FSMStateId.ESCAPE, orchestrator.ctx)
        self.orch = orchestrator

    def step(self, state: GameState):
        self.orch.escape_mgr.execute_escape()
        # After escaping, perform archive cleanup
        self.orch.archive_mgr.maintain_archives()
        self.ctx.transition_to(StoppedState(self.orch))


class StoppedState(StateNode):
    def __init__(self, orchestrator: ChaosOrchestrator):
        super().__init__(FSMStateId.STOPPED, orchestrator.ctx)
        self.orch = orchestrator

    def step(self, state: GameState):
        logger.info("FSM in STOPPED state. Halting execution loop.")
        self.ctx.running = False

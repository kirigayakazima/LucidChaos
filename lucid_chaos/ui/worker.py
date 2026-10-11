"""
Backend Worker Thread for ChaosBot.
Runs ChaosOrchestrator in a separate QThread, communicating with UI via Qt Signals.
"""

import time
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

from PySide6.QtCore import QThread, Signal, QObject

class QtLogHandler(logging.Handler):
    """Logging handler that emits log records to a Qt Signal."""
    def __init__(self, log_signal: Signal):
        super().__init__()
        self.log_signal = log_signal

    def emit(self, record):
        try:
            msg = self.format(record)
            level = record.levelname
            self.log_signal.emit(msg, level)
        except Exception:
            pass

SCENE_NAMES_CN = {
    "HOME_LOBBY": "游戏主城大厅",
    "CHAOS_SELECT": "卡厄思分支",
    "SYSTEM_ZERO_LOBBY": "零号系统大厅",
    "CODEX_SELECT": "法典选择",
    "ROGUELIKE_MAP": "肉鸽迷宫选路",
    "EVENT_CHOICE": "局内奇遇事件",
    "EPIPHANY_HERO": "灵光一闪选人",
    "EPIPHANY_CARDS": "灵光一闪选牌",
    "BATTLE": "战斗进行中",
    "CAMP_SHOP": "营地与商店",
    "EQUIP_ASSIGN": "获得装备分配",
    "STAGE_MAP": "主线关卡大地图",
    "CUTSCENE_DIALOG": "主线剧情演出",
    "SETTLEMENT_LOOT": "战利品结算",
    "UNKNOWN": "识别中/过渡帧",
}

class BotWorker(QThread):
    """
    Worker thread that runs the Orchestrator loop.
    Emits signals for state updates, status changes, and logs.
    """
    # Signals
    log_received = Signal(str, str)             # (message, level)
    status_changed = Signal(str, str)           # (status_text, color)
    scene_changed = Signal(str)                 # (scene_name)
    running_state_changed = Signal(bool)        # (is_running)
    stats_updated = Signal(dict)                # ({run_time: int, loop_count: int, ...})

    def __init__(self, config: Dict[str, Any], parent: Optional[QObject] = None):
        super().__init__(parent)
        self.config = config
        self._is_running = False
        self._orchestrator = None
        self._start_time = 0.0
        self._loop_count = 0

    def run(self):
        from ..fsm.orchestrator import ChaosOrchestrator

        self._is_running = True
        self.running_state_changed.emit(True)
        self.status_changed.emit("正在初始化引擎...", "#FFA500")

        # Attach QtLogHandler to root logger with INFO level
        handler = QtLogHandler(self.log_received)
        handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
        handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

        # Attach RunCappedFileHandler to root logger if not already attached
        from ..core.log_rotator import RunCappedFileHandler
        has_file_handler = any(isinstance(h, RunCappedFileHandler) for h in root_logger.handlers)
        if not has_file_handler:
            log_cfg = self.config.get("logging", {}) if isinstance(self.config, dict) else {}
            max_runs = int(log_cfg.get("max_runs", 3))
            max_lines = int(log_cfg.get("max_lines_per_run", 200))
            log_path = Path(__file__).resolve().parent.parent / "chaos_bot.log"
            file_handler = RunCappedFileHandler(str(log_path), max_runs=max_runs, max_lines_per_run=max_lines)
            file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", "%H:%M:%S"))
            root_logger.addHandler(file_handler)

        worker_logger = logging.getLogger("BotWorker")
        worker_logger.info("==================================================")
        worker_logger.info("【自动化控制台】挂机任务已启动，实时详细流程日志已开启...")

        try:
            self._orchestrator = ChaosOrchestrator(self.config)
            self._start_time = time.time()
            self._loop_count = 0
            win_info = self._orchestrator.ctx.window.get_info()
            if win_info.get("status") == "not_found":
                self.status_changed.emit("等待游戏《卡厄思梦境》...", "#FFA500")
                worker_logger.info("【窗口检测】等待《卡厄思梦境》游戏窗口就绪中...")
            else:
                self.status_changed.emit(f"已锁定: {win_info.get('title')}", "#00E676")
                worker_logger.info(f"【窗口检测】已成功锁定游戏窗口: {win_info.get('title')}")

            while self._is_running and self._orchestrator.ctx.running:
                self._orchestrator.step()
                self._loop_count += 1

                # Send stats
                elapsed = int(time.time() - self._start_time)
                self.stats_updated.emit({
                    "elapsed_seconds": elapsed,
                    "loop_count": self._loop_count
                })

                # Check scene name
                if hasattr(self._orchestrator, "_last_logged_screen") and self._orchestrator._last_logged_screen:
                    raw_sc = self._orchestrator._last_logged_screen
                    sc_cn = SCENE_NAMES_CN.get(raw_sc, raw_sc)
                    self.scene_changed.emit(sc_cn)

                time.sleep(0.1)

        except Exception as e:
            logging.getLogger("BotWorker").exception(f"挂机异常退出: {e}")
            self.status_changed.emit(f"异常退出: {e}", "#FF5252")
        finally:
            if self._orchestrator:
                self._orchestrator.stop()
            root_logger.removeHandler(handler)
            self._is_running = False
            self.running_state_changed.emit(False)
            self.status_changed.emit("已停止", "#888888")

    def stop_worker(self):
        """Request the worker thread to stop cleanly."""
        self._is_running = False
        if self._orchestrator and hasattr(self._orchestrator, "ctx"):
            self._orchestrator.ctx.stop()
        self.wait(3000)

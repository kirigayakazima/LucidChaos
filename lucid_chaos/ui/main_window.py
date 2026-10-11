"""
Main Window (FluentWindow) for ChaosBot PC Client.
Built with PySide6 & QFluentWidgets (Fluent Design).
"""

import sys
import os
import json
from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

from qfluentwidgets import (
    FluentWindow, NavigationInterface, NavigationItemPosition,
    FluentIcon as FIF, InfoBar, InfoBarPosition, setTheme, Theme
)

from .home_interface import HomeInterface
from .settings_interface import SettingsInterface
from .toolbox_interface import ToolboxInterface
from .worker import BotWorker

class MainWindow(FluentWindow):
    """Modern Fluent Desktop GUI for Chaos Bot."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LucidChaos 控制台 · 卡厄思自动化助手")
        self.resize(1000, 680)

        # Config path
        self.config_path = str(Path(__file__).resolve().parent.parent / "config.json")
        self.config_data = self._load_config()

        # Bot background worker thread
        self.worker = None

        # Init Sub-Interfaces
        self.home_interface = HomeInterface(self)
        self.settings_interface = SettingsInterface(self.config_path, self)
        self.toolbox_interface = ToolboxInterface(self)

        self._init_navigation()
        self._wire_signals()
        self._setup_shortcuts()

        # Center on screen
        self._center_window()

    def _load_config(self) -> dict:
        p = Path(self.config_path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _init_navigation(self):
        # 1. 主页看板
        self.addSubInterface(
            self.home_interface,
            FIF.HOME,
            "控制看板",
            position=NavigationItemPosition.TOP
        )

        # 2. 策略配置
        self.addSubInterface(
            self.settings_interface,
            FIF.SETTING,
            "策略配置",
            position=NavigationItemPosition.TOP
        )

        # 3. 工具箱
        self.addSubInterface(
            self.toolbox_interface,
            FIF.DEVELOPER_TOOLS,
            "视觉工具箱",
            position=NavigationItemPosition.TOP
        )

    def _wire_signals(self):
        # Home interface buttons
        self.home_interface.start_requested.connect(self.start_bot)
        self.home_interface.stop_requested.connect(self.stop_bot)

        # Settings save
        self.settings_interface.config_saved.connect(self._on_config_updated)

    def _setup_shortcuts(self):
        # F8: Start Bot
        self.shortcut_start = QShortcut(QKeySequence("F8"), self)
        self.shortcut_start.activated.connect(self.start_bot)

        # F9: Stop Bot
        self.shortcut_stop = QShortcut(QKeySequence("F9"), self)
        self.shortcut_stop.activated.connect(self.stop_bot)

    def _center_window(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = (screen.height() - self.height()) // 2
        self.move(x, y)

    def _on_config_updated(self, new_cfg: dict):
        self.config_data = new_cfg
        InfoBar.info("配置同步", "策略已实时同步至内存中！", parent=self, duration=2000)

    def start_bot(self):
        if self.worker is not None and self.worker.isRunning():
            InfoBar.warning("提示", "挂机脚本已经在运行中！", parent=self)
            return

        self.config_data = self._load_config()
        self.worker = BotWorker(self.config_data, self)
        self.worker.log_received.connect(self.home_interface.append_log)
        self.worker.status_changed.connect(self.home_interface.set_status)
        self.worker.scene_changed.connect(self.home_interface.set_scene)
        self.worker.stats_updated.connect(self.home_interface.set_stats)
        self.worker.running_state_changed.connect(self.home_interface.set_running_state)

        self.worker.start()
        InfoBar.success("已启动", "自动化挂机已开始运行，按 F9 可随时快捷停止", parent=self, position=InfoBarPosition.TOP_RIGHT, duration=2500)

    def stop_bot(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop_worker()
            InfoBar.warning("已停止", "自动化挂机已安全停止。", parent=self, position=InfoBarPosition.TOP_RIGHT, duration=2000)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop_worker()
        event.accept()

def launch_app():
    # Enable High DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    setTheme(Theme.AUTO)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

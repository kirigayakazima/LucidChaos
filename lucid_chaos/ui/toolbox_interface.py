"""
Toolbox Interface (视觉调试与工具箱) for ChaosBot.
Provides 2K template online cropper, fixture browser, and game window test.
"""

import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from qfluentwidgets import (
    CardWidget, PrimaryPushButton, PushButton, LineEdit,
    FluentIcon as FIF, TitleLabel, SubtitleLabel, CaptionLabel,
    InfoBar, InfoBarPosition
)

class ToolboxInterface(QWidget):
    """Toolbox and Vision Calibration Page."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ToolboxInterface")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(20)

        # Title
        title_box = QVBoxLayout()
        title_lbl = TitleLabel("视觉调试与工具箱", self)
        sub_lbl = CaptionLabel("针对 2K 高精度模板快速截取、窗口句柄检测与自动化校准工具", self)
        sub_lbl.setTextColor("#777777", "#AAAAAA")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)
        layout.addLayout(title_box)

        # 1. 2K Template Cropper Card
        card_crop = CardWidget(self)
        layout_crop = QVBoxLayout(card_crop)
        layout_crop.setContentsMargins(20, 18, 20, 18)
        layout_crop.setSpacing(12)

        crop_title = SubtitleLabel("高精度模板在线截取工具", card_crop)
        crop_desc = CaptionLabel("从当前运行的游戏画面中一键截屏，并在弹出窗口中用鼠标拉框快速保存到 templates 目录", card_crop)
        crop_desc.setTextColor("#888888", "#AAAAAA")
        layout_crop.addWidget(crop_title)
        layout_crop.addWidget(crop_desc)

        action_row = QHBoxLayout()
        self.input_tpl_name = LineEdit(card_crop)
        self.input_tpl_name.setPlaceholderText("请输入模板文件名 (例如: rogue_arrow_cyan.png)")
        self.input_tpl_name.setText("custom_template.png")
        self.input_tpl_name.setFixedHeight(36)

        self.btn_launch_cropper = PrimaryPushButton(FIF.PHOTO, "开始截取模板", card_crop)
        self.btn_launch_cropper.setFixedHeight(36)
        self.btn_launch_cropper.clicked.connect(self._launch_cropper)

        action_row.addWidget(self.input_tpl_name, 1)
        action_row.addWidget(self.btn_launch_cropper)
        layout_crop.addLayout(action_row)

        layout.addWidget(card_crop)

        # 2. Window Handle Test Card
        card_win = CardWidget(self)
        layout_win = QVBoxLayout(card_win)
        layout_win.setContentsMargins(20, 18, 20, 18)
        layout_win.setSpacing(12)

        win_title = SubtitleLabel("游戏窗口连接测试", card_win)
        win_desc = CaptionLabel("检测《卡厄思梦境》PC端窗口句柄、分辨率以及后台截图权限", card_win)
        win_desc.setTextColor("#888888", "#AAAAAA")
        layout_win.addWidget(win_title)
        layout_win.addWidget(win_desc)

        self.btn_test_win = PushButton(FIF.SEARCH, "测试游戏窗口连接", card_win)
        self.btn_test_win.setFixedSize(180, 36)
        self.btn_test_win.clicked.connect(self._test_window)
        layout_win.addWidget(self.btn_test_win)

        self.lbl_win_res = CaptionLabel("点击上方按钮开始测试...", card_win)
        layout_win.addWidget(self.lbl_win_res)

        layout.addWidget(card_win)

        layout.addStretch(1)

    def _launch_cropper(self):
        tpl_name = self.input_tpl_name.text().strip()
        if not tpl_name:
            InfoBar.warning("提示", "请输入模板文件名", parent=self.window())
            return
        if not tpl_name.endswith(".png"):
            tpl_name += ".png"

        helper_script = Path(__file__).resolve().parent.parent / "tools" / "crop_template_helper.py"
        if not helper_script.exists():
            InfoBar.error("错误", "未找到 crop_template_helper.py 脚本", parent=self.window())
            return

        cmd = [sys.executable, str(helper_script), "--capture", "--save-as", tpl_name]
        try:
            subprocess.Popen(cmd)
            InfoBar.info("截取工具已启动", f"正在捕获画面并打开选区窗口，保存文件名: {tpl_name}", parent=self.window(), duration=3000)
        except Exception as e:
            InfoBar.error("启动失败", str(e), parent=self.window())

    def _test_window(self):
        try:
            from ..core.window import WindowManager
            from ..core.capture import WindowCapture
            import json

            cfg_path = Path(__file__).resolve().parent.parent / "config.json"
            cfg = {}
            if cfg_path.exists():
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)

            wm = WindowManager(cfg.get("window", {}))
            if wm.find_window():
                rect = wm.client_rect
                cap = WindowCapture(wm)
                img = cap.capture()
                if img is not None:
                    h, w = img.shape[:2]
                    self.lbl_win_res.setText(f"✅ 成功连接游戏窗口！HWND: {wm.hwnd} | 客户区尺寸: {w}x{h} (Rect: {rect})")
                    InfoBar.success("连接成功", f"找到游戏窗口，分辨率为 {w}x{h}", parent=self.window())
                else:
                    self.lbl_win_res.setText(f"⚠️ 找到窗口句柄但未能成功后台捕获画面 (HWND: {wm.hwnd})")
            else:
                self.lbl_win_res.setText("❌ 未检测到《卡厄思梦境》游戏窗口，请确保游戏已启动并运行。")
                InfoBar.warning("未找到游戏", "请先启动游戏《卡厄思梦境》", parent=self.window())
        except Exception as e:
            self.lbl_win_res.setText(f"❌ 测试发生异常: {e}")

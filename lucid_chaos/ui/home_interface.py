"""
Home Dashboard Interface (主控看板) for ChaosBot.
Displays system status, start/stop toggle, running stats, and real-time log stream.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame
)
from qfluentwidgets import (
    PrimaryPushButton, PushButton, TextEdit, CardWidget,
    IconWidget, FluentIcon as FIF, BodyLabel, TitleLabel, SubtitleLabel,
    CaptionLabel, StrongBodyLabel, ProgressBar
)

class StatCard(CardWidget):
    """Small card showing a single stat metric (e.g. 运行时间 / 循环次数)."""
    def __init__(self, icon: FIF, title: str, init_val: str = "--", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        self.icon_widget = IconWidget(icon, self)
        self.icon_widget.setFixedSize(28, 28)
        layout.addWidget(self.icon_widget)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        self.title_lbl = CaptionLabel(title, self)
        self.title_lbl.setTextColor("#888888", "#AAAAAA")
        self.val_lbl = StrongBodyLabel(init_val, self)
        text_layout.addWidget(self.title_lbl)
        text_layout.addWidget(self.val_lbl)

        layout.addLayout(text_layout)
        layout.addStretch(1)

    def set_value(self, val: str):
        self.val_lbl.setText(val)

class HomeInterface(QWidget):
    """Home Dashboard Page."""
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HomeInterface")
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(32, 24, 32, 24)
        main_layout.setSpacing(16)

        # 1. Header Banner
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        
        self.title_lbl = TitleLabel("卡厄思梦境 · 自动化控制台", self)
        self.sub_lbl = CaptionLabel("基于 OpenCV + HSV 多模态场景路由的高精度挂机助手", self)
        self.sub_lbl.setTextColor("#777777", "#AAAAAA")
        title_box.addWidget(self.title_lbl)
        title_box.addWidget(self.sub_lbl)
        header_layout.addLayout(title_box)
        header_layout.addStretch(1)

        # Big Action Button
        self.btn_toggle = PrimaryPushButton(FIF.PLAY, "启动挂机 (F8)", self)
        self.btn_toggle.setFixedSize(160, 42)
        self.btn_toggle.clicked.connect(self._on_toggle_clicked)
        header_layout.addWidget(self.btn_toggle)

        main_layout.addLayout(header_layout)

        # 2. Stats & Status Grid
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(12)

        self.card_status = StatCard(FIF.INFO, "当前状态", "就绪 (未运行)")
        self.card_scene = StatCard(FIF.TAG, "当前场景", "--")
        self.card_time = StatCard(FIF.DATE_TIME, "已运行时间", "00:00:00")
        self.card_loops = StatCard(FIF.SYNC, "执行步数", "0 次")

        stats_layout.addWidget(self.card_status)
        stats_layout.addWidget(self.card_scene)
        stats_layout.addWidget(self.card_time)
        stats_layout.addWidget(self.card_loops)
        main_layout.addLayout(stats_layout)

        # 3. Log Stream Card
        log_card = CardWidget(self)
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.setSpacing(10)

        log_header = QHBoxLayout()
        log_title = SubtitleLabel("实时运行日志", log_card)
        log_header.addWidget(log_title)
        log_header.addStretch(1)

        self.btn_clear_log = PushButton(FIF.DELETE, "清空日志", log_card)
        self.btn_clear_log.setFixedSize(100, 32)
        self.btn_clear_log.clicked.connect(self._clear_logs)
        log_header.addWidget(self.btn_clear_log)
        log_layout.addLayout(log_header)

        self.log_text = TextEdit(log_card)
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("等待任务启动中，日志将在此处实时刷新...")
        log_layout.addWidget(self.log_text)

        main_layout.addWidget(log_card, 1)

        self._is_running = False

    def _on_toggle_clicked(self):
        if not self._is_running:
            self.start_requested.emit()
        else:
            self.stop_requested.emit()

    def set_running_state(self, is_running: bool):
        self._is_running = is_running
        if is_running:
            self.btn_toggle.setText("停止挂机 (F9)")
            self.btn_toggle.setIcon(FIF.PAUSE)
        else:
            self.btn_toggle.setText("启动挂机 (F8)")
            self.btn_toggle.setIcon(FIF.PLAY)

    def set_status(self, text: str, color: str = ""):
        self.card_status.set_value(text)

    def set_scene(self, scene: str):
        self.card_scene.set_value(scene)

    def set_stats(self, stats: dict):
        elapsed = stats.get("elapsed_seconds", 0)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        self.card_time.set_value(f"{h:02d}:{m:02d}:{s:02d}")
        loops = stats.get("loop_count", 0)
        self.card_loops.set_value(f"{loops} 次")

    def append_log(self, msg: str, level: str):
        import html
        safe_msg = html.escape(msg)

        color = "#E0E0E0"
        font_weight = "normal"

        if level in ["WARNING", "WARN"]:
            color = "#FFA726"
        elif level == "ERROR":
            color = "#FF5252"
            font_weight = "bold"
        elif "【流程进度" in safe_msg or "【自动化控制台】" in safe_msg or "【系统就绪】" in safe_msg:
            color = "#00E5FF"
            font_weight = "bold"
        elif "【法典选择】" in safe_msg or "【零号系统】" in safe_msg or "【卡厄思" in safe_msg or "【大厅" in safe_msg:
            color = "#40C4FF"
        elif "【战斗" in safe_msg or "【出牌" in safe_msg or "【战斗技能】" in safe_msg or "【回合】" in safe_msg:
            color = "#FF4081"
        elif "【肉鸽选路】" in safe_msg or "【肉鸽前行】" in safe_msg or "【路线】" in safe_msg:
            color = "#29B6F6"
        elif "【局内事件" in safe_msg or "【多对话框" in safe_msg or "【灵光一闪" in safe_msg or "【获得装备" in safe_msg or "成功" in safe_msg or "锁定" in safe_msg:
            color = "#00E676"
        elif "【关卡结算】" in safe_msg or "【战后结算】" in safe_msg or "【胜利】" in safe_msg or "【获得】" in safe_msg:
            color = "#FFD54F"
        elif "路由" in safe_msg or "ScenePool" in safe_msg:
            color = "#80D8FF"

        html_line = f"<div style='color:{color}; font-weight:{font_weight}; font-family:Consolas, monospace; font-size:12px; line-height:1.4;'>{safe_msg}</div>"
        self.log_text.append(html_line)

        # Keep buffer to max 600 lines for smooth performance
        doc = self.log_text.document()
        if doc.blockCount() > 600:
            from PySide6.QtGui import QTextCursor
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            for _ in range(doc.blockCount() - 500):
                cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
                cursor.removeSelectedText()
                cursor.deleteChar()

        # Scroll to bottom
        from PySide6.QtGui import QTextCursor
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def _clear_logs(self):
        self.log_text.clear()

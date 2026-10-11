"""
Settings Interface (策略配置界面) for ChaosBot.
Provides visual controls using SettingCard + ComboBox/SwitchButton to adjust config.json.
"""

import json
from pathlib import Path
from typing import Dict, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QVBoxLayout
from qfluentwidgets import (
    ScrollArea, ExpandLayout, SettingCardGroup, SettingCard,
    ComboBox, SwitchButton, PrimaryPushButton, InfoBar,
    InfoBarPosition, FluentIcon as FIF, TitleLabel, CaptionLabel
)

class SettingsInterface(ScrollArea):
    """Configuration Page with QFluentWidgets SettingCards."""
    config_saved = Signal(dict)

    def __init__(self, config_path: str, parent=None):
        super().__init__(parent)
        self.setObjectName("SettingsInterface")
        self.config_path = config_path
        self.config_data = self._load_config()

        self.scroll_widget = QWidget()
        self.expand_layout = ExpandLayout(self.scroll_widget)
        self.setWidget(self.scroll_widget)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._init_ui()

    def _load_config(self) -> dict:
        p = Path(self.config_path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_config(self, show_toast: bool = True):
        p = Path(self.config_path)
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.config_data, f, ensure_ascii=False, indent=2)
            if show_toast:
                InfoBar.success(
                    title="保存成功",
                    content="配置已成功写入 config.json！",
                    parent=self.window(),
                    position=InfoBarPosition.TOP_RIGHT,
                    duration=2000
                )
            self.config_saved.emit(self.config_data)
        except Exception as e:
            if show_toast:
                InfoBar.error(
                    title="保存失败",
                    content=f"写入配置文件失败: {e}",
                    parent=self.window(),
                    position=InfoBarPosition.TOP_RIGHT,
                    duration=3000
                )

    def _init_ui(self):
        self.expand_layout.setContentsMargins(32, 24, 32, 24)
        self.expand_layout.setSpacing(20)

        # Title
        title_container = QWidget(self.scroll_widget)
        title_box = QVBoxLayout(title_container)
        title_box.setContentsMargins(0, 0, 0, 0)
        title_box.setSpacing(4)
        title_lbl = TitleLabel("策略配置", title_container)
        sub_lbl = CaptionLabel("实时调整自动化策略、战斗规则与视觉参数，保存后即刻生效", title_container)
        sub_lbl.setTextColor("#777777", "#AAAAAA")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)
        self.expand_layout.addWidget(title_container)

        # 1. 运行模式与基础规则
        group_mode = SettingCardGroup("基础与运行模式", self.scroll_widget)

        # 模式切换
        card_mode = SettingCard(FIF.APPLICATION, "运行模式", "选择主线推进、纯肉鸽爬塔迷宫或零号系统法典刷取", group_mode)
        self.combo_mode = ComboBox(card_mode)
        self.combo_mode.addItems([
            "零号系统 (法典自选 + 纯事件刷取)",
            "主线推进 (剧情+关卡+全自动探索)",
            "纯迷宫肉鸽爬塔 (无限循环选路)"
        ])
        curr_mode = self.config_data.get("mode", "system_zero")
        if curr_mode == "system_zero":
            mode_idx = 0
        elif curr_mode == "main_story":
            mode_idx = 1
        else:
            mode_idx = 2
        self.combo_mode.setCurrentIndex(mode_idx)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        card_mode.hBoxLayout.addWidget(self.combo_mode, 0)
        card_mode.hBoxLayout.addSpacing(16)
        group_mode.addSettingCard(card_mode)

        # 全局悬浮蒙版
        card_overlay = SettingCard(FIF.VIEW, "目标聚焦悬浮蒙版", "在游戏窗口上方实时绘制青色箭头与目标绿色锚框（开发调试建议开启）", group_mode)
        self.switch_overlay = SwitchButton(card_overlay)
        overlay_enabled = self.config_data.get("overlay", {}).get("enabled", True)
        self.switch_overlay.setChecked(overlay_enabled)
        self.switch_overlay.checkedChanged.connect(lambda v: self._set_cfg(["overlay", "enabled"], v))
        card_overlay.hBoxLayout.addWidget(self.switch_overlay, 0)
        card_overlay.hBoxLayout.addSpacing(16)
        group_mode.addSettingCard(card_overlay)

        self.expand_layout.addWidget(group_mode)

        # 2. 零号系统专属配置
        group_zero = SettingCardGroup("零号系统策略配置", self.scroll_widget)

        # 法典选择
        card_codex = SettingCard(FIF.BOOK_SHELF, "目标法典选择", "指定在零号系统法典列表中选择几号法典进行挑战 (01 / 02 / 03)", group_zero)
        self.combo_codex = ComboBox(card_codex)
        self.combo_codex.addItems(["01 号法典", "02 号法典", "03 号法典"])
        curr_codex = self.config_data.get("system_zero", {}).get("target_codex", 1) - 1
        self.combo_codex.setCurrentIndex(max(0, min(2, curr_codex)))
        self.combo_codex.currentIndexChanged.connect(lambda idx: self._set_cfg(["system_zero", "target_codex"], idx + 1))
        card_codex.hBoxLayout.addWidget(self.combo_codex, 0)
        card_codex.hBoxLayout.addSpacing(16)
        group_zero.addSettingCard(card_codex)

        # 寻路策略提示
        card_zero_route = SettingCard(FIF.PIN, "局内路线策略", "零号系统模式下自动执行纯事件/非战斗路线，严禁进入危险战斗", group_zero)
        self.switch_zero_event = SwitchButton(card_zero_route)
        self.switch_zero_event.setChecked(True)
        self.switch_zero_event.setEnabled(False)  # 默认强制锁定纯事件
        card_zero_route.hBoxLayout.addWidget(self.switch_zero_event, 0)
        card_zero_route.hBoxLayout.addSpacing(16)
        group_zero.addSettingCard(card_zero_route)

        self.expand_layout.addWidget(group_zero)

        # 3. 战斗与 BOSS 规则
        group_battle = SettingCardGroup("战斗与主线策略", self.scroll_widget)

        # BOSS 行为
        card_boss = SettingCard(FIF.GAME, "首领战 (BOSS) 应对策略", "卡厄思梦境中首领战原生 AUTO 会被强制锁定，选择应对策略", group_battle)
        self.combo_boss = ComboBox(card_boss)
        self.combo_boss.addItems(["全自动接管 (数字键出牌 + E结束回合)", "暂停交由手动 (弹出提醒并等待主人操作)"])
        boss_act = self.config_data.get("main_story", {}).get("boss_action", "auto")
        self.combo_boss.setCurrentIndex(0 if boss_act == "auto" else 1)
        self.combo_boss.currentIndexChanged.connect(lambda idx: self._set_cfg(["main_story", "boss_action"], "auto" if idx == 0 else "pause"))
        card_boss.hBoxLayout.addWidget(self.combo_boss, 0)
        card_boss.hBoxLayout.addSpacing(16)
        group_battle.addSettingCard(card_boss)

        # 局内常驻 2X
        card_speed = SettingCard(FIF.SPEED_HIGH, "自动开启常驻 2X 倍速", "在对白与战斗等界面检测右上角 ⏩ 图标，若未点亮则自动点亮", group_battle)
        self.switch_speed = SwitchButton(card_speed)
        auto_speed = self.config_data.get("main_story", {}).get("auto_speed", True)
        self.switch_speed.setChecked(auto_speed)
        self.switch_speed.checkedChanged.connect(lambda v: self._set_cfg(["main_story", "auto_speed"], v))
        card_speed.hBoxLayout.addWidget(self.switch_speed, 0)
        card_speed.hBoxLayout.addSpacing(16)
        group_battle.addSettingCard(card_speed)

        self.expand_layout.addWidget(group_battle)

        # 4. 肉鸽路线偏好
        group_rogue = SettingCardGroup("肉鸽地图寻路偏好", self.scroll_widget)

        card_rogue = SettingCard(FIF.FOLDER, "寻路优先偏好", "设定多步骤前瞻评估路线权重 (问号/营地 > 普通怪 > 其余不走)", group_rogue)
        self.combo_rogue = ComboBox(card_rogue)
        self.combo_rogue.addItems(["非战斗优先 (问号 & 篝火优先，跳过精英怪)", "全收益兼顾"])
        pref = self.config_data.get("path_planning", {}).get("preference", "non_battle")
        self.combo_rogue.setCurrentIndex(0 if pref == "non_battle" else 1)
        self.combo_rogue.currentIndexChanged.connect(lambda idx: self._set_cfg(["path_planning", "preference"], "non_battle" if idx == 0 else "balanced"))
        card_rogue.hBoxLayout.addWidget(self.combo_rogue, 0)
        card_rogue.hBoxLayout.addSpacing(16)
        group_rogue.addSettingCard(card_rogue)

        self.expand_layout.addWidget(group_rogue)

        # Save Button
        self.btn_save = PrimaryPushButton(FIF.SAVE, "保存所有配置", self.scroll_widget)
        self.btn_save.setFixedHeight(40)
        self.btn_save.clicked.connect(self._save_config)
        self.expand_layout.addWidget(self.btn_save)

    def _on_mode_changed(self, idx: int):
        if idx == 0:
            val = "system_zero"
        elif idx == 1:
            val = "main_story"
        else:
            val = "labyrinth"
        self._set_cfg(["mode"], val)

    def _set_cfg(self, keys: list, val: Any):
        cur = self.config_data
        for k in keys[:-1]:
            if k not in cur or not isinstance(cur[k], dict):
                cur[k] = {}
            cur = cur[k]
        cur[keys[-1]] = val
        self._save_config(show_toast=False)

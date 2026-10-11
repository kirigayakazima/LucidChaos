"""
Phase 5: Screen Observer
Translates raw captured images into a structured GameState.
"""

from typing import List
import re
import cv2
import numpy as np
from ..core.vision_service import VisionService, VisionTarget
from .screen_state import ScreenState, GameState

class ScreenObserver:
    def __init__(self, vision_service: VisionService):
        self.vision = vision_service

    def observe(self, img_bgr: np.ndarray) -> GameState:
        """Analyze current screen image and classify into a GameState."""
        if img_bgr is None or img_bgr.size == 0:
            return GameState(screen=ScreenState.UNKNOWN, confidence=0.0)
        targets = self.vision.locate_all_texts(img_bgr, min_score=0.5)
        return self._classify(img_bgr, targets)

    def observe_frame(self, frame) -> GameState:
        """复用 FrameContext 里**已经跑过**的 OCR（一帧只跑一次），避免重复识别。"""
        if frame is None or not getattr(frame, "valid", False):
            return GameState(screen=ScreenState.UNKNOWN, confidence=0.0)
        if not frame.targets:
            frame.observe(self.vision)
        gs = self._classify(frame.img, frame.targets)
        gs.frame = frame
        return gs

    def _classify(self, img_bgr: np.ndarray, targets) -> GameState:
        h, w = img_bgr.shape[:2]
        raw_texts = [t.text for t in targets]
        elements = {t.text: t for t in targets}
        text_corpus = " ".join(raw_texts)

        def _state(**kw) -> GameState:
            """elements 会按 text 去重，这里同时带上不去重的完整 targets 列表。"""
            gs = GameState(**kw)
            gs.targets = targets
            return gs

        # 1. Check Disconnect Dialog
        if any(w in text_corpus for w in ["断线", "重新连接", "网络异常", "连线失败"]):
            reconnect_btn = next((t for t in targets if "重连" in t.text or "确认" in t.text), None)
            if reconnect_btn:
                elements["reconnect_btn"] = reconnect_btn
            return _state(
                screen=ScreenState.DISCONNECTED,
                confidence=0.95,
                raw_texts=raw_texts,
                elements=elements
            )

        # Check bottom-right orange button [▷ 離開 / ✓ 獲得 / ✓ 確認]
        crop_btn = img_bgr[int(0.85 * h):int(0.96 * h), int(0.75 * w):int(0.98 * w)]
        hsv_loot = cv2.cvtColor(crop_btn, cv2.COLOR_BGR2HSV) if crop_btn.size > 0 else None
        has_orange_btn = (hsv_loot is not None and 
                          np.count_nonzero(cv2.inRange(hsv_loot, np.array([8, 120, 120]), np.array([25, 255, 255]))) >= 2000)
        has_purple_btn = (hsv_loot is not None and 
                          np.count_nonzero(cv2.inRange(hsv_loot, np.array([125, 90, 120]), np.array([165, 255, 255]))) >= 1500)

        # 1.5 Check Epiphany Card Selection (靈光一閃 3选1 卡牌效果挑选界面)
        # Must NOT be an event dialogue choice containing reward description like "靈光一閃 1次" or "分享資訊"
        is_event_choice_text = any(k in text_corpus for k in ["分享資訊", "分享资讯", "所選戰鬥員", "所选战斗员", "1次", "一次"])
        is_epiphany = not is_event_choice_text and (
            ("長按卡牌" in text_corpus or "长按卡牌" in text_corpus) or
            any(k in text_corpus for k in ["請選擇靈光一閃效果", "请选择灵光一闪效果", "選擇靈光一閃效果", "选择灵光一闪效果"]) or
            (any(k in text_corpus for k in ["靈光一閃", "灵光一闪", "靈光一", "灵光一", "靈光", "灵光"]) and
             any(k in text_corpus for k in ["查看內容", "查看内容", "CAPS", "效果選擇", "效果选择"])) or
            (any(k in text_corpus for k in ["靈光一閃", "灵光一闪", "靈光", "灵光", "【靈光一】", "靈光一"]) and
             any(k in text_corpus for k in ["戰鬥員", "战斗员", "戰鬥", "战斗", "隊員", "队员", "欲獲得", "欲获得", "請選擇", "请选择", "撐欲瘦得", "撑欲瘦得", "門眞", "门真", "的門眞", "的门真", "的眞", "的真"]))
        )
        if is_epiphany:
            return _state(
                screen=ScreenState.EPIPHANY_SELECTION,
                confidence=0.95,
                raw_texts=raw_texts,
                elements=elements
            )

        # 2. Check Equip Selection (获得装备 / 选择战斗员)
        crop_red = img_bgr[int(0.88 * h):int(0.97 * h), int(0.55 * w):int(0.75 * w)]
        hsv_red = cv2.cvtColor(crop_red, cv2.COLOR_BGR2HSV) if crop_red.size > 0 else None
        m1 = cv2.inRange(hsv_red, np.array([0, 100, 100]), np.array([10, 255, 255])) if hsv_red is not None else None
        m2 = cv2.inRange(hsv_red, np.array([170, 100, 100]), np.array([180, 255, 255])) if hsv_red is not None else None
        has_red_btn = (np.count_nonzero(m1 | m2) >= 1500) if (m1 is not None and m2 is not None) else False

        is_equip = (has_red_btn and has_orange_btn) or any(w in text_corpus for w in ["選擇戰鬥員", "选择战斗员", "請選擇戰鬥員", "请选择战斗员", "提煉", "提炼"]) or (any(w in text_corpus for w in ["裝備", "装备"]) and has_red_btn)
        if is_equip:
            return _state(
                screen=ScreenState.EQUIP_SELECTION,
                confidence=0.95,
                raw_texts=raw_texts,
                elements=elements
            )

        # 3. Check Victory / Settlement / Post-battle Loot (排除全力战等非结算界面)
        is_all_out = any(w in text_corpus for w in ["全力", "通關配置", "通关配置", "巨像", "阿巴爾克", "阿巴尔克", "準備戰鬥", "准备战斗"])
        is_settlement = (not is_all_out) and (has_orange_btn or 
                         has_purple_btn or
                         any(w in text_corpus for w in ["戰鬥結束", "战斗结束", "胜利", "勝利", "獲得分數", "战斗分数", "点击任意", "确定", "确认", "確定", "確認", "獎勵", "奖励"]))
        if is_settlement:
            confirm_btn = next((t for t in targets if any(k in t.text for k in ["确定", "确认", "確定", "確認", "继续", "完成", "獲得", "获得"])), None)
            if confirm_btn:
                elements["confirm_btn"] = confirm_btn
            if has_orange_btn or has_purple_btn or confirm_btn:
                return _state(
                    screen=ScreenState.VICTORY_SETTLEMENT,
                    confidence=0.95 if (has_orange_btn or has_purple_btn) else 0.88,
                    raw_texts=raw_texts,
                    elements=elements
                )

        # 3. Check Battle State
        # Indicators: "EP", "回合", "連鎖結晶", or skill counts like "01/10"
        is_lobby = any(w in text_corpus for w in ["方舟都市", "救援", "Seasonal", "機率提升", "机率提升"]) or (
            "/240" in text_corpus and any(w in text_corpus for w in ["卡厄思", "故事", "出擊", "出击", "模擬", "模拟"])
        )
        is_battle = (not is_lobby) and ("EP" in text_corpus or 
                     "連鎖結晶" in text_corpus or 
                     "回合" in text_corpus or 
                     any("/" in t and len(t) <= 6 and "240" not in t and t not in ("0/1", "0/4", "0/9") for t in raw_texts))
        if is_battle:
            return _state(
                screen=ScreenState.BATTLE,
                confidence=0.92,
                raw_texts=raw_texts,
                elements=elements
            )

        # 4. Check Opening Reroll (物主代理人)
        is_opening = any(w in text_corpus for w in ["代理人", "物主", "选择初始", "开局"])
        if is_opening:
            return _state(
                screen=ScreenState.OPENING_REROLL,
                confidence=0.88,
                raw_texts=raw_texts,
                elements=elements
            )

        # 5. Check Shop (德朗商店)
        is_shop = any(w in text_corpus for w in ["德朗", "商店", "购买", "删卡", "金币"])
        if is_shop:
            return _state(
                screen=ScreenState.SHOP,
                confidence=0.90,
                raw_texts=raw_texts,
                elements=elements
            )

        # 6. Check Camp / Safe Area (安全区 / 营地)
        is_camp = any(w in text_corpus for w in ["安全区", "营地", "训练", "休息"])
        if is_camp:
            train_btn = next((t for t in targets if "训练" in t.text), None)
            rest_btn = next((t for t in targets if "休息" in t.text), None)
            if train_btn: elements["train_btn"] = train_btn
            if rest_btn: elements["rest_btn"] = rest_btn
            return _state(
                screen=ScreenState.CAMP,
                confidence=0.90,
                raw_texts=raw_texts,
                elements=elements
            )

        # 7. Check Event (问号事件 / 多对话框交互)
        event_keywords = ["事件", "未知", "抉择", "抉擇", "搭話", "搭话", "搜索", "屍體", "尸体", "靈光一閃", "灵光一闪"]
        has_event_kw = any(w in text_corpus for w in event_keywords)
        has_event_card = any(0.65 <= t.center_rel[1] <= 0.95 and any(k in t.text for k in ["搭話", "搭话", "搜索", "屍體", "尸体", "靈光一閃", "灵光一闪", "獲得隨機", "获得随机", "【複製品】", "【复制品】"]) for t in targets)
        is_event = has_event_kw or has_event_card
        if is_event:
            return _state(
                screen=ScreenState.EVENT,
                confidence=0.90 if has_event_card else 0.80,
                raw_texts=raw_texts,
                elements=elements
            )

        # 8. Check Map Navigation: Outer Stage Map (主线大地图选关) & Roguelike Map (肉鸽迷宫路线)
        has_story_header = any("故事" in t.text and t.center_rel[0] < 0.25 and t.center_rel[1] < 0.15 for t in targets)
        has_chap_header = any(re.search(r'第\d+章', t.text) and t.center_rel[1] < 0.15 for t in targets)
        has_stage_badge = any(re.search(r'#0\d|#1\d', t.text) for t in targets)
        is_outer_stage_map = has_story_header or has_chap_header or has_stage_badge

        # Check bottom-left mini-map rogue template match
        h, w = img_bgr.shape[:2]
        crop_map = img_bgr[int(0.62 * h):int(0.98 * h), int(0.01 * w):int(0.42 * w)]
        is_rogue_map = False
        if crop_map.size > 0:
            from ..logic.path_planner import PathPlanner
            tpl_sword = PathPlanner.get_template('SWORD')
            if tpl_sword is not None:
                scale = h / 576.0
                ts = cv2.resize(tpl_sword, (0, 0), fx=0.22 * scale, fy=0.22 * scale)
                if crop_map.shape[0] >= ts.shape[0] and crop_map.shape[1] >= ts.shape[1]:
                    res = cv2.matchTemplate(crop_map, ts, cv2.TM_CCOEFF_NORMED)
                    _, mv, _, _ = cv2.minMaxLoc(res)
                    if mv >= 0.65:
                        is_rogue_map = True

        if is_outer_stage_map or is_rogue_map or any(w in text_corpus for w in ["航路", "迷宫", "层", "地图"]):
            return _state(
                screen=ScreenState.MAP_NAVIGATION,
                confidence=0.95 if (is_outer_stage_map or is_rogue_map) else 0.75,
                raw_texts=raw_texts,
                elements=elements
            )

        # 9. Check Story / Dialogue / Subtitles (剧情对话、台词、CG)
        # Indicators: subtitle text, quotation marks, ellipsis, or skip button
        is_dialog = any(w in text_corpus for w in ["跳过", "跳過", "……", "...", "。", "！", "？"]) or (len(raw_texts) <= 4 and len(text_corpus) > 4)
        if is_dialog:
            skip_btn = next((t for t in targets if any(k in t.text for k in ["跳过", "跳過"])), None)
            if skip_btn: elements["skip_btn"] = skip_btn
            return _state(
                screen=ScreenState.DIALOG,
                confidence=0.88,
                raw_texts=raw_texts,
                elements=elements
            )

        # Fallback to UNKNOWN
        return _state(
            screen=ScreenState.UNKNOWN,
            confidence=0.3,
            raw_texts=raw_texts,
            elements=elements
        )

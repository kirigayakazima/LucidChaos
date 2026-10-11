"""
Main Story Campaign Auto-Pusher (主线剧情与自动战斗推图引擎) - Accurate Top-Right Icon Model
Implements user-specified rules:
1. Top-Right Icon Detection:
   - Slot 1 (rx: 0.77 ~ 0.83): Diode Skip Icon (▷|)
   - Slot 2 (rx: 0.85 ~ 0.91): Middle Icon (Auto Play ⏩ in story, or AUTO ⚔️ in battle)
   - Slot 3 (rx: 0.93 ~ 0.98): Menu Icon (三)
2. Story Dialogue:
   - When Slot 1 Skip Icon is present (3 icons): Click Skip (rx=0.805, ry=0.052), confirms popups.
3. Battle vs Boss:
   - In battle, if only 2 icons (Boss fight): Skip dialog if any, do NOT take over card play (manual mode).
   - In battle, if 3 icons (Normal fight): Ensure native AUTO is on (rx=0.88, ry=0.05).
4. Stage Map & Drawer:
   - Locks onto uncompleted stage with [▶/N] badge.
   - Clicks [進入] on side drawer and team prep screen.
5. Settlement:
   - Clicks purple [確認] button at bottom-right (rx=0.88, ry=0.91).
"""

import time
import logging
import re
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2

from ..fsm.engine import FSMContext
from ..state.screen_state import GameState, ScreenState
from ..core.stage_detector import StageMapDetector

logger = logging.getLogger("StoryPusher")

class StoryPusher:
    def __init__(self, ctx: FSMContext, config: Dict[str, Any]):
        self.ctx = ctx
        self.config = config
        story_cfg = config.get("main_story", {})
        self.boss_action = story_cfg.get("boss_action", "pause")
        self.auto_speed = story_cfg.get("auto_speed", True)
        self.map_swipe_count = 0
        from .path_planner import PathPlanner
        from .battle_manager import BattleManager
        from .node_engine import NodeDecisionEngine
        from .target_manager import TargetManager
        from ..router.router import SceneRouter
        mode = config.get("mode", "main_story")
        sz_cfg = config.get("system_zero", {})
        event_only = (mode == "system_zero" or sz_cfg.get("event_only", False))
        pref = "event_only" if event_only else config.get("path_planning", {}).get("preference", "non_battle")
        self.path_planner = PathPlanner(preference=pref, event_only=event_only)
        self.battle_mgr = BattleManager(self.ctx)
        self.node_engine = NodeDecisionEngine(self.ctx, TargetManager(), config)
        self.router = SceneRouter(
            self.ctx, config,
            story_pusher=self,
            node_engine=self.node_engine,
            path_planner=self.path_planner,
            battle_mgr=self.battle_mgr
        )

    _tpl_skip_cache = None
    _tpl_arrow_cache = None

    @classmethod
    def _get_skip_template(cls) -> Optional[np.ndarray]:
        from ..core.template_bank import BANK
        return BANK.get("skip_btn.png")

    @classmethod
    def _get_arrow_template(cls) -> Optional[np.ndarray]:
        from ..core.template_bank import BANK
        return BANK.get("next_chapter_arrow.png")

    @classmethod
    def _get_search_bubble_template(cls) -> Optional[np.ndarray]:
        from ..core.template_bank import BANK
        return BANK.get("explore_search_bubble.png")

    @classmethod
    def inspect_top_right_icons(cls, img_bgr: np.ndarray) -> Tuple[int, bool, bool, bool, bool]:
        """
        Inspect the top-right control capsule using template/feature matching:
        Slot 1 (rx: 0.77 ~ 0.83): Skip Icon (▷|) - verified via template match (>= 0.75)
        Slot 2 (rx: 0.85 ~ 0.91): Middle Icon (⏩ Speed or AUTO)
        Slot 3 (rx: 0.93 ~ 0.98): Menu Icon (三)
        Returns (icon_count, has_skip, has_mid, has_menu, is_mid_active).
        where is_mid_active indicates if Slot 2 is golden/highlighted (2x speed or AUTO active).
        """
        if img_bgr is None or img_bgr.size == 0:
            return 0, False, False, False, False

        h, w = img_bgr.shape[:2]
        scale = h / 576.0

        # 1. Feature verification of Skip Icon (▷|)
        has_slot1 = False
        tpl_skip = cls._get_skip_template()
        if tpl_skip is not None:
            roi_skip = img_bgr[:int(0.12 * h), int(0.72 * w):int(0.86 * w)]
            if roi_skip.size > 0:
                for s in [0.90, 1.0, 1.10]:
                    ts = cv2.resize(tpl_skip, (0, 0), fx=scale * s, fy=scale * s)
                    if roi_skip.shape[0] >= ts.shape[0] and roi_skip.shape[1] >= ts.shape[1]:
                        res = cv2.matchTemplate(roi_skip, ts, cv2.TM_CCOEFF_NORMED)
                        _, max_v, _, _ = cv2.minMaxLoc(res)
                        if max_v >= 0.75:
                            has_slot1 = True
                            break

        # 2. Contour check for slots 2 and 3 (Auto/Speed and Menu)
        x1, x2 = int(0.84 * w), int(0.98 * w)
        y1, y2 = int(0.02 * h), int(0.09 * h)
        crop = img_bgr[y1:y2, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        has_slot2 = False
        has_slot3 = False
        for c in contours:
            area = cv2.contourArea(c)
            if 10 < area < 600:
                bx, by, bw, bh = cv2.boundingRect(c)
                if bw < 50 and bh < 45:
                    rx = (x1 + bx + bw / 2) / w
                    if 0.85 <= rx <= 0.91:
                        has_slot2 = True
                    elif 0.93 <= rx <= 0.98:
                        has_slot3 = True

        # 3. Check if Slot 2 is golden/highlighted (2x fast-forward or AUTO enabled)
        is_mid_active = False
        if has_slot2:
            crop2 = img_bgr[int(0.025 * h):int(0.08 * h), int(0.855 * w):int(0.910 * w)]
            hsv2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)
            lower_gold = np.array([10, 80, 110])
            upper_gold = np.array([35, 255, 255])
            gold_count = np.count_nonzero(cv2.inRange(hsv2, lower_gold, upper_gold))
            if gold_count > 30:
                is_mid_active = True

        icon_count = sum([has_slot1, has_slot2, has_slot3])
        return icon_count, has_slot1, has_slot2, has_slot3, is_mid_active

    def ensure_speed_2x(self, img_bgr: np.ndarray) -> bool:
        """
        局内常驻 2X 加速：右上角 ⏩/AUTO 按钮没点亮就点亮它（主人要求："局内应该帮我激活"）。

        判据复用 `inspect_top_right_icons`（形状+金色像素，不是文字）：
          - `has_mid`  ：slot2（rx 0.85~0.91）有图标
          - `is_mid_active`：该区域金色像素 > 30（已点亮）
        实测 6 张夹具里只有「局内事件」和「对白」两张有这个按钮且未点亮，
        肉鸽选路 / 结算 / 选牌 / 选关 都不检测到 -> 不会误点顶部菜单。

        ⚠️ 只应由"自己不处理加速的画面"调用：对白界面由 `handle_cutscene_dialog` 自己点，
        战斗界面由 BattleManager 管 AUTO —— 重复点同一个开关会把它**关掉**，
        所以调用方必须先过 `should_activate_speed(scene)`。
        """
        if img_bgr is None or img_bgr.size == 0:
            return False
        count, _skip, has_mid, _menu, is_active = self.inspect_top_right_icons(img_bgr)
        if has_mid and not is_active:
            logger.info("【常驻加速】右上角 ⏩/AUTO 图标未点亮，点击激活（局内常驻 2X）...")
            self.ctx.input.click_area_rel(0.865, 0.035, 0.895, 0.068, delay_after=0.35,
                                          label="【常驻加速】2X")
            return True
        return False

    def _find_settlement_confirm_button(self, img_bgr: np.ndarray) -> Optional[tuple]:
        """Detect the bright purple [確認] button on the settlement screen."""
        if img_bgr is None:
            return None
        h, w = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_purple = np.array([125, 90, 120])
        upper_purple = np.array([165, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            # Settlement confirm button is wide (>60px), area > 500, aspect ratio bw/bh >= 1.6
            # Avoid matching square reward item icons (e.g. 40x40 crystals)
            if area > 500:
                x, y, bw, bh = cv2.boundingRect(c)
                if bh > 0 and (bw / bh) >= 1.5:
                    cx, cy = x + bw // 2, y + bh // 2
                    rx, ry = cx / w, cy / h
                    if rx > 0.70 and ry > 0.80:
                        return (cx, cy, (rx, ry))
        return None

    @classmethod
    def _find_next_chapter_entry(cls, img_bgr: np.ndarray, ocr_targets: Optional[List[Any]] = None) -> Optional[Tuple[int, int, Tuple[float, float], str]]:
        """
        Locate next chapter entrance when current chapter stages are all completed:
        1. Feature template match for next chapter arrow [>] at right edge (rx: 0.88~0.99, ry: 0.40~0.60)
        2. OCR check on right chapter timeline card (e.g. '02', '03', or chapter title)
        3. OCR check on top navigation bar for next chapter (e.g. '第2章', '第3章'...) at rx: 0.52~0.70, ry < 0.08
        """
        if img_bgr is None or img_bgr.size == 0:
            return None
        h, w = img_bgr.shape[:2]
        scale = h / 576.0

        # 1. Feature template match for right chevron [>]
        tpl_arrow = cls._get_arrow_template()
        if tpl_arrow is not None:
            roi_arrow = img_bgr[int(0.40 * h):int(0.60 * h), int(0.88 * w):int(0.99 * w)]
            if roi_arrow.size > 0:
                for s in [0.90, 1.0, 1.10]:
                    ts = cv2.resize(tpl_arrow, (0, 0), fx=scale * s, fy=scale * s)
                    if roi_arrow.shape[0] >= ts.shape[0] and roi_arrow.shape[1] >= ts.shape[1]:
                        res = cv2.matchTemplate(roi_arrow, ts, cv2.TM_CCOEFF_NORMED)
                        _, max_v, _, max_l = cv2.minMaxLoc(res)
                        if max_v >= 0.75:
                            cx = int(0.88 * w + max_l[0] + ts.shape[1] // 2)
                            cy = int(0.40 * h + max_l[1] + ts.shape[0] // 2)
                            return (cx, cy, (cx / w, cy / h), f"右侧章节大箭头 [>] (置信度={max_v:.2f})")

        # Convert OCR targets to uniform format
        targets = []
        if ocr_targets:
            for t in ocr_targets:
                if hasattr(t, 'text'):
                    targets.append((t.text.strip(), t.center, t.center_rel))
                elif isinstance(t, dict):
                    targets.append((t.get('text', '').strip(), t.get('center_client', (0, 0)), t.get('center_rel', (0.0, 0.0))))

        # 2. OCR check on right chapter card (02, 03, 04... or Chapter Title)
        for text, center, (rx, ry) in targets:
            if 0.76 <= rx <= 0.98 and 0.38 <= ry <= 0.62:
                if re.match(r'^0[2-9]$|^1[0-9]$', text):
                    return (center[0], center[1], (rx, ry), f"右侧章节标号 [{text}]")
                if any(w in text for w in ["封印", "禁忌", "進化", "进化", "催化", "催化劑", "章", "節"]):
                    return (center[0], center[1], (rx, ry), f"右侧章节标题 [{text}]")

        # 3. Top navigation next chapter (rx: 0.52 ~ 0.70, ry < 0.08)
        for text, center, (rx, ry) in targets:
            if 0.52 <= rx <= 0.70 and ry < 0.08:
                if re.match(r'第[2-9一二三四五六七八九十]+章', text) or ("章" in text and any(d in text for d in "23456789")):
                    return (center[0], center[1], (rx, ry), f"顶部导航下一章 [{text}]")

        return None

    def _find_story_dialogue_choices(self, raw_img: np.ndarray, ocr_targets: Optional[List[Any]] = None) -> List[Any]:
        """
        Locate floating choice banners in the center of the screen during main story dialogues.
        Geometry:
        - Horizontally centered: 0.30 <= rx <= 0.70
        - Vertically upper-middle: 0.25 <= ry <= 0.68
        Per user instruction: "主线出现选项的时候，其实应该是随便点一个的"
        """
        if ocr_targets is None or len(ocr_targets) == 0:
            if hasattr(self.ctx, 'vision') and raw_img is not None:
                ocr_targets = self.ctx.vision.locate_all_texts(raw_img, min_score=0.4)
            else:
                return []

        choices = []
        for t in ocr_targets:
            rx, ry = t.center_rel
            # Exclude stage badges and chapter titles
            if any(k in t.text for k in ["#0", "#1", "#2", "第", "章", "節", "關", "关"]):
                continue
            # Exclude top bar (ry < 0.22) and bottom subtitles (ry > 0.70)
            if 0.30 <= rx <= 0.70 and 0.25 <= ry <= 0.68:
                choices.append(t)

        choices.sort(key=lambda c: c.center_rel[1])
        return choices

    def _execute_battle_turn(self, raw_img: np.ndarray, text_corpus: str) -> bool:
        """
        Execute an automated battle turn according to user rules:
        1. Left Character Skills:
           - Check 1, 2, 3 hero slots on the left (rx: 0.10, ry: 0.725 / 0.815 / 0.905).
           - Detect golden glowing border (HSV gold > 150).
           - Sequentially trigger: 1号位 -> 2号位 -> 3号位.
        2. Hand Card Drag Play:
           - Hand cards MUST be dragged upwards to the battlefield (drag from rx, 0.84 to rx, 0.42).
           - Cost rule: Total 3 AP spent per turn.
           - 0-cost cards must be played completely first!
           - Priority: 0-cost (200) > ATTACK [攻擊] (100) > SKILL [技能] (60) > ENHANCE [強化] (30, max 1 per turn).
        3. Turn End:
           - When cards are played to 0 AP / exhausted, click the bottom-right Turn End button (✓ at rx: 0.89, ry: 0.86).
        """
        h, w = raw_img.shape[:2]

        # 1. Left Character Skills Rotation
        hero_slots = [
            (1, 0.100, 0.725, 0.68, 0.77),
            (2, 0.100, 0.815, 0.77, 0.86),
            (3, 0.100, 0.905, 0.86, 0.95),
        ]
        lit_heroes = []
        for slot_num, cx_rel, cy_rel, ry1, ry2 in hero_slots:
            crop = raw_img[int(ry1 * h):int(ry2 * h), int(0.05 * w):int(0.14 * w)]
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([12, 100, 150]), np.array([32, 255, 255]))
            if np.count_nonzero(mask) > 150:
                lit_heroes.append((slot_num, cx_rel, cy_rel))

        # Check sequential rotation
        if lit_heroes:
            for offset in range(3):
                cand_slot = (self.hero_rot_idx + offset) % 3 + 1
                match = next((m for m in lit_heroes if m[0] == cand_slot), None)
                if match:
                    slot_num, cx_rel, cy_rel = match
                    logger.info(f"【战斗技能】触发 {slot_num} 号位角色技能/绝招 (按 1->2->3 轮转)...")
                    self.ctx.input.click_rel(cx_rel, cy_rel, delay_after=0.4)
                    self.hero_rot_idx = slot_num % 3
                    break

        # 2. Hand Card Layout & Identification
        crop_hand = raw_img[int(0.70 * h):int(0.96 * h), int(0.18 * w):int(0.82 * w)]
        hand_ocr = self.ctx.vision.locate_all_texts(crop_hand, min_score=0.4)

        # Detect hand count (e.g. '05/10', '07/10')
        hand_count = 5
        for t in hand_ocr:
            m = re.search(r'0([3-9])/10', t.text)
            if m:
                hand_count = int(m.group(1))
                break

        # Calculate horizontal positions of hand cards
        card_positions = []
        for i in range(hand_count):
            rx = 0.23 + i * (0.76 - 0.23) / max(1, hand_count - 1)
            card_positions.append({
                'index': i + 1,
                'rx': rx,
                'type': 'SKILL',
                'priority': 50,
                'cost': 1,
                'title': ''
            })

        # Match OCR items to cards
        for t in hand_ocr:
            text = t.text
            cx_abs = (int(0.18 * w) + t.center[0]) / w
            closest = min(card_positions, key=lambda c: abs(c['rx'] - cx_abs))

            # Check 0-cost
            if "0" in text and ("0/" not in text) and ("05" not in text) and ("07" not in text):
                closest['cost'] = 0
                closest['priority'] = 200

            if any(k in text for k in ['攻擊', '攻击', '子彈', '子弹', '破碎', '閃光', '闪光']):
                closest['type'] = 'ATTACK'
                if closest['priority'] < 100:
                    closest['priority'] = 100
                closest['title'] = text
            elif any(k in text for k in ['强化', '強化', '回聲', '回声', '成對', '成对']):
                closest['type'] = 'ENHANCE'
                if closest['priority'] < 100:
                    closest['priority'] = 30
                closest['title'] = text
            elif any(k in text for k in ['技能', '命運', '命运', '鬥志', '斗志', '永凍', '永冻', '凝聚']):
                if closest['priority'] < 100:
                    closest['priority'] = 60
                closest['title'] = text

        # Sort cards by priority: 0-cost (200) > ATTACK (100) > SKILL (60) > ENHANCE (30)
        sorted_cards = sorted(card_positions, key=lambda c: c['priority'], reverse=True)

        # Plan play queue: All 0-cost cards, max 1 Enhance card, up to 4 cards total (spending 3 AP to 0)
        play_queue = []
        used_enhance = False
        for c in sorted_cards:
            if c['type'] == 'ENHANCE':
                if used_enhance:
                    continue
                used_enhance = True
            play_queue.append(c)
            if len(play_queue) >= 4:
                break

        # 3. Execute Upward Drag for each card
        for c in play_queue:
            rx = c['rx']
            c_desc = f"#{c['index']} [{c['type']}:{c['title'] or '手牌'}] (费用={c['cost']})"
            logger.info(f"【自动出牌】长按并上滑打出 {c_desc} (rx: {rx:.3f} -> 战场中轴)...")
            self.ctx.input.drag_rel(rx, 0.84, rx, 0.42, steps=15, duration=0.25)
            time.sleep(0.5)

        # 4. End Turn: Click the bottom-right turn end button (✓ at rx: 0.89, ry: 0.86)
        logger.info("【回合结束】本轮手牌出牌完毕 (费用清零)，点击右下角【回合结束 ✓】(rx: 0.89, ry: 0.86)...")
        self.ctx.input.click_area_rel(0.87, 0.83, 0.91, 0.89, delay_after=1.5)
        return True

    def handle_stage_map(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """Handle outer chapter/stage map navigation, drawer, and team prep."""
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
            if raw_img is None:
                return False

        text_corpus = " ".join(state.raw_texts)

        # 1. Check [進入] Buttons: Full-Screen Stage, Side Drawer, OR Team Prep (配置隊伍)
        # Find 【進入】 button.
        # ⚠️ 真机踩坑：抽屉里有正文 `和阿黛海特一起進入夢中的世界` 含「進入」，
        # 旧写法 `locate_text("進入")` 会命中**正文行**（rx .742 / ry .335），
        # 位置校验 ry>0.75 失败 -> 跳过本分支 -> 反复点节点 -> 抽屉反复打开，**永远进不去**。
        # 现在：① 只在"按钮该在的区域"里做模糊匹配（文字+位置校验）
        #       ② 若 OCR 没读到，用**形状兜底**：右下角那条紫色/橙色宽按钮（实测质心 (2357,1338) 就是【進入】）
        target_list = list(getattr(state, "targets", None) or [])
        BTN_REGION = (0.55, 0.80, 1.00, 1.00)
        enter_btn = None
        if target_list:
            enter_btn = self.ctx.vision.pick_text(target_list, "進入", min_score=0.4, region=BTN_REGION)
            if enter_btn is None:
                enter_btn = self.ctx.vision.pick_text(target_list, "进入", min_score=0.4, region=BTN_REGION)
        enter_pos = None
        if enter_btn is not None and enter_btn.center_rel[0] > 0.65 and enter_btn.center_rel[1] > 0.75:
            enter_pos = (enter_btn.center_rel[0], enter_btn.center_rel[1], f"OCR『{enter_btn.text}』")
        else:
            from ..core.features import find_button as _find_button
            btn = _find_button(raw_img, region=(0.55, 0.84, 1.00, 0.99),
                               colors=("purple", "orange"), min_pixels=4000)
            if btn is not None:
                enter_pos = (btn['cx'] / w, btn['cy'] / h, f"形状/右下宽按钮({btn['color']})")
        if enter_pos is not None:
            rx, ry, how = enter_pos
            if any(w2 in text_corpus for w2 in ["配置", "隊伍", "队伍", "无法配置", "無法配置"]):
                logger.info(f"【配置隊伍】预设配队就绪，点击右下角【進入】(坐标: {rx:.3f}, {ry:.3f}｜{how})...")
                self.ctx.input.click_area_rel(rx - 0.02, ry - 0.02, rx + 0.02, ry + 0.02,
                                              delay_after=2.0, label="【配置隊伍】進入")
            else:
                logger.info(f"【关卡详情】关卡就绪（抽屉节点），点击右下角【進入】(坐标: {rx:.3f}, {ry:.3f}｜{how})...")
                self.ctx.input.click_area_rel(rx - 0.02, ry - 0.02, rx + 0.02, ry + 0.02,
                                              delay_after=2.0, label="【关卡详情】進入")
            return True

        # Check if an already-cleared stage drawer was opened by mistake
        is_cleared_drawer = any(w in text_corpus for w in ["重新觀看", "重新观看", "重新挑戰", "重新挑战", "再次觀看", "再次观看"])
        if is_cleared_drawer:
            logger.warning("【关卡抽屉】检测到已通关关卡详情（显示【重新觀看/重新挑戰】而非【進入】），判定为误触！")
            logger.info("【关卡抽屉】点击右上角 [X] 或左侧暗色背景退回主线大地图...")
            self.ctx.input.click_area_rel(0.935, 0.045, 0.965, 0.085, delay_after=0.6)
            self.ctx.input.click_rel(0.25, 0.50, delay_after=1.0)
            self.map_swipe_count = getattr(self, "map_swipe_count", 0) + 1
            return True

        # Find active stage button
        # 用不去重的完整 targets（elements 按 text 去重，重名关卡名会丢）
        ocr_targets = list(getattr(state, "targets", None) or [])
        if not ocr_targets:
            ocr_targets = list(state.elements.values()) if state.elements else self.ctx.vision.locate_all_texts(raw_img)
        active_node = StageMapDetector.find_active_stage_button(raw_img, ocr_targets=ocr_targets)
        if active_node:
            self.map_swipe_count = 0
            cx, cy, (rx, ry) = active_node
            logger.info(f"【外层选关】锁定可推进新关卡 (坐标: {rx:.3f}, {ry:.3f})，点击展开详情...")
            self.ctx.input.click_client(cx, cy, delay_after=1.5, jitter=8, label="【外层选关】可推进关卡")
            return True

        # Check next chapter entry
        next_chap = self._find_next_chapter_entry(raw_img, ocr_targets=ocr_targets)
        if next_chap:
            cx, cy, (rx, ry), desc = next_chap
            logger.info(f"【章节切换】当前章节已通关！锁定下一章节入口 ({desc}, 坐标: {rx:.3f}, {ry:.3f})，点击进入下一章...")
            self.ctx.input.click_client(cx, cy, delay_after=3.0, jitter=6, label="【章节切换】下一章节入口")
            self.map_swipe_count = 0
            return True

        # Swipe to reveal subsequent stages
        self.map_swipe_count = getattr(self, "map_swipe_count", 0) + 1
        if self.map_swipe_count <= 4:
            logger.info(f"【外层选关】当前视图未见未通关高亮关卡(均为已通关✓)，向左长按拖动地图展现右侧后续关卡/章节 (第 {self.map_swipe_count} 次)...")
            self.ctx.input.drag_rel(0.80, 0.50, 0.25, 0.50, steps=25, duration=0.45)
            time.sleep(1.0)
            return True
        else:
            logger.info("【外层选关】多次向右滚动仍未发现新关卡，向右拖回起始位置重新定位...")
            self.ctx.input.drag_rel(0.25, 0.50, 0.80, 0.50, steps=20, duration=0.35)
            time.sleep(1.0)
            self.map_swipe_count = 0
            return True

    def handle_settlement(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """Handle victory and reward settlement screens."""
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
            if raw_img is None:
                return False

        h, w = raw_img.shape[:2]
        text_corpus = " ".join(state.raw_texts) if state else ""

        # 0. 优先检测右下角橙色【ENTER 獲得】按钮（肉鸽战后战利品/结算页最常见）
        crop_orange = raw_img[int(0.85 * h):int(0.96 * h), int(0.75 * w):int(0.98 * w)]
        hsv_o = cv2.cvtColor(crop_orange, cv2.COLOR_BGR2HSV) if crop_orange.size > 0 else None
        has_orange_loot = (np.count_nonzero(cv2.inRange(hsv_o, np.array([8, 120, 120]), np.array([25, 255, 255]))) >= 2000) if hsv_o is not None else False
        has_loot_kw = any(w in text_corpus for w in ["獲得", "获得", "未顯現的靈光一閃", "未显现的灵光一闪", "信用點數", "信用点数", "戰鬥結束", "战斗结束"])

        if has_orange_loot or (has_loot_kw and ("ENTER" in text_corpus or "F1" in text_corpus)):
            logger.info("【关卡结算】检测到战后战利品【ENTER 獲得】画面 -> 键盘优先按 ENTER 极速领取并推进...")
            self.ctx.input.tap_action("confirm", delay_after=0.6, label="【战后结算】ENTER獲得")
            self.ctx.input.click_area_rel(0.84, 0.88, 0.94, 0.95, delay_after=0.8, label="【战后结算】橙色獲得")
            return True

        purple_btn = self._find_settlement_confirm_button(raw_img)
        if purple_btn:
            cx, cy, (rx, ry) = purple_btn
            logger.info(f"【关卡结算】锁定紫色【確認】按钮 (坐标: {rx:.3f}, {ry:.3f})，"
                        f"键盘优先按 ENTER（官方：確認 = ENTER）...")
            if not self.ctx.input.tap_action("confirm", delay_after=0.8, label="【关卡结算】確認(ENTER)"):
                self.ctx.input.click_client(cx, cy, delay_after=0.8, label="【关卡结算】紫色確認")
            return True

        if self.battle_mgr.check_battle_ended(state, raw_img=raw_img):
            return True

        # 与"点進入"同类的坑：正文里也可能出现「確認/完成/關閉」，
        # 所以必须 **区域限定**（结算按钮在右下 y>0.60）+ 只在已 OCR 结果里挑，不重新 OCR。
        target_list = list(getattr(state, "targets", None) or [])
        confirm_ocr = None
        if target_list:
            confirm_ocr = self.ctx.vision.pick_texts(
                target_list, ("確認", "确认", "確定", "完成", "關閉", "关闭"),
                min_score=0.4, region=(0.55, 0.85, 1.00, 1.00))
        if confirm_ocr and confirm_ocr.center_rel[1] > 0.60:
            logger.info(f"【关卡结算】OCR 锁定确认/完成按钮 [{confirm_ocr.text}]，点击退回外层...")
            self.ctx.input.click_target(confirm_ocr, delay_after=0.8, label="【关卡结算】确认按钮")
            return True

        logger.info("【关卡结算】点击结算确认区域 (rx: 0.83~0.93, ry: 0.88~0.94)...")
        self.ctx.input.click_area_rel(0.83, 0.88, 0.93, 0.94, delay_after=0.8)
        return True

    def handle_dream_mode(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        Handle Dream Mode (夢境模式) triggered by characters with Deep Trauma (深層創傷).
        主人 2026/10/11 规则：
        "如果战局失败，再次进入刷0号系统的时候出现中显示深层创伤，有治疗中心的字，就先去治疗，
         治疗记得先选角色，然后点消除记忆，因为战局失败的话，其实我们不保存，因为肯定卡片数量不够，不要了"
        - 界面：中央显示「夢境模式」4张卡牌，左下角为浅绿色药丸按钮【< 治療中心】(rx: 0.088, ry: 0.917)
        - 动作：点击左下角【治療中心】进入治疗室，绝不点击右下角【確認】(进入梦境模式)！
        """
        logger.info("【梦境模式】检测到战斗员处于「深层创伤」，已显示梦境模式 -> 点击左下角【治療中心】进行治疗...")
        # 优先通过 OCR 定位「治療中心」按钮
        targets = getattr(state, "targets", None) or []
        for t in targets:
            txt = getattr(t, "text", "") or ""
            if any(w in txt for w in ["治療中心", "治疗中心", "治療", "治疗"]):
                rx, ry = getattr(t, "center_rel", (0.088, 0.917))
                if ry >= 0.80 and rx <= 0.30:
                    logger.info(f"【梦境模式】OCR 定位到【治療中心】按钮 (坐标: {rx:.3f}, {ry:.3f})，点击进入...")
                    self.ctx.input.click_rel(rx, ry, delay_after=1.5, label="【梦境模式】点击治療中心")
                    return True

        # 视觉/固定坐标兜底：左下角浅绿色胶囊框 (0.088, 0.917)
        self.ctx.input.click_area_rel(0.05, 0.89, 0.16, 0.94, delay_after=1.5, label="【梦境模式】点击治療中心")
        return True

    def handle_treatment_center(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        Handle Treatment Center (治療中心) memory wipe routine.
        主人 2026/10/11 规则：
        "先选角色，然后点消除记忆，因为战局失败的话，其实我们不保存，因为肯定卡片数量不够，不要了"
        - 步骤：
          1. 若出现确认弹窗（"確定" / "確認" / "是否消除記憶"），按 ENTER 确认。
          2. 使用紫光创伤徽章检测当前界面上处于深层创伤的角色（区域: ry 0.44~0.53, rx 0.25~0.85）。
          3. 若检测到受创角色徽章：
             - 点击该徽章上方的角色舱体选中角色 (rx: badge_rx + 0.08, ry: 0.360)。
             - 等待 0.4s。
             - 点击底部【消除記憶】按钮 (rx: 0.550, ry: 0.925)。
             - 等待 0.5s。
             - 按 ENTER (或点击確認) 完成二次确认。
          4. 若未检测到任何受创角色徽章（所有角色记忆均已消除完毕）：
             - 点击左上角【<】返回按钮 (0.040, 0.045) 或按 ESC 退出治疗中心，返回配队/外层界面！
        """
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
            if raw_img is None:
                return False

        h, w = raw_img.shape[:2]
        text_corpus = " ".join(state.raw_texts) if state else ""

        # 1. 优先检测与处理「消除記憶」二次确认弹窗（真机截图 media_1791714616227_d1e35cca.png）
        # 弹窗特征：文案含「損壞資料」「消滅」「已消滅的損壞資料將無法復原」「即將消滅的損壞資料」「以解除深層創傷」或
        # 弹窗包含「取消」与 橙色「[⦿] 消除記憶 30,000」(rx: 0.66, ry: 0.70)。
        targets = getattr(state, "targets", None) or []
        is_confirm_modal = any(k in text_corpus for k in [
            "損壞資料", "损坏资料", "即將消滅", "即将消灭", "無法復原", "无法复原",
            "消除戰鬥員的記憶", "消除战斗员的记忆", "解除深層創傷", "解除深层创伤",
            "是否消除", "確定消除", "确认消除", "消滅", "消灭"
        ]) or (
            "取消" in text_corpus and any("消除" in getattr(t, "text", "") and 0.65 <= getattr(t, "center_rel", (0, 0))[1] <= 0.76 for t in targets)
        )
        if is_confirm_modal:
            logger.info("【治疗中心】检测到「消除記憶」二次确认弹窗 -> 正在点击右侧橙色【消除記憶】确认按钮...")
            # 优先通过 OCR 定位弹窗中右侧的「消除記憶」或「30,000」按钮 (rx: 0.50~0.85, ry: 0.64~0.76)
            modal_btn_target = next((
                t for t in targets
                if (any(w in (getattr(t, "text", "") or "") for w in ["消除記憶", "消除记忆", "30,000", "30000", "000"])
                    and 0.50 <= getattr(t, "center_rel", (0, 0))[0] <= 0.85
                    and 0.64 <= getattr(t, "center_rel", (0, 0))[1] <= 0.76)
            ), None)
            if modal_btn_target:
                mx, my = modal_btn_target.center_rel
                logger.info(f"【治疗中心】OCR 定位到弹窗橙色确认按钮 (坐标: {mx:.3f}, {my:.3f})，点击确认消除...")
                self.ctx.input.click_rel(mx, my, delay_after=0.8, label="【治疗中心】点击弹窗消除記憶")
            else:
                logger.info("【治疗中心】点击弹窗右侧橙色确认按钮区域 (坐标: 0.660, 0.701)...")
                self.ctx.input.click_rel(0.660, 0.701, delay_after=0.8, label="【治疗中心】点击弹窗消除記憶")

            # 辅以 ENTER 键盘按键（官方快捷键）
            self.ctx.input.tap_action("confirm", delay_after=1.2, label="【治疗中心】ENTER确认消除")
            return True

        # 2. 检测紫色深层创伤徽章（HSV: 125~165, S>=80, V>=80，区域: y 0.44~0.53, x 0.25~0.85）
        crop = raw_img[int(0.44 * h):int(0.53 * h), int(0.25 * w):int(0.85 * w)]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([125, 80, 80]), np.array([165, 255, 255]))
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        badges = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 100:
                cx = int(0.25 * w + centroids[i][0])
                cy = int(0.44 * h + centroids[i][1])
                badges.append((cx / w, cy / h, area))

        # 按从左到右排序
        badges.sort(key=lambda b: b[0])

        if badges:
            bx, by, area = badges[0]
            pod_rx = min(0.85, max(0.20, bx + 0.08))
            pod_ry = 0.360
            logger.info(f"【治疗中心】检测到 {len(badges)} 位待治疗角色，选中第 1 位受创角色 (坐标: {pod_rx:.3f}, {pod_ry:.3f})...")
            self.ctx.input.click_rel(pod_rx, pod_ry, delay_after=0.4, label="【治疗中心】选中受创角色")
            time.sleep(0.3)

            logger.info("【治疗中心】点击底部【消除記憶】按钮 (坐标: 0.550, 0.925)...")
            # 优先通过 OCR 定位「消除記憶」
            clicked_btn = False
            for t in targets:
                txt = getattr(t, "text", "") or ""
                if "消除記憶" in txt or "消除记忆" in txt or "消除" in txt:
                    tx, ty = getattr(t, "center_rel", (0.550, 0.925))
                    if ty >= 0.85:
                        self.ctx.input.click_rel(tx, ty, delay_after=0.5, label="【治疗中心】点击消除記憶")
                        clicked_btn = True
                        break
            if not clicked_btn:
                self.ctx.input.click_area_rel(0.48, 0.89, 0.65, 0.96, delay_after=0.5, label="【治疗中心】点击消除記憶")

            time.sleep(0.4)
            # 弹窗弹出后立即辅助点击二次确认橙色按钮 (0.660, 0.701) 并发送 ENTER
            logger.info("【治疗中心】点击二次确认弹窗橙色按钮 (0.660, 0.701) 并发送 ENTER...")
            self.ctx.input.click_rel(0.660, 0.701, delay_after=0.4, label="【治疗中心】点击弹窗消除記憶")
            self.ctx.input.tap_action("confirm", delay_after=1.0, label="【治疗中心】ENTER确认")
            return True

        # 3. 如果没有检测到任何受创角色徽章：
        # 先确认画面是否确实仍处于治疗中心界面（有标题「治療中心」/「治療」或底部「消除記憶」/「諮商」）
        clean_corpus = " ".join(
            t for t in (text_corpus or "").split(" ")
            if "路由" not in t and not (t.startswith("【") and ("分" in t or "%" in t or "％" in t))
        )
        is_in_treatment = (
            any(w in clean_corpus for w in ["治療中心", "治疗中心", "治療", "治疗"])
            or any(w in clean_corpus for w in ["消除記憶", "消除记忆", "諮商", "咨询"])
        )
        if not is_in_treatment:
            logger.info("【治疗中心】当前画面已不处于治疗中心界面，跳过退出动作。")
            return False

        logger.info("【治疗中心】未检测到受创角色，所有战斗员记忆已消除完成 -> 点击左上角【<】退出治疗中心...")
        self.ctx.input.click_area_rel(0.02, 0.02, 0.06, 0.07, delay_after=1.2, label="【治疗中心】退出返回")
        return True

    def handle_cutscene_dialog(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """Handle cutscenes, story dialogue choices, and skip buttons."""
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
            if raw_img is None:
                return False

        icon_count, has_skip, has_mid, has_menu, is_mid_active = self.inspect_top_right_icons(raw_img)

        # 1. Permanent Fast-Forward (⏩) —— 键盘优先：CTRL = 切換倍速（官方快捷键）
        #    注意 CTRL 是**切换型**，所以只在"判定为未点亮"时才按（否则会把 2X 关掉）
        if has_mid and not is_mid_active:
            logger.info("【剧情加速】右上角 ⏩ 未点亮 -> 按 CTRL 切换倍速（官方快捷键：切換倍速）...")
            self.ctx.input.tap_action("toggle_speed", delay_after=0.35, label="【常驻加速】2X(CTRL键)")

        # 2. Skip Button —— 键盘优先：Z = 跳過（官方快捷键）
        if has_skip:
            logger.info(f"【剧情跳过】检测到右上角【▷| 跳过】(图标数: {icon_count}) -> 按 Z 跳过...")
            self.ctx.input.tap_action("skip_cutscene", delay_after=0.5, label="【剧情】跳过(Z键)")
            time.sleep(0.3)
            sub_img = self.ctx.capture.capture()
            if sub_img is not None:
                sub_targets = self.ctx.vision.locate_all_texts(sub_img)
                skip_confirm = next((t for t in sub_targets 
                                     if any(k in t.text for k in ["跳过", "跳過", "确定", "確定", "确认"]) 
                                     and 0.30 < t.center_rel[1] < 0.80), None)
                if skip_confirm:
                    logger.info(f"【主线剧情】点击跳过确认弹窗 [{skip_confirm.text}]...")
                    self.ctx.input.click_target(skip_confirm, delay_after=1.0, label="【弹窗】确认跳过")
                    return True
                else:
                    # 弹窗确认按钮找不到时，回车通常就是"确定"（官方：確認 = ENTER）
                    self.ctx.input.tap_action("confirm", delay_after=0.5, label="【弹窗】确认跳过(ENTER)")
            return True

        # 3. Skip Modal popup check
        skip_modal_words = ["跳过", "跳過", "确定", "確定", "确认", "是否跳過", "是否跳过"]
        modal_btn = next((t for t in state.elements.values() 
                          if any(w in t.text for w in skip_modal_words) and 0.30 < t.center_rel[1] < 0.80), None)
        if modal_btn:
            logger.info(f"【跳过弹窗】锁定确认跳过按钮 [{modal_btn.text}]，点击确认...")
            self.ctx.input.click_target(modal_btn, delay_after=1.0)
            return True

        # 4. Check Dialogue Choices (用户明确要求：出现选项时，随便选一个推进)
        ocr_targets = list(state.elements.values()) if state.elements else self.ctx.vision.locate_all_texts(raw_img)
        story_choices = self._find_story_dialogue_choices(raw_img, ocr_targets=ocr_targets)
        if story_choices:
            chosen = story_choices[0]
            logger.info(f"【剧情选项】检测到居中对白选项 [{chosen.text}]，按指令任选一项推进剧情...")
            self.ctx.input.click_target(chosen, delay_after=1.2)
            return True

        # 5. Advance Dialogue —— 键盘优先：SPACE = 對話（官方快捷键）
        logger.info("【主线剧情】推进剧情台词与演出（按 SPACE）...")
        self.ctx.input.tap_action("advance_dialogue", delay_after=0.5, label="【主线剧情】推进台词(SPACE)")
        return True

    def handle_battle(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """Handle battle execution (Boss detection and round play)."""
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
            if raw_img is None:
                return False

        text_corpus = " ".join(state.raw_texts)
        auto_status = self.battle_mgr.inspect_auto_battle_status(raw_img)
        is_locked = (auto_status == 'LOCKED')
        is_boss = (is_locked or any(w in text_corpus for w in ["BOSS", "首领", "首領", "領主", "首脑"]))

        # 0. Check Epiphany selection during combat
        if state.screen == ScreenState.EPIPHANY_SELECTION or self.battle_mgr.is_epiphany_screen(raw_img, state):
            return self.battle_mgr.handle_epiphany_selection(raw_img, state)

        # 0b. Check if In-Combat Card Discover/Generation Modal is visible
        if self.battle_mgr.is_card_discover_screen(raw_img, state):
            return self.battle_mgr.handle_card_discover_selection(raw_img, state)

        # 0c. Check if In-Combat Attack ENTER Button (⚔ 攻擊 / ENTER) is visible
        # 主人 2026/10/11 规则与真机特征：
        # "有时候进入战斗了已经是激活的R的状态，但是应该要点一下屏幕的，上面有一个攻击按钮，是enter按键，看看把这个适配一下"
        if self.battle_mgr.is_attack_enter_screen(raw_img, state):
            return self.battle_mgr.handle_attack_enter(raw_img, state)

        # BOSS 战策略（`main_story.boss_action`）：
        #   pause —— 不接管，交主人手动（老行为，默认）
        #   auto  —— **接管出牌**（主人 2026/10/6 要求："boss战我希望还是搞一下自动，因为现在我们有打牌了"）
        #            BOSS 战游戏会**锁掉原生 AUTO**，所以只能用我们自己的编号键出牌逻辑。
        if is_boss and self.boss_action == "pause":
            logger.warning("=" * 55)
            logger.warning("【BOSS战】检测到当前为首领战 (AUTO带锁/首领特征)！")
            logger.warning("【BOSS战】脚本不接管打牌与技能，全权交由主人手动操作！"
                           "（想让它自动打就把 config.json 的 boss_action 改成 \"auto\"）")
            logger.warning("=" * 55)
            time.sleep(3.0)
            return True

        # 普通战斗：如果原生 AUTO 已经开启，直接观战！
        if auto_status == 'ACTIVE' and not is_boss:
            logger.info("【战斗托管】原生【自动战斗】运行中 (金色高亮)，脚本处于观战状态...")
            time.sleep(1.5)
            return True

        # 普通战斗：若原生 AUTO 未开启，依次尝试 点击 / 按 R 激活，绝不手动出牌
        if auto_status == 'INACTIVE' and not is_boss:
            logger.info("【战斗状态】检测到右上角 AUTO 图标未激活，正在激活原生【自动战斗】...")
            # 1. 优先点击右上角 AUTO 按钮 (rx: 0.879, ry: 0.052)
            self.ctx.input.click_rel(0.879, 0.052, delay_after=0.6, label="【战斗】激活AUTO(点击)")
            recheck_img = self.ctx.capture.capture()
            if recheck_img is not None and self.battle_mgr.inspect_auto_battle_status(recheck_img) == 'ACTIVE':
                logger.info("【战斗托管】成功激活原生【自动战斗】！由游戏原生托管出牌，脚本进入观战状态...")
                time.sleep(1.5)
                return True

            # 2. 若点击后仍未高亮，尝试快捷键 R
            self.ctx.input.tap_action("auto_battle", delay_after=0.6, label="【战斗】激活AUTO(R键)")
            recheck_img = self.ctx.capture.capture()
            if recheck_img is not None and self.battle_mgr.inspect_auto_battle_status(recheck_img) == 'ACTIVE':
                logger.info("【战斗托管】成功激活原生【自动战斗】(R键)！由游戏原生托管出牌，脚本进入观战状态...")
                time.sleep(1.5)
                return True

            # 3. 游戏刚入场可能有出牌/开场动画导致点击暂未生效，耐心等待并进入下一轮重试，绝不能擅自出牌！
            logger.info("【战斗托管】原生【自动战斗】正在启动中 (等待开场动画就绪)...")
            time.sleep(1.0)
            return True

        # 首领战 (BOSS / LOCKED)：仅在确认是 BOSS 战且配置允许时接管出牌
        if is_boss:
            if self.boss_action == "auto":
                logger.info("【BOSS战】首领战 -> 按 boss_action=auto **接管出牌**"
                            "（BOSS 战原生 AUTO 被锁，用编号键出牌 + E 结束回合）...")
                return self.battle_mgr.play_round(state, raw_img=raw_img)
            else:
                logger.warning("【BOSS战】检测到当前为首领战且 boss_action!=auto，脚本等待玩家手动操作...")
                time.sleep(2.0)
                return True

        # 兜底：非 boss 战一律不手动出牌，保持观战重试
        logger.info("【战斗托管】等待战斗自动推进中...")
        time.sleep(1.2)
        return True

    def _check_search_bubble(self, raw_img: np.ndarray) -> bool:
        """Check interactive in-scene search bubble [🔍]."""
        tpl_bubble = self._get_search_bubble_template()
        if tpl_bubble is None or raw_img is None:
            return False

        h, w = raw_img.shape[:2]
        scale = h / 576.0
        best_score = 0.0
        best_loc = None
        for s in [0.85, 0.95, 1.0, 1.05, 1.15, 1.25]:
            tw = int(tpl_bubble.shape[1] * scale * s)
            th = int(tpl_bubble.shape[0] * scale * s)
            if tw <= 0 or th <= 0 or tw >= w or th >= h:
                continue
            ts = cv2.resize(tpl_bubble, (tw, th))
            res = cv2.matchTemplate(raw_img, ts, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv > best_score:
                best_score = mv
                best_loc = (ml[0] + tw // 2, ml[1] + th // 2)

        if best_score >= 0.70 and best_loc:
            rx, ry = best_loc[0] / w, best_loc[1] / h
            if 0.10 < ry < 0.90:
                logger.info(f"【局内探索】锁定场景内搜索交互图标 [🔍] (置信度={best_score:.2f}, 坐标: {rx:.3f}, {ry:.3f})，点击执行调查...")
                if getattr(self.ctx, "overlay", None):
                    bw = int(36 * (h / 576.0))
                    self.ctx.overlay.show_boxes([{
                        'bbox': (best_loc[0] - bw, best_loc[1] - bw, bw * 2, bw * 2),
                        'label': f"【调查】搜索图标 [🔍] ({best_score:.2f})",
                        'click': best_loc,
                        'color': '#00FFFF'
                    }], duration=1.2)
                self.ctx.input.click_client(best_loc[0], best_loc[1], delay_after=1.5, jitter=6)
                return True
        return False

    def step(self, state: GameState) -> bool:
        """Process one loop of main story automation via dynamic scene router."""
        # 优先复用本帧已经抓好的图（orchestrator 每帧只截一次屏），
        # 这样 router 与 observer 看的是**同一帧**，也省掉一次全屏截取。
        frame = getattr(state, "frame", None)
        if frame is not None and getattr(frame, "valid", False):
            return self.router.dispatch_frame(frame, state)
        raw_img = self.ctx.capture.capture()
        if raw_img is None:
            return False
        return self.router.dispatch(raw_img, state)

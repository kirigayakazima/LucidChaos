"""
Phase 11: BattleManager
Automates combat execution, card playing, and victory/defeat detection.
Fully aligned with user combat rules:
1. Native AUTO Battle status check:
   - If present & ACTIVE (golden glowing ring): DO NOT take over, let game native auto fight!
   - If present & INACTIVE: tap once (0.880, 0.055) to attempt activation.
   - If LOCKED (padlock icon / special scene): switch to our automated card-play logic.
2. Automated card-play logic (for locked / manual combat):
   - Left hero skills rotation (1 -> 2 -> 3) upon glowing golden border
   - Hand cards upward drag (drag from rx, 0.84 -> rx, 0.42) to battle arena
   - Priority: 0-cost (exhausted first) > Attack [攻擊] > Skill [技能] > Enhance [強化] (max 1/turn)
   - Spend AP to 0 then click Turn End checkmark [✓] (rx: 0.89, ry: 0.86)
3. Post-battle Loot Screen (战后战利品/结算):
   - If no choices required, automatically click orange [獲得] button (rx: 0.890, ry: 0.915)
   - If choice cards exist, select the optimal card then click [獲得]
"""

import os
import re
import cv2
import time
import logging
import numpy as np
from typing import Optional, Tuple
from ..fsm.engine import FSMContext
from ..state.screen_state import GameState, ScreenState

logger = logging.getLogger("BattleManager")

def load_image_unicode(file_path: str) -> Optional[np.ndarray]:
    """Load image from path supporting Chinese / unicode characters."""
    if not os.path.exists(file_path):
        return None
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Failed to load image from {file_path}: {e}")
        return None

class BattleManager:
    _tpl_lock_cache: Optional[np.ndarray] = None

    # 手牌上限（游戏里 1~10 张；第 10 张对应键盘 `0`）
    MAX_HAND = 10
    # 一轮最多打出几张（手牌数最多 10）
    MAX_PLAY_PER_TURN = 10
    # 读不到技能费用时的保守门槛：技能一般是 4~5 点自我意識
    SKILL_MIN_EP = 4

    # -------------------------------------------------------------
    # 自我意識點數（EP）
    # -------------------------------------------------------------
    @staticmethod
    def parse_ep(targets) -> Optional[int]:
        """
        从 OCR 结果里读**当前自我意識點數（EP）**。

        真机证据（主人 2026/10/6 截图）：左侧竖排显示 `EP 3` / `EP 7`。
        ⚠️ 为什么要读它：EP 不够时按编号键，游戏会弹出
        `自我意識點數不足。` 的选择界面（ENTER 使用卡牌 / ESC 取消選擇）-> 脚本卡死。
        所以**出牌前先看 EP**，买不起就不按。
        """
        for t in (targets or []):
            text = (getattr(t, "text", "") or "").upper().replace(" ", "")
            if "EP" not in text:
                continue
            m = re.search(r'EP[^0-9]{0,3}(\d{1,2})', text)
            if m:
                try:
                    return int(m.group(1))
                except Exception:  # noqa: BLE001
                    continue
            # 形如 `EP3`（已经去掉空格）或独立数字块紧邻 EP 的情况
            m2 = re.search(r'EP(\d{1,2})$', text)
            if m2:
                return int(m2.group(1))
        return None

    @classmethod
    def parse_skill_costs(cls, targets) -> dict:
        """
        读左侧 F1/F2/F3 的费用（截图里每个 F 键旁边有一个数字，例如 4/4/5）。
        读不到就返回空 dict，调用方用 SKILL_MIN_EP 当保守门槛。
        """
        costs = {}
        for t in (targets or []):
            rx, ry = getattr(t, "center_rel", (None, None))
            text = (getattr(t, "text", "") or "").strip()
            if rx is None or rx > 0.10 or not (0.60 <= ry <= 0.96):
                continue
            m = re.fullmatch(r'(\d)', text)
            if m:
                # 左侧竖排从上到下：F1/F2/F3 -> 用 y 排序映射
                costs.setdefault("list", []).append((ry, int(m.group(1))))
        if costs.get("list"):
            ordered = [c for _, c in sorted(costs["list"])]
            return {i + 1: c for i, c in enumerate(ordered[:3])}
        return {}

    def detect_zero_cost_cards(self, img_bgr: np.ndarray, card_positions: list) -> list:
        """
        识别"**0 费**"的手牌，返回它们的编号列表（1-based）。

        背景（主人 2026/10/6 规则）："如果有0费的牌，优先使用，这个很重要"。
        但卡牌费用数字是**装饰性字体**：OCR 读不到（实测连放大 4 倍后单独 OCR 也读不到），
        所以只能用**模板匹配**认 0 费徽章（卡牌左上角那个白色圆角方块里的数字）。

        该模板需要从真机截图里裁准 → 放在 `templates/battle/cost_0.png`。
        **模板不存在时直接返回 []**（不影响其它逻辑：0 费牌就不享受"优先"待遇，
        其余按 攻击 > 技能 + EP 预算 出牌）。
        """
        from pathlib import Path as _P
        tpl_path = _P(__file__).resolve().parent.parent / "templates" / "battle" / "cost_0.png"
        if not tpl_path.exists():
            if not getattr(self, "_zero_tpl_warned", False):
                logger.info("【自动出牌】0 费徽章模板尚未标定（templates/battle/cost_0.png 不存在）-> "
                            "本轮不做 0 费优先，仅按 攻击/技能 + EP 预算 出牌")
                self._zero_tpl_warned = True
            return []
        try:
            tpl = cv2.imdecode(np.fromfile(str(tpl_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            return []
        if tpl is None or img_bgr is None or img_bgr.size == 0:
            return []
        h, w = img_bgr.shape[:2]
        # 只查手牌带上半部（费用徽章所在），多尺度匹配
        x1, x2 = int(0.15 * w), int(0.85 * w)
        y1, y2 = int(0.70 * h), int(0.86 * h)
        band = img_bgr[y1:y2, x1:x2]
        hits = []
        for s in (0.85, 0.95, 1.0, 1.05, 1.15):
            ts = cv2.resize(tpl, (0, 0), fx=s, fy=s)
            if band.shape[0] < ts.shape[0] or band.shape[1] < ts.shape[1]:
                continue
            res = cv2.matchTemplate(band, ts, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv >= 0.72:
                hits.append((x1 + ml[0] + ts.shape[1] // 2, mv))
        out = []
        for cx, _score in hits:
            nearest = min(card_positions, key=lambda c: abs(c["rx"] - cx / w))
            if nearest["index"] not in out:
                out.append(nearest["index"])
        return out

    def __init__(self, ctx: FSMContext):
        self.ctx = ctx
        self.round_count = 0
        self.hero_rot_idx = 0
        self._load_lock_template()

    @classmethod
    def _load_lock_template(cls):
        if cls._tpl_lock_cache is not None:
            return
        # 统一走全项目共用的模板库（原来这里是唯一一个「每次调用都读盘」的加载器）
        from ..core.template_bank import BANK
        cls._tpl_lock_cache = BANK.get("auto_locked.png")

    def inspect_auto_battle_status(self, raw_img: np.ndarray) -> str:
        """
        判断右上角【自动战斗(AUTO)】按钮的状态（区域 rx 0.855~0.905, ry 0.025~0.085）。

        ⚠️ 2026/10/6 真机修正（主人："不对啊，没自动战斗啊，没高亮"）：
          对比 5 张真机截图的图标后发现：
            - 圆圈 + **锁🔒** + 回转箭头 = **被锁**（BOSS 战，游戏禁止 AUTO）
            - 圆圈 + **交叉双剑⚔** + `R`  = **可用但未开启**（应去按 R 开启）
            - 开启时是同一个⚔图标但**高亮**（金色光圈）
          旧代码判据是 `金色像素>=40 **或** S均值>=38` —— 第二个条件太松，
          实测"未开启"那张图 S 均值 156.8，于是被误判成 ACTIVE -> 脚本"观战"什么都不做 ✗
          （表现就是主人看到的"没自动战斗"）。现在：
            **只认"强金色高亮"**（金色像素数门槛大幅提高），不再看整体饱和度。
        """
        if raw_img is None or raw_img.size == 0:
            return 'UNKNOWN'

        h, w = raw_img.shape[:2]
        crop = raw_img[int(0.025 * h):int(0.085 * h), int(0.855 * w):int(0.905 * w)]
        if crop.size == 0:
            return 'UNKNOWN'

        # 1. 被锁？(auto_locked.png 模板)
        self._load_lock_template()
        if self._tpl_lock_cache is not None:
            scale = h / 576.0
            ts_lock = cv2.resize(self._tpl_lock_cache, (0, 0), fx=scale, fy=scale)
            if crop.shape[0] >= ts_lock.shape[0] and crop.shape[1] >= ts_lock.shape[1]:
                res_lock = cv2.matchTemplate(crop, ts_lock, cv2.TM_CCOEFF_NORMED)
                _, mv_lock, _, _ = cv2.minMaxLoc(res_lock)
                if mv_lock >= 0.70:
                    return 'LOCKED'

        # 2. 已开启？只认**强金色高亮**
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_gold = cv2.inRange(hsv, np.array([12, 60, 120]), np.array([32, 255, 255]))
        gold_count = int(np.count_nonzero(mask_gold))
        # 实测真机：未开启 = 0~25；真正激活时高亮金圈 68~200+。
        # 阈值取 40，安全稳定区分。
        if gold_count >= 40:
            return 'ACTIVE'

        # 3. 其余情况（⚔ 白圈 + R）= 可用但未开启 -> 可以按 R 打开
        return 'INACTIVE'

    @classmethod
    def is_epiphany_hero_screen(cls, raw_img: Optional[np.ndarray] = None, state: Optional[GameState] = None) -> bool:
        """Check if current screen is Epiphany Combatant Selection (選擇欲獲得【靈光一閃】的戰鬥員)."""
        text_corpus = " ".join(state.raw_texts) if state else ""
        if any(k in text_corpus for k in ["零號系統", "零号系统", "零号系統", "選擇法典", "选择法典"]):
            return False
        has_epiphany = any(k in text_corpus for k in ["靈光一閃", "灵光一闪", "靈光", "灵光", "一閃", "一闪", "【靈光一】", "靈光一"])
        has_hero = any(k in text_corpus for k in ["戰鬥員", "战斗员", "戰鬥", "战斗", "隊員", "队员", "欲獲得", "欲获得", "請選擇", "请选择", "撐欲瘦得", "撑欲瘦得", "門眞", "门真", "的門眞", "的门真", "的眞", "的真"])
        return has_epiphany and has_hero

    @classmethod
    def is_epiphany_screen(cls, raw_img: Optional[np.ndarray] = None, state: Optional[GameState] = None) -> bool:
        """Check if current screen is the Epiphany effect card selection screen."""
        if cls.is_epiphany_hero_screen(raw_img, state):
            return True
        text_corpus = " ".join(state.raw_texts) if state else ""
        if any(k in text_corpus for k in ["零號系統", "零号系统", "零号系統", "選擇法典", "选择法典", "撐法典", "撑法典"]):
            return False
        if ((any(k in text_corpus for k in ["靈光", "灵光", "一閃", "一闪"]) and any(k in text_corpus for k in ["選擇", "选择", "選", "选", "效果", "查看内容", "查看內容", "CAPS"])) or
            (("查看內容" in text_corpus or "查看内容" in text_corpus) and ("長按卡牌" in text_corpus or "长按卡牌" in text_corpus or "CAPS" in text_corpus)) or
            any(k in text_corpus for k in ["選擇您的命運", "选择您的命运", "您的命運", "您的命运", "選擇命運", "选择命运", "欲適用的命運", "欲适用的命运"]) or
            (any(k in text_corpus for k in ["命運卡牌", "命运卡牌", "命運效果", "命运效果"]) and any(k in text_corpus for k in ["ENTER", "F1", "AP1", "確認", "确认"]))):
            return True
        return False

    def _epiphany_scene_same(self, img_bgr: np.ndarray) -> bool:
        """更新灵光一闪卡死计数器：如果同一画面未推进，stuck 计数自增。"""
        det = getattr(self, "_stuck_det", None)
        if det is None:
            from .action_runner import StuckDetector
            det = self._stuck_det = StuckDetector()
        same, stuck = det.update(img_bgr)
        self._epiphany_stuck = stuck
        return same

    def handle_epiphany_selection(self, raw_img: Optional[np.ndarray] = None, state: Optional[GameState] = None) -> bool:
        """
        Handle Epiphany (靈光一閃) combatant / card selection:
        主人 2026/10/11 核心规则与真机教训：
        - 默认优先选 1 号位（角色或卡牌）；
        - 若 1 号位点击后【確認】置灰/画面未推进，绝不能原地死点 5 次！
          必须智能轮换阶梯：1 号位不行立即切换 2 号位，依此类推到 3 号位！
        - 3 个槽位分布（水平排列）：
          1 号位 (左): rx 0.245 (区域 0.18~0.31, 0.35~0.65), 快捷键 1
          2 号位 (中): rx 0.500 (区域 0.43~0.57, 0.35~0.65), 快捷键 2
          3 号位 (右): rx 0.755 (区域 0.68~0.82, 0.35~0.65), 快捷键 3
        """
        if raw_img is None:
            raw_img = self.ctx.capture.capture()

        if raw_img is not None and getattr(raw_img, "size", 0) > 0:
            self._epiphany_scene_same(raw_img)
        stuck = getattr(self, "_epiphany_stuck", 0)

        hero_coords = [
            (1, (0.18, 0.35, 0.31, 0.65), 0.245, 0.500),
            (2, (0.43, 0.35, 0.57, 0.65), 0.500, 0.500),
            (3, (0.68, 0.35, 0.82, 0.65), 0.755, 0.500),
        ]

        slot_idx = stuck % len(hero_coords)
        slot_num, area, cx, cy = hero_coords[slot_idx]
        is_hero_screen = self.is_epiphany_hero_screen(raw_img, state)
        kind_label = "选人" if is_hero_screen else "选牌"

        # 如果是选牌界面且未发生卡死，优先检测并选择「神闪」强化卡牌
        if not is_hero_screen and stuck == 0 and raw_img is not None:
            from ..core.features import detect_god_epiphany_card
            god_slot, god_det = detect_god_epiphany_card(raw_img)
            if god_slot is not None:
                slot_num = god_slot
                slot_idx = god_slot - 1
                _, area, cx, cy = hero_coords[slot_idx]
                logger.info(
                    f"【灵光一闪·神闪】发现场景存在「神闪」专属强化卡牌！"
                    f"卡牌 #{god_slot} 带有金色专属包边 (得分={god_det['gold']:.0f}) -> "
                    f"最高优先级选定第 {god_slot} 张神闪卡牌 (键盘 {god_slot} 号键)..."
                )

        if stuck > 0:
            logger.warning(
                f"【灵光一闪{kind_label}】上一位目标未成功推进 (stuck={stuck}) -> "
                f"智能轮换切换至第 {slot_num} 号位 (坐标: {cx:.3f}, {cy:.3f})..."
            )
        else:
            logger.info(
                f"【灵光一闪{kind_label}】默认选择第 {slot_num} 号位 (坐标: {cx:.3f}, {cy:.3f})..."
            )

        # 1. 优先发送官方编号快捷键 (键 1/2/3) + 点击对应槽位
        self.ctx.input.tap_card(slot_num, delay_after=0.3, label=f"【灵光一闪】按键{slot_num}")
        self.ctx.input.click_area_rel(area[0], area[1], area[2], area[3], delay_after=0.5, label=f"【灵光一闪】第{slot_num}槽位")
        time.sleep(0.4)

        # 2. 点击右下角【確認】按钮 (ENTER / 0.890, 0.930)
        logger.info(f"【灵光一闪{kind_label}】已选中第 {slot_num} 号位，按 ENTER / 点击【確認】按钮完成选择...")
        if not self.ctx.input.tap_action("confirm", delay_after=1.2, label=f"【灵光一闪{kind_label}】确认(ENTER)"):
            self.ctx.input.click_area_rel(0.83, 0.88, 0.95, 0.96, delay_after=1.2, label=f"【灵光一闪{kind_label}】确认")
        self.ctx.input.click_rel(0.890, 0.930, delay_after=1.2, label=f"【灵光一闪{kind_label}】点击确认按钮")
        return True

    @classmethod
    def is_attack_enter_screen(cls, raw_img: Optional[np.ndarray] = None, state: Optional[GameState] = None) -> bool:
        """
        Check if in-combat '⚔ 攻擊 / [ENTER]' attack initiation button is visible.
        主人 2026/10/11 规则与真机特征：
        "有时候进入战斗了已经是激活的R的状态，但是应该要点一下屏幕的，上面有一个攻击按钮，是enter按键，看看把这个适配一下"
        - 画面手牌上方中央显示「⚔ 攻擊」胶囊框，底部带有「ENTER」按钮指示 (rx: 0.500, ry: 0.858)。
        - 处于该状态时，无论原生 AUTO(R) 是否激活，游戏均暂停等待玩家发起攻击！
        """
        targets = (getattr(state, "targets", None) or []) if state is not None else []
        for t in targets:
            txt = (getattr(t, "text", "") or "").strip().upper()
            rx, ry = getattr(t, "center_rel", (0, 0))
            if 0.42 <= rx <= 0.58 and 0.81 <= ry <= 0.90:
                if any(w in txt for w in ["ENTER", "ENTE", "NTER"]):
                    return True
            if 0.40 <= rx <= 0.60 and 0.77 <= ry <= 0.86:
                if any(w in getattr(t, "text", "") for w in ["攻擊", "攻击"]):
                    corpus = " ".join(getattr(x, "text", "") for x in targets)
                    if "ENTER" in corpus.upper():
                        return True

        if raw_img is not None and raw_img.size > 0:
            from ..core.template_bank import BANK
            tpl = BANK.get("battle_attack_enter.png")
            if tpl is not None:
                h, w = raw_img.shape[:2]
                band = raw_img[int(0.80 * h):int(0.92 * h), int(0.40 * w):int(0.60 * w)]
                if band.shape[0] >= tpl.shape[0] and band.shape[1] >= tpl.shape[1]:
                    scale = h / 576.0
                    for s in (0.90, 1.0, 1.10):
                        ts = cv2.resize(tpl, (0, 0), fx=scale * s, fy=scale * s)
                        if band.shape[0] >= ts.shape[0] and band.shape[1] >= ts.shape[1]:
                            res = cv2.matchTemplate(band, ts, cv2.TM_CCOEFF_NORMED)
                            _, mv, _, _ = cv2.minMaxLoc(res)
                            if mv >= 0.72:
                                return True

        return False

    def handle_attack_enter(self, raw_img: Optional[np.ndarray] = None, state: Optional[GameState] = None) -> bool:
        """
        Handle in-combat '⚔ 攻擊 / [ENTER]' attack initiation:
        1. Send ENTER key to initiate attack sequence (官方快捷键：ENTER 攻擊).
        2. Click center attack button (0.500, 0.825).
        """
        logger.info("【战斗攻击】检测到屏幕中央「⚔ 攻擊 / [ENTER]」攻击确认按钮 -> 发送 ENTER 并点击攻击按钮发起攻击...")
        self.ctx.input.tap_action("confirm", delay_after=0.8, label="【战斗】ENTER攻击")
        self.ctx.input.click_rel(0.500, 0.825, delay_after=1.0, label="【战斗】点击攻擊按钮")
        return True

    @classmethod
    def is_card_discover_screen(cls, raw_img: Optional[np.ndarray] = None, state: Optional[GameState] = None) -> bool:
        """Check if in-combat card discovery / generation selection modal is visible."""
        text_corpus = " ".join(state.raw_texts) if state else ""
        return (
            any(w in text_corpus for w in [
                "欲產生", "欲产生", "欲生成", "欲保留", "欲檢索", "欲检索",
                "欲丟棄", "欲丢弃", "欲消耗", "欲抽取", "欲領取", "欲领取",
                "產生的卡牌", "产生的卡牌", "生成的卡牌", "保留的卡牌", "檢索的卡牌", "检索的卡牌"
            ]) or (
                any(w in text_corpus for w in ["最多", "最少"]) and any(w in text_corpus for w in ["卡牌", "卡", "牌"])
            ) or (
                any(w in text_corpus for w in ["0/1", "0 / 1", "0/2", "0 / 2", "0/3", "0 / 3"])
                and any(w in text_corpus for w in ["ENTER", "選擇", "选择"])
                and not any(w in text_corpus for w in ["0/4", "補充", "凭证", "憑證"])
            )
        ) and any(w in text_corpus for w in ["ENTER", "選擇", "选择", "確認", "确认", "隱藏", "隐藏"])

    def handle_card_discover_selection(self, raw_img: Optional[np.ndarray] = None, state: Optional[GameState] = None) -> bool:
        """Handle in-combat card discovery / generation selection modal."""
        import re
        text_corpus = " ".join(state.raw_texts) if state else ""
        ocr_targets = getattr(state, "targets", None) or []
        logger.info("【战斗选牌】检测到局内卡牌产生/检索选择弹窗（请选择欲产生的卡牌）...")

        need_count = 1
        m_count = re.search(r'最多\s*(\d+)\s*[张張]', text_corpus) or re.search(r'0\s*/\s*(\d+)', text_corpus)
        if m_count:
            try:
                need_count = max(1, min(5, int(m_count.group(1))))
            except Exception:
                need_count = 1

        card_xs = []
        if ocr_targets:
            raw_xs = sorted([t.center_rel[0] for t in ocr_targets if 0.22 <= t.center_rel[1] <= 0.65 and 0.05 <= t.center_rel[0] <= 0.95])
            clusters = []
            for x in raw_xs:
                if not clusters or abs(x - clusters[-1][-1]) > 0.08:
                    clusters.append([x])
                else:
                    clusters[-1].append(x)
            card_xs = [float(np.mean(c)) for c in clusters if len(c) >= 1]

        default_5_xs = [0.120, 0.296, 0.475, 0.652, 0.829]
        target_xs = card_xs if len(card_xs) >= need_count else default_5_xs

        for idx in range(min(need_count, len(target_xs))):
            click_x = target_xs[idx]
            click_y = 0.440
            logger.info(f"【战斗选牌】点击选中第 {idx + 1} 张卡牌 (坐标: {click_x:.3f}, {click_y:.3f})...")
            self.ctx.input.click_rel(click_x, click_y, delay_after=0.3, label=f"【战斗选牌】第{idx+1}张卡牌")
            time.sleep(0.2)

        time.sleep(0.3)
        logger.info("【战斗选牌】卡牌已选择，按 ENTER 确认（官方快捷键：ENTER 選擇）...")
        if not self.ctx.input.tap_action("confirm", delay_after=1.2, label="【战斗选牌】ENTER确认"):
            self.ctx.input.click_area_rel(0.84, 0.89, 0.95, 0.96, delay_after=1.2, label="【战斗选牌】点击選擇按钮")
        self.ctx.input.click_rel(0.891, 0.931, delay_after=1.0, label="【战斗选牌】点击橙色選擇")
        return True

    def play_round(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        Execute one complete combat turn:
        1. Checks Auto Battle button:
           - If ACTIVE: DO NOT take over, let game native auto fight!
           - If INACTIVE: Click to activate.
           - If LOCKED: Execute manual intelligent card play.
        """
        self.round_count += 1
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
        if raw_img is None or raw_img.size == 0:
            return False

        h, w = raw_img.shape[:2]
        text_corpus = " ".join(state.raw_texts)

        # -------------------------------------------------------------
        # 0. Check if Epiphany (靈光一閃) is visible during combat
        # -------------------------------------------------------------
        if state.screen == ScreenState.EPIPHANY_SELECTION or self.is_epiphany_screen(raw_img, state):
            return self.handle_epiphany_selection(raw_img, state)

        # -------------------------------------------------------------
        # 0b. Check if In-Combat Card Discover/Generation Modal is visible
        # -------------------------------------------------------------
        if self.is_card_discover_screen(raw_img, state):
            return self.handle_card_discover_selection(raw_img, state)

        # -------------------------------------------------------------
        # 0c. Check if In-Combat Attack ENTER Button (⚔ 攻擊 / ENTER) is visible
        # -------------------------------------------------------------
        if self.is_attack_enter_screen(raw_img, state):
            return self.handle_attack_enter(raw_img, state)

        # -------------------------------------------------------------
        # 1. Check Native Auto Battle Status
        # -------------------------------------------------------------
        auto_status = self.inspect_auto_battle_status(raw_img)

        if auto_status == 'ACTIVE':
            logger.info("【战斗托管】原生【自动战斗】运行中 (金色高亮)，脚本处于观战状态...")
            time.sleep(1.5)
            return True

        if auto_status == 'INACTIVE':
            logger.info("【战斗状态】检测到原生【自动战斗】未激活，正在激活 AUTO...")
            # 1. 优先点击右上角 AUTO 图标 (rx: 0.879, ry: 0.052)
            self.ctx.input.click_rel(0.879, 0.052, delay_after=0.6, label="【战斗】激活AUTO(点击)")
            recheck_img = self.ctx.capture.capture()
            if recheck_img is not None and self.inspect_auto_battle_status(recheck_img) == 'ACTIVE':
                logger.info("【战斗托管】成功激活原生【自动战斗】！由游戏原生托管出牌，脚本处于观战状态...")
                time.sleep(1.5)
                return True

            # 2. 若点击未生效，尝试 R 快捷键
            self.ctx.input.tap_action("auto_battle", delay_after=0.6, label="【战斗】激活AUTO(R键)")
            recheck_img = self.ctx.capture.capture()
            if recheck_img is not None:
                new_status = self.inspect_auto_battle_status(recheck_img)
                if new_status == 'ACTIVE':
                    logger.info("【战斗托管】成功激活原生【自动战斗】(R键)！由游戏原生托管出牌，脚本处于观战状态...")
                    time.sleep(1.5)
                    return True
                elif new_status == 'LOCKED':
                    auto_status = 'LOCKED'

        # 普通战斗绝不手动出牌！只有被锁（LOCKED）才允许进入智能出牌
        if auto_status != 'LOCKED':
            logger.info("【战斗托管】原生【自动战斗】正在启动中 (等待开场动画就绪)...")
            time.sleep(1.0)
            return True

        logger.info("【特殊战斗】检测到【自动战斗】带锁 (特殊/Boss场景，禁止原生Auto)！转入全自动智能出牌逻辑...")
        logger.info(f"【智能出牌】执行出牌回合 (第 {self.round_count} 轮)...")

        # -------------------------------------------------------------
        # 1.5 先读「自我意識點數（EP）」——决定这一轮能出什么
        # 主人 2026/10/6：EP 不够时按卡/技能会弹出「自我意識點數不足」并卡死，
        # 所以这里先算清预算：技能/卡牌只在**买得起**时才按。
        # -------------------------------------------------------------
        ep = self.parse_ep(getattr(state, "targets", None))
        skill_costs = self.parse_skill_costs(getattr(state, "targets", None))
        if ep is None:
            ep_budget = 99          # 读不到 EP 时不在技能上做限制，交给"试探 + ESC 自救"兜住
            logger.info("【智能出牌】没读到 EP（自我意識點數）-> 技能不做预算限制，"
                        "改由『试探出牌 + 点数不足时 ESC 收手』兜底")
        else:
            ep_budget = ep
            logger.info(f"【智能出牌】当前 EP（自我意識點數）= {ep}；技能费用读取: {skill_costs or '未读到(用保守门槛)'}")

        # 出牌预算：**卡牌用的是"行动值"（画面中间那个大数字），不是左侧 EP**。
        # 主人 2026/10/6 说明：左侧 EP 随时间变化(最大 7)供技能用；中间数字(如 3)是卡牌行动值。
        # ⚠️ 这两个大数字都是装饰字体，OCR 读不出来 -> 卡牌预算走"**试探**"：
        #    逐张打，打到游戏提示「點數不足」就 ESC 收手，**能打的都打完**（修掉"牌没打完就退出"）。
        card_budget = 99
        card_budget_known = False

        # -------------------------------------------------------------
        # 2. Left Character Skills Rotation (1 -> 2 -> 3)——只在 EP 够时按
        # -------------------------------------------------------------
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

        if lit_heroes:
            for offset in range(3):
                cand_slot = (self.hero_rot_idx + offset) % 3 + 1
                match = next((m for m in lit_heroes if m[0] == cand_slot), None)
                if match:
                    slot_num, cx_rel, cy_rel = match
                    need = skill_costs.get(slot_num, self.SKILL_MIN_EP)
                    if ep_budget < need:
                        logger.info(f"【战斗技能】{slot_num} 号位技能需要 {need} 点，剩余 EP={ep_budget} -> "
                                    f"**跳过不按**（避免弹出「自我意識點數不足」卡死）")
                        continue
                    logger.info(f"【战斗技能】触发 {slot_num} 号位角色技能（键盘优先：F{slot_num}，"
                                f"需 {need} 点，剩余 EP={ep_budget}，按 1->2->3 轮转）...")
                    if not self.ctx.input.tap_hero_skill(slot_num, delay_after=0.45):
                        self.ctx.input.click_rel(cx_rel, cy_rel, delay_after=0.4)
                    ep_budget = max(0, ep_budget - need)
                    self.hero_rot_idx = slot_num % 3
                    break

        # -------------------------------------------------------------
        # 3. Hand Card Layout & Identification
        # -------------------------------------------------------------
        crop_hand = raw_img[int(0.70 * h):int(0.96 * h), int(0.18 * w):int(0.82 * w)]
        hand_ocr = self.ctx.vision.locate_all_texts(crop_hand, min_score=0.4)

        # 手牌数量：底部中间那个 `NN/10` 指示器（实测看到 `04/10`、`09/10`）。
        # 老正则只认 `0[3-9]/10`，遇到 `04/10`、`10/10` 会漏 -> 手牌数退化成默认 5、按错编号。
        # 现在用更宽松的 `(\d{1,2})\s*/\s*(\d{1,2})`，取前一个数当手牌数（上限 10）。
        hand_count = None
        for t in hand_ocr:
            m = re.search(r'(\d{1,2})\s*/\s*(\d{1,2})', t.text or "")
            if m:
                first, second = int(m.group(1)), int(m.group(2))
                if 1 <= first <= 10 and second <= 12:
                    hand_count = first
                    break
        hand_known = hand_count is not None
        if not hand_known:
            # 实测这个 `NN/10` 指示器 OCR 很不稳（同一帧被读成 `0/60`）。
            # 安全兜底：**按满 10 个编号** —— 战斗界面里数字键只有「使用N號卡牌」一个含义，
            # 空位/费用不够的按了都是空操作，所以不会出错，只是慢一点。
            hand_count = self.MAX_HAND
        logger.info(f"【自动出牌】手牌数 = {hand_count}"
                    f"{'（OCR 读到）' if hand_known else '（读不到指示器 -> 按满 10 个编号兜底，空位无副作用）'}"
                    f"，每个编号按 {getattr(self.ctx.input, 'DIGIT_PRESS_TIMES', 2)} 下")

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

        for t in hand_ocr:
            text = t.text
            cx_abs = (int(0.18 * w) + t.center[0]) / w
            closest = min(card_positions, key=lambda c: abs(c['rx'] - cx_abs))

            # 费用：**0 费牌优先**是主人明确规则（"如果有0费的牌，优先使用，这个很重要"）。
            # ⚠️ 卡牌费用数字是装饰字体，OCR（含放大后单独 OCR）实测**读不出来**，
            #    所以这里只有"**0 费**"能可靠识别 —— 见 detect_zero_cost_cards()（模板匹配）。
            #    其余费用按默认 1 点估算，靠 EP 预算兜住"买不起就别按"。
            if any(k in text for k in ['攻擊', '攻击', '子彈', '子弹', '破碎', '閃光', '闪光']):
                closest['type'] = 'ATTACK'
                closest['priority'] = 100
                closest['title'] = text
            elif any(k in text for k in ['强化', '強化', '回聲', '回声', '成對', '成对']):
                closest['type'] = 'ENHANCE'          # 按主人规则：强化牌不参与出牌
                closest['priority'] = 10
                closest['title'] = text
            elif any(k in text for k in ['技能', '命運', '命运', '鬥志', '斗志', '永凍', '永冻', '凝聚']):
                closest['type'] = 'SKILL'
                closest['priority'] = 60
                closest['title'] = text

        # 0 费牌识别（模板匹配）：命中就把它排到**最前面**并把费用记 0
        zero_idx = self.detect_zero_cost_cards(raw_img, card_positions)
        for i in zero_idx:
            c = next((x for x in card_positions if x['index'] == i), None)
            if c is not None:
                c['cost'] = 0
                c['priority'] = 200
                logger.info(f"【自动出牌】#{i} 识别为 **0 费** -> 排到最前面优先使用")

        sorted_cards = sorted(card_positions, key=lambda c: c['priority'], reverse=True)
        play_queue = []
        for c in sorted_cards:
            # 主人 2026/10/6 的出牌规则：
            #   ① **0 费牌优先**（最重要）② 然后 1 费攻击/技能 ③ 没有 1 费再找 2 费
            #   ④ **不打强化牌**（"为了简单我们不打强化牌"）
            if c['type'] == 'ENHANCE':
                logger.info(f"【自动出牌】#{c['index']} 是强化牌 -> 按规则**跳过**（不打强化）")
                continue
            # 类型优先级：攻击 > 技能（0 费的在前面已经给了最高优先级）
            if c['type'] not in ('ATTACK', 'SKILL'):
                continue
            # 行动值预算：读不到（装饰字体 OCR 认不出）时按 99 放行，靠下面的"试探 + ESC 收手"兜住
            need = c['cost'] if isinstance(c.get('cost'), int) else 1
            if need > card_budget:
                logger.info(f"【自动出牌】#{c['index']} 需要 {need} 点行动值，剩余 {card_budget} -> 跳过")
                continue
            play_queue.append(c)
            card_budget -= max(0, need)
            if len(play_queue) >= self.MAX_PLAY_PER_TURN:      # 最多 10 张（游戏手牌上限）
                break
        if play_queue:
            logger.info("【自动出牌】本轮出牌顺序: " + " -> ".join(
                f"#{c['index']}({c['type']},{c['cost']}费)" for c in play_queue))
        logger.info(f"【自动出牌】本轮计划打出 {len(play_queue)} 张")

        # -------------------------------------------------------------
        # 4. 按编号出牌（键盘优先）+ **试探**：打到点数不足就 ESC 收手
        # 主人 2026/10/6 确认键盘更准：每张手牌上方圆圈标着位次，官方快捷键「使用N號卡牌 = 1~9、0(=10號)」，
        # 且**要按两下**（选中+确认）。
        # ⚠️ 卡牌的行动值/EP 都是装饰字体、OCR 读不出来 -> 逐张试探：
        #    打完一张就复查画面，若出现「點數不足」就按 ESC 取消并结束本轮出牌。
        #    这样"能打的牌一定都打完"（修掉之前"牌没打完就退出"的 bug）。
        # -------------------------------------------------------------
        played = 0
        for c in play_queue:
            c_desc = f"#{c['index']} [{c['type']}:{c['title'] or '手牌'}] (费用={c['cost']})"
            logger.info(f"【自动出牌】按编号打出 {c_desc}（键盘 {c['index']} 号键，按两下）...")
            self.ctx.input.tap_card(c['index'], delay_after=(0.45 if hand_known else 0.3),
                                    label=f"【战斗】出牌 {c_desc} (键 {c['index']})")
            played += 1
            time.sleep(0.3)

            # 试探复查：点数不足 -> ESC 收手
            # 性能：只 OCR "错误提示所在的那一小块"（画面上方居中 y 0.14~0.32），
            # 全屏 OCR 要 1.4s/张，小区域 ~0.2s，10 张牌省下十几秒。
            if not card_budget_known:
                probe = self.ctx.capture.capture() if getattr(self.ctx, "capture", None) else None
                if probe is not None and getattr(probe, "size", 0):
                    try:
                        ph, pw = probe.shape[:2]
                        region = probe[int(0.12 * ph):int(0.34 * ph), int(0.22 * pw):int(0.78 * pw)]
                        pt = self.ctx.vision.locate_all_texts(region, min_score=0.4)
                        pc = " ".join(getattr(t, "text", "") or "" for t in pt)
                    except Exception:  # noqa: BLE001
                        pc = ""
                    if any(w in pc for w in ("點數不足", "点数不足")):
                        logger.warning(f"【自动出牌】第 {played} 张出不起（游戏提示『點數不足』）-> "
                                       f"按 ESC 取消，本轮不再出牌（已出 {played - 1} 张）")
                        self.ctx.input.tap_action("cancel", delay_after=0.8, label="【战斗】取消選擇(ESC)")
                        break

        # -------------------------------------------------------------
        # 5. 结束回合（键盘 E）
        # -------------------------------------------------------------
        logger.info(f"【回合结束】本轮已出 {played} 张 -> 按 E 键结束回合（官方快捷键：結束回合）...")
        self.ctx.input.tap_action("end_turn", delay_after=1.5, label="【战斗】结束回合(E键)")
        return True

    def check_battle_ended(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        Check if combat has concluded (Victory settlement, loot acquisition, or clear).
        If post-battle loot screen has no choices, automatically click orange [獲得] button.
        """
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
        if raw_img is None or raw_img.size == 0:
            return False

        h, w = raw_img.shape[:2]
        text_corpus = " ".join(state.raw_texts)

        # 0. Check if Epiphany (靈光一閃) is visible in post-battle settlement
        if state.screen == ScreenState.EPIPHANY_SELECTION or self.is_epiphany_screen(raw_img, state):
            logger.info("【结算灵光一闪】战后结算触发靈光一閃！按规则直接选择第 1 张牌 (坐标: 0.245, 0.500)...")
            self.handle_epiphany_selection(raw_img, state)
            return True

        # 1. Check for bright orange [獲得] button at bottom-right (rx: 0.78~0.98, ry: 0.86~0.96)
        crop_btn = raw_img[int(0.85 * h):int(0.96 * h), int(0.75 * w):int(0.98 * w)]
        hsv = cv2.cvtColor(crop_btn, cv2.COLOR_BGR2HSV)
        mask_orange = cv2.inRange(hsv, np.array([8, 120, 120]), np.array([25, 255, 255]))
        has_orange_btn = np.count_nonzero(mask_orange) >= 2000

        is_loot_screen = has_orange_btn or any(w in text_corpus for w in ["戰鬥結束", "战斗结束", "信用點", "信用点", "獲得", "获得"])

        if is_loot_screen:
            logger.info("【战斗结算】检测到战后战利品/结算画面...")

            # Check if there are cards/relics choices to pick in center (ry: 0.35~0.75, rx: 0.20~0.80)
            choice_words = ["选择", "選擇", "三选一", "替換", "替换"]
            has_choices = any(w in text_corpus for w in choice_words)

            if has_choices:
                logger.info("【战斗结算】检测到待选卡牌/遗物，默认选择第 1 项...")
                self.ctx.input.click_area_rel(0.26, 0.40, 0.40, 0.60, delay_after=0.8, label="【结算】选择第1项")

            # "如果没有选择的内容，那么就点获取"
            logger.info("【战斗结算】按 ENTER / 点击右下角橙色【獲得】按钮领取所有奖励...")
            self.ctx.input.tap_action("confirm", delay_after=0.6, label="【结算】ENTER獲得")
            self.ctx.input.click_area_rel(0.84, 0.88, 0.94, 0.95, delay_after=0.8, label="【结算】獲得奖励")
            self.round_count = 0
            return True

        # 2. Check standard victory settlement screen
        if state.screen == ScreenState.VICTORY_SETTLEMENT or any(w in text_corpus for w in ["獲得分數", "获得分数", "胜利", "VICTORY"]):
            logger.info("【战斗结算】检测到关卡胜利结算，点击确认退回外层...")
            confirm_btn = (state.get_element("確認") or state.get_element("确认") or 
                           state.get_element("確定") or state.get_element("完成"))
            if confirm_btn:
                self.ctx.input.click_target(confirm_btn, delay_after=1.8)
            else:
                self.ctx.input.click_area_rel(0.83, 0.88, 0.93, 0.94, delay_after=2.0, label="【结算】确认完成")
            self.round_count = 0
            return True

        return False

"""
Phase 10: NodeDecisionEngine
Implements business handling for Event, Shop, and Camp/Safe Area.
Fully aligned with user rules:
1. Camp/Bonfire:
   - Must choose Free Rest [休息 (免費)] at (rx: 0.334, ry: 0.735)
   - Click [確認] confirmation button at (rx: 0.660, ry: 0.670)
   - Click orange [▷ 離開] exit button at (rx: 0.885, ry: 0.915) to leave
2. In-room Exit:
   - Any combat, camp, or event screen showing the orange arrow button [▷ 離開 / 獲得] is automatically clicked to leave.
"""

import time
import logging
import cv2
import re
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from ..fsm.engine import FSMContext
from ..state.screen_state import GameState, ScreenState
from ..router.scene_types import SceneType
from .target_manager import TargetManager
from .action_runner import ActionLadder, Attempt, StuckDetector

logger = logging.getLogger("NodeEngine")

class NodeDecisionEngine:
    _tpl_recommend_badge_cache = None

    # 探索/交互的「升级阶梯」：先换目标，再换手段。
    # 目的：任何一步只要画面真的变了就回到第 1 步；若整条阶梯都点不动，
    # 也不会像以前那样同一坐标空点到 Watchdog STOP。
    #
    # ⚠️ 2026/10/6 主人定的规则（有游戏截图证据）：
    #   - **探索交互点 = F1/F2/F3**（场景里的放大镜图标旁边就标着 `F1`；"有几个就是 F 几"）
    #   - **事件选项框 = 1/2/3**（框上有编号）
    #   - **确认/下一步 = ENTER**
    # 这几条同时解决了"世界空间交互点不动"的老大难：**根本不用点它，按键就行**。
    _EVT_LADDER = (
        ("按 F1 触发 1 号交互点（键盘优先）", "key", "F1"),
        ("按 F2 触发 2 号交互点", "key", "F2"),
        ("按 F3 触发 3 号交互点", "key", "F3"),
        ("改用鼠标点击第 1 个搜索气泡 [🔍]（回退）", "bubble", 0),
        ("改用 SPACE 推进对白/演出", "key", "SPACE"),
        ("点击气泡正下方地面发光物", "ground", 0),
        ("按 Z 跳过剧情（官方快捷键）", "key", "Z"),
    )
    _EVT_KEY_LABELS = {"F1": "【探索】交互点 1 (F1)",
                       "F2": "【探索】交互点 2 (F2)",
                       "F3": "【探索】交互点 3 (F3)",
                       "SPACE": "【剧情】推进台词 (SPACE)",
                       "Z": "【剧情】跳过 (Z)"}

    # 64x36 灰度小图当画面指纹；实测同一幕的雾/呼吸动画漂移约 1.2~1.7，故阈值取 4.0
    _EVT_SIG_W = 64
    _EVT_SIG_H = 36
    _EVT_SAME_SCENE_TOL = 4.0

    # 【仅局内搜索图标专用】点击参数（其它所有点击仍走 input.py 的默认值，行为不变）
    # 不动真实光标、不抢前台，只发窗口消息。
    _BUBBLE_CLICK_PRESS_MS = 80       # 每次按下 -> 抬起的时长（≈人点 50~120ms）
    # ⚠️ 真机观察：主人发现"把真实鼠标移到问号上，那几下点击才生效；不移过去没反应"。
    # 说明场景里的问号是**世界空间**交互（可能靠 hover 状态/射线拾取），比界面按钮更挑。
    # 因此这里把前置时间从 120ms 提到 300ms，并让 input.click_client 走「人手式靠近」
    # （先发一个偏离几十像素的中间点，再发目标点，最后按下）——**依然不移动真实光标**。
    # 若这样仍不生效，下一步需要主人决定是否允许"临时移动真实光标再复位"。
    _BUBBLE_CLICK_MOVE_GAP_MS = 300   # 移动 -> 第一次按下（含两段 move + 更多帧时间）
    _BUBBLE_CLICK_COUNT = 3           # 在同一个点上快速连点几次
    _BUBBLE_CLICK_GAP_MS = 110        # 连点之间的间隔（≈人手双击节奏）

    def __init__(self, ctx: FSMContext, target_mgr: TargetManager, config: Dict[str, Any]):
        self.ctx = ctx
        self.target_mgr = target_mgr
        self.config = config
        # 「是否还停在同一幕」的检测状态（见 _event_scene_same）
        self._stuck_det = StuckDetector(tol=self._EVT_SAME_SCENE_TOL)
        self._evt_sig: Optional[np.ndarray] = None
        self._evt_stuck: int = 0

    @classmethod
    def _get_recommend_badge_template(cls) -> Optional[np.ndarray]:
        from ..core.template_bank import BANK
        return BANK.get("rogue/recommend_badge.png")

    @classmethod
    def _get_close_x_template(cls) -> Optional[np.ndarray]:
        from ..core.template_bank import BANK
        return BANK.get("btn_close_x.png")

    @classmethod
    def _get_search_bubble_template(cls) -> Optional[np.ndarray]:
        from ..core.template_bank import BANK
        return BANK.get("explore_search_bubble.png")

    def detect_search_bubbles(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Detect interactive exploration search cards/bubbles [🔍] in scene."""
        from ..core.template_bank import BANK
        tpl = self._get_search_bubble_template()
        if tpl is None or img_bgr is None or img_bgr.size == 0:
            return []

        ih, iw = img_bgr.shape[:2]
        scale = ih / 576.0
        th, tw = tpl.shape[:2]

        # 性能：5 档全图 matchTemplate 实测 **3083ms**（每档约 600ms）。
        # 先对**同样的 5 档**在 1/4 缩略图上各粗筛一次（合计约 85ms，比全分辨率便宜 ~35 倍）：
        # 连粗筛都够不到 0.55（真实阈值 0.70）时，说明这一帧根本没有搜索气泡，直接返回。
        # ⚠️ 必须把 5 档都探一遍：只探 0.85 那档时，真实气泡落在 1.15 档的夹具会被漏掉
        #   （回归用例 narration_with_markers 当场抓到过这个问题）。
        coarse_max = 0.0
        try:
            small = cv2.resize(img_bgr, (max(1, iw // 4), max(1, ih // 4)), interpolation=cv2.INTER_AREA)
            for s in [0.85, 0.95, 1.0, 1.05, 1.15]:
                cw0 = max(1, int(tw * scale * s) // 4)
                ch0 = max(1, int(th * scale * s) // 4)
                if small.shape[0] < ch0 or small.shape[1] < cw0:
                    continue
                ts0 = cv2.resize(tpl, (cw0, ch0))
                res0 = cv2.matchTemplate(small, ts0, cv2.TM_CCOEFF_NORMED)
                _, mv0, _, _ = cv2.minMaxLoc(res0)
                coarse_max = max(coarse_max, float(mv0))
        except Exception:  # noqa: BLE001
            coarse_max = 1.0          # 快筛本身出错就别省
        if coarse_max < 0.55:
            return []

        bubbles = []
        for s in [0.85, 0.95, 1.0, 1.05, 1.15]:
            cw = int(tw * scale * s)
            ch = int(th * scale * s)
            if cw <= 0 or ch <= 0 or cw >= iw or ch >= ih:
                continue
            ts = BANK.scaled_to("explore_search_bubble.png", cw, ch)   # 同一份缩放缓存
            if ts is None:
                continue
            res = cv2.matchTemplate(img_bgr, ts, cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= 0.70)
            for y, x in zip(*locs):
                cx = x + cw // 2
                cy = y + ch // 2
                bubbles.append({
                    'cx': cx,
                    'cy': cy,
                    'rx': cx / iw,
                    'ry': cy / ih,
                    'score': float(res[y, x]),
                    'bbox': (x, y, cw, ch)
                })

        clusters = []
        for b in sorted(bubbles, key=lambda it: -it['score']):
            if not any(abs(b['cx'] - c['cx']) < 30 * scale and abs(b['cy'] - c['cy']) < 30 * scale for c in clusters):
                clusters.append(b)

        clusters.sort(key=lambda it: it['cx'])
        return clusters

    def reset_stuck_state(self) -> None:
        """清空"卡在同一幕"的计数与阶梯状态（全局卡死守卫升级时调用，让下一轮重新评估）。"""
        try:
            self._stuck_det.reset()
        except Exception:  # noqa: BLE001
            pass
        self._evt_sig = None
        self._evt_stuck = 0

    def _event_scene_sig(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """画面指纹：缩到 64x36 灰度，用来判断「是不是还停在同一幕」。"""
        return self._stuck_det.signature(img_bgr)

    def _event_scene_same(self, img_bgr: np.ndarray) -> bool:
        """
        更新「卡在同一幕」计数：画面内容变了 -> 归零；还是同一幕 -> +1。
        返回值仅表示“和上一轮是同一幕”。

        实现在 `logic/action_runner.StuckDetector`（全项目共用），
        这里只把结果镜像回 `_evt_sig` / `_evt_stuck`（其它代码与测试仍在读这两个属性）。
        """
        det = getattr(self, "_stuck_det", None)
        if det is None:          # 兼容用 object.__new__ 构造的诊断脚本
            det = self._stuck_det = StuckDetector(tol=self._EVT_SAME_SCENE_TOL)
        same, stuck = det.update(img_bgr)
        self._evt_sig = self._stuck_det.sig
        self._evt_stuck = stuck
        return same

    def is_equip_screen(self, raw_img: Optional[np.ndarray] = None, state: Optional[GameState] = None) -> bool:
        """Check if current screen is the equipment acquisition / assignment screen."""
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
        if raw_img is None or raw_img.size == 0:
            return False

        h, w = raw_img.shape[:2]
        crop_red = raw_img[int(0.88 * h):int(0.97 * h), int(0.55 * w):int(0.75 * w)]
        hsv_red = cv2.cvtColor(crop_red, cv2.COLOR_BGR2HSV) if crop_red.size > 0 else None
        m1 = cv2.inRange(hsv_red, np.array([0, 100, 100]), np.array([10, 255, 255])) if hsv_red is not None else None
        m2 = cv2.inRange(hsv_red, np.array([170, 100, 100]), np.array([180, 255, 255])) if hsv_red is not None else None
        has_red = (np.count_nonzero(m1 | m2) >= 1500) if (m1 is not None and m2 is not None) else False

        crop_orange = raw_img[int(0.88 * h):int(0.97 * h), int(0.78 * w):int(0.98 * w)]
        hsv_orange = cv2.cvtColor(crop_orange, cv2.COLOR_BGR2HSV) if crop_orange.size > 0 else None
        has_orange = (np.count_nonzero(cv2.inRange(hsv_orange, np.array([8, 120, 120]), np.array([25, 255, 255]))) >= 2000) if hsv_orange is not None else False

        text_corpus = " ".join(state.raw_texts) if state else ""
        # Veto: 局内事件选项界面（含选项气泡且非装备穿戴）绝非装备界面
        if any(w in text_corpus for w in ["詢問", "询问", "競技場", "竞技场", "【競技場", "【竞技场"]) or (
            any(w in text_corpus for w in ["獲得獎勵", "获得奖励"]) and any(w in text_corpus for w in ["資訊", "资讯", "事件"])
        ):
            return False

        has_equip_kw = any(w in text_corpus for w in [
            "選擇戰鬥員", "选择战斗员", "請選擇戰鬥員", "请选择战斗员", "提煉", "提炼",
            "重鑄裝備", "重装备", "重装備", "神之錘", "神之锤", "神之鈺", "神之槌", "欲套用"
        ]) or (any(w in text_corpus for w in ["裝備", "装备", "装備"]) and has_red)
        return (has_red and has_orange) or has_equip_kw

    def handle_equip_selection(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        Handle Equipment Choice (二选一装备挑选) & Equipment Assignment (分配给战斗员).
        主人 2026/10/11 规则与真机适配：
        1. 阶段一：装备二选一（提示「請選擇裝備。」/ 画面中间上下两件装备）
           - 点击第 1 件装备（坐标: 0.460, 0.350）
           - 短暂等待后按 ENTER 或点击右下角【確認】(0.890, 0.925) 进入角色分配
        2. 阶段二：角色分配（提示「請選擇戰鬥員。」/ 右侧三名战斗员）
           - 核心原则：如果某些角色已有该类装备，必须分配给另外没有该装备的角色！
           - 通过装备槽位发光高亮（outline / glow）定位目标装备槽位（槽位1=武器, 槽位2=防具, 槽位3=饰品）
           - 检查 3 名战斗员在该发光槽位是否已装备（未穿戴槽位背景暗，已穿戴槽位高亮且有星级/装备图）
           - 优先分配给该栏位为空的战斗员！若均为空或均满，则优先参考系统推荐，无推荐选第 1 人
           - 选中目标战斗员后，按 ENTER 或点击右下角橙色【確認】(0.890, 0.925) 完成装备分配！
        """
        logger.info("【获得装备】正在处理装备分配逻辑...")
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
        if raw_img is None or raw_img.size == 0:
            return False

        h, w = raw_img.shape[:2]
        text_corpus = " ".join(state.raw_texts) if state else ""

        # -------------------------------------------------------------
        # 分支 1：卡牌奖励战斗员分配（「請選擇欲領取卡牌的戰鬥員」）
        # 特征：出现「欲領取卡牌」/「卡牌獎勵」/「欲領取」/「卡牌樊」等
        # 主人规则：遇到中立卡片奖励时，若某些角色显示「無法獲得」，必须智能过滤并选择可用角色！
        # -------------------------------------------------------------
        is_card_reward_assign = any(k in text_corpus for k in [
            "欲領取", "欲领取", "欲取卡牌", "撑欲取", "撐欲取", "領取卡牌", "领取卡牌",
            "卡牌獎勵", "卡牌奖励", "卡牌樊", "卡牌奖"
        ]) and not any(k in text_corpus for k in ["請選擇裝備", "请选择装备", "選擇裝備", "选择装备", "请摆装備", "請擺裝備"])

        if is_card_reward_assign:
            logger.info("【卡牌领奖分配】检测到卡牌奖励分配界面（請選擇欲領取卡牌的戰鬥員）...")
            skip_neutral = self.config.get("card_reward", {}).get("skip_neutral_cards", True)
            stuck_cnt = getattr(self, "_evt_stuck", 0)

            targets = self._ocr_targets(raw_img, state)

            # 主人 2026/10/11 核心规则："凡是中立卡的我们跳过，不要保存到角色里面，这个很重要，我们不需要中立卡"
            # 判据：只有当确实为中立卡牌（含有「中立」字样，或某些角色出现「無法獲得」等职业限制）时才跳过；
            # 普通卡牌奖励（角色专属或全员可用）必须选择战斗员并确认领取！
            is_neutral_card = (
                any(k in text_corpus for k in ["中立卡牌", "中立卡", "中立"]) or
                any(any(k in getattr(t, "text", "") for k in ["無法獲得", "无法获得", "無法得", "无法得", "無法瘦得", "无法瘦得"])
                    for t in targets if t.center_rel[0] >= 0.35)
            )

            # 官方快捷键：[F1] 跳過 (0.700, 0.930) / [ENTER] 確認 (0.890, 0.925)
            if skip_neutral and is_neutral_card and stuck_cnt < 3:
                logger.info("【中立卡跳过】检测到中立卡牌分配，按主人规则跳过不存入战斗员（官方快捷键: [F1] 跳過 / 点击 0.700, 0.930）...")
                import win32con
                self.ctx.input.send_key(win32con.VK_F1, delay_after=1.0, label="【中立卡跳过】跳过(F1)")
                self.ctx.input.click_rel(0.700, 0.930, delay_after=1.5, label="【中立卡跳过】点击跳过")
                return True

            row_bounds = [(0.20, 0.40), (0.41, 0.61), (0.62, 0.82)]
            hero_click_ys = [0.290, 0.510, 0.730]

            available_heroes = []
            recommended_heroes = []

            for idx, (y1, y2) in enumerate(row_bounds):
                hero_num = idx + 1
                row_tgts = [t for t in targets if t.center_rel[0] >= 0.35 and y1 <= t.center_rel[1] <= y2]
                row_texts = [getattr(t, "text", "") for t in row_tgts]

                # 检测是否被禁用（中立卡职业限制：如「無法獲得獵人」、「無法獲得心靈術士」）
                is_disabled = any(
                    any(k in txt for k in ["無法獲得", "无法获得", "無法", "无法", "瘦得", "無法得", "无法得"])
                    for txt in row_texts
                )

                if is_disabled:
                    logger.info(f"【卡牌领奖分配】战斗员 {hero_num} 无法获取此卡牌（包含无法获得标签: {row_texts}），跳过...")
                else:
                    available_heroes.append(hero_num)
                    if any(any(k in txt for k in ["推薦", "推荐"]) for txt in row_texts):
                        recommended_heroes.append(hero_num)

            if not available_heroes:
                logger.warning("【卡牌领奖分配】未能识别出可用战斗员，兜底尝试第 1 位战斗员...")
                available_heroes = [1]

            ladder_idx = ActionLadder.index_for(getattr(self, "_evt_stuck", 0), len(available_heroes))
            if recommended_heroes and ladder_idx == 0:
                chosen_hero = recommended_heroes[0]
                logger.info(f"【卡牌领奖分配】采纳系统推荐战斗员 {chosen_hero}...")
            else:
                chosen_hero = available_heroes[ladder_idx]
                logger.info(f"【卡牌领奖分配】选择可用战斗员 {chosen_hero} (可用列表: {available_heroes}, 尝试序号: {ladder_idx + 1})...")

            target_ry = hero_click_ys[chosen_hero - 1]
            self.ctx.input.click_rel(0.600, target_ry, delay_after=0.6, label=f"【卡牌领奖】选择第{chosen_hero}战斗员")
            time.sleep(0.4)
            logger.info(f"【卡牌领奖分配】已选中第 {chosen_hero} 战斗员，按 ENTER 确认领取...")
            if not self.ctx.input.tap_action("confirm", delay_after=1.5, label="【卡牌领奖】确认(ENTER)"):
                self.ctx.input.click_area_rel(0.84, 0.88, 0.95, 0.95, delay_after=1.5, label="【卡牌领奖】确认")
            return True

        # -------------------------------------------------------------
        # 分支 2：二选一装备挑选界面（提示「請選擇裝備。」/ 画面中间上下两件装备）
        # 特征：文本出现「請選擇裝備」/「選擇裝備」/「请摆装備」，且无战斗员/装备槽特征（无「0/120pt」）
        # -------------------------------------------------------------
        is_equip_pick = any(k in text_corpus for k in ["請選擇裝備", "请选择装备", "選擇裝備", "选择装备", "请摆装備", "請擺裝備"]) \
                        and not any(k in text_corpus for k in ["0/120pt", "120pt", "長按穿戴中", "长按穿戴中", "請選擇戰鬥員", "选择战斗员"])

        if is_equip_pick:
            logger.info("【获得装备 -> 装备挑选】检测到二选一装备界面（請選擇裝備），点击选中第 1 件装备 (坐标: 0.460, 0.350)...")
            self.ctx.input.click_area_rel(0.35, 0.25, 0.58, 0.45, delay_after=0.4, label="【装备挑选】选中第1件装备")
            time.sleep(0.3)
            logger.info("【获得装备 -> 确认挑选】按 ENTER 确认挑选，进入战斗员分配界面...")
            if not self.ctx.input.tap_action("confirm", delay_after=1.2, label="【装备挑选】ENTER确认"):
                self.ctx.input.click_area_rel(0.84, 0.88, 0.95, 0.95, delay_after=1.2, label="【装备挑选】点击確認")
            return True

        # -------------------------------------------------------------
        # 分支 3：神之锤强化效果对比 / 二次确认界面（「決定」/「跳過」）
        # 特征：出现「神之」/「神之錘」/「神之锤」/「神之鈺」/「重鑄」等，且底部包含「決定」/「决定」按钮
        # 主人 2026/10/11 规则与真机特征：
        #   "确认后，会二次确认，依旧确认"
        # 动作：
        #   按 ENTER 确认（官方快捷键），辅以点击右侧橙色【決定】按钮 (0.637, 0.927)，完成最终强化确认！
        # -------------------------------------------------------------
        is_god_hammer_confirm = (
            any(k in text_corpus for k in ["神之錘", "神之锤", "神之鈺", "神之槌", "重鑄", "重铸", "重裝備", "重装备"])
            and any(k in text_corpus for k in ["決定", "决定", "跳過", "跳过"])
            and not any(k in text_corpus for k in ["請選擇欲使用", "请选择欲使用", "請選擇", "请选择"])
        ) or (
            any(k in text_corpus for k in ["決定", "决定"])
            and any(0.50 <= getattr(t, "center_rel", (0, 0))[0] <= 0.75 and getattr(t, "center_rel", (0, 0))[1] >= 0.85 and any(w in getattr(t, "text", "") for w in ["決定", "决定"]) for t in (getattr(state, "targets", None) or []))
            and any(k in text_corpus for k in ["神之", "重鑄", "重铸", "裝備", "装备", "強化", "强化"])
        )

        if is_god_hammer_confirm:
            logger.info("【神之锤强化 -> 二次确认】检测到重铸强化效果对比二次确认界面（[決定] / [跳過]），按主人规则执行确认...")
            # 1. 优先发送官方确认快捷键 (ENTER)
            self.ctx.input.tap_action("confirm", delay_after=0.6, label="【神之锤】二次确认(ENTER)")
            # 2. 点击橙色【決定】按钮中心 (0.637, 0.927)
            self.ctx.input.click_area_rel(0.54, 0.89, 0.74, 0.96, delay_after=1.2, label="【神之锤】点击決定按钮")
            time.sleep(0.3)
            return True

        # -------------------------------------------------------------
        # 分支 3.5：神之锤强化重铸装备界面（「神之錘」/「請選擇欲使用神之錘的重鑄裝備」）
        # 特征：出现「神之錘」/「神之锤」/「神之鈺」/「神之」/「使用神之」/「只能持有1個使用」
        # 主人 2026/10/11 规则与真机特征：
        #   "可以看到亮起的装备就是可以被重铸的，因此需要点击亮起的装备然后确认"
        #   该界面左侧无词条选项，仅有铁砧说明；右侧只有被重铸过、亮起发光(含橙金图标/高亮)的装备可被选中！
        # 动作：
        #   1. 遍历 3 名战斗员的 3 个装备槽（共 9 个槽位）
        #   2. 计算各槽位的亮起/重铸特征分（均值亮度 + 高亮像素数 > 200 + 橙金重铸角标/光效），精准筛选出「亮起的可重铸装备」
        #   3. 点击该亮起装备槽位激活（默认首选最高分槽位，若停留未推进按 ladder 选下一候选项）
        #   4. 按 ENTER 确认（或点击右下角橙色【確認】0.890, 0.925）
        #   5. 若未检测到亮起装备（异常兜底），按 F1 跳过 (0.700, 0.930)
        # -------------------------------------------------------------
        is_god_hammer = any(k in text_corpus for k in [
            "神之錘", "神之锤", "神之鈺", "神之", "使用神之", "欲使用神之",
            "只能持有1個使用", "只能持有1个使用", "只能持有1個", "只能持有1个"
        ]) and any(k in text_corpus for k in [
            "重鑄", "重铸", "重裝備", "重装备", "重装備", "裝備", "装备", "装備", "強力效果", "强力效果", "強化的裝備", "强化的装备"
        ])

        if is_god_hammer:
            logger.info("【神之锤强化】检测到「請選擇欲使用神之錘的重鑄裝備」界面...")
            slot_xs = [(0.735, 0.785), (0.805, 0.855), (0.875, 0.925)]
            row_ys = [(0.260, 0.380), (0.470, 0.590), (0.690, 0.810)]
            slot_names = {1: "武器", 2: "防具", 3: "饰品"}

            glowing_slots = []
            for r in range(3):
                for s in range(3):
                    x1, x2 = int(slot_xs[s][0] * w), int(slot_xs[s][1] * w)
                    y1, y2 = int(row_ys[r][0] * h), int(row_ys[r][1] * h)
                    crop = raw_img[y1:y2, x1:x2]
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    badge_mask = cv2.inRange(hsv, np.array([10, 120, 120]), np.array([25, 255, 255]))
                    badge_cnt = int(np.count_nonzero(badge_mask))
                    bright_cnt = int(np.count_nonzero(crop > 200))
                    mean_val = float(crop.mean())
                    score = mean_val + (bright_cnt / 5.0) + (badge_cnt * 1.5)
                    cx = (slot_xs[s][0] + slot_xs[s][1]) / 2
                    cy = (row_ys[r][0] + row_ys[r][1]) / 2
                    if score > 150.0 or (mean_val > 80.0 and (badge_cnt > 30 or bright_cnt > 100)):
                        glowing_slots.append({
                            'hero': r + 1, 'slot': s + 1, 'slot_name': slot_names[s + 1],
                            'cx': cx, 'cy': cy, 'score': score, 'mean': mean_val,
                            'badge': badge_cnt, 'bright': bright_cnt
                        })

            glowing_slots.sort(key=lambda x: x['score'], reverse=True)

            if glowing_slots:
                stuck_cnt = getattr(self, "_evt_stuck", 0)
                ladder_idx = ActionLadder.index_for(stuck_cnt, len(glowing_slots))
                target_item = glowing_slots[ladder_idx]
                logger.info(f"【神之锤强化】发现 {len(glowing_slots)} 个亮起的可重铸装备，选择第 {ladder_idx + 1} 个：战斗员 {target_item['hero']} 的 {target_item['slot_name']} "
                            f"(坐标: {target_item['cx']:.3f}, {target_item['cy']:.3f}, 分数: {target_item['score']:.1f})...")
                self.ctx.input.click_rel(target_item['cx'], target_item['cy'], delay_after=0.5,
                                         label=f"【神之锤强化】选择战斗员{target_item['hero']}的{target_item['slot_name']}")
                time.sleep(0.4)

                logger.info("【神之锤强化】已选中亮起装备，按 ENTER / 点击【確認】(0.890, 0.925) 完成强化...")
                if not self.ctx.input.tap_action("confirm", delay_after=1.5, label="【神之锤强化】确认(ENTER)"):
                    self.ctx.input.click_area_rel(0.84, 0.88, 0.95, 0.95, delay_after=1.5, label="【神之锤强化】点击確認")
                self.ctx.input.click_rel(0.890, 0.925, delay_after=1.0, label="【神之锤强化】点击橙色確認")

                # 主人 2026/10/11 规则："确认后，会二次确认，依旧确认"
                # 检查是否立即弹出二次确认界面（若弹出直接点击决定）
                time.sleep(0.6)
                if hasattr(self.ctx, "capture") and self.ctx.capture is not None:
                    after_img = self.ctx.capture.capture()
                    if after_img is not None and after_img.size > 0:
                        h_a, w_a = after_img.shape[:2]
                        strip_a = after_img[int(0.88 * h_a):int(0.97 * h_a), int(0.50 * w_a):int(0.80 * w_a)]
                        hsv_a = cv2.cvtColor(strip_a, cv2.COLOR_BGR2HSV)
                        mask_a = cv2.inRange(hsv_a, np.array([8, 120, 120]), np.array([25, 255, 255]))
                        if np.count_nonzero(mask_a) >= 5000:
                            logger.info("【神之锤强化】检测到已触发二次确认界面，连贯点击橙色【決定】(0.637, 0.927)...")
                            self.ctx.input.tap_action("confirm", delay_after=0.5, label="【神之锤】二次确认(ENTER)")
                            self.ctx.input.click_area_rel(0.54, 0.89, 0.74, 0.96, delay_after=1.2, label="【神之锤】点击決定按钮")
                return True
            else:
                logger.warning("【神之锤强化】未检测到任何亮起的可重铸装备，按 F1 / 点击跳过 (0.700, 0.930)...")
                if not self.ctx.input.send_key(112, delay_after=1.2, label="【神之锤强化】跳过(F1)"):
                    self.ctx.input.click_rel(0.700, 0.930, delay_after=1.2, label="【神之锤强化】点击跳过")
                return True

        # -------------------------------------------------------------
        # 分支 4：装备重铸界面（「重鑄裝備」/「請選擇欲套用重鑄效果的裝備」）
        # 特征：出现「重鑄」/「重铸」/「重裝備」/「重新整理」等
        # 动作：
        #   1. 选定重铸效果（默认点击第 2 项/攻击效果或第 1 项）
        #   2. 扫描右侧 3 位战斗员的已穿戴装备槽（寻找 mean > 70 的槽位）
        #   3. 优先选择武器（槽位 1），其次饰品/防具；点击该已穿戴装备槽位激活确认
        #   4. 按 ENTER（或点击确认）完成重铸；若无任何已穿戴装备，点击 F1 跳过
        # -------------------------------------------------------------
        is_equip_reforge = any(k in text_corpus for k in [
            "重鑄", "重铸", "重鑄裝備", "重铸装备", "重裝備", "重装备", "重装備",
            "欲套用", "重鑄效果", "重铸效果", "重新整理"
        ])

        if is_equip_reforge:
            logger.info("【装备重铸】检测到装备重铸界面（重鑄裝備 / 請選擇欲套用重鑄效果的裝備）...")

            # 1. 点击选择左侧重铸效果（选项 1/2/3 竖排坐标：y=0.356, 0.523, 0.690）
            option_ys = [0.356, 0.523, 0.690]
            ladder_opt = ActionLadder.index_for(getattr(self, "_evt_stuck", 0), 3)
            opt_idx = 1 if ladder_opt == 0 else (0 if ladder_opt == 1 else 2)
            opt_y = option_ys[opt_idx]
            logger.info(f"【装备重铸】选择第 {opt_idx + 1} 个重铸词条效果 (坐标: 0.280, {opt_y:.3f})...")
            self.ctx.input.click_area_rel(0.12, opt_y - 0.05, 0.45, opt_y + 0.05, delay_after=0.4, label="【装备重铸】选择重铸效果")
            time.sleep(0.3)

            # 2. 扫描右侧 3 名战斗员的已穿戴装备槽
            # 槽位定义：槽位1=武器 (rx: 0.735~0.785), 槽位2=防具 (rx: 0.805~0.855), 槽位3=饰品 (rx: 0.875~0.925)
            # 战斗员行：行1=(0.26~0.38), 行2=(0.47~0.59), 行3=(0.69~0.81)
            slot_xs = [(0.735, 0.785), (0.805, 0.855), (0.875, 0.925)]
            row_ys = [(0.260, 0.380), (0.470, 0.590), (0.690, 0.810)]

            equipped_items = []
            for r in range(3):
                for s in range(3):
                    x1, x2 = int(slot_xs[s][0] * w), int(slot_xs[s][1] * w)
                    y1, y2 = int(row_ys[r][0] * h), int(row_ys[r][1] * h)
                    crop = raw_img[y1:y2, x1:x2]
                    # 空槽位均值通常在 30~45，已穿戴装备均值在 70~120
                    if crop.mean() > 70:
                        cx = (slot_xs[s][0] + slot_xs[s][1]) / 2
                        cy = (row_ys[r][0] + row_ys[r][1]) / 2
                        equipped_items.append({'hero': r + 1, 'slot': s + 1, 'cx': cx, 'cy': cy, 'mean': crop.mean()})

            if equipped_items:
                weapons = [it for it in equipped_items if it['slot'] == 1]
                accessories = [it for it in equipped_items if it['slot'] == 3]
                target_item = weapons[0] if weapons else (accessories[0] if accessories else equipped_items[0])

                slot_names = {1: "武器", 2: "防具", 3: "饰品"}
                logger.info(f"【装备重铸】已锁定目标装备：战斗员 {target_item['hero']} 的 {slot_names.get(target_item['slot'], '装备')} "
                            f"(坐标: {target_item['cx']:.3f}, {target_item['cy']:.3f})，点击选中...")
                self.ctx.input.click_rel(target_item['cx'], target_item['cy'], delay_after=0.5,
                                         label=f"【装备重铸】选择战斗员{target_item['hero']}的{slot_names.get(target_item['slot'], '装备')}")
                time.sleep(0.4)

                logger.info("【装备重铸】已选定装备，按 ENTER 确认重铸...")
                if not self.ctx.input.tap_action("confirm", delay_after=1.5, label="【装备重铸】确认(ENTER)"):
                    self.ctx.input.click_area_rel(0.84, 0.88, 0.95, 0.95, delay_after=1.5, label="【装备重铸】点击確認")
                return True
            else:
                logger.warning("【装备重铸】未检测到任何战斗员拥有已穿戴装备，点击跳过(F1)...")
                if not self.ctx.input.send_key(112, delay_after=1.2, label="【装备重铸】跳过(F1)"):
                    self.ctx.input.click_rel(0.700, 0.930, delay_after=1.2, label="【装备重铸】点击跳过")
                return True

        # -------------------------------------------------------------
        # 阶段四：战斗员分配界面（請選擇戰鬥員）
        # -------------------------------------------------------------
        # 三名战斗员与三个装备槽的坐标定义：
        # 槽位1=武器 (rx: 0.74~0.785), 槽位2=防具 (rx: 0.81~0.855), 槽位3=饰品 (rx: 0.88~0.925)
        # 战斗员行：行1=(0.26~0.36), 行2=(0.47~0.57), 行3=(0.68~0.78)
        slot_xs = [(0.740, 0.785), (0.810, 0.855), (0.880, 0.925)]
        row_ys = [(0.260, 0.360), (0.470, 0.570), (0.680, 0.780)]
        hero_click_ys = [0.290, 0.510, 0.730]

        # 1. 寻找当前发光的装备槽（主人："发光的就是当前的装备栏"）
        glow_scores = [0, 0, 0]
        for s in range(3):
            for r in range(3):
                x1, x2 = int(slot_xs[s][0] * w), int(slot_xs[s][1] * w)
                y1, y2 = int(row_ys[r][0] * h), int(row_ys[r][1] * h)
                crop_slot = raw_img[y1:y2, x1:x2]
                # 只有未装备的槽位（背景较暗，均值 < 75）上的高亮轮廓（白边/发光图标 > 180）才代表发光槽位
                if crop_slot.mean() < 75:
                    bright_cnt = np.count_nonzero(crop_slot > 180)
                    glow_scores[s] += bright_cnt

        target_slot_idx = int(np.argmax(glow_scores))
        slot_names = ["武器(槽位1)", "防具(槽位2)", "饰品(槽位3)"]
        logger.info(f"【获得装备 -> 槽位分析】发光特征分析: {glow_scores} -> 锁定目标装备类型: {slot_names[target_slot_idx]}")

        # 2. 分析各战斗员在 target_slot_idx 上的装备状态（均值 < 75 判定为未穿戴/空）
        empty_heroes = []
        for r in range(3):
            x1, x2 = int(slot_xs[target_slot_idx][0] * w), int(slot_xs[target_slot_idx][1] * w)
            y1, y2 = int(row_ys[r][0] * h), int(row_ys[r][1] * h)
            slot_mean = raw_img[y1:y2, x1:x2].mean()
            if slot_mean < 75:
                empty_heroes.append(r + 1)
            else:
                logger.info(f"【获得装备 -> 状态检测】战斗员 {r + 1} 的 {slot_names[target_slot_idx]} 已有装备 (亮度={slot_mean:.1f})，跳过该角色")

        # 3. 决定目标战斗员（主人："如果某些角色有装备了，就分配给另外的没有装备的角色"）
        chosen_hero = 1
        if empty_heroes:
            # 优先分配给装备栏为空的角色！
            chosen_hero = empty_heroes[0]
            logger.info(f"【获得装备 -> 智能分配】检测到战斗员 {empty_heroes} 的 {slot_names[target_slot_idx]} 仍为空，优先分配给战斗员 {chosen_hero}！")
        else:
            # 如果三名角色均已有装备，检查是否有推荐角色
            rec_loc = None
            targets = self._ocr_targets(raw_img, state)
            for t in targets:
                txt = getattr(t, "text", "") or (t.get("text", "") if isinstance(t, dict) else "")
                if any(k in txt for k in ["推薦", "推荐"]):
                    rec_loc = t.center_rel if hasattr(t, "center_rel") else None
                    break
            if rec_loc is not None:
                if rec_loc[1] < 0.38:
                    chosen_hero = 1
                elif rec_loc[1] < 0.60:
                    chosen_hero = 2
                else:
                    chosen_hero = 3
                logger.info(f"【获得装备 -> 智能分配】所有战斗员均已有装备，采纳系统推荐战斗员 {chosen_hero}...")
            else:
                chosen_hero = 1
                logger.info(f"【获得装备 -> 智能分配】所有战斗员均已有装备且无推荐标签，默认分配给战斗员 {chosen_hero}...")

        # 4. 点击选择目标战斗员
        target_ry = hero_click_ys[chosen_hero - 1]
        logger.info(f"【获得装备】点击选中战斗员 {chosen_hero} (坐标: 0.700, {target_ry:.3f})...")
        self.ctx.input.click_area_rel(0.65, target_ry - 0.05, 0.78, target_ry + 0.05, delay_after=0.4, label=f"【装备】选择第{chosen_hero}战斗员")
        time.sleep(0.3)

        # 5. 点击右下角橙色【確認】按钮 (0.890, 0.925 / ENTER)
        logger.info("【获得装备】已选择目标战斗员，按 ENTER / 点击【確認】完成装备分配...")
        if not self.ctx.input.tap_action("confirm", delay_after=1.2, label="【装备】ENTER确认分配"):
            self.ctx.input.click_area_rel(0.84, 0.88, 0.95, 0.95, delay_after=1.5, label="【装备】确认分配")
        self.ctx.input.click_rel(0.890, 0.925, delay_after=1.2, label="【装备】点击橙色確認")
        return True

    def is_epiphany_hero_screen(self, raw_img: Optional[np.ndarray] = None, state: Optional[GameState] = None) -> bool:
        """Check if current screen is Epiphany Combatant Selection (選擇欲獲得【靈光一閃】的戰鬥員)."""
        text_corpus = " ".join(state.raw_texts) if state else ""
        if any(k in text_corpus for k in ["零號系統", "零号系统", "零号系統", "選擇法典", "选择法典"]):
            return False
        has_epiphany = any(k in text_corpus for k in ["靈光一閃", "灵光一闪", "靈光", "灵光", "一閃", "一闪", "【靈光一】", "靈光一"])
        has_hero = any(k in text_corpus for k in ["戰鬥員", "战斗员", "戰鬥", "战斗", "隊員", "队员", "欲獲得", "欲获得", "請選擇", "请选择", "撐欲瘦得", "撑欲瘦得", "門眞", "门真", "的門眞", "的门真", "的眞", "的真"])
        return has_epiphany and has_hero

    def is_epiphany_screen(self, raw_img: Optional[np.ndarray] = None, state: Optional[GameState] = None) -> bool:
        """Check if current screen is the Epiphany effect card selection screen."""
        if self.is_epiphany_hero_screen(raw_img, state):
            return True
        text_corpus = " ".join(state.raw_texts) if state else ""
        is_event_choice_text = any(k in text_corpus for k in ["分享資訊", "分享资讯", "所選戰鬥員", "所选战斗员", "1次", "一次"])
        if is_event_choice_text:
            return False
        # 零号系统与法典界面绝非局内灵光一闪
        if any(k in text_corpus for k in ["零號系統", "零号系统", "零号系統", "零號系统", "選擇法典", "选择法典", "撐法典", "撑法典", "選撑法典"]):
            return False
        return (
            ("長按卡牌" in text_corpus or "长按卡牌" in text_corpus) or
            (any(k in text_corpus for k in ["靈光一閃", "灵光一闪", "靈光一", "灵光一", "靈光", "灵光"]) and
             any(k in text_corpus for k in ["查看內容", "查看内容", "CAPS", "效果選擇", "效果选择"])) or
            any(k in text_corpus for k in ["選擇您的命運", "选择您的命运", "您的命運", "您的命运", "選擇命運", "选择命运", "欲適用的命運", "欲适用的命运"]) or
            (any(k in text_corpus for k in ["命運卡牌", "命运卡牌", "命運效果", "命运效果"]) and any(k in text_corpus for k in ["ENTER", "F1", "AP1", "確認", "确认"]))
        )

    # 选牌界面（灵光一闪 3选1 / 卡牌獎勵 3选1）：三张卡各自的 x 范围，统一 y 0.35~0.65
    _CARD_COLUMNS = ((0.19, 0.38), (0.39, 0.58), (0.63, 0.82))
    _CARD_Y = (0.35, 0.65)
    _CARD_CONFIRM = (0.84, 0.88, 0.94, 0.95)   # 右下角【確認】
    _CARD_CHOICE_MAX = 3                        # 最多试 3 张

    # 「选择战斗员领奖」类界面（卡牌奖励 / 装备奖励）：每行左侧有一个「等級」标签，
    # 点它右边一点就是那一行的战斗员
    _ASSIGN_ROW_LABEL_KW = ("等級", "等级", "等摄", "等", "レベル", "LV", "Lv")
    _ASSIGN_ROW_CLICK_DX = 0.09
    _ASSIGN_ROW_MIN_RX = 0.35
    _ASSIGN_ROW_MAX = 3

    def _ocr_targets(self, raw_img: np.ndarray, state: Optional[GameState] = None) -> List[Any]:
        """取 OCR 命中项。优先用 state.targets（不去重的完整列表），否则退回 OCR。
        注意不要用 state.elements —— 它按 text 去重，重名元素（例如三个「等級」标签）会丢。"""
        targets = list(getattr(state, "targets", None) or []) if state is not None else []
        if targets:
            return targets
        return self.ctx.vision.locate_all_texts(raw_img)

    def find_assign_rows(self, raw_img: np.ndarray, state: Optional[GameState] = None) -> List[Dict[str, Any]]:
        """找「選擇欲領取卡牌的戰鬥員 / 获得装备分配」这类界面的战斗员行（按 y 从上到下）。"""
        if raw_img is None or raw_img.size == 0:
            return []
        rows = []
        for t in self._ocr_targets(raw_img, state):
            rx, ry = t.center_rel
            if rx < self._ASSIGN_ROW_MIN_RX:
                continue
            if any(k in t.text for k in self._ASSIGN_ROW_LABEL_KW):
                rows.append({'rx': min(0.95, rx + self._ASSIGN_ROW_CLICK_DX), 'ry': ry, 'text': t.text})
        rows.sort(key=lambda r: r['ry'])
        # 同一行的「等級 / 50」两行标签会挨得很近，去重
        dedup: List[Dict[str, Any]] = []
        for r in rows:
            if not dedup or abs(r['ry'] - dedup[-1]['ry']) > 0.03:
                dedup.append(r)
        return dedup[: self._ASSIGN_ROW_MAX]

    def _click_assign_row_then_confirm(self, rows: List[Dict[str, Any]], confirm_rect, tag: str) -> bool:
        """主人规则：点第 k 个战斗员 -> 点【確認】；画面没变就换下一个，最多 3 个。"""
        n = min(len(rows), self._ASSIGN_ROW_MAX)
        idx = ActionLadder.index_for(getattr(self, "_evt_stuck", 0), n)   # 共用「卡住就换下一个」规则
        row = rows[idx]
        if idx > 0:
            logger.warning(f"【{tag}】同一画面已停留 {idx} 轮，前 {idx} 个战斗员点了没反应 -> "
                           f"改点第 {idx + 1}/{n} 个战斗员 (坐标: {row['rx']:.3f}, {row['ry']:.3f})...")
        else:
            logger.info(f"【{tag}】按规则选择第 1 个战斗员 (坐标: {row['rx']:.3f}, {row['ry']:.3f})...")
        self.ctx.input.click_rel(row['rx'], row['ry'], delay_after=0.6, label=f"【{tag}】第{idx + 1}个战斗员")
        time.sleep(0.4)
        c1, c2, c3, c4 = confirm_rect
        # 键盘优先（主人 2026/10/6）：装备/卡牌奖励这类界面底部都标着 `ENTER 確認`，
        # 所以"确认"这一步改按 ENTER，选人/选牌仍然是鼠标点（那些行上没有编号键）。
        logger.info(f"【{tag}】已选第 {idx + 1} 个战斗员 -> 按 ENTER 确认（官方：確認 = ENTER）...")
        if not self.ctx.input.tap_action("confirm", delay_after=1.5, label=f"【{tag}】确认(ENTER)"):
            self.ctx.input.click_area_rel(c1, c2, c3, c4, delay_after=1.5, label=f"【{tag}】确认")
        return True

    def handle_epiphany_selection(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        选牌界面（灵光一闪 3选1 / 卡牌獎勵「請選擇1張欲獲得的卡牌」）:
        主人规则 —— 点第 1 张牌，再点右下角【確認】；如果点了没反应（画面不变），
        就改点第 2 张 -> 【確認】，依此类推，最多试 3 张。
        """
        if raw_img is None:
            raw_img = self.ctx.capture.capture()

        # 针对选择欲获得灵光一闪战斗员界面，直接交由 handle_epiphany_hero_selection
        if self.is_epiphany_hero_screen(raw_img, state):
            return self.handle_epiphany_hero_selection(state, raw_img=raw_img)

        # 更新「是否还停在同一幕」计数：卡住轮数 = 已经试过几张牌
        self._event_scene_same(raw_img)
        stuck_cnt = getattr(self, "_evt_stuck", 0)

        # -------------------------------------------------------------
        # 1. 优先检查是否存在「神闪」（金色包边专属卡牌）！
        # 主人 2026/10/11 核心规则与真机特征：
        # "又遇到了新的样式，灵感一闪，但是有神闪，看到左边卡的1下面的那个图标没有，而且卡片带有金色的包边就是神闪，
        #  但是那个神闪的图标不一样都是这个样式，但是你可以通过包边金色来区分"
        # 只要检测到神闪卡牌，最高优先级选定该神闪卡牌！
        # -------------------------------------------------------------
        from ..core.features import detect_god_epiphany_card
        god_slot, god_det = detect_god_epiphany_card(raw_img)
        if god_slot is not None and stuck_cnt == 0:
            logger.info(
                f"【灵光一闪·神闪】发现场景中存在「神闪」专属强化卡牌！"
                f"卡牌 #{god_slot} 带有金色专属包边 (金色得分={god_det['gold']:.0f}, 其余卡均分={god_det.get('other_avg', 0):.0f}) -> "
                f"最高优先级选定第 {god_slot} 张神闪卡牌 (键盘 {god_slot} 号键)..."
            )
            g_idx = god_slot - 1
            grx1, grx2 = self._CARD_COLUMNS[g_idx]
            y1, y2 = self._CARD_Y
            self.ctx.input.tap_card(god_slot, delay_after=0.25, label=f"【选牌·神闪】按键{god_slot}")
            self.ctx.input.click_area_rel(grx1, y1, grx2, y2, delay_after=0.3, label=f"【选牌·神闪】第{god_slot}张卡牌")
            self.ctx.input.tap_action("confirm", delay_after=0.6, label="【选牌·神闪】确认(ENTER)")
            return True

        idx = ActionLadder.index_for(stuck_cnt,
                                     min(self._CARD_CHOICE_MAX, len(self._CARD_COLUMNS)))
        rx1, rx2 = self._CARD_COLUMNS[idx]

        if idx > 0:
            logger.warning(f"【选牌】同一画面已停留 {idx} 轮，前 {idx} 张牌没反应 -> "
                           f"改选第 {idx + 1}/{self._CARD_CHOICE_MAX} 张卡牌（键盘 {idx + 1} 号键）...")
        else:
            logger.info(f"【选牌】检测到卡牌选择界面，按规则选第 1 张卡牌（键盘 1 号键）...")

        # 键盘优先（主人 2026/10/6 确认）：灵光一闪选牌界面上，三张牌**自己标着 1/2/3**，
        # 官方快捷键表也写着「使用N號卡牌 / 選擇N號 = 1~9」。按数字键比点列坐标稳得多。
        # 若按键不可用（返回 False）再回退到点击列中心。
        y1, y2 = self._CARD_Y
        text_corpus = " ".join(state.raw_texts) if state is not None else ""

        # -------------------------------------------------------------
        # 针对「請選擇1張欲觸發靈光一閃的卡牌。」/ 竖排列表选卡牌触发灵光一闪：
        # 特征：出现「欲觸發靈光一閃」/「欲觸發」/ 顶部有「選擇卡牌 SPACE」
        # 主人规则：当前可看到2张卡，支持滑动浏览；按规则选中卡牌（SPACE/点击）后按 ENTER 确认
        # -------------------------------------------------------------
        is_trigger_epiphany_card = any(k in text_corpus for k in [
            "欲觸發靈光一閃", "欲触发灵光一闪", "觸發靈光一閃", "触发灵光一闪",
            "欲發靈光", "欲发灵光", "欲觸發", "欲触发"
        ]) or (
            any(k in text_corpus for k in ["選擇卡牌", "选择卡牌"]) and any(k in text_corpus for k in ["SPACE", "CAPS", "靈光一閃", "灵光一闪"])
        )

        if is_trigger_epiphany_card:
            logger.info("【灵光一闪选牌】检测到「請選擇1張欲觸發靈光一閃的卡牌」界面...")
            card_ys = [0.310, 0.715]

            # 若卡住 2 轮以上，尝试向下滚动/拖拽以露显后续卡牌
            if stuck_cnt >= 2:
                logger.warning("【灵光一闪选牌】前两张卡牌未成功推进，向下滚动浏览更多卡牌...")
                self.ctx.input.drag_rel(0.355, 0.750, 0.355, 0.250, delay_after=0.6, label="【灵光选牌】向上拖拽浏览")
                time.sleep(0.3)
                ladder_idx = 0
            else:
                ladder_idx = ActionLadder.index_for(stuck_cnt, len(card_ys))

            click_y = card_ys[ladder_idx]
            logger.info(f"【灵光一闪选牌】点击选中第 {ladder_idx + 1} 张候选卡牌 (坐标: 0.355, {click_y:.3f})，辅以 SPACE 键...")
            self.ctx.input.click_rel(0.355, click_y, delay_after=0.4, label=f"【灵光选牌】第{ladder_idx + 1}张卡牌")
            time.sleep(0.2)
            self.ctx.input.send_key(32, delay_after=0.4, label="【灵光选牌】SPACE选择")
            time.sleep(0.3)

            logger.info("【灵光一闪选牌】已选中卡牌，按 ENTER / 点击【靈光一閃】确认触发...")
            if not self.ctx.input.tap_action("confirm", delay_after=1.2, label="【灵光选牌】确认(ENTER)"):
                self.ctx.input.click_area_rel(0.82, 0.88, 0.98, 0.96, delay_after=1.2, label="【灵光选牌】点击靈光一閃")
            self.ctx.input.click_rel(0.890, 0.925, delay_after=1.0, label="【灵光选牌】点击橙色靈光一閃")
            return True

        is_epiphany_3cards = any(k in text_corpus for k in [
            "靈光一閃效果", "灵光一闪效果", "請選擇靈光一閃", "请选择灵光一闪"
        ]) or (
            any(k in text_corpus for k in ["靈光一閃", "灵光一闪", "靈光一", "灵光一", "靈光", "灵光"]) and
            any(k in text_corpus for k in ["查看內容", "查看内容", "CAPS", "效果選擇", "效果选择"])
        ) or (
            state is not None and getattr(state, "route", None) == SceneType.EPIPHANY_CARDS
        )

        if is_epiphany_3cards:
            # 针对 3选1 灵光一闪效果界面（卡片下方明确标有 ①/②/③ 官方快捷键）：
            # 键盘按两下编号键（第一下选中、第二下确认），辅以点击卡牌与 ENTER
            logger.info(f"【灵光一闪选牌】检测到 3选1 效果界面 -> 优先按快捷键 {idx + 1} 极速选定...")
            self.ctx.input.tap_card(idx + 1, delay_after=0.25, label=f"【选牌】按键{idx + 1}")
            self.ctx.input.click_area_rel(rx1, y1, rx2, y2, delay_after=0.25, label=f"【选牌】第{idx + 1}张卡牌")
            self.ctx.input.tap_action("confirm", delay_after=0.6, label="【选牌】确认(ENTER)")
            return True

        # 普通卡牌奖励选择（卡牌奖励界面，右下角有【確認】按钮）：
        self.ctx.input.click_area_rel(rx1, y1, rx2, y2, delay_after=0.4, label=f"【选牌】第{idx + 1}张卡牌")
        logger.info(f"【选牌】已点第 {idx + 1} 张牌 -> 按 ENTER 确认（官方：確認 = ENTER）...")
        c1, c2, c3, c4 = self._CARD_CONFIRM
        if not self.ctx.input.tap_action("confirm", delay_after=0.8, label="【选牌】确认(ENTER)"):
            self.ctx.input.click_area_rel(c1, c2, c3, c4, delay_after=0.8, label="【选牌】确认选择")
        return True

    def handle_epiphany_hero_selection(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        Handle Epiphany Combatant Selection (選擇欲獲得【靈光一閃】的戰鬥員):
        主人 2026/10/11 核心规则与真机教训：
        - 默认优先选 1 号位战斗员；
        - 若某位战斗员没有可灵光一闪的卡牌或达到上限，无法被选中，确认按钮置灰/无法推进时，
          绝不能死死在 1 号位原地盲点！必须智能轮换：1 号位不行立即去 2 号位，依次类推到 3 号位！
        - 3 位战斗员卡牌水平分布：
          1 号位 (左): rx 0.245 (区域 0.18~0.31, 0.35~0.65), 快捷键 1
          2 号位 (中): rx 0.500 (区域 0.43~0.57, 0.35~0.65), 快捷键 2
          3 号位 (右): rx 0.755 (区域 0.68~0.82, 0.35~0.65), 快捷键 3
        """
        logger.info("【灵光一闪选人】处理选择灵光一闪战斗员...")
        if raw_img is None:
            raw_img = self.ctx.capture.capture()

        # 跟踪同一幕停留轮数
        self._event_scene_same(raw_img)
        stuck = getattr(self, "_evt_stuck", 0)

        hero_coords = [
            (1, (0.18, 0.35, 0.31, 0.65), 0.245, 0.500),
            (2, (0.43, 0.35, 0.57, 0.65), 0.500, 0.500),
            (3, (0.68, 0.35, 0.82, 0.65), 0.755, 0.500),
        ]

        # 战斗员轮换阶梯 (1号位 -> 2号位 -> 3号位，依次类推)
        slot_idx = stuck % len(hero_coords)
        slot_num, area, cx, cy = hero_coords[slot_idx]

        if stuck > 0:
            logger.warning(
                f"【灵光一闪选人】上一位战斗员未成功推进 (stuck={stuck}) -> "
                f"智能轮换切换至第 {slot_num} 号位战斗员 (坐标: {cx:.3f}, {cy:.3f})..."
            )
        else:
            logger.info(
                f"【灵光一闪选人】默认选择第 {slot_num} 号位战斗员 (坐标: {cx:.3f}, {cy:.3f})..."
            )

        # 1. 优先发送官方编号快捷键 (键 1/2/3) + 点击对应角色卡片
        self.ctx.input.tap_card(slot_num, delay_after=0.3, label=f"【灵光一闪】按键{slot_num}")
        self.ctx.input.click_area_rel(area[0], area[1], area[2], area[3], delay_after=0.5, label=f"【灵光一闪】第{slot_num}战斗员")
        time.sleep(0.4)

        # 2. 点击右下角【確認】按钮 (ENTER / 0.890, 0.930)
        logger.info(f"【灵光一闪选人】已选中第 {slot_num} 号位战斗员，按 ENTER / 点击【確認】按钮完成选择...")
        if not self.ctx.input.tap_action("confirm", delay_after=1.2, label="【灵光一闪】确认战斗员(ENTER)"):
            self.ctx.input.click_area_rel(0.83, 0.88, 0.95, 0.96, delay_after=1.2, label="【灵光一闪】确认战斗员")
        self.ctx.input.click_rel(0.890, 0.930, delay_after=1.2, label="【灵光一闪】点击确认按钮")
        return True

    def _draw_bubble_overlay(self, bubbles: List[Dict[str, Any]]) -> None:
        """把全部搜索气泡画到 overlay 上（第 1 个青色、其余黄色）。"""
        if not getattr(self.ctx, "overlay", None):
            return
        overlay_boxes = []
        for i, b in enumerate(bubbles):
            overlay_boxes.append({
                'bbox': b['bbox'],
                'label': f"【搜索】[🔍] ({b['score']:.2f})",
                'click': (b['cx'], b['cy']),
                'color': '#00FFFF' if i == 0 else '#FFFF00'
            })
        self.ctx.overlay.show_boxes(overlay_boxes, duration=1.2)

    def _build_bubble_ladder(self, bubbles: List[Dict[str, Any]], h: int, w: int) -> ActionLadder:
        """
        把 `_EVT_LADDER` 的三类动作编成执行阶梯（统一由 ActionLadder 选级/封顶）：
          - `bubble`：点第 N 个搜索气泡（同点快速连点）
          - `ground`：点气泡正下方的地面发光物（物体本体）
          - `rel`   ：相对坐标点击（剧情推进 / 右上角跳过），保持默认点击时序
        """
        steps: List[Attempt] = []

        def _make(kind: str, payload, label: str):
            def _run() -> None:
                if kind == "key":
                    # 键盘优先：探索交互点 = F1/F2/F3；推进 = SPACE；跳过 = Z
                    from ..core.keys import resolve, KEYS
                    vk = resolve(payload)
                    if vk is None:
                        raise ValueError(f"未知按键 {payload}")
                    nice = self._EVT_KEY_LABELS.get(str(payload).upper(), f"【按键】{payload}")
                    self.ctx.input.send_key(vk, delay_after=1.2, label=nice)
                elif kind == "bubble":
                    i = min(payload, len(bubbles) - 1)
                    b = bubbles[i]
                    self.ctx.input.click_client(b['cx'], b['cy'], delay_after=1.5, jitter=4,
                                                label=f"【搜索】气泡#{i} x{self._BUBBLE_CLICK_COUNT}",
                                                press_ms=self._BUBBLE_CLICK_PRESS_MS,
                                                move_gap_ms=self._BUBBLE_CLICK_MOVE_GAP_MS,
                                                clicks=self._BUBBLE_CLICK_COUNT,
                                                click_gap_ms=self._BUBBLE_CLICK_GAP_MS)
                elif kind == "ground":
                    i = min(payload, len(bubbles) - 1)
                    b = bubbles[i]
                    gy = min(h - 2, b['cy'] + int(0.13 * h))
                    logger.info(f"【局内探索】改点气泡正下方的地面发光物体 (坐标: {b['cx'] / w:.3f}, {gy / h:.3f})...")
                    self.ctx.input.click_client(int(b['cx']), int(gy), delay_after=1.5, jitter=6,
                                                label=f"【搜索】地面物体 x{self._BUBBLE_CLICK_COUNT}",
                                                press_ms=self._BUBBLE_CLICK_PRESS_MS,
                                                move_gap_ms=self._BUBBLE_CLICK_MOVE_GAP_MS,
                                                clicks=self._BUBBLE_CLICK_COUNT,
                                                click_gap_ms=self._BUBBLE_CLICK_GAP_MS)
                else:  # "rel"
                    rx, ry = payload
                    self.ctx.input.click_rel(rx, ry, delay_after=1.5, label=label)
            return _run

        for label, kind, payload in self._EVT_LADDER:
            steps.append(Attempt(label, _make(kind, payload, label)))
        return ActionLadder(self.ctx, steps, tag="局内探索")

    def get_event_choices(self, raw_img: np.ndarray, state: Optional[GameState] = None) -> List[Dict[str, Any]]:
        """
        Extract and prioritize multi-choice dialog boxes:
        Priority rule:
        1. 灵光一闪 (Priority 1: score = 100)
        2. 装备 (Priority 2: score = 50)
        3. 其他 (Priority 3: score = 10)
        """
        if raw_img is None or raw_img.size == 0:
            return []

        h, w = raw_img.shape[:2]
        # Get OCR targets（用不去重的完整列表：底部可能有多个同名文本块）
        targets = self._ocr_targets(raw_img, state)

        choice_items = []
        for t in targets:
            rx, ry = t.center_rel
            if 0.65 <= ry <= 0.95 and 0.10 <= rx <= 0.90:
                choice_items.append({'text': t.text, 'rx': rx, 'ry': ry, 'cx': t.center[0], 'cy': t.center[1]})

        if not choice_items:
            return []

        # Horizontal clustering into option cards
        choice_items.sort(key=lambda item: item['cx'])
        clusters = []
        for item in choice_items:
            if not clusters or (item['cx'] - clusters[-1]['items'][-1]['cx'] > int(100 * (w / 1024.0))):
                clusters.append({'items': [item]})
            else:
                clusters[-1]['items'].append(item)

        options = []
        for c in clusters:
            all_text = " ".join(item['text'] for item in c['items'])
            mean_rx = float(np.mean([item['rx'] for item in c['items']]))
            mean_ry = float(np.mean([item['ry'] for item in c['items']]))

            # If text consists only of numbers/symbols (e.g. item counts "20 13", "30", "84,000")
            cleaned_letters = re.sub(r'[\d\s,./+%\-–]+', '', all_text).strip()
            if not cleaned_letters:
                priority = -1
                tag = "数值/物品数量"
            elif any(k in all_text for k in ["取消", "離開", "离开", "返回"]):
                priority = -1000
                tag = "取消"
            elif any(k in all_text for k in ["擲骰", "掷骰", "骰子", "般子", "相子", "般影"]):
                priority = -1
                tag = "掷骰提示"
            elif any(k in all_text for k in ["確認", "确认", "確定", "确定"]):
                priority = 200
                tag = "确认"
            elif any(k in all_text for k in ["靈光一閃", "灵光一闪", "靈光", "灵光", "一閃", "一闪"]):
                priority = 100
                tag = "灵光一闪"
            elif any(k in all_text for k in ["裝備", "装备", "装備"]):
                priority = 50
                tag = "装备"
            elif any(k in all_text for k in ["中立卡牌", "中立卡", "中立稀有", "中立傳說", "中立传说"]):
                # 主人 2026/10/11 核心规则：凡是中立卡尽量不选/跳过，优先级降至 5，有多选分支时优先选其他奖励
                priority = 5
                tag = "中立卡(降权)"
            elif any(k in all_text for k in ["跳過", "跳过", "放棄", "放弃"]):
                priority = 8
                tag = "跳过/放弃"
            elif any(k in all_text for k in ["選擇", "选择", "搜索", "搭話", "搭话", "調查", "调查", "前進", "前进", "觀察", "观察", "觸碰", "触碰", "無視", "无视", "獲得", "获得",
                                             # 交互提示框常用词（旧表里没有 -> 曾被判成「台词/旁白」丢弃）
                                             "搜查", "觸發", "触发", "結束", "结束", "事件", "互動", "互动",
                                             "移除", "删除", "刪除", "忘卻", "忘却", "銷毀", "销毁", "丟棄", "丢弃", "卡牌", "卡片"]):
                priority = 10
                tag = "其他"
            elif len(clusters) >= 2 and not any(k in all_text for k in ["……", "...", "？", "！", "。"]):
                priority = 10
                tag = "其他"
            else:
                # 形态兜底：底部中央的「事件框」（标题+效果说明的深色面板，例：
                # 「尋找生存訊號 / 選擇3張隨機中立稀有～傳說卡 / 牌的其中1張獲得」，
                # 「結束搜查 / 觸發【結束搜查】事件」）。这类框的文字往往一个白名单词都不沾，
                # 旧逻辑会把它判成「台词/旁白」丢掉 -> 永远不点 -> 卡死。
                # 所以这里只按位置与形态兜底。
                is_box_shape = (0.15 <= mean_rx <= 0.85 and 0.68 <= mean_ry <= 0.96
                                and len(all_text) <= 60
                                and not any(k in all_text for k in ["長按卡牌", "长按卡牌", "查看內容", "查看内容", "效果選擇", "效果选择"]))
                if is_box_shape:
                    priority = 10
                    tag = "事件框"
                else:
                    priority = -1
                    tag = "台词/旁白"

            options.append({
                'text': all_text,
                'rx': mean_rx,
                'ry': mean_ry,
                'priority': priority,
                'tag': tag
            })

        options = [o for o in options if o['priority'] > 0]
        options.sort(key=lambda o: (-o['priority'], o['rx']))
        return options

    def has_event_choices(self, raw_img: np.ndarray, state: Optional[GameState] = None) -> bool:
        """Check if any multi-choice dialog boxes are currently on screen."""
        return len(self.get_event_choices(raw_img, state)) > 0

    def check_and_click_close_x(self, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        Check and click the top-right close [X] button.
        Used to close drawers, info popups, or accidental overlays.
        """
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
        if raw_img is None or raw_img.size == 0:
            return False

        h, w = raw_img.shape[:2]
        crop = raw_img[:int(0.15 * h), int(0.85 * w):]
        if crop.size == 0:
            return False

        tpl_x = self._get_close_x_template()
        if tpl_x is None:
            return False

        scale = h / 576.0
        ts = cv2.resize(tpl_x, (0, 0), fx=scale, fy=scale)
        if crop.shape[0] < ts.shape[0] or crop.shape[1] < ts.shape[1]:
            return False

        res = cv2.matchTemplate(crop, ts, cv2.TM_CCOEFF_NORMED)
        _, mv, _, ml = cv2.minMaxLoc(res)
        if mv >= 0.70:
            cx = int(0.85 * w + ml[0] + ts.shape[1] // 2)
            cy = int(ml[1] + ts.shape[0] // 2)
            rx, ry = cx / w, cy / h
            logger.info(f"【关闭弹窗】检测到右上角 [X] 关闭按钮 (置信度={mv:.2f}, 坐标: {rx:.3f}, {ry:.3f})，点击关闭...")
            if getattr(self.ctx, "overlay", None):
                self.ctx.overlay.show_boxes([{
                    'bbox': (cx - 20, cy - 20, 40, 40),
                    'label': "【关闭】X 按钮",
                    'click': (cx, cy),
                    'color': '#FF5555'
                }], duration=1.0)
            self.ctx.input.click_client(cx, cy, delay_after=1.2)
            return True
        return False

    def check_and_click_leave_button(self, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        Check and click the universal bottom-right orange button [▷ 離開 / ✓ 獲得]
        present at the end of combat, events, and campfire rest.
        Note: If this is an info drawer with a close [X] and a "撤退" button, NEVER click retreat!
        """
        if raw_img is None:
            raw_img = self.ctx.capture.capture()
        if raw_img is None or raw_img.size == 0:
            return False

        # If there's an X button on screen (like 关卡资讯), prioritize X and NEVER click retreat!
        if self.check_and_click_close_x(raw_img):
            return True

        h, w = raw_img.shape[:2]
        crop_btn = raw_img[int(0.85 * h):int(0.96 * h), int(0.75 * w):int(0.98 * w)]
        if crop_btn.size == 0:
            return False

        hsv = cv2.cvtColor(crop_btn, cv2.COLOR_BGR2HSV)
        mask_orange = cv2.inRange(hsv, np.array([8, 120, 120]), np.array([25, 255, 255]))
        orange_pixels = np.count_nonzero(mask_orange)

        if orange_pixels >= 2200:
            logger.info("【局内出口】检测到右下角橙色带箭头【▷ 離開/獲得】按钮，点击离开返回路线 (坐标: 0.885, 0.915)...")
            self.ctx.input.click_area_rel(0.82, 0.87, 0.95, 0.96, delay_after=2.0, label="【出口】▷ 離開/獲得")
            return True

        return False

    def handle_event(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        Handle Question Mark (Event) selection and multi-choice dialog boxes:
        1. If close [X] button exists (e.g. opened drawer/info popup): click [X] to close
        2. If an Epiphany / equip-assign screen is showing: delegate to its dedicated handler
        3. 底部「事件框 / 选项框」——这才是局内事件真正可点的目标（主人指定规则）：
           只有 1 个就直接点；多个就点第 1 个；同一幕反复触发（点了没反应）就依次改点
           第 2、第 3 个，最多 3 个。优先级 确认 > 灵光一闪 > 装备 仍然保留。
           框的识别带「形态兜底」，不依赖关键字白名单（见 get_event_choices）。
        4. If in-scene search bubbles [🔍] exist: click them; if the screen stays the same
           (click seems ineffective), escalate along _EVT_LADDER instead of retrying forever
        5. If orange leave button is visible: click to leave
        6. Otherwise: advance dialog / interaction
        """
        logger.info("【神秘事件】处理局内事件抉择与交互...")
        if raw_img is None:
            raw_img = self.ctx.capture.capture()

        # 每轮更新「是否还停在同一幕」计数：下面的「事件框轮换」和「气泡升级阶梯」共用它
        self._event_scene_same(raw_img)
        stuck = getattr(self, "_evt_stuck", 0)

        # 0. Check if close [X] exists (e.g. opened stage info or drawer)
        if self.check_and_click_close_x(raw_img):
            return True

        # 1. 优先检查：底部「事件框 / 选项框」！
        # 主人规则与真机证据：事件界面往往有多个选项卡片（如「詢問...」「獲得獎勵...」），
        # 选项描述里可能包含"裝備"、"靈光一閃"、"重鑄"等奖励名词。
        # 只要存在事件选项框，必须优先进行事件选择，绝不能被后续子界面判定抢走！
        if raw_img is not None:
            choices = self.get_event_choices(raw_img, state)
            if choices:
                prio = [c for c in choices if c['priority'] >= 50]   # 确认 / 灵光一闪 / 装备
                if prio:
                    n = min(len(prio), 3)
                    idx = min(stuck, n - 1)
                    chosen = prio[idx]
                    why = f"按优先级 [{chosen['tag']}]" + (f" (第{idx + 1}/{n}项)" if n > 1 else "")
                else:
                    # 主人规则：1 个就直接点；多个先点第 1 个；同一幕卡住就依次改点第 2、第 3 个（最多 3 个）
                    n = min(len(choices), 3)
                    idx = min(stuck, n - 1)
                    chosen = choices[idx]
                    why = f"事件框第 {idx + 1}/{n} 个" + ("（上一轮点击后画面未变，轮换中）" if idx > 0 else "")
                if min(stuck, min(len(choices), 3) - 1) > 0:
                    logger.warning(f"【局内事件轮换】检测到 {len(choices)} 个选项，{why} 尝试第 {idx + 1} 项: "
                                   f"[{chosen['tag']}] \"{chosen['text']}\" (得分: {chosen['priority']})，"
                                   f"点击坐标 ({chosen['rx']:.3f}, {chosen['ry']:.3f})...")
                else:
                    logger.info(f"【局内事件选择】检测到 {len(choices)} 个选项，{why} 选定: "
                                f"[{chosen['tag']}] \"{chosen['text']}\" (得分: {chosen['priority']})，"
                                f"执行选择 (坐标: {chosen['rx']:.3f}, {chosen['ry']:.3f})...")
                if getattr(self.ctx, "overlay", None):
                    h, w = raw_img.shape[:2]
                    overlay_boxes = []
                    for b_idx, c in enumerate(choices):
                        cx, cy = int(c['rx'] * w), int(c['ry'] * h)
                        overlay_boxes.append({
                            'bbox': (cx - 240, cy - 60, 480, 120),
                            'label': f"[{c['tag']}] {c['text'][:8]}",
                            'click': (cx, cy),
                            'color': '#00FF00' if c is chosen else '#FFD700'
                        })
                    self.ctx.overlay.show_boxes(overlay_boxes, duration=1.2)
                # 键盘优先（主人 2026/10/6 定的规则）：**事件选项框 = 1/2/3**，框上就标着编号。
                # ⚠️ 关键修正：界面上的 1/2/3 对应屏幕上【从左到右】的物理顺序 (rx)，
                # 不能用按优先级重排后的列表下标（否则选第 3 个高优先级选项会误按 1 号键）
                sorted_by_pos = sorted(choices, key=lambda c: c['rx'])
                choice_index = sorted_by_pos.index(chosen) + 1 if chosen in sorted_by_pos else 1
                if choice_index <= 3 and not self.ctx.input.tap_card(
                        choice_index, delay_after=0.6,
                        label=f"【事件框】选项 {choice_index} (键 {choice_index})"):
                    self.ctx.input.click_rel(chosen['rx'], chosen['ry'], delay_after=0.8,
                                             label=f"【事件框】{chosen['tag']}")
                elif choice_index > 3:
                    self.ctx.input.click_rel(chosen['rx'], chosen['ry'], delay_after=0.8,
                                             label=f"【事件框】{chosen['tag']}")
                return True

        # 2. Check if this is an Epiphany selection screen
        if (state is not None and state.screen == ScreenState.EPIPHANY_SELECTION) or self.is_epiphany_screen(raw_img, state):
            if self.is_epiphany_hero_screen(raw_img, state):
                return self.handle_epiphany_hero_selection(state, raw_img=raw_img)
            return self.handle_epiphany_selection(state, raw_img=raw_img)

        # 3. Check if this is an equipment acquisition screen
        if self.is_equip_screen(raw_img, state):
            return self.handle_equip_selection(state, raw_img=raw_img)

        # 4. Check if this is a card removal screen (請選擇1張欲移除的卡牌 / 忘卻卡牌)
        text_corpus = " ".join(state.raw_texts) if state is not None else ""
        is_remove_card = any(w in text_corpus for w in ["欲移除的卡牌", "欲移除", "移除的卡牌", "移除卡牌", "移除1張", "移除1张", "忘卻", "忘却"]) or (
            "移除" in text_corpus and any(w in text_corpus for w in ["F1", "跳過", "跳过"]) and "ENTER" in text_corpus
        )
        if is_remove_card:
            logger.info("【局内事件 -> 卡牌移除】检测到移除卡牌界面 -> 点击选中第 1 张基础卡牌 (0.35, 0.30)...")
            self.ctx.input.click_area_rel(0.30, 0.22, 0.40, 0.38, delay_after=0.5, label="【删牌】选中第1张卡牌")
            time.sleep(0.3)
            logger.info("【卡牌移除】按 ENTER 确认移除该卡牌（官方：ENTER 移除）...")
            if not self.ctx.input.tap_action("confirm", delay_after=1.0, label="【删牌】ENTER确认移除"):
                self.ctx.input.click_area_rel(0.84, 0.88, 0.96, 0.94, delay_after=1.0, label="【删牌】点击移除按钮")
            return True

        # 5. Check if this is a dice roll screen (Tap / 請點擊畫面查看骰子結果 / 擲骰影響)
        has_dice_kw = any(w in text_corpus for w in [
            "骰子結果", "骰子结果", "般子结果", "相子结果",
            "查看骰子", "查看般子", "查看相子",
            "擲骰影響", "掷骰影响", "般影", "擲骰", "掷骰",
        ]) or (
            any(w in text_corpus for w in ["Tap", "TAP"]) and any(w in text_corpus for w in ["查看", "结果", "結果", "影響", "影响", "影", "骰", "般", "相"])
        ) or (
            any(w in text_corpus for w in ["點擊畫面查看", "點擎書面查看", "点击画面查看"]) and any(w in text_corpus for w in ["結果", "结果"])
        )
        if has_dice_kw:
            logger.info("【局内事件 -> 掷骰】检测到局内掷骰判定界面 -> 极速按 ENTER 并点击中央骰子...")
            self.ctx.input.tap_action("confirm", delay_after=0.3, label="【掷骰】ENTER确认")
            self.ctx.input.click_rel(0.50, 0.46, delay_after=0.6, label="【掷骰】点击中央骰子")
            return True

        # 3.5 Check if treasure chests exist in scene
        if raw_img is not None:
            from ..core.features import detect_treasure_chests
            chests = detect_treasure_chests(raw_img)
            if chests:
                t_idx = min(stuck, len(chests) - 1)
                cx, cy, sc = chests[t_idx]
                logger.info(f"【局内探索 -> 宝物库】发现场景中存在 {len(chests)} 个未开启宝箱 (置信度={sc:.2f}) -> "
                            f"点击开启第 {t_idx + 1} 个宝箱 (坐标: {cx:.3f}, {cy:.3f})...")
                self.ctx.input.click_rel(cx, cy, delay_after=1.8, label=f"【宝物库】开启宝箱 #{t_idx + 1}")
                return True

        # 4. Check if in-scene interactive search bubbles [🔍] exist
        if raw_img is not None:
            bubbles = self.detect_search_bubbles(raw_img)
            if bubbles:
                h, w = raw_img.shape[:2]
                # 打印「全部」气泡坐标（以前只打印第 1 个，排查时看不到其它可选目标）
                coords_text = " | ".join(
                    f"#{i} ({b['rx']:.3f},{b['ry']:.3f}) 置信度={b['score']:.2f}" for i, b in enumerate(bubbles)
                )

                # 卡死检测 + 升级阶梯：统一走 logic/action_runner（选级规则 min(stuck, n-1)、封顶）
                ladder = self._build_bubble_ladder(bubbles, h, w)
                n_steps = ladder.n

                def _on_pick(idx: int, label: str) -> None:
                    if stuck > 0 and idx > 0:
                        logger.warning(
                            f"【局内探索】同一画面已停留 {idx} 轮，上次点击无效 -> 升级措施 "
                            f"[{idx + 1}/{n_steps}]: {label}（气泡: {coords_text}）"
                        )
                    else:
                        logger.info(
                            f"【局内探索】发现场景内 {len(bubbles)} 个搜索气泡 [🔍]: {coords_text}，"
                            f"执行措施 [{idx + 1}/{n_steps}]: {label}"
                            f"（搜索点击: 同点快速连点 {self._BUBBLE_CLICK_COUNT} 次 / 每次按住 {self._BUBBLE_CLICK_PRESS_MS}ms"
                            f" / 间隔 {self._BUBBLE_CLICK_GAP_MS}ms）"
                        )
                    self._draw_bubble_overlay(bubbles)

                ladder.escalate(stuck, on_pick=_on_pick)
                return True

        # 5. If orange leave button is visible, click to leave
        if self.check_and_click_leave_button(raw_img):
            return True

        # 6. Check if standard choice buttons exist (center area: rx: 0.25~0.75, ry: 0.40~0.75)
        text_corpus = " ".join(state.raw_texts) if state is not None else ""
        if any(w in text_corpus for w in ["選擇", "选择", "確認", "确认", "前進", "前进"]):
            # Click primary event option
            self.ctx.input.click_rel(0.50, 0.45, delay_after=1.0)
            time.sleep(0.5)
            # Confirm if prompt appeared
            self.ctx.input.click_rel(0.50, 0.70, delay_after=1.0)
            return True

        # Advance dialog
        self.ctx.input.click_rel(0.50, 0.50, delay_after=1.0)
        return True

    def handle_shop(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """Handle Shop (德朗商店): auto buy target items and leave."""
        logger.info("【德朗商店】正在处理商店选购...")
        if raw_img is None:
            raw_img = self.ctx.capture.capture()

        # 1. If orange leave button is visible, click to leave
        if self.check_and_click_leave_button(raw_img):
            return True

        # 2. Check if targets are on shop shelves
        matched_cards, matched_equips = self.target_mgr.check_matches(state.raw_texts)
        for item in matched_cards + matched_equips:
            elem = state.get_element(item)
            if elem:
                logger.info(f"【商店购买】发现目标物品 [{item}]！执行购买...")
                cx, cy = elem.center
                self.ctx.input.click_client(cx, cy, delay_after=0.8)
                self.ctx.input.click_rel(0.65, 0.75, delay_after=1.0)
                self.target_mgr.mark_acquired(item)

        # 3. Leave shop
        time.sleep(0.5)
        leave_btn = (state.get_element("离开") or 
                     state.get_element("離開") or 
                     state.get_element("退出"))
        if leave_btn:
            self.ctx.input.click_target(leave_btn, delay_after=1.5)
        else:
            self.ctx.input.click_rel(0.92, 0.08, delay_after=1.5)
        return True

    def handle_camp(self, state: GameState, raw_img: Optional[np.ndarray] = None) -> bool:
        """
        Handle Camp/Safe Area (篝火区):
        1. If orange [▷ 離開] button is present (rest already applied): click to leave!
        2. If confirmation dialog is visible (HP恢復 / 取消 / 確認): click orange [確認] at (0.660, 0.670)
        3. If main camp screen: click free rest [休息 (免費)] at (0.334, 0.735)
        """
        logger.info("【篝火区】处理安全区域功能与休息...")
        if raw_img is None:
            raw_img = self.ctx.capture.capture()

        # 1. Step 3: Check if already finished rest and orange [▷ 離開] button is visible
        if self.check_and_click_leave_button(raw_img):
            logger.info("【篝火区】已顺利离开篝火区，返回路线地图！")
            return True

        text_corpus = " ".join(state.raw_texts)

        # Check if rest was already consumed / unavailable: click bottom-right to leave
        if any(w in text_corpus for w in ["無法使用", "无法使用", "自前无法使用", "目前無法使用"]):
            logger.info("【篝火区】免费休息已使用完毕，点击右下角离开营地/商店 (坐标: 0.885, 0.915)...")
            self.ctx.input.click_area_rel(0.84, 0.88, 0.94, 0.95, delay_after=1.8, label="【营地】离开")
            return True

        # 2. Step 2: Check if confirmation dialog is visible
        is_confirm_dialog = any(w in text_corpus for w in ["HP恢復", "HP恢复", "壓力恢復", "压力恢复", "取消", "自前无法使用", "目前無法使用"])
        if is_confirm_dialog:
            logger.info("【篝火区】检测到休息确认弹窗，点击右侧橙色【確認】按钮 (坐标: 0.660, 0.670)...")
            self.ctx.input.click_area_rel(0.61, 0.63, 0.71, 0.71, delay_after=1.5, label="【营地】确认休息")
            time.sleep(0.5)
            # Recheck if leave button appeared immediately
            after_img = self.ctx.capture.capture()
            if self.check_and_click_leave_button(after_img):
                return True
            return True

        # 3. Step 1: In main camp screen, click the Free Rest option [休息 (免費)]
        logger.info("【篝火区】按规则必须选择免费休息，点击【休息 (免費)】选项 (坐标: 0.334, 0.735)...")
        # Click the rest icon / button
        self.ctx.input.click_area_rel(0.28, 0.68, 0.38, 0.78, delay_after=1.2, label="【营地】免费休息")
        return True

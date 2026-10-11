"""
Global Scene Router (场景路由器).
Binds the Dynamic Scene Feature Scoring Pool to operational phase handlers:
- ROGUELIKE_MAP    -> PathPlanner.execute_branch_selection
- EVENT_CHOICE     -> NodeDecisionEngine.handle_event / in-scene search bubble
- EPIPHANY_CARDS   -> NodeDecisionEngine.handle_epiphany_selection
- BATTLE           -> BattleManager.play_round / StoryPusher.handle_battle
- CAMP_SHOP        -> NodeDecisionEngine.handle_camp
- EQUIP_ASSIGN     -> NodeDecisionEngine.handle_equip_selection
- STAGE_MAP        -> StoryPusher.handle_stage_map
- CUTSCENE_DIALOG  -> StoryPusher.handle_cutscene_dialog
- SETTLEMENT_LOOT  -> StoryPusher.handle_settlement / BattleManager.check_battle_ended
- UNKNOWN          -> Interstitial cruise click / leave button check
"""

import time
import logging
import re
import cv2
import numpy as np
import win32con
from typing import Dict, Any, Optional

from .scene_types import SceneType, SCENE_TO_SCREEN, text_locked_scene
from .scene_pool import ScenePoolClassifier, SceneResult
from ..state.screen_state import GameState
from ..core.features import is_blank_frame, blank_stats

logger = logging.getLogger("SceneRouter")

class SceneRouter:
    def __init__(self, ctx, config: Dict[str, Any], story_pusher=None, node_engine=None, path_planner=None, battle_mgr=None):
        self.ctx = ctx
        self.config = config
        router_cfg = config.get("router", {})
        self.classifier = ScenePoolClassifier(
            min_threshold=router_cfg.get("min_threshold", 50.0),
            min_margin=router_cfg.get("min_margin", 15.0)
        )
        self.story_pusher = story_pusher
        self.node_engine = node_engine
        self.path_planner = path_planner
        self.battle_mgr = battle_mgr
        # 「谁说了算」：默认 observer（= 改造前的行为不变）；真机验证后可切成 router
        cls_cfg = config.get("classification", {}) if isinstance(config, dict) else {}
        self.authority = cls_cfg.get("authority", "observer")

    def classify_frame(self, frame, state: Optional[GameState] = None) -> SceneResult:
        """
        **唯一的场景判定入口**：吃 FrameContext（含本帧 OCR + 共用模板库），
        产出单一结论 `(场景, 置信度, 证据)`。observer 不再参与"这是哪个场景"的判断。
        """
        if frame is None or not getattr(frame, "valid", False):
            return SceneResult(scene=SceneType.UNKNOWN, confidence=0.0)
        locked = text_locked_scene(frame.corpus)
        if locked is not None:
            return SceneResult(scene=locked[0], confidence=100.0, second_scene=SceneType.UNKNOWN,
                               second_score=0.0, margin=100.0, all_scores={},
                               details={"text_lock": locked[1]})
        return self.classifier.classify(frame.img, frame.corpus, frame.targets)

    def _record(self, state: GameState, decision: str, res: Optional[SceneResult] = None) -> None:
        """把本帧的决策写到 state 上（一行日志/诊断就能看清"这一帧到底判成什么、为什么"）。"""
        if state is None:
            return
        state.decision = decision
        if res is not None:
            state.route = res.scene
            state.scene_result = res

    def _reconcile_screen(self, state: GameState, res: SceneResult) -> None:
        """
        把 router 的结论挂到 state 上，并按配置决定「谁说了算」。

        - `state.route` / `state.scene_result` 始终记录 router 的结论（供 handler 与日志使用）
        - `classification.authority == "router"` 时，用映射把 `state.screen` 也改成路由结论
          （默认 "observer"：保持改造前的行为不变，只记录分歧，等真机验证后再切）
        """
        self._record(state, f"route:{res.scene.value}", res)

        mapped = SCENE_TO_SCREEN.get(res.scene)
        obs_screen = getattr(state, "screen", None)
        obs_conf = float(getattr(state, "confidence", 0.0) or 0.0)
        obs_conf_pct = obs_conf * 100.0 if obs_conf <= 1.0 else obs_conf      # 观察者是 0~1，路由是 0~100
        if mapped is not None and obs_screen is not None and obs_screen != mapped:
            logger.info(f"【判定分歧】observer={obs_screen.name}({obs_conf_pct:.0f}分) "
                        f"路由={res.scene.value}({res.confidence:.0f}分) -> 路由映射为 {mapped.name}"
                        f"（authority={self.authority}）")
            # **谁分高听谁的**：观察者只是"另一个候选人"。
            # 真机教训（主人 2026/10/6）：肉鸽里的「注視希爾德的欲望」事件框，
            # 观察者判 VICTORY_SETTLEMENT(50)，路由判 局内互动事件(65)，
            # 旧逻辑无条件采信观察者 -> 去点"关闭橙色大按钮" -> 动作完全错（主人："搞麻了"）。
            if res.confidence > obs_conf_pct + 5.0:
                logger.info(f"【判定裁决】路由分更高（{res.confidence:.0f} > 观察者 {obs_conf_pct:.0f}）"
                            f"-> 采用路由结论: {res.scene.value}")
                state.screen = mapped
        if self.authority == "router" and mapped is not None:
            state.screen = mapped

    def dispatch_frame(self, frame, state: GameState) -> bool:
        """按 FrameContext 路由（与 dispatch 同一套逻辑，只是不再自己重新截图/OCR）。"""
        if frame is None or not getattr(frame, "valid", False):
            return False
        return self._dispatch_impl(frame.img, state, frame.corpus, list(frame.targets))

    def dispatch(self, raw_img: np.ndarray, state: GameState) -> bool:
        """
        Evaluate current frame in the dynamic scene pool and route to the winning handler.
        """
        if raw_img is None or raw_img.size == 0:
            return False

        text_corpus = " ".join(state.raw_texts)
        # 必须用 state.targets（不去重的完整列表）：elements 按 text 去重，
        # 「卡牌獎勵→選擇欲領取卡牌的戰鬥員」界面三个同名「等級」标签会只剩一个，
        # 打分池的「>=2 行」结构特征就永远命中不了。
        ocr_targets = (list(getattr(state, "targets", None) or [])
                       or (list(state.elements.values()) if state.elements
                           else self.ctx.vision.locate_all_texts(raw_img)))
        return self._dispatch_impl(raw_img, state, text_corpus, ocr_targets)

    def _dispatch_impl(self, raw_img: np.ndarray, state: GameState,
                       text_corpus: str, ocr_targets: list) -> bool:
        """真正的路由主体：dispatch 与 dispatch_frame 共用。"""

        # -------------------------------------------------------------
        # Universal Safety Check: 过渡/全黑帧（进出关卡、剧情的淡入淡出）
        # 黑屏时所有评估器都是 0 分，防抖会沿用上一帧场景，于是"拿旧场景在黑屏上乱点"
        # （真机踩到：黑屏上拖动地图，直到看门狗 20 秒超时 ESC）。这里直接跳过，等画面恢复。
        # -------------------------------------------------------------
        if is_blank_frame(raw_img, targets=ocr_targets):
            mean, std = blank_stats(raw_img)
            self._record(state, "skip:blank_frame")
            logger.info(f"【过渡帧】画面近乎全黑 (均值={mean:.1f}, 标准差={std:.1f}) 且无任何游戏文字，"
                        f"本轮不动作，等画面恢复...")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: 游戏设置/系统设置弹窗（ESC误按导致开启「圖像設定」等）
        # 真机证据：ESC导致打开了「圖像設定」，包含「解析度」「全螢幕」「水面反射」等。
        # 弹窗右上角有 [X] (约 rx: 0.938, ry: 0.095)，官方快捷键也是 ESC。
        # -------------------------------------------------------------
        if any(w in text_corpus for w in ["圖像設定", "图像设定", "聲音設定", "声音设定",
                                          "快捷鍵設定", "快捷键设定", "系統設定", "系统设定",
                                          "遊戲設定", "游戏设定", "水面反射", "全螢幕", "全屏幕",
                                          "邊界陰影", "边界阴影", "陰影效果", "阴影效果",
                                          "顯示模式", "显示模式"]):
            logger.warning("【安全兜底】检测到游戏设置/系统设置窗口意外开启 -> 点击右上角 [X] 并发送 ESC 关闭...")
            self._record(state, "safety:settings_dialog")
            self.ctx.input.click_area_rel(0.925, 0.080, 0.950, 0.110, delay_after=0.6, label="【设置】点击关闭[X]")
            self.ctx.input.tap_action("cancel", delay_after=1.0, label="【设置】退出设置(ESC)")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: 账号资讯/个人资料弹窗（误点头像进入，点击左上角[<]或按ESC退回）
        # 真机证据（主人 2026/10/11 截图 media_1791716405752_574259e2.png）：
        #   标题「帳號資訊」，包含「ID卡」「展示櫃」「好友列表」「UID:」。
        # -------------------------------------------------------------
        if any(w in text_corpus for w in ["帳號資訊", "账号资讯", "展示櫃", "展示柜", "UID:"]) and any(w in text_corpus for w in ["ID卡", "好友列表", "代表戰鬥員", "代表战斗员", "封鎖列表", "封锁列表", "編輯ID卡", "编辑ID卡"]):
            logger.info("【账号资讯】检测到账号资讯/个人资料窗口，点击左上角【<】或按ESC退出返回主界面...")
            self._record(state, "safety:account_info_exit")
            self.ctx.input.click_area_rel(0.02, 0.02, 0.06, 0.07, delay_after=0.8, label="【账号资讯】点击返回")
            self.ctx.input.tap_action("cancel", delay_after=1.0, label="【账号资讯】ESC返回")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: 全力战界面（誤入全力戰內部/強敵挑戰，點擊左上角返回上層）
        # 真机证据（主人 2026/10/11 截图 media_1791715910129_97cbdfcc.png）：
        #   标题为「全力戰」(OCR可能只认出「全力」)，展示渴望的啟動、巨像、阿巴爾克等Boss，右下角「準備戰鬥」。
        #   注意：必须排除主界面大厅（大厅右下角仅为全力战入口Banner，绝不能在大厅点左上角返回误开账号信息！）
        # -------------------------------------------------------------
        is_home_lobby_signs = any(w in text_corpus for w in ["出擊", "出击", "方舟都市", "救援", "戰鬥員", "战斗员"]) or (
            "/240" in text_corpus and any(w in text_corpus for w in ["卡厄思", "故事", "模擬", "模拟"])
        )
        is_all_out_interior = (not is_home_lobby_signs) and (
            any(w in text_corpus for w in ["巨像", "阿巴爾克", "阿巴尔克", "渴望的啟動", "渴望的启动"])
            or (any(w in text_corpus for w in ["我的排名", "排名"]) and any(w in text_corpus for w in ["準備戰鬥", "准备战斗", "0/9"]))
        )
        if is_all_out_interior:
            logger.info("【安全回退】检测到误入「全力戰」强敌挑战界面 -> 点击左上角【<】安全返回...")
            self._record(state, "safety:all_out_battle_exit")
            self.ctx.input.click_area_rel(0.02, 0.02, 0.06, 0.07, delay_after=1.2, label="【全力战】点击返回")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: Item detail popup (误触物品详情弹窗兜底关闭)
        # -------------------------------------------------------------
        if any(w in text_corpus for w in ["持有数量", "持有數量", "获得途径", "獲得途徑", "经验值材料", "經驗值材料", "支援资料", "支援資料"]):
            logger.info("【物品详情】检测到物品/材料信息弹窗，点击右上角 [X] 或空白处关闭...")
            self._record(state, "safety:item_detail_popup")
            self.ctx.input.click_area_rel(0.81, 0.26, 0.84, 0.29, delay_after=0.6)
            self.ctx.input.click_rel(0.15, 0.50, delay_after=0.8)
            return True

        # -------------------------------------------------------------
        # Universal Safety Check:「自我意識點數不足」选择界面（按了买不起的卡/技能会弹出来）
        # 真机证据（主人 2026/10/6 截图）：EP 不够时按编号键 -> 游戏弹出卡牌预览 +
        # `自我意識點數不足。`，左侧提示 `ENTER 使用卡牌` / `ESC 取消選擇`。
        # 脚本如果不管它就会一直卡在这个选择态里 -> 这里统一按 **ESC 取消選擇**（官方快捷键取消=ESC）。
        # -------------------------------------------------------------
        if any(w in text_corpus for w in ["點數不足", "点数不足", "自我意識點數", "自我意识点数",
                                          "取消選擇", "取消选择"]):
            if any(w in text_corpus for w in ["點數不足", "点数不足"]):
                logger.warning("【战斗自救】检测到「自我意識點數不足」-> 按 ESC 取消選擇（官方：取消/返回 = ESC）...")
                self._record(state, "safety:ep_insufficient")
                self.ctx.input.tap_action("cancel", delay_after=1.0, label="【战斗】取消選擇(ESC)")
                return True

        # -------------------------------------------------------------
        # -------------------------------------------------------------
        # Universal Safety Check: 宝物库/场景未开启宝箱 (Treasure Chests)
        # 主人 2026/10/11 核心规则："这里没点击箱子，看到箱子应该点击，然后开完了在继续前进"
        # ⚠️ 绝对不能在宝箱尚未开启时直接按「繼續進行」跳过房间！
        # -------------------------------------------------------------
        from ..core.features import detect_treasure_chests
        chests = detect_treasure_chests(raw_img)
        if chests:
            click_cnt = getattr(self, "_chest_click_count", 0)
            target_idx = click_cnt % len(chests)
            cx, cy, sc = chests[target_idx]
            self._chest_click_count = click_cnt + 1

            logger.info(f"【宝物库】发现场景中存在 {len(chests)} 个未开启宝箱 (置信度={sc:.2f}) -> "
                        f"优先点击第 {target_idx + 1} 个宝箱 (坐标: {cx:.3f}, {cy:.3f}) 开启战利品...")
            self._record(state, "safety:open_treasure_chest")
            if getattr(self.ctx, "overlay", None):
                bw = int(120 * (raw_img.shape[0] / 576.0))
                bh = int(70 * (raw_img.shape[0] / 576.0))
                px, py = int(cx * raw_img.shape[1]), int(cy * raw_img.shape[0])
                self.ctx.overlay.show_boxes([{
                    'bbox': (px - bw // 2, py - bh // 2, bw, bh),
                    'label': f"【宝箱】开启宝箱 #{target_idx + 1}",
                    'click': (px, py),
                    'color': '#00FFFF'
                }], duration=1.2)
            self.ctx.input.click_rel(cx, cy, delay_after=1.8, label=f"【宝物库】开启宝箱 #{target_idx + 1}")
            return True
        else:
            self._chest_click_count = 0

        # -------------------------------------------------------------
        # Universal Safety Check:「繼續進行」门（肉鸽里的"继续前进"提示）
        # 真机证据（主人 2026/10/6 截图）：肉鸽里一扇门 + 右下角橙色大按钮 `ENTER 繼續進行`。
        # OCR 把「繼續」读成了「缆」（`缆進行`），所以这里用**短子串 + 区域限定**：
        # 右下角出现含 `進行`/`继续`/`繼續` 的文字 -> 按 ENTER。
        #
        # ⚠️ 教训（主人立刻纠正："不能看到了 enter 就去点，搞麻了"）：
        #   **不能**写成"右下角有 ENTER 字样就按 ENTER" —— 很多场景（交互点、探索提示）
        #   右下角都有个通用 `⚔ ENTER` 图标，照着按会抢掉真正该做的动作（比如 `① 注視…`）。
        #   所以只认这条**具体文案**。
        # -------------------------------------------------------------
        try:
            proceed_hits = [t for t in (ocr_targets or [])
                            if any(w in (getattr(t, "text", "") or "") for w in ("進行", "进行", "繼續", "继续"))
                            and not any(neg in (getattr(t, "text", "") or "") for neg in ("進行中", "进行中", "進行度", "进行度", "條約", "条约", "協議", "协议", "任務", "任务"))
                            and t.center_rel[0] >= 0.55 and t.center_rel[1] >= 0.70]
        except Exception:  # noqa: BLE001
            proceed_hits = []
        if proceed_hits:
            logger.info(f"【按键提示】右下角是「繼續進行」按钮（{proceed_hits[0].text!r}）-> 按 ENTER 前进...")
            self._record(state, "safety:proceed_button")
            self.ctx.input.tap_action("confirm", delay_after=1.5, label="【肉鸽前进】繼續進行(ENTER)")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check:「還有可用的卡牌…是否要結束這回合?」确认弹窗
        # 真机证据（主人 2026/10/6 截图）：按 E 结束回合时，如果手牌里还有能用的牌，
        # 游戏会弹这个确认框：`[ESC] 取消` / `[ENTER] 確認`。
        # 我们已经决定结束回合 -> 直接按 **ENTER 確認**（否则会一直停在这个弹窗上）。
        # -------------------------------------------------------------
        if any(w in text_corpus for w in ["結束這回合", "结束这回合", "不使用它們", "不使用它们",
                                          "還有可用的卡牌", "还有可用的卡牌"]):
            logger.info("【战斗】检测到「是否結束這回合」确认弹窗 -> 按 ENTER 確認结束回合...")
            self._record(state, "safety:end_turn_confirm")
            self.ctx.input.tap_action("confirm", delay_after=1.5, label="【战斗】確認结束回合(ENTER)")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: 局内战斗攻击确认按钮（⚔ 攻擊 / ENTER）
        # 真机证据（主人 2026/10/11 截图 media_1791711090058_fab13cef.jpg）：
        #   "有时候进入战斗了已经是激活的R的状态，但是应该要点一下屏幕的，上面有一个攻击按钮，是enter按键"
        #   战斗中手牌上方中央弹出「⚔ 攻擊」椭圆胶囊框，底部带有「ENTER」按钮提示 (rx: 0.500, ry: 0.858)。
        #   处于该状态时，即便原生 AUTO(R) 已激活，游戏仍暂停等待玩家发起攻击！
        # 动作：
        #   发送 ENTER 确认键发起攻击，并辅以点击中央【攻擊】胶囊框 (0.500, 0.825)！
        # -------------------------------------------------------------
        from ..logic.battle_manager import BattleManager as _BM
        if _BM.is_attack_enter_screen(raw_img, state):
            logger.info("【战斗攻击】检测到屏幕中央「⚔ 攻擊 / [ENTER]」攻击确认按钮 -> 发送 ENTER 并点击发起攻击...")
            self._record(state, "safety:battle_attack_enter")
            self.ctx.input.tap_action("confirm", delay_after=0.8, label="【战斗】ENTER攻击")
            self.ctx.input.click_rel(0.500, 0.825, delay_after=1.0, label="【战斗】点击攻擊按钮")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: 底部中央的橙色大按钮（弹窗的【關閉】/【確認】）
        # 例：「查看詳情」卡牌详情弹窗 —— 它没有右上角 [X]，只有一个橙色【關閉】，
        # 而且 OCR 实测**完全认不出**「關閉」两个字，只能靠视觉识别。
        # 只认底部中央 (x 0.30~0.70)，右下角的【確認】/【回合結束】不在此范围，不会误伤。
        # ⚠️ 注意：若处于 3 卡牌等距布局或灵光一闪选牌界面（背景眼球火光会产生橙色像素），
        # 绝不能当成弹窗关闭按钮误点，必须放行至正常选牌逻辑！
        # -------------------------------------------------------------
        from ..core.features import card_layout
        has_3cards_layout = bool(card_layout(ocr_targets).get('ok', False))
        is_epiphany_screen = (
            any(w in text_corpus for w in ["靈光一閃", "灵光一闪", "靈光一", "靈光"])
            and any(w in text_corpus for w in ["查看內容", "查看内容", "CAPS", "效果選擇", "效果选择"])
            and not any(w in text_corpus for w in ["查看詳情", "查看祥情", "詳情", "详情"])
        )
        if not (has_3cards_layout or is_epiphany_screen):
            h_img, w_img = raw_img.shape[:2]
            strip = raw_img[int(0.88 * h_img):int(0.98 * h_img), int(0.30 * w_img):int(0.70 * w_img)]
            if strip.size > 0:
                hsv_s = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
                mask_o = cv2.inRange(hsv_s, np.array([8, 120, 120]), np.array([25, 255, 255]))
                if np.count_nonzero(mask_o) >= 8000:
                    ys, xs = np.nonzero(mask_o)
                    if (xs.max() - xs.min()) >= int(0.10 * w_img):     # 必须是"横向长条"按钮，不是零散色块
                        cx = int(0.30 * w_img + xs.mean())
                        cy = int(0.88 * h_img + ys.mean())
                        logger.info(f"【弹窗关闭】检测到底部中央橙色大按钮（關閉/確認），点击退回上一层 (坐标: {cx}, {cy})...")
                        self._record(state, "safety:bottom_center_button")
                        self.ctx.input.click_client(cx, cy, delay_after=1.5, label="【弹窗】關閉/確認")
                        return True

        # -------------------------------------------------------------
        # Universal Safety Check:「點擊畫面可關閉視窗」/ 底部中央 [ENTER] 关闭视窗弹窗
        # 真机证据（主人 2026/10/11 截图与 fixtures_auto 深度取证）：
        #   1. 战前灵光一闪卡牌预览：「靈光一閃卡牌」+ 底部「點擊畫面可關閉視窗。ENTER」
        #   2. 局内分析/突发事件结果：研究技术弹窗 + 底部「點擊畫面可關閉視窗。ENTER」
        # 原先因未识别此弹窗，导致在此处原地等待（卡死守卫从 stuck=4 一直升到 stuck=10 后才按 ESC/关闭脱困）。
        # 动作：
        #   立即按 ENTER 确认键（官方快捷键），辅以点击屏幕中央任意处，0.5秒干净关闭弹窗并推进！
        # -------------------------------------------------------------
        has_close_window_text = any(
            any(w in (getattr(t, "text", "") or "") for w in ["關閉視窗", "关闭视窗", "可關閉", "可关闭", "點擊畫面", "点击画面", "點擎", "關朋"])
            and t.center_rel[1] >= 0.85
            for t in (ocr_targets or [])
        ) or any(w in text_corpus for w in ["關閉視窗", "关闭视窗", "點擊畫面可關閉", "点击画面可关闭"])

        has_bottom_enter_center = any(
            (getattr(t, "text", "") or "").strip().upper() == "ENTER"
            and 0.40 <= t.center_rel[0] <= 0.60 and t.center_rel[1] >= 0.92
            for t in (ocr_targets or [])
        )

        if has_close_window_text or has_bottom_enter_center:
            logger.info("【弹窗推进】检测到「點擊畫面可關閉視窗 / [ENTER]」提示 -> 立即发送 ENTER 并点击屏幕推进...")
            self._record(state, "safety:click_to_close_modal")
            self.ctx.input.tap_action("confirm", delay_after=0.8, label="【弹窗】ENTER关闭")
            self.ctx.input.click_rel(0.50, 0.50, delay_after=0.8, label="【弹窗】点击屏幕任意处")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: 掷骰检定画面（Tap / 點擊畫面查看骰子結果 / 擲骰影響）
        # 真机证据（主人 2026/10/11 截图 media_1791693902093_ad3a2d11.png）：
        #   画面中央为一个 D20 骰子 (rx: 0.50, ry: 0.46)，提示文案：
        #   `Tap`、`請點擊畫面查看骰子結果`、`ENTER`、`擲骰影響+3`。
        #   OCR 可能会把 骰 读成 般 或 相。
        # 动作：
        #   立即发送 ENTER 确认键（官方快捷键直接掷骰），并辅以点击中央骰子 (0.50, 0.46)，瞬间完成掷骰！
        # -------------------------------------------------------------
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
            logger.info("【掷骰检定】检测到局内掷骰判定画面 -> 立即按 ENTER 并点击骰子完成检定...")
            self._record(state, "safety:dice_roll")
            self.ctx.input.tap_action("confirm", delay_after=0.6, label="【掷骰】ENTER确认")
            self.ctx.input.click_rel(0.50, 0.46, delay_after=1.2, label="【掷骰】点击中央骰子")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: 移除卡牌事件画面（請選擇1張欲移除的卡牌 / 德朗商店删牌）
        # 真机证据（主人 2026/10/11 截图 media_1791694012716_2a78cac9.png）：
        #   画面显示「請選擇1張欲移除的卡牌。」，卡牌排列展示（左上角为基础攻击牌），
        #   右下角有「F1 跳過」和「ENTER 移除」。
        # ⚠️ 反否决注意：
        #   「卡牌獎勵 / 請選擇欲領取卡牌的戰鬥員」界面（真机 media_1791726996655_dfd87921.png），
        #   卡牌描述正文可能包含「[ 移除 ]」机制说明，右下角带有 [F1 跳過] 与 [ENTER 確認]，
        #   绝不是删牌界面！必须严格反否决，避免误触卡牌而未选战斗员导致卡死。
        # 动作：
        #   根据肉鸽删牌策略（精简牌组删除初始基础攻击牌）：
        #   1. 点击左上角第 1 张基础卡牌（长剑斩击 / 基本攻击）(rx: 0.35, ry: 0.30)
        #   2. 发送 ENTER 键（官方：ENTER 移除）或点击右下角【移除】(0.88, 0.90)
        #   如果配置明确禁用删牌 (shop_auto_delete_card == False)，则按 F1 跳过。
        # -------------------------------------------------------------
        is_card_reward_assign_screen = any(w in text_corpus for w in [
            "卡牌獎勵", "卡牌奖励", "卡牌樊", "卡牌奖",
            "欲領取", "欲领取", "領取卡牌", "领取卡牌", "欲取卡牌",
            "戰鬥員", "战斗员", "載門", "载门"
        ])
        is_remove_card_screen = (not is_card_reward_assign_screen) and (
            any(w in text_corpus for w in ["欲移除的卡牌", "欲移除", "移除的卡牌", "移除卡牌", "移除1張", "移除1张", "移除1"]) or (
                ("移除" in text_corpus or "欲移除" in text_corpus)
                and any(w in text_corpus for w in ["F1", "跳過", "跳过"])
                and any(w in (getattr(t, "text", "") or "") for t in (ocr_targets or [])
                        if t.center_rel[0] >= 0.80 and t.center_rel[1] >= 0.85 and "移除" in t.text)
            )
        )

        if is_remove_card_screen:
            auto_delete = self.config.get("path_planning", {}).get("shop_auto_delete_card", True)
            if not auto_delete:
                logger.info("【卡牌移除】根据配置已禁用自动删牌 -> 发送 F1 跳过删牌...")
                self._record(state, "safety:card_removal_skip")
                self.ctx.input.send_key(win32con.VK_F1, delay_after=1.5, label="【删牌】F1跳过")
                return True

            logger.info("【卡牌移除】检测到移除卡牌界面 -> 点击选中第 1 张基础卡牌 (0.35, 0.30)...")
            self._record(state, "safety:card_removal")
            self.ctx.input.click_area_rel(0.30, 0.22, 0.40, 0.38, delay_after=0.6, label="【删牌】选中第1张卡牌")
            time.sleep(0.3)
            logger.info("【卡牌移除】按 ENTER 确认移除该卡牌（官方：ENTER 移除）...")
            if not self.ctx.input.tap_action("confirm", delay_after=1.5, label="【删牌】ENTER确认移除"):
                self.ctx.input.click_area_rel(0.84, 0.88, 0.96, 0.94, delay_after=1.5, label="【删牌】点击移除按钮")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: 补充战利品认证卡牌弹窗 (0/4 凭证不足进入时的二次确认)
        # 真机证据 (20261011-142934_BATTLE_肉鸽选路.png):
        #   标题: `補充戰利品認證卡牌` (OCR 可能认成 `補充戟利品懿證卡牌`)
        #   正文: `消耗60個埃忒耳，補充1個戰利品認證卡牌`
        #   左键: `取消` (rx: 0.349, ry: 0.762)
        #   右键: `確認` (橙色大按钮, rx: 0.621, ry: 0.762 或 ENTER)
        # 动作:
        #   若配置 use_travel_ticket 为 True (默认)：按 ENTER 确认补充 (或点击 0.621, 0.762)
        #   若为 False：按 ESC 取消 (或点击 0.349, 0.762)
        # -------------------------------------------------------------
        is_voucher_refill = any(w in text_corpus for w in [
            "補充戰利品", "补充战利品", "補充戟利品", "补充戟利品",
            "戰利品認證卡牌", "战利品认证卡牌", "戟利品懿證卡牌", "战利品认证", "戰利品認證",
            "消耗60個埃忒耳", "消耗60个埃忒耳", "埃忒耳"
        ]) and any(w in text_corpus for w in ["補充", "补充", "卡牌", "取消", "確認", "确认", "0/4"])

        if is_voucher_refill:
            use_ticket = self.config.get("archive", {}).get("use_travel_ticket", True)
            if use_ticket:
                logger.info("【凭证补充】检测到战利品凭证补充弹窗，按配置确认消耗埃忒耳补充 (ENTER / 0.621, 0.762)...")
                self._record(state, "safety:voucher_refill_confirm")
                self.ctx.input.tap_action("confirm", delay_after=1.2, label="【凭证】ENTER确认补充")
                self.ctx.input.click_rel(0.621, 0.762, delay_after=1.5, label="【凭证】点击橙色确认")
            else:
                logger.info("【凭证补充】检测到战利品凭证补充弹窗，按配置取消补充 (ESC / 0.349, 0.762)...")
                self._record(state, "safety:voucher_refill_cancel")
                self.ctx.input.tap_action("cancel", delay_after=1.2, label="【凭证】ESC取消补充")
                self.ctx.input.click_rel(0.349, 0.762, delay_after=1.5, label="【凭证】点击取消")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: 局内战斗卡牌发现/产生/检索选择弹窗
        # 真机证据（主人 2026/10/11 截图 media_1791702494094_2f4759ef.png / stuck4~7 夹具）：
        #   战斗中打出特定卡牌（如定位雷射、生成卡牌等）触发了选牌弹窗：
        #   顶部提示：「請選擇最多1張欲產生的卡牌。」（OCR 可能读成「摆最多1张欲產生的卡牌」或「暹撑最多1张...」）
        #   计数器：「0 / 1」或「0/1」
        #   卡牌排列：中央水平并列 1~5 张卡牌（定位雷射），卡牌费用 0/1
        #   右下角：「ENTER 選擇」（橙色大按钮，rx: 0.891, ry: 0.931）
        #   右上角：「隱藏 [U]」（rx: 0.916, ry: 0.132）
        # 动作：
        #   1. 根据 OCR 目标识别或聚类出第 1 张卡牌（或按最多数量顺次点击），默认点击第 1 张 (rx: 0.12, ry: 0.44)
        #   2. 短暂等待 0.3s 后，按 ENTER 确认（官方：ENTER 選擇），并辅以点击右下角橙色【選擇】按钮 (0.891, 0.931)
        # -------------------------------------------------------------
        # 排除装备重铸、装备/卡牌奖励分配、凭证补充等（绝非局内战斗卡牌发现）
        is_reforge_or_assign = any(w in text_corpus for w in [
            "重鑄", "重铸", "重裝備", "重装备", "重新整理", "欲套用",
            "120pt", "pt", "請選擇裝備", "请选择装备", "選擇裝備", "选择装备",
            "請選擇戰鬥員", "请选择战斗员", "選擇戰鬥員", "选择战斗员",
            "欲領取", "欲领取", "欲取卡牌", "撐欲取", "撑欲取",
            "0/4", "補充", "补充", "凭证", "憑證"
        ])

        is_combat_card_discover = (not is_reforge_or_assign) and (
            any(w in text_corpus for w in [
                "欲產生", "欲产生", "欲生成", "欲保留", "欲檢索", "欲检索",
                "欲丟棄", "欲丢弃", "欲消耗", "欲抽取",
                "產生的卡牌", "产生的卡牌", "生成的卡牌", "保留的卡牌", "檢索的卡牌", "检索的卡牌"
            ]) or (
                any(w in text_corpus for w in ["最多", "最少"]) and any(w in text_corpus for w in ["卡牌", "欲產生", "欲生成", "欲檢索"])
            )
        ) and any(w in text_corpus for w in ["ENTER", "選擇", "选择", "確認", "确认", "隱藏", "隐藏"])

        if is_combat_card_discover:
            logger.info("【战斗选牌】检测到局内卡牌产生/检索选择弹窗（请选择欲产生的卡牌）...")
            self._record(state, "safety:combat_card_discover")

            need_count = 1
            m_count = re.search(r'最多\s*(\d+)\s*[张張]', text_corpus) or re.search(r'0\s*/\s*([1-5])\b', text_corpus)
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

        # -------------------------------------------------------------
        # Universal Safety Check: 命运选择画面（選擇您的命運。 / 3选1命运卡牌）
        # 真机证据（主人 2026/10/11 截图 media_1791697303809_b18d8ceb.png）：
        #   画面显示「選擇您的命運。」，三张大卡牌（1: 勤勞, 2: 共情, 3: 開朗）等距并排展示，
        #   卡牌下方有 1/2/3 编号指示，右下角为【ENTER 確認】和【F1 ▷|】。
        # 动作：
        #   按主人规则：三选一选牌，优先选第 1 张牌（官方编号键 1，或点击 rx: 0.255, ry: 0.50），
        #   短暂等待后按 ENTER 确认（官方：ENTER 確認）推进！
        # -------------------------------------------------------------
        is_fate_select_screen = (
            not any(w in text_corpus for w in ["零號系統", "零号系统", "零号系統", "選擇法典", "选择法典", "撐法典", "撑法典"])
            and (
                any(w in text_corpus for w in ["選擇您的命運", "选择您的命运", "您的命運", "您的命运", "選擇命運", "选择命运", "欲適用的命運", "欲适用的命运"])
                or (any(w in text_corpus for w in ["命運卡牌", "命运卡牌", "命運效果", "命运效果"]) and any(w in text_corpus for w in ["ENTER", "F1", "AP1", "確認", "确认"]))
            )
        )

        if is_fate_select_screen:
            logger.info("【命运选择】检测到「選擇您的命運」3选1界面 -> 优先选择第 1 张命运卡牌 (按键1)...")
            self._record(state, "safety:fate_selection")
            # 键盘优先：按 1 选中第 1 张命运卡牌，辅以点击第 1 张卡牌中心 (0.255, 0.500)
            self.ctx.input.tap_card(1, delay_after=0.4, label="【命运选择】第1张卡牌(按键1)")
            self.ctx.input.click_area_rel(0.18, 0.35, 0.33, 0.65, delay_after=0.4, label="【命运选择】点击第1张卡牌")
            time.sleep(0.3)
            logger.info("【命运选择】已选中第 1 张命运卡牌，按 ENTER 确认选择...")
            if not self.ctx.input.tap_action("confirm", delay_after=1.2, label="【命运选择】ENTER确认"):
                self.ctx.input.click_area_rel(0.84, 0.88, 0.96, 0.95, delay_after=1.2, label="【命运选择】点击确认按钮")
            return True

        # -------------------------------------------------------------
        # Universal Safety Check: 梦境模式 (深层创伤启动 -> 点击【治療中心】)
        # 真机证据（主人 2026/10/11 截图 media_1791713114494_e1bdcd8a.png）：
        #   战局失败后再次进入零号系统，由于战斗员处于「深層創傷」启动了梦境模式，
        #   主人规则："如果战局失败，再次进入刷0号系统的时候出现中显示深层创伤，有治疗中心的字，就先去治疗"
        #   动作：点击左下角【治療中心】进入治疗室，绝不点击右下角【確認】！
        # -------------------------------------------------------------
        is_dream_mode_screen = (
            any(w in text_corpus for w in ["夢境模式", "梦境模式", "啟動夢境模式", "启动梦境模式"])
            or (any(w in text_corpus for w in ["深層創傷", "深层创伤", "深创傷", "深唇創傷"]) and any(w in text_corpus for w in ["治療中心", "治疗中心"]))
        )
        if is_dream_mode_screen:
            logger.info("【梦境模式】检测到深层创伤 / 梦境模式提示 -> 点击左下角【治療中心】进行治疗...")
            self._record(state, "safety:dream_mode")
            if self.story_pusher and self.story_pusher.handle_dream_mode(state, raw_img=raw_img):
                return True

        # -------------------------------------------------------------
        # Universal Safety Check: 治疗中心 (选择受创战斗员并消除记忆)
        # 真机证据（主人 2026/10/11 截图 media_1791713173059_1edd98fe.png）：
        #   主人规则："治疗记得先选角色，然后点消除记忆，因为战局失败的话，其实我们不保存，因为肯定卡片数量不够，不要了"
        # 净化 text_corpus，排除 bot 自身 HUD 徽章文本（防 OCR 自激死循环）
        clean_corpus = " ".join(
            t for t in (text_corpus or "").split(" ")
            if "路由" not in t and not (t.startswith("【") and ("分" in t or "%" in t or "％" in t))
        )

        is_treatment_screen = (
            (any(w in clean_corpus for w in ["治療中心", "治疗中心"]) and any(w in clean_corpus for w in ["消除記憶", "消除记忆", "諮商", "咨询", "深層創傷", "深层创伤", "戰鬥員", "战斗员"]))
            or any(w in clean_corpus for w in ["損壞資料", "损坏资料", "即將消滅", "即将消灭", "無法復原", "无法复原", "以解除深層創傷", "以解除深层创伤"])
            or (any(w in clean_corpus for w in ["消除記憶", "消除记忆"]) and any(w in clean_corpus for w in ["諮商", "咨询", "艦長", "舰长", "取消", "費用"]))
        )
        if is_treatment_screen:
            logger.info("【治疗中心】检测到治疗中心界面 -> 进入消除记忆治疗流程...")
            self._record(state, "safety:treatment_center")
            if self.story_pusher and self.story_pusher.handle_treatment_center(state, raw_img=raw_img):
                return True

        # -------------------------------------------------------------
        # 1. Dynamic Pool Classification
        # -------------------------------------------------------------
        # L1「文字锁定」优先：稳定 UI 文字（例如配队界面的 `配置隊伍`）直接定案，
        # 不参加打分池竞争 —— 真机上它就是因为去竞争才被"局内互动事件/肉鸽选路(各65)"抢走。
        locked = text_locked_scene(clean_corpus)
        if locked is not None:
            res = SceneResult(scene=locked[0], confidence=100.0, second_scene=SceneType.UNKNOWN,
                              second_score=0.0, margin=100.0, all_scores={},
                              details={"text_lock": locked[1]})
            logger.info(f"【文字锁定】{locked[1]} -> 直接判定 [{locked[0].value}]")
        else:
            res = self.classifier.classify(raw_img, clean_corpus, ocr_targets)
        self._reconcile_screen(state, res)

        logger.info(f"【场景路由执行】-> [{res.scene.value}] (置信度={res.confidence:.1f}, 领先分差={res.margin:.1f})")

        # -------------------------------------------------------------
        # 2. Scene Handler Routing
        # -------------------------------------------------------------
        if res.scene == SceneType.ROGUELIKE_MAP:
            logger.info("【流程进度 -> 肉鸽选路】已到达肉鸽路线选择界面，正在长线规划最优路线...")
            planner = self.path_planner or (self.story_pusher.path_planner if self.story_pusher else None)
            if planner:
                return planner.execute_branch_selection(self.ctx, raw_img)

        elif res.scene == SceneType.EVENT_CHOICE:
            logger.info("【流程进度 -> 局内事件】触发奇遇互动对话框，正在选择事件选项...")
            engine = self.node_engine or (self.story_pusher.node_engine if self.story_pusher else None)
            if engine:
                return engine.handle_event(state, raw_img=raw_img)

        elif res.scene == SceneType.EPIPHANY_HERO:
            engine = self.node_engine or (self.story_pusher.node_engine if self.story_pusher else None)
            if engine:
                logger.info("【流程进度 -> 灵光一闪选人】进入选择灵光一闪战斗员界面，按规则选择第 1 位战斗员并确认...")
                return engine.handle_epiphany_hero_selection(state, raw_img=raw_img)

        elif res.scene == SceneType.EPIPHANY_CARDS:
            engine = self.node_engine or (self.story_pusher.node_engine if self.story_pusher else None)
            if engine:
                logger.info("【流程进度 -> 灵光一闪选牌】进入 3选1 选牌界面，按规则选择推荐卡牌...")
                engine.handle_epiphany_selection(state, raw_img=raw_img)
                return True

        elif res.scene == SceneType.BATTLE:
            logger.info("【流程进度 -> 局内战斗】识别到局内战斗进行中，优先确保原生自动战斗激活...")
            if self.story_pusher:
                return self.story_pusher.handle_battle(state, raw_img=raw_img)
            elif self.battle_mgr:
                return self.battle_mgr.play_round(state, raw_img=raw_img)

        elif res.scene == SceneType.CAMP_SHOP:
            engine = self.node_engine or (self.story_pusher.node_engine if self.story_pusher else None)
            if engine:
                logger.info("【流程进度 -> 营地与商店】进入营地安全区，执行自动免费休息与补给...")
                return engine.handle_camp(state, raw_img=raw_img)

        elif res.scene == SceneType.EQUIP_ASSIGN:
            engine = self.node_engine or (self.story_pusher.node_engine if self.story_pusher else None)
            if engine:
                logger.info("【流程进度 -> 获得装备】触发装备分配，执行战斗员穿戴确认...")
                engine.handle_equip_selection(state, raw_img=raw_img)
                return True

        elif res.scene == SceneType.STAGE_MAP:
            logger.info("【流程进度 -> 关卡大地图】正在扫描可推进的新关卡节点...")
            if self.story_pusher:
                return self.story_pusher.handle_stage_map(state, raw_img=raw_img)

        elif res.scene == SceneType.CUTSCENE_DIALOG:
            if self.story_pusher:
                return self.story_pusher.handle_cutscene_dialog(state, raw_img=raw_img)
            else:
                logger.info("【流程进度 -> 剧情对白】推进对白台词...")
                self.ctx.input.click_rel(0.85, 0.85, delay_after=1.0)
                return True

        elif res.scene == SceneType.SETTLEMENT_LOOT:
            logger.info("【流程进度 -> 战利品结算】关卡/战斗已完成，正在确认领取战利品...")
            if self.story_pusher:
                return self.story_pusher.handle_settlement(state, raw_img=raw_img)
            elif self.battle_mgr:
                return self.battle_mgr.check_battle_ended(state, raw_img=raw_img)

        elif res.scene == SceneType.HOME_LOBBY:
            logger.info("【流程进度 -> 游戏大厅】检测到主界面大厅，执行进入卡厄思模式（发送快捷键 4 + 点击卡厄思）...")
            self.ctx.input.send_key(0x34, delay_after=0.8, label="【大厅】按4进入卡厄思")
            self.ctx.input.click_area_rel(0.86, 0.40, 0.92, 0.48, delay_after=1.8, label="【大厅】点击卡厄思卡片")
            return True

        elif res.scene == SceneType.CHAOS_SELECT:
            logger.info("【流程进度 -> 卡厄思分支】检测到分支选择界面，点击右侧【零號系統】...")
            zero_target = None
            if ocr_targets:
                zero_target = next((t for t in ocr_targets if any(k in getattr(t, 'text', '') for k in ['零號系統', '零号系统', '零號', '零号']) and getattr(t, 'center_rel', (0, 0))[0] > 0.50), None)
            if zero_target:
                self.ctx.input.click_target(zero_target, delay_after=2.0, label="【卡厄思分支】零號系統(OCR)")
            else:
                self.ctx.input.click_area_rel(0.62, 0.65, 0.76, 0.84, delay_after=2.0, label="【卡厄思分支】点击零號系統卡片")
            return True

        elif res.scene == SceneType.SYSTEM_ZERO_LOBBY:
            logger.info("【流程进度 -> 零号系统大厅】锁定右侧【法典】球形图标，点击进入法典选择...")
            codex_target = None
            if ocr_targets:
                codex_target = next((t for t in ocr_targets if '法典' in getattr(t, 'text', '') and getattr(t, 'center_rel', (0, 0))[0] > 0.70), None)
            if codex_target:
                self.ctx.input.click_target(codex_target, delay_after=2.0, label="【零号系统】法典图标(OCR)")
            else:
                self.ctx.input.click_area_rel(0.80, 0.42, 0.90, 0.52, delay_after=2.0, label="【零号系统】点击法典图标")
            return True

        elif res.scene == SceneType.CODEX_SELECT:
            target_codex = int(self.config.get("system_zero", {}).get("target_codex", 1))
            try:
                import json
                from pathlib import Path
                cfg_path = Path(__file__).resolve().parent.parent / "config.json"
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        disk_cfg = json.load(f)
                    target_codex = int(disk_cfg.get("system_zero", {}).get("target_codex", target_codex))
            except Exception:
                pass
            codex_coords = {
                1: (0.098, 0.373),
                2: (0.098, 0.582),
                3: (0.098, 0.781),
            }
            c_pos = codex_coords.get(target_codex, (0.098, 0.373))
            logger.info(f"【流程进度 -> 法典选择】锁定配置目标: 0{target_codex} 号法典，点击选中 (坐标: {c_pos[0]:.3f}, {c_pos[1]:.3f})...")
            self.ctx.input.click_area_rel(c_pos[0] - 0.025, c_pos[1] - 0.035, c_pos[0] + 0.025, c_pos[1] + 0.035,
                                          delay_after=0.8, label=f"【法典选择】0{target_codex}号法典")
            logger.info(f"【流程进度 -> 发起挑战】已选中 0{target_codex} 号法典，点击右下角【進入】按钮进入关卡...")
            self.ctx.input.click_area_rel(0.86, 0.86, 0.96, 0.94, delay_after=2.5, label="【法典选择】進入")
            return True

        elif res.scene == SceneType.DREAM_MODE:
            logger.info("【流程进度 -> 梦境模式】进入梦境模式处理 -> 点击左下角【治療中心】...")
            if self.story_pusher:
                return self.story_pusher.handle_dream_mode(state, raw_img=raw_img)
            return True

        elif res.scene == SceneType.TREATMENT_CENTER:
            logger.info("【流程进度 -> 治疗中心】进入治疗中心处理 -> 消除受创战斗员记忆...")
            if self.story_pusher:
                return self.story_pusher.handle_treatment_center(state, raw_img=raw_img)
            return True

        # -------------------------------------------------------------
        # 3. Fallback for UNKNOWN / Interstitial Transition
        # -------------------------------------------------------------
        # Check if orange leave / obtain button is present
        engine = self.node_engine or (self.story_pusher.node_engine if self.story_pusher else None)
        if engine and engine.check_and_click_leave_button(raw_img):
            return True

        # 零号系统自愈仅在外层非局内大厅菜单（如误点入图鉴、纪录室等）时才返回首页；
        # 严禁在局内/战斗中按 ESC 或点左上角返回（主人明确要求：局内绝不擅自退出）
        is_lobby_submenu = any(w in text_corpus for w in ["紀錄室", "记录室", "備援", "备援", "圖鑑", "图鉴", "郵件", "邮件", "任務", "任务"])
        in_dungeon_signs = any(w in text_corpus for w in ["EP", "回合", "TURN", "連鎖", "连锁", "生命值", "卡牌", "裝備", "遺物", "傷害", "護盾", "骰", "Tap", "移除", "探索", "事件", "休息", "商店", "安全區", "戰鬥", "BOSS", "首領"])

        if self.config.get("mode") == "system_zero" and is_lobby_submenu and not in_dungeon_signs:
            logger.info("【零号系统大厅自愈】当前处于大厅次级菜单，执行退回首页（ESC / 点击左上角[<]）...")
            self.ctx.input.click_area_rel(0.03, 0.03, 0.065, 0.07, delay_after=0.8, label="【自愈】左上角返回[<]")
            self.ctx.input.tap_action("cancel", delay_after=1.2, label="【自愈】返回首页(ESC)")
            return True

        logger.info("【未知巡航】画面处于过渡状态，点击右下角推进...")
        self.ctx.input.click_rel(0.85, 0.85, delay_after=1.2)
        return True

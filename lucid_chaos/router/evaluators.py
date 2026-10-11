"""
Multi-modal Scene Evaluators for Dynamic Feature Scoring Pool.
Each evaluator computes positive evidence (+) and negative vetoes (-)
across OCR tokens, OpenCV templates, color masks, and geometric layouts.
Scores are normalized from 0.0 to 100.0.
"""

import cv2
import re
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
from .scene_types import SceneType
from ..core.template_bank import BANK
from ..core.features import (card_layout, find_button, bright_ratio, group_rows,
                             find_horizontal_bars, edge_density, hsv_ratio_multi,
                             blank_stats, count_game_texts)

class SceneEvaluators:
    @classmethod
    def _get_template(cls, rel_path: str) -> Optional[np.ndarray]:
        """统一走全项目共用的模板库（原来是本类自己的一份缓存）。"""
        from ..core.template_bank import BANK
        return BANK.get(rel_path)

    # -------------------------------------------------------------
    # 1. ROGUELIKE_MAP (内部肉鸽选路)
    # -------------------------------------------------------------
    @classmethod
    def eval_roguelike_map(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        h, w = img_bgr.shape[:2]
        scale = h / 576.0

        # Feature 1: Mini-map in bottom-left [0.62*h:0.98*h, 0.01*w:0.42*w]
        # 取「节点模板里的最高分」，不要只看 SWORD 一个：
        # 实测某张选路图上 SWORD 只打到 0.633（差 0.017 没过 0.65 阈值），
        # 而同一位置的 QUESTION 能打到 0.731 —— 单模板依赖会把整个界面判成"未知画面"。
        crop_map = img_bgr[int(0.62 * h):int(0.98 * h), int(0.01 * w):int(0.42 * w)]
        best_map, best_map_name = 0.0, ""
        # 先算"青色面板占比"（1~2ms），用它当**便宜的预筛**：
        # 小地图面板本身就是青色调的（H/S 不受整体压暗影响，实测亮/暗两张都是 3.5%~5.8%），
        # 面板都不存在时没必要去跑 4 个大模板（590x438，合计约 264ms）。
        #
        # ⚠️ 关键区分（真机踩坑）：**面板是"暗青"，角色美术是"亮青"**。
        # 实测 5.32% 青色的结算界面（青发角色 + 青色光环装饰）被误判成肉鸽选路 100 分：
        #   青发结算图：暗青(V<150) 0.89% / 亮青(V>=150) 4.44%
        #   三张真选路图：暗青 4.54% / 53.72% / 57.49%，亮青 0.55% / 0.55% / 1.54%
        # 所以只认**暗青**（面板底色的明度区间），亮青（头发/光效）不算。
        cyan_ratio = 0.0            # 暗青占比（判定用）
        cyan_bright = 0.0           # 亮青占比（诊断用）
        if crop_map.size > 0:
            hsv_map = cv2.cvtColor(crop_map, cv2.COLOR_BGR2HSV)
            m_dark = cv2.inRange(hsv_map, np.array([80, 60, 35]), np.array([105, 255, 150]))
            m_bright = cv2.inRange(hsv_map, np.array([80, 60, 151]), np.array([105, 255, 255]))
            cyan_ratio = float(np.count_nonzero(m_dark)) / float(m_dark.size)
            cyan_bright = float(np.count_nonzero(m_bright)) / float(m_bright.size)
        panel_present = cyan_ratio >= 0.015      # 实测选路图 >=4.5%，青发结算图 0.89%，其它 <=0.22%
        if panel_present and crop_map.size > 0:
            for tname in ("rogue_sword.png", "rogue_question.png", "rogue_fire.png", "rogue_fire_shop.png"):
                rel = str(Path("rogue") / tname)
                tpl = cls._get_template(rel)
                if tpl is None:
                    continue
                ts = BANK.scaled(rel, 0.22 * scale)      # 同一份缩放缓存
                if ts is None:
                    continue
                if crop_map.shape[0] < ts.shape[0] or crop_map.shape[1] < ts.shape[1]:
                    continue
                res = cv2.matchTemplate(crop_map, ts, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_map:
                    best_map, best_map_name = float(max_val), tname
            if best_map >= 0.65:
                score += 65.0

        # Feature 1b: 青色小地图面板占比。
        # 画面被压暗（进出房间的渐变蒙版）时模板分数会整体下滑（实测 question 0.731 -> 0.632，
        # 刚好掉到 0.65 阈值以下），但 HSV 的 H/S 基本不受整体压暗影响，用「青色面板占比」很稳。
        # 实测：选路图 5.31% / 5.79%，其它界面 <=1.79%（卡牌奖励选战斗员）-> 阈值取 3.0%
        # （cyan_ratio 已在上面算好，这里直接复用，不再重复 cvtColor/inRange）
        if cyan_ratio >= 0.03 and score == 0.0:
            score += 65.0

        # Feature 2: 3D scene cyan arrows on right
        # 预筛：青色箭头只出现在选路界面上（右侧 3D 房间预览里），
        # 面板青色占比过低时直接跳过这次 194ms 的全区域匹配。
        tpl_arrow = cls._get_template(str(Path("rogue") / "cyan_arrow.png"))
        if tpl_arrow is not None and panel_present:
            crop_right = img_bgr[:, int(0.40 * w):]
            ts_a = cv2.resize(tpl_arrow, (0, 0), fx=scale, fy=scale)
            if crop_right.shape[0] >= ts_a.shape[0] and crop_right.shape[1] >= ts_a.shape[1]:
                res_a = cv2.matchTemplate(crop_right, ts_a, cv2.TM_CCOEFF_NORMED)
                _, max_a, _, _ = cv2.minMaxLoc(res_a)
                if max_a >= 0.75:
                    score += 35.0

        # Vetoes
        if any(w in text_corpus for w in ["EP", "連鎖結晶", "查看內容", "長按卡牌"]):
            score -= 60.0

        # Veto: 梦境模式与治疗中心
        if any(w in text_corpus for w in ["夢境模式", "梦境模式", "治療中心", "治疗中心", "消除記憶", "消除记忆"]):
            return 0.0, {"veto": "dream_or_treatment_screen", "cyan_ratio": round(cyan_ratio, 4)}

        # 反否决（形状）：右上角那条**橙色 XP/进度条**是关卡奖励结算界面的形状特征，
        # 有它就绝不可能是肉鸽选路（实测只有结算界面有，选路图 0 条）。
        # 这是"青发角色把青色判据骗了"那次的纵深防御。
        if find_horizontal_bars(img_bgr, "orange", region=(0.55, 0.08, 1.0, 0.30)):
            return 0.0, {"veto": "settlement_xp_bar", "cyan_ratio": round(cyan_ratio, 4)}

        # 反否决（文字）：**底部中央出现"交互提示框"字样** -> 这是局内交互场景，不是选路图。
        # 真机踩坑：蘑菇森林的青色环境光让暗青占比到 4.71%（>=3%），于是这里拿到 65 分，
        # 与"局内互动事件"(65) 打平 -> 按候选顺序被本评估器抢走 -> 路径规划器去点空地。
        # 提示框才有的词（事件/觸發/搜查/調查/互動）是干净的判别依据：
        # 真正的选路图即使叠了对白（rogue_map_real），对白里也不会出现这些词。
        prompt_words = ["事件", "觸發", "触发", "搜查", "調查", "调查", "互動", "互动", "拾取", "採集", "采集"]
        prompt_band = [t for t in ocr_targets
                       if 0.65 <= t.center_rel[1] <= 0.96 and 0.15 <= t.center_rel[0] <= 0.85
                       and any(w in (t.text or "") for w in prompt_words)]
        if not find_horizontal_bars(img_bgr, "orange", region=(0.55, 0.08, 1.0, 0.30)) and prompt_band:
            return 0.0, {"veto": "in_scene_prompt_box",
                         "prompt": [t.text for t in prompt_band][:3],
                         "cyan_ratio": round(cyan_ratio, 4)}

        # 反否决（按键提示）：右下角出现 `ENTER` 字样的按钮 -> 这是"确认/繼續進行"类提示画面，
        # 不是选路图。真机证据：肉鸽里那扇门 + 右下角 `ENTER 繼續進行`（OCR 把 繼續 读成 缆，
        # 但 ENTER 读出来了）被误判成肉鸽选路(65分，青色洞穴地面让 cyan 到 27.5%)，
        # 然后路径规划器在"没有任何分支"的情况下盲按了 2 号键。
        # 真正的选路图**没有**这种右下角 ENTER 按钮（分支用 1/2/3 数字键）。
        enter_hint = [t for t in ocr_targets
                      if "ENTER" in (getattr(t, "text", "") or "").upper()
                      and t.center_rel[0] >= 0.60 and t.center_rel[1] >= 0.70]
        if enter_hint:
            return 0.0, {"veto": "enter_hint_button", "cyan_ratio": round(cyan_ratio, 4)}

        return max(0.0, min(100.0, score)), {"has_mini_map": score >= 50,
                                             "mini_map_score": round(best_map, 3),
                                             "mini_map_tpl": best_map_name,
                                             "cyan_ratio": round(cyan_ratio, 4),
                                             "cyan_bright": round(cyan_bright, 4)}

    # -------------------------------------------------------------
    # 2. EVENT_CHOICE (局内互动事件选项 / 放大镜搜索)
    # -------------------------------------------------------------
    @classmethod
    def eval_event_choice(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        h, w = img_bgr.shape[:2]

        # Feature 1: Option dialogue bubbles in lower half [0.65 <= ry <= 0.95, 0.10 <= rx <= 0.90]
        choice_targets = [t for t in ocr_targets if 0.65 <= t.center_rel[1] <= 0.95 and 0.10 <= t.center_rel[0] <= 0.90]
        # 关键字表（真机补充）：`事件/觸發/触发/互動/互动` 是**局内交互提示框**最常见的词。
        # 踩坑记录：`觸發【費洛蒙孢子】事件` 被 OCR 拆成 `【洛蒙孢子】`/`】事件` —— 关键字里没有"事件"
        # 就只剩"≥2 个文本块"的弱判据(45分)，于是跟"肉鸽选路"(65) 打平 -> 被路径规划器抢走 -> 点空地。
        choice_words = ["分享", "資訊", "资讯", "靈光", "灵光", "裝備", "装备", "搜索", "前進", "前进",
                        "選擇", "选择", "調查", "调查", "無視", "无视",
                        "事件", "觸發", "触发", "互動", "互动", "搜查", "查看", "拾取", "採集", "采集"]
        has_choice_text = any(k in t.text for t in choice_targets for k in choice_words)
        if choice_targets and has_choice_text:
            score += 75.0
        elif choice_targets and len(choice_targets) >= 2:
            score += 45.0

        # Feature 2: Search bubble [🔍] in scene
        # 性能：这张模板在全图上做一次 matchTemplate 要 **421ms**（实测）。
        # 先用 1/4 缩略图做一次 17ms 的快筛，只有粗略分够高才去做全分辨率匹配。
        # 阈值 0.55 与真实阈值 0.70 之间留了余量（实测有气泡的夹具粗筛分很高）。
        tpl_bubble = cls._get_template("explore_search_bubble.png")
        if tpl_bubble is not None:
            scale = h / 576.0
            ts_b = cv2.resize(tpl_bubble, (int(tpl_bubble.shape[1] * scale), int(tpl_bubble.shape[0] * scale)))
            if img_bgr.shape[0] >= ts_b.shape[0] and img_bgr.shape[1] >= ts_b.shape[1]:
                coarse = 0.0
                try:
                    small = cv2.resize(img_bgr, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
                    ts_small = cv2.resize(ts_b, (max(1, ts_b.shape[1] // 4), max(1, ts_b.shape[0] // 4)))
                    if small.shape[0] >= ts_small.shape[0] and small.shape[1] >= ts_small.shape[1]:
                        res_s = cv2.matchTemplate(small, ts_small, cv2.TM_CCOEFF_NORMED)
                        _, coarse, _, _ = cv2.minMaxLoc(res_s)
                except Exception:  # noqa: BLE001
                    coarse = 1.0        # 快筛本身出错就别省，老老实实全分辨率匹配
                if coarse >= 0.55:
                    res_b = cv2.matchTemplate(img_bgr, ts_b, cv2.TM_CCOEFF_NORMED)
                    _, max_b, _, _ = cv2.minMaxLoc(res_b)
                    if max_b >= 0.70:
                        score += 70.0

        # Feature 3: Bottom-right circular action button (⚔️ / 🔍 / ✋)
        crop_action = img_bgr[int(0.80 * h):int(0.96 * h), int(0.88 * w):int(0.98 * w)]
        if crop_action.size > 0:
            gray = cv2.cvtColor(crop_action, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30, param1=100, param2=30,
                                       minRadius=int(22 * (h / 576.0)), maxRadius=int(55 * (h / 576.0)))
            if circles is not None:
                score += 20.0

        # Vetoes: Epiphany 3-card screen
        if ("長按卡牌" in text_corpus or "长按卡牌" in text_corpus) and "分享資訊" not in text_corpus:
            score -= 80.0

        # Vetoes: Epiphany hero selection screen ("請選擇欲獲得【靈光一閃】的戰鬥員")
        if any(k in text_corpus for k in ["欲獲得", "欲获得", "的戰鬥員", "的战斗员", "的眞", "撑欲瘦得"]) and any(k in text_corpus for k in ["靈光", "灵光", "一閃", "一闪", "【靈光一】"]):
            score -= 80.0

        # Veto: 大厅与卡厄思分支
        if any(k in text_corpus for k in ["出擊", "出击", "模擬", "模拟", "方舟都市", "戰鬥員", "救援", "卡厄思顯現", "卡厄思显现", "零號系統", "零号系统"]):
            return 0.0, {"veto": "home_lobby_screen"}

        # Veto: 图像/系统设置弹窗（ESC误按打开）
        if any(k in text_corpus for k in ["圖像設定", "图像设定", "解析度", "分辨率", "全螢幕", "全屏幕", "水面反射", "陰影", "阴影", "顯示模式", "显示模式", "聲音設定", "声音设定"]):
            return 0.0, {"veto": "settings_dialog"}

        hand_edge = edge_density(img_bgr, (0.15, 0.70, 0.78, 0.97))
        if hand_edge >= 0.08:
            return 0.0, {"veto": "battle_hand_row", "hand_edge": round(hand_edge, 4),
                         "choice_count": len(choice_targets)}

        return max(0.0, min(100.0, score)), {"choice_count": len(choice_targets),
                                             "hand_edge": round(hand_edge, 4)}

    # -------------------------------------------------------------
    # 2.5 EPIPHANY_HERO (灵光一闪选战斗员: 請選擇欲獲得【靈光一閃】的戰鬥員)
    # -------------------------------------------------------------
    @classmethod
    def eval_epiphany_hero(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        h, w = img_bgr.shape[:2]

        # Veto if this is an event choice bubble with "分享資訊" or "所選戰鬥員"
        if any(k in text_corpus for k in ["分享資訊", "分享资讯", "所選戰鬥員", "所选战斗员"]):
            return 0.0, {"veto": "event_choice_text"}

        # Veto: 卡牌奖励分配界面不是灵光一闪选人
        if any(k in text_corpus for k in ["欲領取", "欲领取", "欲取卡牌", "撑欲取", "撐欲取", "領取卡牌", "领取卡牌"]):
            return 0.0, {"veto": "card_reward_assign_screen"}

        # Feature 1: Prompt text keywords ("請選擇欲獲得【靈光一閃】的戰鬥員")
        has_epiphany_word = any(k in text_corpus for k in ["靈光一閃", "灵光一闪", "靈光", "灵光", "一閃", "一闪", "【靈光一】", "靈光一"])
        has_hero_word = any(k in text_corpus for k in ["戰鬥員", "战斗员", "欲獲得", "欲获得", "請選擇", "请选择", "的眞", "撑欲瘦得"])
        if has_epiphany_word and has_hero_word:
            score += 85.0
        elif has_epiphany_word and any(k in text_corpus for k in ["戰", "战", "獲", "获"]):
            score += 65.0

        # Feature 2: Orange confirm button at bottom right (rx: 0.82~0.96, ry: 0.88~0.96)
        crop_btn = img_bgr[int(0.85 * h):int(0.98 * h), int(0.75 * w):int(0.98 * w)]
        if crop_btn.size > 0:
            hsv = cv2.cvtColor(crop_btn, cv2.COLOR_BGR2HSV)
            mask_o = cv2.inRange(hsv, np.array([8, 120, 120]), np.array([25, 255, 255]))
            if np.count_nonzero(mask_o) >= 2000:
                score += 20.0

        # Feature 3: Golden light bulb glow on top-middle combatant (rx: 0.45~0.55, ry: 0.05~0.25)
        crop_bulb = img_bgr[int(0.05 * h):int(0.25 * h), int(0.42 * w):int(0.58 * w)]
        if crop_bulb.size > 0:
            hsv_bulb = cv2.cvtColor(crop_bulb, cv2.COLOR_BGR2HSV)
            mask_yellow = cv2.inRange(hsv_bulb, np.array([15, 120, 150]), np.array([35, 255, 255]))
            if np.count_nonzero(mask_yellow) >= 300:
                score += 15.0

        return max(0.0, min(100.0, score)), {}

    # -------------------------------------------------------------
    # 3. EPIPHANY_CARDS (局内灵光一闪 3选1 选牌界面)
    # -------------------------------------------------------------
    @classmethod
    def eval_epiphany_cards(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0

        # Must NOT be an event choice bubble with "分享資訊" or "所選戰鬥員"
        if any(k in text_corpus for k in ["分享資訊", "分享资讯", "所選戰鬥員", "所选战斗员"]):
            return 0.0, {"veto": "event_choice_text"}

        # Veto: 零号系统与法典界面绝非局内灵光一闪选牌！
        if any(k in text_corpus for k in ["零號系統", "零号系统", "零号系統", "零號系统", "選擇法典", "选择法典", "撐法典", "撑法典", "選撑法典"]):
            return 0.0, {"veto": "system_zero_codex"}

        # Veto: 卡牌奖励分配战斗员界面不是 3选1 选牌！
        if any(k in text_corpus for k in ["欲領取", "欲领取", "欲取卡牌", "撑欲取", "撐欲取", "領取卡牌", "领取卡牌"]):
            return 0.0, {"veto": "card_reward_assign_screen"}

        # Feature 1: Prompt words
        if "長按卡牌" in text_corpus or "长按卡牌" in text_corpus:
            score += 70.0
        if any(k in text_corpus for k in [
            "欲觸發靈光一閃", "欲触发灵光一闪", "觸發靈光一閃", "触发灵光一闪",
            "欲發靈光", "欲发灵光", "欲觸發", "欲触发"
        ]) or (
            any(k in text_corpus for k in ["選擇卡牌", "选择卡牌"]) and any(k in text_corpus for k in ["SPACE", "CAPS", "靈光一閃", "灵光一闪"])
        ):
            score += 90.0
        elif any(k in text_corpus for k in ["請選擇靈光一閃效果", "请选择灵光一闪效果", "選擇靈光一閃效果", "选择灵光一闪效果"]):
            score += 80.0
        elif any(k in text_corpus for k in ["選擇您的命運", "选择您的命运", "您的命運", "您的命运", "選擇命運", "选择命运", "欲適用的命運", "欲适用的命运"]):
            score += 90.0
        elif any(k in text_corpus for k in ["命運卡牌", "命运卡牌", "命運效果", "命运效果"]) and any(k in text_corpus for k in ["ENTER", "F1", "AP1", "確認", "确认"]):
            score += 85.0
        elif any(k in text_corpus for k in ["靈光一閃", "灵光一闪", "靈光一", "灵光一", "靈光", "灵光"]) and any(k in text_corpus for k in ["查看內容", "查看内容", "CAPS", "效果選擇"]):
            score += 75.0

        # Feature 2: Center card layout
        center_cards = [t for t in ocr_targets if 0.28 <= t.center_rel[1] <= 0.65]
        if len(center_cards) >= 3 and score >= 40:
            score += 20.0

        # Feature 3（形状优先，主人定的原则）：**三列等距卡牌**布局。
        # 不依赖 OCR 关键字——实测 OCR 会把这些认歪：
        # 「請選擇欲領取卡牌的戰鬥員」-> '撑欲取卡牌的戟国眞'、「長按卡牌以查看資訊」-> '長按卡牌以查看餐讯'、
        # 「選擇3張隨機中立稀有～傳說卡」-> '摆3张随機中立稀有～傅説卡'。
        # 目前只作为**加分/兜底**（+30，不足以单独过 50 分门槛）：要单独成判还需要更强的形状证据，
        # 等 fixtures_auto 收到"关键字被认歪"的真实帧后再调权重。
        layout = card_layout(ocr_targets)
        if layout['ok']:
            score += 30.0

        return max(0.0, min(100.0, score)), {'card_layout': layout}

    # -------------------------------------------------------------
    # 4. BATTLE (战斗中)
    # -------------------------------------------------------------
    @classmethod
    def eval_battle(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        det: Dict[str, Any] = {}
        h, w = img_bgr.shape[:2]

        # Feature 0（形状优先，主人定的原则）：**底部手牌成排 -> 边缘密度极高**。
        # 这是战斗界面最可靠的判据，因为它的文字根本不可靠：
        # 实测那一帧 OCR 把血量读成 `19431984`、地名读成 `劍之樂園`，
        # 而手牌是一叠带边框的卡牌 -> 底部中央边缘密度 **12.04%**，其它 17 张夹具最高 4.93%（2.4 倍余量）。
        hand_edge = edge_density(img_bgr, (0.15, 0.70, 0.78, 0.97))
        det['hand_edge'] = round(hand_edge, 4)
        if hand_edge >= 0.08:
            score += 55.0

        # Feature 0b（形状）：Boss 血条 —— 顶部窄带里的红/粉色横向长条。
        # 实测该带占比 9.72%，其它界面最高 4.98%。
        boss_pink = hsv_ratio_multi(img_bgr, (
            ((145, 70, 110), (180, 255, 255)),      # 洋红/粉
            ((0, 70, 110), (12, 255, 255)),         # 红
        ), region=(0.40, 0.13, 0.80, 0.20))
        det['boss_pink'] = round(boss_pink, 4)
        if boss_pink >= 0.07:
            score += 25.0

        # Feature 1: Battle HUD elements (EP, 連鎖結晶, 回合) —— 文字已是辅助判据
        if any(k in text_corpus for k in ["EP", "回合", "TURN", "連鎖結晶", "连锁结晶"]):
            score += 60.0

        # Feature 2: Top-right auto battle capsule (rx: 0.85~0.91, ry: 0.025~0.085)
        crop_top = img_bgr[int(0.025 * h):int(0.085 * h), int(0.85 * w):int(0.91 * w)]
        if crop_top.size > 0:
            hsv = cv2.cvtColor(crop_top, cv2.COLOR_BGR2HSV)
            mask_gold = cv2.inRange(hsv, np.array([14, 45, 140]), np.array([30, 255, 255]))
            if np.count_nonzero(mask_gold) >= 35 or np.mean(hsv[:, :, 1]) >= 35:
                score += 35.0

        # Feature 3: Team HP bar anchor at top-left [rx: 0.10~0.40, ry <= 0.08] (e.g. 805/805, 1943/1984)
        hp_match = any(
            re.search(r'\d+/\d+', getattr(t, 'text', ''))
            and 0.10 <= getattr(t, 'center_rel', (0, 0))[0] <= 0.40
            and getattr(t, 'center_rel', (0, 0))[1] <= 0.08
            for t in ocr_targets
        )
        if hp_match:
            score += 50.0
            det['has_hp_bar'] = True

        # Feature 4: Deck card counter at bottom-center [rx: 0.38~0.62, ry >= 0.90] (e.g. 00/10, 05/10)
        deck_match = any(
            re.search(r'\d+/10|/10', getattr(t, 'text', ''))
            and 0.38 <= getattr(t, 'center_rel', (0, 0))[0] <= 0.62
            and getattr(t, 'center_rel', (0, 0))[1] >= 0.90
            for t in ocr_targets
        )
        if deck_match:
            score += 50.0
            det['has_deck_counter'] = True

        # Feature 5: Combat shortcut key badges (E/R/K)
        combat_keys = any(
            getattr(t, 'text', '').strip() in ("E", "R", "K")
            and (getattr(t, 'center_rel', (0, 0))[0] >= 0.80 or getattr(t, 'center_rel', (0, 0))[1] >= 0.80)
            for t in ocr_targets
        )
        if combat_keys:
            score += 25.0
            det['has_combat_keys'] = True

        # Ambience HUD (HP bar at top-left and pause button at top-right) also exist during exploration.
        # Without actual combat evidence (hand cards >= 0.08, or EP/TURN/連鎖結晶, or boss bar), cap score at 45.0.
        has_real_combat_evidence = (
            hand_edge >= 0.08 or
            boss_pink >= 0.07 or
            any(k in text_corpus for k in ["EP", "回合", "TURN", "連鎖結晶", "连锁结晶"]) or
            combat_keys or deck_match
        )
        if not has_real_combat_evidence:
            score = min(score, 45.0)

        # Veto: Settlement screen
        if any(k in text_corpus for k in ["戰鬥結束", "战斗结束", "獲得分數", "获得分数", "VICTORY"]):
            score -= 70.0

        # Veto: 卡牌选择/奖励界面。这类界面的 OCR 噪声里常含 "EPENFT" 之类，
        # 会被上面的 "EP" 子串命中而误判成战斗中（实测「卡牌獎勵」界面因此拿到 60 分，
        # 与「灵光一闪选牌」70 分只差 10 -> 分差不足 -> 掉进 UNKNOWN 乱点）。
        if any(k in text_corpus for k in ["長按卡牌", "长按卡牌", "欲獲得的卡牌", "欲获得的卡牌",
                                          "卡牌獎勵", "卡牌奖励", "欲獲得", "卡牌以查看"]):
            return 0.0, {"veto": "card_choice_screen"}

        # Veto: 移除卡牌事件界面（请选择1张欲移除的卡牌）
        if any(k in text_corpus for k in ["欲移除的卡牌", "欲移除", "移除的卡牌", "移除卡牌", "移除1張", "移除1张"]):
            return 0.0, {"veto": "card_remove_screen"}

        # Veto: 命运选择界面（選擇您的命運 / 您的命運 / 命运3选1）
        if any(k in text_corpus for k in ["選擇您的命運", "选择您的命运", "您的命運", "您的命运", "選擇命運", "选择命运"]) or (
            any(k in text_corpus for k in ["命運", "命运"]) and any(k in text_corpus for k in ["ENTER", "F1", "AP1", "確認", "确认"])
        ):
            return 0.0, {"veto": "fate_select_screen"}

        # Veto: 零号系统/卡厄思分支/大厅界面（背景科技台座边缘密度高，但包含零号系统/法典文字）
        if any(k in text_corpus for k in ["零號系統", "零号系统", "零號", "零号", "選擇法典", "选择法典", "法典紀錄", "備援協議", "卡厄思顯現", "卡厄思显现", "卡厄思", "方舟都市", "戰鬥員", "出擊"]):
            return 0.0, {"veto": "non_battle_lobby_screen"}

        # Veto: 装备重铸界面
        if any(k in text_corpus for k in ["重鑄", "重铸", "重裝備", "重装备", "重装備", "神之錘", "神之锤", "神之鈺", "神之", "重新整理", "欲套用"]):
            return 0.0, {"veto": "equip_reforge_screen"}

        # Veto: 灵光一闪触发选牌界面
        if any(k in text_corpus for k in ["欲觸發", "欲触发", "欲發靈光", "欲发灵光", "觸發靈光", "触发灵光"]):
            return 0.0, {"veto": "epiphany_trigger_card_screen"}

        return max(0.0, min(100.0, score)), det

    # -------------------------------------------------------------
    # 5. CAMP_SHOP (营地与商店安全区)
    # -------------------------------------------------------------
    @classmethod
    def eval_camp_shop(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0

        if any(k in text_corpus for k in ["休息 (免費)", "休息(免費)", "安全區域", "安全区", "德朗商店"]):
            score += 85.0
        elif any(k in text_corpus for k in ["HP恢復", "HP恢复", "壓力恢復", "压力恢复"]) and any(k in text_corpus for k in ["確認", "确认", "取消"]):
            score += 90.0
        elif "營地" in text_corpus or "营地" in text_corpus:
            score += 55.0

        if any(k in text_corpus for k in ["EP", "連鎖結晶", "故事"]):
            score -= 60.0

        return max(0.0, min(100.0, score)), {}

    # -------------------------------------------------------------
    # 6. EQUIP_ASSIGN (获得装备分配)
    # -------------------------------------------------------------
    @classmethod
    def eval_equip_assign(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        h, w = img_bgr.shape[:2]

        # Veto: 灵光一闪触发选牌界面不是装备分配
        if any(k in text_corpus for k in ["欲觸發", "欲触发", "欲發靈光", "欲发灵光", "觸發靈光", "触发灵光"]):
            return 0.0, {"veto": "epiphany_trigger_card_screen"}

        # Veto: 梦境模式与治疗中心不是装备分配
        if any(k in text_corpus for k in ["夢境模式", "梦境模式", "治療中心", "治疗中心", "深層創傷", "深层创伤", "消除記憶", "消除记忆"]):
            return 0.0, {"veto": "dream_or_treatment_screen"}

        # Veto: 局内事件选项界面（出现选项气泡且无装备槽/点数/分配提示）绝非装备分配
        has_event_clues = any(w in text_corpus for w in ["詢問", "询问", "競技場", "竞技场", "【競技場", "【竞技场"]) or (
            any(w in text_corpus for w in ["獲得獎勵", "获得奖励"]) and any(w in text_corpus for w in ["資訊", "资讯", "事件", "對話", "对话"])
        )
        if has_event_clues and not any(w in text_corpus for w in ["120pt", "pt", "請選擇", "请选择", "選擇戰鬥員", "选择战斗员", "選擇裝備", "选择装备", "重鑄裝備", "神之錘"]):
            return 0.0, {"veto": "event_choice_screen"}

        crop_red = img_bgr[int(0.88 * h):int(0.97 * h), int(0.55 * w):int(0.75 * w)]
        if crop_red.size > 0:
            hsv = cv2.cvtColor(crop_red, cv2.COLOR_BGR2HSV)
            m1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
            if np.count_nonzero(m1 | m2) >= 1500:
                score += 45.0

        reforge_god_kws = [
            "重鑄裝備", "重装备", "重装備", "重新整理", "欲套用",
            "神之錘", "神之锤", "神之鈺", "神之槌", "使用神之", "強化的裝備", "强化的装备", "强化的装備"
        ]
        if any(k in text_corpus for k in [
            "選擇戰鬥員", "选择战斗员", "請選擇戰鬥員", "请选择战斗员",
            "選擇裝備", "选择装备", "請選擇裝備", "请选择装备", "请摆装備", "請擺裝備",
            "提煉", "提炼", "提烦", "0/120pt",
            "欲領取卡牌", "欲领取卡牌", "撑欲取卡牌", "撐欲取卡牌",
        ] + reforge_god_kws):
            score += 55.0
        elif any(k in text_corpus for k in ["裝備", "装备", "装備"]) and score >= 40:
            score += 30.0

        # Feature: 「選擇欲領取卡牌的戰鬥員」/ 装备分配 这类界面，右侧有三行战斗员，
        # 每行左侧都有一个「等級」标签 —— 这是比文字关键字更稳的结构特征
        # （实测 OCR 会把「請選擇欲領取卡牌的戰鬥員」认成 '撑欲取卡牌的戟国眞'，关键字根本命中不了）。
        level_labels = [t for t in ocr_targets
                        if t.center_rel[0] > 0.35 and any(k in t.text for k in ["等級", "等级", "等摄", "等", "LV", "Lv"])]
        if len(level_labels) >= 2 or any(k in text_corpus for k in [
            "欲領取卡牌", "欲领取卡牌", "撑欲取卡牌", "撐欲取卡牌",
        ] + reforge_god_kws):
            score += 80.0

        return max(0.0, min(100.0, score)), {}

    # -------------------------------------------------------------
    # 7. STAGE_MAP (外层大地图选关)
    # -------------------------------------------------------------
    @classmethod
    def stage_map_shape(cls, img_bgr: np.ndarray, ocr_targets: list) -> Dict[str, Any]:
        """
        「外侧选关」的**形状特征**（与关键字无关，供 STAGE_MAP 加分、CUTSCENE 反否决共用）：

          - `bright`：浅色/雪白大地图背景占比（实测该界面中央 V≈211/S≈18，87.5% 像素又亮又不饱和；
            其它界面背景都偏暗）
          - `badge_rows`：`#0N` 关卡徽章分几行、最多一行有几个（选关界面是一排关卡胶囊）

        为什么要它：实测这个界面 OCR 只认出 `第1章 / 故事 / #04 / #05 / #06 / 关卡名`，
        「主線/章節/進行觀測」这些关键字一个都没有 → 只靠文字时 `外层选关` 只有 35 分，
        被 `剧情对白演出`(55，因为关卡名落进"对白行"竖带) 抢走路由 → 脚本去点屏幕中央。
        """
        bright = bright_ratio(img_bgr, region=(0.05, 0.20, 0.95, 0.90))
        badges = [t for t in ocr_targets if re.search(r'#\s?\d', t.text)]
        rows = group_rows(badges, y_tol=0.06)
        badge_row_max = max((len(r) for r in rows), default=0)
        return {'bright': round(bright, 4), 'badges': len(badges),
                'badge_rows': len(rows), 'badge_row_max': badge_row_max}

    @classmethod
    def eval_stage_map(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        det = cls.stage_map_shape(img_bgr, ocr_targets)

        if any(k in text_corpus for k in ["進行觀測", "进行观测", "主線", "主线", "章節", "章节"]):
            score += 65.0
        if det['badges'] >= 1:
            score += 35.0

        # 形状特征（主判据）：浅色雪白大地图背景
        if det['bright'] >= 0.60:
            score += 35.0
        # 形状特征：一排关卡徽章（>=3 个在同一行）
        if det['badge_row_max'] >= 3:
            score += 25.0
        # 辅助：顶部章节标题
        if any("章" in t.text and t.center_rel[1] < 0.12 for t in ocr_targets):
            score += 15.0

        # Drawer check
        if any(k in text_corpus for k in ["進入", "进入", "關卡資訊", "关卡资讯"]):
            score += 30.0

        # Vetoes: rogue mini map
        tpl_sword = cls._get_template(str(Path("rogue") / "rogue_sword.png"))
        if tpl_sword is not None:
            h, w = img_bgr.shape[:2]
            scale = h / 576.0
            crop_map = img_bgr[int(0.62 * h):int(0.98 * h), int(0.01 * w):int(0.42 * w)]
            if crop_map.size > 0:
                ts = cv2.resize(tpl_sword, (0, 0), fx=0.22 * scale, fy=0.22 * scale)
                if crop_map.shape[0] >= ts.shape[0] and crop_map.shape[1] >= ts.shape[1]:
                    res = cv2.matchTemplate(crop_map, ts, cv2.TM_CCOEFF_NORMED)
                    _, max_v, _, _ = cv2.minMaxLoc(res)
                    if max_v >= 0.65:
                        score -= 90.0

        return max(0.0, min(100.0, score)), det

    # -------------------------------------------------------------
    # 8. CUTSCENE_DIALOG (纯剧情演出 / 对白推进)
    # -------------------------------------------------------------
    @classmethod
    def eval_cutscene_dialog(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        h, w = img_bgr.shape[:2]

        # Feature 1: Skip button [▷|] in top-right slot 1 (rx: 0.77~0.83, ry: 0.02~0.08)
        tpl_skip = cls._get_template("skip_btn.png")
        if tpl_skip is not None:
            scale = h / 576.0
            ts_s = cv2.resize(tpl_skip, (0, 0), fx=scale, fy=scale)
            crop_tr = img_bgr[int(0.02 * h):int(0.09 * h), int(0.76 * w):int(0.84 * w)]
            if crop_tr.shape[0] >= ts_s.shape[0] and crop_tr.shape[1] >= ts_s.shape[1]:
                res_s = cv2.matchTemplate(crop_tr, ts_s, cv2.TM_CCOEFF_NORMED)
                _, max_s, _, _ = cv2.minMaxLoc(res_s)
                if max_s >= 0.70:
                    score += 75.0

        # Feature 2: Skip modal popup
        if any(k in text_corpus for k in ["是否跳過", "是否跳过", "跳過劇情", "跳过剧情"]):
            score += 90.0

        # Feature 3: Dialogue text lines with no choice bubbles
        dialog_lines = [t for t in ocr_targets if 0.15 <= t.center_rel[1] <= 0.60]
        choice_lines = [t for t in ocr_targets if 0.65 <= t.center_rel[1] <= 0.95]
        if len(dialog_lines) >= 1 and len(choice_lines) == 0 and score < 50:
            score += 55.0

        # Feature 4（真机新增）：**黑底白字的剧情演出**。
        # 这游戏的剧情文本就是黑底白字，整图亮度跟"淡入淡出全黑帧"几乎一样
        # （实测黑底剧情 均值 6.4 / 过渡帧 3.3），原先被当成过渡帧 -> 脚本什么都不做
        # -> 看门狗超时 -> 硬停止（主人原话："对话的过程不去点跳过和快进，很麻烦"）。
        # 判据：整图**极黑**（均值 <=20；实测其它所有界面都 >=33.7）**且 有游戏文字**（排除脚本角标）。
        DARK_STORY_MAX_MEAN = 20.0
        mean, _std = blank_stats(img_bgr)
        game_texts = count_game_texts(ocr_targets)
        det = {"dark_story": False, "game_texts": game_texts, "mean": round(mean, 1)}
        if mean <= DARK_STORY_MAX_MEAN and game_texts >= 1:
            score += 70.0
            det["dark_story"] = True
            # 黑底时"下方有台词"再加一点（剧情台词通常在下方）
            if any(t.center_rel[1] >= 0.55 for t in ocr_targets if count_game_texts([t])):
                score += 10.0

        # Veto if event choices are present
        if len(choice_lines) >= 1 and any(k in t.text for t in choice_lines for k in ["分享", "資訊", "靈光", "裝備", "搜索", "前進"]):
            score -= 60.0

        # Veto: 这是"外侧选关"大地图（浅色背景 + 一排关卡徽章），不是剧情演出。
        # 这个误判真实发生过：关卡名落进"对白行"竖带 -> 剧情对白演出拿到 55 分抢走路由
        # -> 脚本去点屏幕中央（正确动作其实是点高亮节点上的 ▶）。
        shape = cls.stage_map_shape(img_bgr, ocr_targets)
        if shape['bright'] >= 0.60 and shape['badge_row_max'] >= 3:
            return 0.0, {"veto": "stage_map_shape", **shape}

        return max(0.0, min(100.0, score)), det

    # -------------------------------------------------------------
    # 9. SETTLEMENT_LOOT (战后结算与战利品)
    # -------------------------------------------------------------
    @classmethod
    def eval_settlement_loot(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        det: Dict[str, Any] = {}
        h, w = img_bgr.shape[:2]

        # Feature 0（形状优先，主人定的原则）：右上角**橙色横向进度条**（关卡奖励结算的 XP 条）。
        # 这是这个界面最干净的形状特征：实测「结算」界面 1 条，而"选关抽屉"0 条
        # （抽屉也有紫色宽按钮，所以光靠按钮颜色区分不开）。
        # 真机教训：这张结算图 OCR 只认出 'LV.50 / +100(13900/48000) / 罪之劇'，
        # 「確認/獲得/勝利」一个都没认出来 -> 旧逻辑只靠右下按钮给 50 分，
        # 被「局内互动事件」(65) 抢走路由 -> 脚本去点屏幕中央，结算永远做不完。
        bars = find_horizontal_bars(img_bgr, "orange", region=(0.55, 0.08, 1.0, 0.30))
        det['xp_bars'] = len(bars)
        if bars:
            score += 45.0

        # Feature 1: Orange or Purple bottom-right button（宽按钮；注意很多界面都有，故只给辅证分）
        crop_btn = img_bgr[int(0.85 * h):int(0.96 * h), int(0.75 * w):int(0.98 * w)]
        has_btn_color = False
        if crop_btn.size > 0:
            hsv = cv2.cvtColor(crop_btn, cv2.COLOR_BGR2HSV)
            mask_o = cv2.inRange(hsv, np.array([8, 120, 120]), np.array([25, 255, 255]))
            mask_p = cv2.inRange(hsv, np.array([125, 90, 120]), np.array([165, 255, 255]))
            if np.count_nonzero(mask_o) >= 2000 or np.count_nonzero(mask_p) >= 1500:
                has_btn_color = True
        det['bottom_right_button'] = has_btn_color
        if has_btn_color:
            score += 20.0 if bars else 50.0     # 有 XP 条时按钮是辅证；没有时仍按原来的强证据

        # Feature 2: Settlement keywords
        if any(k in text_corpus for k in ["戰鬥結束", "战斗结束", "獲得分數", "获得分数", "勝利", "胜利", "VICTORY", "信用點", "信用点"]):
            score += 50.0
        elif any(k in text_corpus for k in ["獲得", "获得", "確認", "确认", "確定", "确定"]) and score >= 40:
            score += 25.0
        # 关卡奖励结算的其它辅助字样
        if any(k in text_corpus for k in ["獎勵", "奖励", "LV.", "Lv.", "LV"]) :
            score += 15.0

        # Veto 1: Never treat stage info drawer with "撤退" / "關卡資訊" as settlement!
        if any(k in text_corpus for k in ["撤退", "關卡資訊", "关卡资讯"]):
            score -= 80.0

        # Veto 2: 图像/系统设置弹窗（ESC误按打开）
        if any(k in text_corpus for k in ["圖像設定", "图像设定", "解析度", "分辨率", "全螢幕", "全屏幕", "水面反射", "陰影", "阴影", "顯示模式", "显示模式", "聲音設定", "声音设定"]):
            return 0.0, {"veto": "settings_dialog"}

        # Veto 3: 梦境模式、治疗中心、消除记忆绝非关卡结算
        if any(k in text_corpus for k in ["夢境模式", "梦境模式", "治療中心", "治疗中心", "消除記憶", "消除记忆"]):
            return 0.0, {"veto": "dream_or_treatment_screen"}

        # Veto 4: 配置队伍/登场怪物界面（非战胜结算画面）
        if any(k in text_corpus for k in ["配置隊伍", "配置队伍", "配置伍", "登場怪物", "登场怪物"]) and not any(k in text_corpus for k in ["戰鬥結束", "战斗结束", "VICTORY", "勝利", "胜利"]):
            return 0.0, {"veto": "team_config_screen"}

        return max(0.0, min(100.0, score)), det

    # -------------------------------------------------------------
    # 10. SYSTEM_ZERO_LOBBY (零号系统主界面)
    # -------------------------------------------------------------
    @classmethod
    def eval_system_zero_lobby(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        det: Dict[str, Any] = {}

        # 区域1: 左上方标题区 (rx <= 0.30, ry <= 0.15) 必须有 "零號系統" / "零号系统"
        has_zero_title = any(
            any(k in t.text for k in ["零號系統", "零号系统", "零號", "零号", "系統", "系统"])
            and t.center_rel[0] <= 0.30 and t.center_rel[1] <= 0.15
            for t in ocr_targets
        ) or any(k in text_corpus for k in ["零號系統", "零号系统"])
        if has_zero_title:
            score += 45.0

        # 区域2: 右侧球形入口 (rx >= 0.70, 0.35 <= ry <= 0.60) 出现 "法典"
        has_codex_sphere = any(
            "法典" in t.text and t.center_rel[0] >= 0.70 and 0.35 <= t.center_rel[1] <= 0.60
            for t in ocr_targets
        ) or ("法典" in text_corpus)
        if has_codex_sphere:
            score += 45.0

        # 区域3: 左侧导航/状态文字
        if any(k in text_corpus for k in ["最高達成難度", "最高达成难度", "升级", "升級", "法典紀錄", "法典纪录", "備援協議", "备援协议"]):
            score += 15.0

        # 反向排他: 如果右下角有 "進入" / "进入" (rx >= 0.85, ry >= 0.85) 或左侧有 "選擇法典" -> 扣分(这是选法典页)
        has_enter_btn = any(
            any(k in t.text for k in ["進入", "进入"]) and t.center_rel[0] >= 0.85 and t.center_rel[1] >= 0.85
            for t in ocr_targets
        )
        if has_enter_btn or any(k in text_corpus for k in ["選擇法典", "选择法典", "選撑法典"]):
            score -= 90.0

        return max(0.0, min(100.0, score)), det

    # -------------------------------------------------------------
    # 11. CODEX_SELECT (零号系统法典选择)
    # -------------------------------------------------------------
    @classmethod
    def eval_codex_select(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        det: Dict[str, Any] = {}

        # 区域1: 左上方包含 "選擇法典" / "选法典" / "選撑法典"
        has_select_title = any(
            any(k in t.text for k in ["選擇法典", "选择法典", "選撑法典", "法典"])
            and t.center_rel[0] <= 0.30 and 0.15 <= t.center_rel[1] <= 0.35
            for t in ocr_targets
        ) or any(k in text_corpus for k in ["選擇法典", "选择法典", "選撑法典"])
        if has_select_title:
            score += 40.0

        # 区域2: 左侧选项列表区域包含编号 (rx <= 0.25, 0.30 <= ry <= 0.85)
        has_nums = any(
            any(k in t.text for k in ["01", "02", "03", "70", "探索度"])
            and t.center_rel[0] <= 0.25 and 0.30 <= t.center_rel[1] <= 0.85
            for t in ocr_targets
        ) or any(k in text_corpus for k in ["01", "02", "03"])
        if has_nums:
            score += 30.0

        # 区域3: 右下角包含 "進入" / "进入" 按钮 (rx >= 0.85, ry >= 0.85)
        has_enter = any(
            any(k in t.text for k in ["進入", "进入"])
            and t.center_rel[0] >= 0.85 and t.center_rel[1] >= 0.85
            for t in ocr_targets
        ) or ("進入" in text_corpus or "进入" in text_corpus)
        if has_enter:
            score += 30.0

        return max(0.0, min(100.0, score)), det

    # -------------------------------------------------------------
    # 12. HOME_LOBBY (游戏大厅主界面)
    # -------------------------------------------------------------
    @classmethod
    def eval_home_lobby(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        det: Dict[str, Any] = {}

        # 区域1: 右侧功能导航条 (rx >= 0.80, 0.15 <= ry <= 0.85)
        right_panel_hits = sum(
            1 for t in ocr_targets
            if t.center_rel[0] >= 0.80 and 0.15 <= t.center_rel[1] <= 0.85
            and any(k in t.text for k in ["出擊", "出击", "出擎", "模擬", "模拟", "模凝", "卡厄思", "方舟都市", "戰鬥員", "战斗员", "救援"])
        )
        if right_panel_hits >= 3:
            score += 70.0
        elif right_panel_hits >= 1:
            score += 45.0
        elif any(k in text_corpus for k in ["卡厄思", "方舟都市", "出擊", "出击", "模擬", "模拟"]):
            score += 35.0

        # 区域2: 顶部体力标识 (ry <= 0.15)
        top_bar_hits = any(
            any(k in t.text for k in ["/240", "F2", "F3", "等級", "等级", "Lv"])
            and t.center_rel[1] <= 0.15
            for t in ocr_targets
        )
        if top_bar_hits:
            score += 30.0

        # 反向排他: 若包含零号系统/选择法典，扣分
        if any(k in text_corpus for k in ["零號系統", "零号系统", "選擇法典", "选择法典"]):
            score -= 80.0

        return max(0.0, min(100.0, score)), det

    # -------------------------------------------------------------
    # 13. CHAOS_SELECT (卡厄思分支选择: 卡厄思顯現 vs 零號系統)
    # -------------------------------------------------------------
    @classmethod
    def eval_chaos_select(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        det: Dict[str, Any] = {}

        # 区域1: 顶部标题栏包含 "卡厄思" (rx <= 0.30, ry <= 0.15)
        has_top_title = any(
            "卡厄思" in t.text and t.center_rel[0] <= 0.30 and t.center_rel[1] <= 0.15
            for t in ocr_targets
        ) or ("卡厄思" in text_corpus)
        if has_top_title:
            score += 30.0

        # 区域2: 左右两张卡片 (ry >= 0.60)
        # 左侧: "卡厄思顯現" (0.15 <= rx <= 0.45)
        has_chaos_card = any(
            any(k in t.text for k in ["卡厄思", "顯現", "显现", "現"])
            and 0.15 <= t.center_rel[0] <= 0.45 and t.center_rel[1] >= 0.60
            for t in ocr_targets
        ) or any(k in text_corpus for k in ["卡厄思顯現", "卡厄思显现", "卡厄思現"])

        # 右侧: "零號系統" (0.55 <= rx <= 0.85)
        has_zero_card = any(
            any(k in t.text for k in ["零號系統", "零号系统", "零號", "零号", "系统", "系統"])
            and 0.55 <= t.center_rel[0] <= 0.85 and t.center_rel[1] >= 0.60
            for t in ocr_targets
        ) or any(k in text_corpus for k in ["零號系統", "零号系统"])

        if has_chaos_card and has_zero_card:
            score += 70.0
        elif has_zero_card or has_chaos_card:
            score += 35.0

        # 反向排他: 选关抽屉或局内事件
        if any(k in text_corpus for k in ["法典紀錄", "法典纪录", "選擇法典", "选择法典"]):
            score -= 80.0

        return max(0.0, min(100.0, score)), det

    # -------------------------------------------------------------
    # 13. DREAM_MODE (梦境模式 / 深层创伤提示)
    # -------------------------------------------------------------
    @classmethod
    def eval_dream_mode(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        det: Dict[str, Any] = {}

        has_dream_title = any(
            any(k in t.text for k in ["夢境模式", "梦境模式", "夢境", "梦境"])
            and t.center_rel[1] <= 0.15
            for t in ocr_targets
        ) or any(k in text_corpus for k in ["夢境模式", "梦境模式", "啟動夢境模式", "启动梦境模式"])
        if has_dream_title:
            score += 60.0

        if any(k in text_corpus for k in ["深層創傷", "深层创伤", "創傷", "创伤"]):
            score += 30.0

        if any(k in text_corpus for k in ["治療中心", "治疗中心", "治療", "治疗"]):
            score += 30.0

        if any(k in text_corpus for k in ["精神異常", "精神异常", "崩潰加劇", "崩溃加剧", "存檔資料", "存档资料", "靈光一閃機率", "灵光一闪机率", "精神翼常"]):
            score += 25.0

        # 反向排他：战斗手牌或对话
        if any(k in text_corpus for k in ["回合結束", "回合结束", "是否跳過", "是否跳过"]):
            score -= 80.0

        return max(0.0, min(100.0, score)), det

    # -------------------------------------------------------------
    # 14. TREATMENT_CENTER (治疗中心 / 消除记忆)
    # -------------------------------------------------------------
    @classmethod
    def eval_treatment_center(cls, img_bgr: np.ndarray, text_corpus: str, ocr_targets: list) -> Tuple[float, Dict[str, Any]]:
        score = 0.0
        det: Dict[str, Any] = {}

        has_treat_title = any(
            any(k in t.text for k in ["治療中心", "治疗中心", "治療", "治疗"])
            and t.center_rel[1] <= 0.15
            for t in ocr_targets
        ) or any(k in text_corpus for k in ["治療中心", "治疗中心"])
        if has_treat_title:
            score += 60.0

        if any(k in text_corpus for k in ["消除記憶", "消除记忆", "消除"]):
            score += 35.0

        if any(k in text_corpus for k in ["諮商", "咨询", "諮商紀錄", "咨询记录"]):
            score += 20.0

        if any(k in text_corpus for k in ["艦長", "舰长", "馬上就去幫您", "马上就去帮您"]):
            score += 20.0

        # 反向排他
        if any(k in text_corpus for k in ["回合結束", "回合结束", "配置隊伍", "配置队伍"]):
            score -= 80.0

        return max(0.0, min(100.0, score)), det


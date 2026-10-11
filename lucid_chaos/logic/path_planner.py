"""
Phase 9: PathPlanner & Roguelike Map Navigator
Calculates optimal route nodes based on user preference, priority rules, and forward lookahead:
1. Priority: 问号 (Event/?) & 篝火 (Camp/Fire) & 篝火带商店 (Shop) > 普通怪 (Battle/Sword)
2. Forbidden nodes: Elite monsters (骷髅/精英), Boss, Hazard nodes ("其余的不走") -> Disqualified
3. Multi-step Lookahead: Evaluates active column nodes combined with subsequent column paths to maximize expected gain
4. Execution: Selects the winning branch by clicking its cyan arrow (▶) or terrain block on the right, or the mini-map node
"""

import os
import cv2
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger("PathPlanner")

def load_image_unicode(file_path: str) -> Optional[np.ndarray]:
    """Load image from path supporting Chinese / unicode characters."""
    if not os.path.exists(file_path):
        return None
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error(f"Failed to load image from {file_path}: {e}")
        return None

@dataclass
class MapNode:
    node_type: str  # 'shop', 'camp', 'event', 'battle', 'boss', 'waypoint', 'unknown'
    client_pos: Tuple[int, int]
    rel_pos: Tuple[float, float]
    score: float = 0.0
    reachable: bool = True

class PathPlanner:
    _templates_cache: Optional[Dict[str, np.ndarray]] = None

    def __init__(self, preference: str = "non_battle", event_only: bool = False):
        self.preference = preference
        self.event_only = event_only
        self._ensure_templates()

        if self.event_only or self.preference == "event_only":
            # 零号系统/纯事件刷取模式：锁定问号神秘事件，强惩罚一切战斗
            self.node_scores = {
                'QUESTION': 150.0,           # 问号神秘事件 (最高优先)
                'FIRE_SHOP': 70.0,           # 篝火（带商店）
                'FIRE': 60.0,                # 篝火
                'SWORD': -500.0,             # 战斗怪 (零号系统强制避战)
                'EMPTY': -100.0,             # 无连线/空位
                'FORBIDDEN_UNKNOWN': -1000.0 # 精英怪/其他未允许节点
            }
        else:
            # Rule scores:
            # 优先走问号和篝火，普通怪可走，其余不走(强惩罚)
            self.node_scores = {
                'FIRE_SHOP': 100.0,           # 篝火（带商店）
                'FIRE': 90.0,                # 篝火
                'QUESTION': 85.0,            # 问号神秘事件
                'SWORD': 40.0,               # 普通怪 (红菱形双剑)
                'EMPTY': -100.0,             # 无连线/空位
                'FORBIDDEN_UNKNOWN': -1000.0 # 精英怪/其他未允许节点 (其余不走)
            }

    @classmethod
    def _ensure_templates(cls):
        if cls._templates_cache is not None:
            return
        # 统一走全项目共用的模板库（原来是本类自己读盘 + 自己缓存）
        from ..core.template_bank import BANK
        cls._tpl_files = {'FIRE_SHOP': "rogue/rogue_fire_shop.png",
                          'FIRE': "rogue/rogue_fire.png",
                          'QUESTION': "rogue/rogue_question.png",
                          'SWORD': "rogue/rogue_sword.png",
                          'cyan_arrow': "rogue/cyan_arrow.png",
                          'cyan_arrow_2k': "rogue_arrow_cyan.png"}
        cls._templates_cache = {}
        for key, fname in cls._tpl_files.items():
            img = BANK.get(fname)
            if img is not None:
                cls._templates_cache[key] = img

    @classmethod
    def get_template(cls, name: str) -> Optional[np.ndarray]:
        cls._ensure_templates()
        return cls._templates_cache.get(name)

    def detect_rogue_screen(self, img_bgr: np.ndarray) -> bool:
        """
        Detect whether current screen is the Roguelike Room / Branch Selection screen.
        Checks for mini-map node templates in the bottom-left window (rx: 0.01~0.42, ry: 0.62~0.98).
        """
        if img_bgr is None or img_bgr.size == 0:
            return False
        h, w = img_bgr.shape[:2]
        crop_map = img_bgr[int(0.62 * h):int(0.98 * h), int(0.01 * w):int(0.42 * w)]
        if crop_map.size == 0:
            return False

        tpl_sword = self.get_template('SWORD')
        if tpl_sword is None:
            return False

        scale = h / 576.0
        # 与 router\evaluators.eval_roguelike_map 保持一致：取四个节点模板的最高分，
        # 只看 SWORD 一个会因为某张图 SWORD 差一点点没过阈值而漏判整个界面。
        best = 0.0
        for key in ('SWORD', 'QUESTION', 'FIRE', 'FIRE_SHOP'):
            tpl = self.get_template(key)
            if tpl is None:
                continue
            ts = cv2.resize(tpl, (0, 0), fx=0.22 * scale, fy=0.22 * scale)
            if crop_map.shape[0] < ts.shape[0] or crop_map.shape[1] < ts.shape[1]:
                continue
            res = cv2.matchTemplate(crop_map, ts, cv2.TM_CCOEFF_NORMED)
            _, mv, _, _ = cv2.minMaxLoc(res)
            best = max(best, float(mv))
        if best >= 0.65:
            return True
        # 与 router 一致：压暗时模板分会掉，用不受亮度影响的青色面板占比兜底
        hsv_map = cv2.cvtColor(crop_map, cv2.COLOR_BGR2HSV)
        m_cyan = cv2.inRange(hsv_map, np.array([80, 60, 80]), np.array([105, 255, 255]))
        return (float(np.count_nonzero(m_cyan)) / float(m_cyan.size)) >= 0.03

    def scan_mini_map(self, img_bgr: np.ndarray) -> Dict[Tuple[int, int], Tuple[str, float]]:
        """
        Scan all grid points in the bottom-left mini-map to classify nodes.
        Grid columns: 0 (Current Pin), 1 (Active next column), 2 (Col 2), 3 (Col 3), 4, 5
        Grid rows: 0 (Top), 1 (Middle), 2 (Bottom)
        """
        h, w = img_bgr.shape[:2]
        crop_map = img_bgr[int(0.62 * h):int(0.98 * h), int(0.01 * w):int(0.42 * w)]
        scale = h / 576.0

        cols_x = [int(x * scale) for x in [42, 102, 164, 225, 285, 346]]
        rows_y = [int(y * scale) for y in [70, 105, 141]]

        grid = {}
        for c in range(len(cols_x)):
            for r in range(len(rows_y)):
                cx, cy = cols_x[c], rows_y[r]
                patch = crop_map[max(0, cy - int(22 * scale)):min(crop_map.shape[0], cy + int(22 * scale)),
                                 max(0, cx - int(30 * scale)):min(crop_map.shape[1], cx + int(30 * scale))]
                if patch.shape[0] < 20 or patch.shape[1] < 25:
                    grid[(c, r)] = ('EMPTY', 0.0)
                    continue

                # Filter out dark background
                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                if np.mean(gray) < 40:
                    grid[(c, r)] = ('EMPTY', 0.0)
                    continue

                best_n, best_s = 'UNKNOWN', 0.0
                from ..core.template_bank import BANK
                for n, tpl in self._templates_cache.items():
                    # 箭头的模板有两个（cyan_arrow / cyan_arrow_2k），都要排除：
                    # 以前只跳过 'cyan_arrow'，'cyan_arrow_2k' 会参与节点分类，
                    # 把当前高亮的篝火节点认成 cyan_arrow_2k(0.51) —— 纯噪声。
                    if 'arrow' in n:
                        continue
                    ts = BANK.scaled(self._tpl_files.get(n, n), 0.22 * scale)   # 同一份缩放缓存
                    if ts is None:
                        continue
                    if patch.shape[0] >= ts.shape[0] and patch.shape[1] >= ts.shape[1]:
                        res = cv2.matchTemplate(patch, ts, cv2.TM_CCOEFF_NORMED)
                        _, mv, _, _ = cv2.minMaxLoc(res)
                        if mv > best_s:
                            best_s, best_n = mv, n

                if best_s < 0.50:
                    grid[(c, r)] = ('EMPTY', best_s)
                else:
                    grid[(c, r)] = (best_n, best_s)

        return grid

    def detect_scene_arrows(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect interactive cyan arrows [▷] on the 3D map scene (x > 0.40*w).
        Returns list of dicts:
        [
            {
                'cx': int, 'cy': int,
                'rx': float, 'ry': float,
                'bbox': (x, y, w, h),
                'score': float
            },
            ...
        ]
        sorted by cy ascending (top to bottom).
        """
        tpl_2k = self.get_template('cyan_arrow_2k')
        tpl_arrow = self.get_template('cyan_arrow')
        if tpl_2k is None and tpl_arrow is None:
            return []
        
        h, w = img_bgr.shape[:2]
        crop = img_bgr[:, int(0.40 * w):]

        if tpl_2k is not None:
            # 模板基准为 2K (1440p)
            scale = h / 1440.0
            ts = cv2.resize(tpl_2k, (0, 0), fx=scale, fy=scale)
        else:
            # 模板基准为 576p
            scale = h / 576.0
            ts = cv2.resize(tpl_arrow, (0, 0), fx=scale, fy=scale)
        if crop.shape[0] < ts.shape[0] or crop.shape[1] < ts.shape[1]:
            return []
        res = cv2.matchTemplate(crop, ts, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= 0.80)
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        matches = []
        for y, x in zip(*locs):
            # Check cyan color in patch to eliminate dark dungeon floor texture false positives
            patch_hsv = hsv_crop[y:y+ts.shape[0], x:x+ts.shape[1]]
            mask_cyan = cv2.inRange(patch_hsv, np.array([75, 50, 100]), np.array([105, 255, 255]))
            cyan_count = np.count_nonzero(mask_cyan)
            if cyan_count >= max(5, int(15 * (scale ** 2))):
                cx = x + int(0.40 * w) + ts.shape[1] // 2
                cy = y + ts.shape[0] // 2
                matches.append({
                    'cx': cx,
                    'cy': cy,
                    'rx': cx / w,
                    'ry': cy / h,
                    'bbox': (cx - ts.shape[1] // 2 - 4, cy - ts.shape[0] // 2 - 4, ts.shape[1] + 8, ts.shape[0] + 8),
                    'score': float(res[y, x])
                })

        # Cluster points within 35px * scale
        clusters = []
        for m in sorted(matches, key=lambda item: -item['score']):
            if not any(abs(m['cx'] - c['cx']) < 35 * scale and abs(m['cy'] - c['cy']) < 35 * scale for c in clusters):
                clusters.append(m)

        clusters.sort(key=lambda item: item['cy'])
        return clusters

    def detect_scene_banners(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect hanging banners (3D挂旗) on building platforms using HSV color segmentation:
        - Pink/Magenta: Battle / SWORD (H: 145~175, S: 80~255, V: 100~255)
        - Orange: Camp / Fire (H: 10~25, S: 120~255, V: 120~255)
        - Blue: Event / Question (H: 100~135, S: 100~255, V: 100~255)
        - Purple: Elite / Boss (H: 130~150, S: 90~255, V: 80~255)
        Completely immune to 3D room background scene variations!
        """
        h, w = img_bgr.shape[:2]
        crop_3d = img_bgr[:, int(0.45 * w):]
        if crop_3d.size == 0:
            return []

        hsv = cv2.cvtColor(crop_3d, cv2.COLOR_BGR2HSV)
        scale = h / 576.0

        color_specs = [
            ('SWORD', np.array([145, 80, 100]), np.array([175, 255, 255])),
            ('FIRE', np.array([10, 120, 120]), np.array([25, 255, 255])),
            ('QUESTION', np.array([100, 100, 100]), np.array([135, 255, 255])),
            ('ELITE', np.array([130, 90, 80]), np.array([150, 255, 255])),
        ]

        banners = []
        for banner_type, lower, upper in color_specs:
            mask = cv2.inRange(hsv, lower, upper)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area >= max(80, int(180 * (scale ** 2))):
                    bx, by, bw, bh = cv2.boundingRect(c)
                    if bh >= int(20 * scale):
                        abs_x = bx + int(0.45 * w)
                        abs_y = by
                        banners.append({
                            'type': banner_type,
                            'cx': abs_x + bw // 2,
                            'cy': abs_y + bh // 2,
                            'bbox': (abs_x, abs_y, bw, bh),
                            'area': area
                        })

        banners.sort(key=lambda b: b['cy'])
        return banners

    def evaluate_branches(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Evaluate selectable branches on the current screen:
        1. Analyzes mini-map to find actual reachable rows.
        2. Detects 3D arrows and corresponding hanging banners.
        3. Coordinates expand between Arrow (Left anchor) and Banner (Right anchor)
           to squarely and perfectly encompass the interactive room platform (Zero guessing!).
        """
        h, w = img_bgr.shape[:2]
        scale = h / 576.0
        grid = self.scan_mini_map(img_bgr)
        scene_arrows = self.detect_scene_arrows(img_bgr)
        scene_banners = self.detect_scene_banners(img_bgr)

        # 1. Identify which rows in Column 1 are actually active / reachable
        active_rows = [r for r in [0, 1, 2] if grid.get((1, r), ('EMPTY', 0.0))[0] != 'EMPTY']

        # Fallback if mini-map OCR temporarily didn't detect active rows
        if not active_rows:
            if len(scene_arrows) == 1:
                active_rows = [1]
            elif len(scene_arrows) == 2:
                active_rows = [0, 1]
            else:
                active_rows = [0, 1, 2]

        row_names = {0: '上方分支 (Top)', 1: '中间分支 (Middle)', 2: '下方分支 (Bottom)'}
        default_arrows = {
            0: {'cx': int(0.631 * w), 'cy': int(0.528 * h), 'rx': 0.631, 'ry': 0.528, 'score': 0.70},
            1: {'cx': int(0.625 * w), 'cy': int(0.524 * h), 'rx': 0.625, 'ry': 0.524, 'score': 0.70},
            2: {'cx': int(0.488 * w), 'cy': int(0.823 * h), 'rx': 0.488, 'ry': 0.823, 'score': 0.70},
        }

        # 2. Match active rows 1-to-1 with detected arrows
        matched_branches = []
        if len(scene_arrows) == len(active_rows):
            for r, arr in zip(active_rows, scene_arrows):
                matched_branches.append((r, arr))
        elif len(scene_arrows) > len(active_rows):
            for idx, r in enumerate(active_rows):
                matched_branches.append((r, scene_arrows[idx]))
        elif len(scene_arrows) > 0:
            for idx, r in enumerate(active_rows):
                arr = scene_arrows[idx] if idx < len(scene_arrows) else default_arrows[r]
                matched_branches.append((r, arr))
        else:
            for r in active_rows:
                matched_branches.append((r, default_arrows[r]))

        results = []
        for r, arr in matched_branches:
            n1 = grid.get((1, r), ('EMPTY', 0.0))[0]
            n2 = grid.get((2, r), ('EMPTY', 0.0))[0]
            n3 = grid.get((3, r), ('EMPTY', 0.0))[0]

            s1 = self.node_scores.get(n1, -1000.0)
            s2 = self.node_scores.get(n2, -1000.0)
            s3 = self.node_scores.get(n3, -1000.0)

            # Discounted path score: s1 + 0.7 * s2 + 0.5 * s3
            total = s1 + 0.7 * s2 + 0.5 * s3
            if n1 == 'FORBIDDEN_UNKNOWN' or n2 == 'FORBIDDEN_UNKNOWN':
                total -= 1000.0

            # Find matching hanging banner for this arrow
            matched_banner = None
            for b in scene_banners:
                if b['cx'] > arr['cx'] + int(40 * scale) and abs(b['cy'] - arr['cy']) < int(130 * scale):
                    matched_banner = b
                    break

            # Coordinate Expansion between Left Anchor (Arrow) and Right Anchor (Banner)
            if matched_banner:
                tile_cx = int((arr['cx'] + matched_banner['cx']) / 2)
                tile_cy = int((arr['cy'] + matched_banner['cy']) / 2)
                tile_w = int(max(200 * scale, (matched_banner['cx'] - arr['cx']) * 1.15))
                tile_h = int(220 * scale)
                banner_click = (matched_banner['cx'], matched_banner['cy'])
                banner_bbox = matched_banner['bbox']
            else:
                if arr['cy'] > 0.65 * h:
                    tile_cy = int(arr['cy'] - int(25 * scale))
                else:
                    tile_cy = int(arr['cy'] + int(10 * scale))
                tile_cx = int(arr['cx'] + int(155 * scale))
                tile_w = int(220 * scale)
                tile_h = int(200 * scale)
                banner_click = None
                banner_bbox = None

            tile_bbox = (
                int(max(0, tile_cx - tile_w // 2)),
                int(max(0, tile_cy - tile_h // 2)),
                int(tile_w),
                int(tile_h)
            )

            results.append({
                'row': r,
                'name': row_names.get(r, f"分支 {r}"),
                'nodes': [n1, n2, n3],
                'scores': [s1, s2, s3],
                'total_score': total,
                'is_forbidden': (n1 == 'FORBIDDEN_UNKNOWN' or n2 == 'FORBIDDEN_UNKNOWN'),
                # Arrow info
                'arrow_click': (arr['cx'], arr['cy']),
                'arrow_score': arr.get('score', 0.99),
                # Banner info
                'banner_click': banner_click,
                'banner_bbox': banner_bbox,
                # Square building platform info
                'tile_click': (tile_cx, tile_cy),
                'bbox': tile_bbox,
                'click_point': (tile_cx, tile_cy)
            })

        # Sort descending by total_score
        results.sort(key=lambda x: x['total_score'], reverse=True)
        return results

    def execute_branch_selection(self, ctx: Any, img_bgr: np.ndarray) -> bool:
        """
        Execute the optimal branch selection:
        1. Evaluates scores and paths without duplicate/overlapping branches.
        2. Renders HUD bounding boxes squarely over the 3D building platforms and cyan arrows.
        3. Clicks both the cyan arrow and the chosen room platform for 100% reliable entry.
        """
        h, w = img_bgr.shape[:2]
        scale = h / 576.0
        branches = self.evaluate_branches(img_bgr)
        if not branches:
            logger.warning("【肉鸽选路】未检测到有效分支，点击中间默认前进...")
            ctx.input.click_rel(0.625, 0.524, delay_after=2.0)
            return True

        best = branches[0]

        # ⚠️ 真机教训（主人 2026/10/6）：如果在"没认出任何节点"（全 EMPTY）的情况下也按键/点击，
        # 就会在一扇门/空地前盲按数字键（实测盲按了 2 号键）。所以这里加一道门槛：
        # **没有任何真实节点证据时，什么都不做**，交给路由层/守卫处理。
        has_evidence = any(
            any(n and n != 'EMPTY' for n in (b.get('nodes') or [])) or b.get('tile_click')
            for b in branches
        )
        all_empty = all(all((n or 'EMPTY') == 'EMPTY' for n in (b.get('nodes') or [])) for b in branches)
        if all_empty and not has_evidence:
            logger.warning("【肉鸽选路】所有分支都没有识别到节点（全 EMPTY）-> **不盲按/不盲点**，"
                           "本轮放弃选路，交给路由层或卡死守卫处理")
            return False

        # Multi-branch / Single-branch logging
        if len(branches) <= 1:
            node_name = best['nodes'][0] if best['nodes'][0] != 'EMPTY' else '重要环节/Boss房'
            logger.info(f"【肉鸽前行】当前为单一通道 (节点: [{node_name}], 评分={best['total_score']:.1f})")
        else:
            logger.info("=" * 60)
            if self.event_only:
                logger.info("【零号系统策略】纯事件优先：强制避战，锁定问号神秘事件路线")
            logger.info("【肉鸽选路】结合左下角路线与右侧分支综合评分 (长线规划):")
            for b in branches:
                path_desc = " -> ".join(b['nodes'])
                logger.info(f"  * {b['name']}: 预测路径 [{path_desc}] | 综合评分={b['total_score']:.1f}")
            logger.info(f"【肉鸽选路】>>> 优选决策: 【{best['name']}】(评分={best['total_score']:.1f})")

        # 1. **键盘优先**（主人 2026/10/6 定的规则）：选路界面上每条分支的 ▶ 旁边都标着编号
        #    （实测截图：两条分支分别标 `1`、`2`），所以"选第 N 条路"= 按 N 号键。
        arrow_order = sorted(branches, key=lambda b: b['arrow_click'][1])      # 从上到下 = 1,2,3
        try:
            branch_index = arrow_order.index(best) + 1
        except ValueError:
            branch_index = 1

        # 简洁高亮：只精准标记选路箭头选项 (1/2/3)，不画多余的假 3D 悬崖/地块边框
        if getattr(ctx, "overlay", None):
            overlay_boxes = []
            box_sz = int(48 * scale)
            # 1. Best branch: 绿色高亮选定路线与按键提示
            overlay_boxes.append({
                'bbox': (best['arrow_click'][0] - box_sz // 2, best['arrow_click'][1] - box_sz // 2, box_sz, box_sz),
                'label': f"【选定路线 {branch_index}】{best['name']} (键 {branch_index})",
                'click': best['arrow_click'],
                'color': '#00FF00'
            })
            # 2. Candidate branches: 黄色轻度标记候选路线
            for b in branches:
                if b is not best:
                    try:
                        b_idx = arrow_order.index(b) + 1
                    except ValueError:
                        b_idx = 0
                    overlay_boxes.append({
                        'bbox': (b['arrow_click'][0] - box_sz // 2, b['arrow_click'][1] - box_sz // 2, box_sz, box_sz),
                        'label': f"候选路线 {b_idx}：{b['name']}",
                        'click': b['arrow_click'],
                        'color': '#FFD700'
                    })
            ctx.overlay.show_boxes(overlay_boxes, duration=1.5)
        used_key = False
        if 1 <= branch_index <= 3:
            try:
                used_key = bool(ctx.input.tap_card(
                    branch_index, delay_after=1.8,
                    label=f"【肉鸽选路】第 {branch_index} 条路线 (键 {branch_index}) {best['name']}"))
            except Exception:  # noqa: BLE001
                used_key = False
            if used_key:
                logger.info(f"【肉鸽选路】按编号键 {branch_index} 选择【{best['name']}】(评分={best['total_score']:.1f})")

        if used_key:
            return True

        # 1b. 回退：点击青色箭头 ▷
        logger.info(f"【肉鸽选路】优先点击选路青色箭头 ▷ (坐标: {best['arrow_click'][0]}, {best['arrow_click'][1]})...")
        ctx.input.click_client(best['arrow_click'][0], best['arrow_click'][1], delay_after=0.3, label=f"【箭头 ▷】{best['name']}")

        # 2. Click center of square building platform (Room selection trigger)
        logger.info(f"【肉鸽选路】同步点击方形建筑区域中心 (坐标: {best['tile_click'][0]}, {best['tile_click'][1]})...")
        ctx.input.click_client(best['tile_click'][0], best['tile_click'][1], delay_after=0.3, label=f"【房间】{best['name']}")

        # 3. Optional click on banner flag
        if best.get('banner_click'):
            ctx.input.click_client(best['banner_click'][0], best['banner_click'][1], delay_after=0.3, label=f"【挂旗】{best['name']}")

        # 4. Confirmation click on arrow to ensure movement commences
        ctx.input.click_client(best['arrow_click'][0], best['arrow_click'][1], delay_after=1.8)
        return True

    # -------------------------------------------------------------
    # Legacy interfaces for backward compatibility
    # -------------------------------------------------------------
    def score_node(self, node: MapNode, missing_cards: int, missing_equips: int) -> float:
        base_scores = {
            "shop": 100.0,
            "camp": 90.0,
            "event": 85.0,
            "battle": 40.0,
            "boss": -1000.0,
            "waypoint": 45.0,
            "unknown": -1000.0
        }
        score = base_scores.get(node.node_type, 10.0)
        node.score = score
        return score

    def select_best_node(self, nodes: List[MapNode], missing_cards: int = 0, missing_equips: int = 0) -> Optional[MapNode]:
        if not nodes:
            return None
        for n in nodes:
            self.score_node(n, missing_cards, missing_equips)
        nodes.sort(key=lambda n: n.score, reverse=True)
        best = nodes[0]
        logger.info(f"Selected best path node: [{best.node_type}] score={best.score:.1f} at rel={best.rel_pos}")
        return best

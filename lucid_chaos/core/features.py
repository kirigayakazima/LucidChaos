"""
可复用的形状 / 颜色 / 布局特征原语 —— 判据的"第一公民"。

主人定的原则：**形状特征匹配为主，文字只做辅助**。为避免每个场景各写一份 HSV 判断
（改造前就是那样，散在 evaluator / node_engine / story_pusher / stage_detector 里），
这里提供通用原语，供各 evaluator 与 handler 组合使用。

全部函数都是**纯函数**（给定图像与区域 -> 结果），不读配置、不写状态，方便单测。
区域统一用相对坐标 `region=(rx1, ry1, rx2, ry2)`（相对整图宽高）。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Region = Optional[Sequence[float]]
BBox = Tuple[int, int, int, int]      # x, y, w, h（整图像素坐标）

# -------------------------------------------------------------
# 常用颜色的 HSV 阈值（H: 0~179, S/V: 0~255）
# 这些值是从改造前散落各处的 inRange 调用里收拢来的，保持一致
# -------------------------------------------------------------
COLORS: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    "orange": ((8, 120, 120), (25, 255, 255)),
    "gold": ((14, 45, 140), (30, 255, 255)),
    "purple": ((125, 90, 120), (165, 255, 255)),
    "green": ((40, 80, 80), (85, 255, 255)),
    "cyan": ((80, 60, 80), (105, 255, 255)),
    "cyan_sat": ((80, 90, 35), (108, 255, 255)),
    "red1": ((0, 100, 100), (10, 255, 255)),
    "red2": ((170, 100, 100), (180, 255, 255)),
}


# -------------------------------------------------------------
# 区域与颜色
# -------------------------------------------------------------
def region_px(img: np.ndarray, region: Region) -> Tuple[Optional[np.ndarray], int, int]:
    """按相对坐标裁剪，返回 `(crop, x0, y0)`；越界自动夹紧，非法返回 (None, 0, 0)。"""
    if img is None or getattr(img, "size", 0) == 0:
        return None, 0, 0
    h, w = img.shape[:2]
    if region is None:
        return img, 0, 0
    x1 = max(0, int(region[0] * w))
    y1 = max(0, int(region[1] * h))
    x2 = min(w, int(region[2] * w))
    y2 = min(h, int(region[3] * h))
    if x2 <= x1 or y2 <= y1:
        return None, 0, 0
    return img[y1:y2, x1:x2], x1, y1


def mask_hsv(img: np.ndarray, color, region: Region = None) -> Optional[np.ndarray]:
    """取某种颜色的掩膜。`color` 可以是 COLORS 里的名字，也可以是 (lo, hi) 元组。"""
    crop, _, _ = region_px(img, region)
    if crop is None:
        return None
    if isinstance(color, str):
        lo, hi = COLORS[color]
    else:
        lo, hi = color
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    if isinstance(color, str) and color.startswith("red"):
        other = "red2" if color == "red1" else "red1"
        lo2, hi2 = COLORS[other]
        m |= cv2.inRange(hsv, np.array(lo2, dtype=np.uint8), np.array(hi2, dtype=np.uint8))
    return m


def hsv_ratio(img: np.ndarray, color, region: Region = None) -> float:
    """区域内该颜色的像素占比（0.0~1.0）。"""
    m = mask_hsv(img, color, region)
    if m is None or m.size == 0:
        return 0.0
    return float(np.count_nonzero(m)) / float(m.size)


def bright_ratio(img: np.ndarray, region: Region = None, *, s_max: int = 50, v_min: int = 150) -> float:
    """
    "又亮又不饱和"的像素占比 —— 用来识别**浅色/雪白大地图**这类背景。

    实测「外侧选关」地图背景：中央 V≈211 / S≈18，87.5% 像素满足 (S<50 & V>150)；
    而其它界面（选牌、事件框、肉鸽地图）背景都偏暗，占比很低。
    """
    crop, _, _ = region_px(img, region)
    if crop is None:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = (hsv[:, :, 1] < s_max) & (hsv[:, :, 2] > v_min)
    return float(np.count_nonzero(m)) / float(m.size)


def blank_stats(img: np.ndarray) -> Tuple[float, float]:
    """整图灰度均值与标准差（诊断/判定过渡帧用）。"""
    if img is None or getattr(img, "size", 0) == 0:
        return 0.0, 0.0
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(g.mean()), float(g.std())


import re

# 脚本自己画在画面上的角标/日志文本（OCR 会读到它们，必须排除，否则"全黑过渡帧"永远判不出来）
OVERLAY_TEXT_PAT = re.compile(r'路由|route|过渡帧|自动取证|INFO|WARNING|ERROR|scene|decision|stuck')


def count_game_texts(items, *, exclude_box: Sequence[float] = (0.0, 0.0, 0.28, 0.07)) -> int:
    """
    数"游戏自己的文本块"（排除脚本画的左上角角标 + 角标样式文本）。

    为什么需要（真机踩坑）：这游戏**剧情文本是黑底白字**，跟"进出关卡的淡入淡出全黑帧"亮度几乎一样
    （实测黑底剧情 均值 6.4 / 过渡帧 3.3）。唯一干净的区别是：
      - 过渡帧：全黑，**没有任何游戏文字**（OCR 只读到脚本自己的 `【路由:...】` 角标）
      - 黑底剧情：全黑，**有剧情文字**（`艾拉具拉`、`曾有受到星辰祝福的始祖存在。`）
    所以"是不是过渡帧"必须把"有没有游戏文字"算进去。
    """
    n = 0
    for it in (items or []):
        try:
            rx, ry = it.center_rel
            text = getattr(it, "text", "") or ""
        except Exception:  # noqa: BLE001
            continue
        if exclude_box[0] <= rx <= exclude_box[2] and exclude_box[1] <= ry <= exclude_box[3]:
            continue                     # 左上角是脚本自己画的路由角标
        if OVERLAY_TEXT_PAT.search(text):
            continue                     # 角标/日志样式文本
        n += 1
    return n


def is_blank_frame(img: np.ndarray, *, mean_max: float = 15.0, std_max: float = 45.0,
                   targets=None, max_game_texts: int = 0) -> bool:
    """
    是不是"过渡/全黑帧"（进出关卡与剧情时的淡入淡出）。

    实测：这类帧亮度均值 3.3、标准差 26（几乎全黑），且**没有任何游戏文字**。
    而最暗的正常界面（肉鸽地图）整图亮度均值也在 30 以上。

    ⚠️ 真机教训（2026/10/6）：最初只判亮度，结果把**黑底白字的剧情演出**也当成过渡帧
    -> 脚本"什么都不做" -> 看门狗超时 -> 硬停止脚本（主人说"对话过程不去点跳过/快进，很麻烦"）。
    现在多一条：**必须没有游戏文字**（`targets` 里排除脚本角标后为 0 块）才算过渡帧。
    """
    mean, std = blank_stats(img)
    if not (mean <= mean_max and std <= std_max):
        return False
    if targets is None:
        return True                      # 未提供文字信息时退回旧的"只看亮度"行为
    return count_game_texts(targets) <= max_game_texts


def group_rows(items: Sequence, *, y_tol: float = 0.02) -> List[List]:
    """把带 `center_rel` 的元素按 y 分成"行"（同一行的 y 差 <= y_tol），从上到下。"""
    rows: List[List] = []
    for it in sorted(items, key=lambda t: t.center_rel[1]):
        if rows and abs(it.center_rel[1] - rows[-1][-1].center_rel[1]) <= y_tol:
            rows[-1].append(it)
        else:
            rows.append([it])
    return rows


def edge_density(img: np.ndarray, region: Region = None, *, low: int = 60, high: int = 160) -> float:
    """
    区域内的**边缘密度**（Canny 边缘像素占比）——纯几何/纹理特征，不看颜色。

    用途（实测）：战斗界面底部手牌是一叠带边框的卡牌，边缘密度极高。
      实测 battle_hand_row: **12.04%**，其它 17 张夹具最高只有 4.93% —— 2.4 倍余量。
    这是战斗判定的主判据（血条/EP 文字都不可靠：实测血条数字被 OCR 读成 `19431984`）。
    """
    crop, _, _ = region_px(img, region)
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low, high)
    return float(np.count_nonzero(edges)) / float(edges.size)


def hsv_ratio_multi(img: np.ndarray, ranges: Sequence[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
                    region: Region = None) -> float:
    """多段 HSV 区间的合计占比（例如"红/粉"要覆盖 0-12 与 145-180 两段）。"""
    total = 0.0
    for lo, hi in ranges:
        total += hsv_ratio(img, (lo, hi), region)
    return total


def mask_span(mask: Optional[np.ndarray]) -> int:
    """掩膜的横向跨度（像素）。"""
    if mask is None or mask.size == 0:
        return 0
    xs = np.nonzero(mask)[1]
    return int(xs.max() - xs.min()) if xs.size else 0


def mask_centroid(mask: Optional[np.ndarray], x0: int = 0, y0: int = 0) -> Optional[Tuple[int, int]]:
    """掩膜质心（加上区域偏移 -> 整图坐标）。"""
    if mask is None or mask.size == 0:
        return None
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(x0 + xs.mean()), int(y0 + ys.mean())


# -------------------------------------------------------------
# 形状：横向长条（HP 条 / 进度条 / 按钮底条）
# -------------------------------------------------------------
def find_horizontal_bars(img: np.ndarray, color, region: Region = None, *,
                         min_w_ratio: float = 0.10, max_h_ratio: float = 0.030,
                         min_aspect: float = 3.0, max_bars: int = 12) -> List[Dict]:
    """
    找横向长条（例如战斗界面左上角的三条绿色 HP 条）。

    判据：颜色掩膜 -> 横向形态学开运算 -> 轮廓 -> 长宽比 >= min_aspect 且宽度占比足够。
    **比例都以整图为基准**（`min_w_ratio` 相对整图宽、`max_h_ratio` 相对整图高）——
    如果相对裁剪区算，区域一矮（例如 y 只占 0.15）就会把真实长条也卡掉。

    返回 `[{'bbox': (x,y,w,h), 'cx','cy','w','h'}]`（整图坐标）。
    """
    crop, x0, y0 = region_px(img, region)
    if crop is None:
        return []
    m = mask_hsv(crop, color)
    if m is None or np.count_nonzero(m) == 0:
        return []
    img_h = img.shape[0]
    hh, ww = crop.shape[:2]
    kx = max(3, int(ww * 0.04))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 1))
    m2 = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(m2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Dict] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh <= 0 or bw / max(1, bh) < min_aspect:
            continue
        if bw < min_w_ratio * ww or bh > max_h_ratio * img_h:
            continue
        out.append({'bbox': (x0 + x, y0 + y, bw, bh), 'cx': x0 + x + bw // 2,
                    'cy': y0 + y + bh // 2, 'w': bw, 'h': bh})
    out.sort(key=lambda d: (d['cy'], d['cx']))
    return out[:max_bars]


# -------------------------------------------------------------
# 形状：按钮（颜色 + 位置 + 横向跨度）
# -------------------------------------------------------------
def find_button(img: np.ndarray, region: Region, colors: Sequence[str] = ("orange", "purple"),
                *, min_pixels: int = 8000, min_span_ratio: float = 0.10) -> Optional[Dict]:
    """
    在区域内找"某个颜色的按钮条"：像素数足够 且 横向跨度足够（排除零散色块）。
    返回 `{'color','cx','cy','pixels','span','bbox'}`（整图坐标）。
    """
    crop, x0, y0 = region_px(img, region)
    if crop is None:
        return None
    best = None
    for name in colors:
        m = mask_hsv(crop, name)
        if m is None:
            continue
        n = int(np.count_nonzero(m))
        if n < min_pixels:
            continue
        span = mask_span(m)
        if span < min_span_ratio * crop.shape[1]:
            continue
        c = mask_centroid(m, x0, y0)
        if c is None:
            continue
        if best is None or n > best['pixels']:
            best = {'color': name, 'cx': c[0], 'cy': c[1], 'pixels': n, 'span': span,
                    'bbox': cv2.boundingRect(m)}
    return best


# -------------------------------------------------------------
# 形状：圆按钮（Hough）
# -------------------------------------------------------------
def find_circles(img: np.ndarray, region: Region = None, *, min_r_ratio: float = 0.035,
                 max_r_ratio: float = 0.090, max_circles: int = 6) -> List[Tuple[int, int, int]]:
    """
    找圆形按钮（例如右下角"回合结束 ✓"）。半径**按整图高度**的比例给（与长条一致的口径），
    返回 `[(cx, cy, r)]`（整图坐标）。
    """
    crop, x0, y0 = region_px(img, region)
    if crop is None:
        return []
    img_h = img.shape[0]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=100, param2=30,
                               minRadius=int(min_r_ratio * img_h), maxRadius=int(max_r_ratio * img_h))
    if circles is None:
        return []
    out = [(int(x0 + c[0]), int(y0 + c[1]), int(c[2])) for c in circles[0][:max_circles]]
    return out


# -------------------------------------------------------------
# 布局：同一竖带内的等宽列（3 选 1 卡牌 / 三行列表）
# -------------------------------------------------------------
def find_columns(items: Sequence, *, band: Sequence[float] = (0.0, 1.0),
                 gap_ratio: float = 0.06, width: float = 1.0) -> List[List]:
    """
    把带 `center_rel` 的元素按 x 聚成"列"。`items` 需有 `.center_rel`（(rx, ry)）。
    `band` 限定 y 范围（相对），`gap_ratio*width` 为分列阈值。
    返回 `[[item, ...], ...]`，从左到右。
    """
    sel = [it for it in items if band[0] <= it.center_rel[1] <= band[1]]
    sel.sort(key=lambda it: it.center_rel[0])
    cols: List[List] = []
    thr = gap_ratio * width
    for it in sel:
        if not cols or (it.center_rel[0] - cols[-1][-1].center_rel[0]) > thr:
            cols.append([it])
        else:
            cols[-1].append(it)
    return cols


def card_layout(items: Sequence, *, band: Sequence[float] = (0.26, 0.80), y_tol: float = 0.02,
                want_columns: int = 3, min_spacing: float = 0.14,
                max_even_dev: float = 0.35, min_rx: float = 0.12, max_rx: float = 0.90) -> Dict:
    """
    **卡牌 N 选 1 的布局特征**（形状/布局为主判据，不依赖 OCR 关键字）。

    做法（踩过坑的版本）：**先在竖带内按 y 分行，再在"某一行内"按 x 聚列**。
    直接对整条竖带聚列会失败——卡面描述文字也在带内，会跟标题交错聚成 5 列。
    只要**任意一行**能聚出 `want_columns` 个间距均匀的列就算成立（标题行/描述行都行）。

    用 OCR 文本块的位置只是把它当**布局信号**（不做关键字语义匹配），
    所以 OCR 把字认歪（`摆3张随機中立稀有～傅説卡`）也照样能判。

    返回 `{'ok','columns','centers','even','spacing','row_ry'}`。
    """
    sel = [it for it in items
           if band[0] <= it.center_rel[1] <= band[1] and min_rx <= it.center_rel[0] <= max_rx]
    rows: List[List] = []
    for it in sorted(sel, key=lambda t: t.center_rel[1]):
        if rows and abs(it.center_rel[1] - rows[-1][-1].center_rel[1]) <= y_tol:
            rows[-1].append(it)
        else:
            rows.append([it])

    best = {'ok': False, 'columns': 0, 'centers': [], 'even': False, 'spacing': [], 'row_ry': None}
    for row in rows:
        cols = find_columns(row, band=(0.0, 1.0), gap_ratio=min_spacing)
        centers = [float(np.mean([it.center_rel[0] for it in c])) for c in cols]
        spacing = [round(centers[i + 1] - centers[i], 4) for i in range(len(centers) - 1)]
        even = False
        if len(centers) == want_columns and spacing:
            mean_s = float(np.mean(spacing))
            if mean_s > 0:
                dev = float(np.max([abs(s - mean_s) for s in spacing]) / mean_s)
                even = dev <= max_even_dev
        cand = {'ok': len(centers) == want_columns and even,
                'columns': len(centers), 'centers': [round(c, 4) for c in centers],
                'even': even, 'spacing': spacing,
                'row_ry': round(float(np.mean([it.center_rel[1] for it in row])), 4)}
        if cand['ok']:
            return cand
        if cand['columns'] > best['columns'] or best['row_ry'] is None:
            best = cand
    return best


def detect_treasure_chests(img_bgr: np.ndarray, min_score: float = 0.65) -> List[Tuple[float, float, float]]:
    """
    检测肉鸽宝物库/场景中的未开启宝箱 (Treasure Chests)。
    返回各宝箱相对坐标与置信度列表: [(rx, ry, score), ...]，按从左到右排序。
    """
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return []
    h, w = img_bgr.shape[:2]
    scale = h / 576.0

    # 1. 区域限制: 地面宝箱区域 (rx: 0.38~0.92, ry: 0.42~0.76)
    x1, x2 = int(0.38 * w), int(0.92 * w)
    y1, y2 = int(0.42 * h), int(0.76 * h)
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    # 颜色快筛: 宝箱发光能量晶体 (H: 85~115, S >= 80, V >= 130)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([85, 80, 130]), np.array([115, 255, 255]))
    if np.count_nonzero(mask) < 200:
        return []

    # 2. 模板匹配
    from .template_bank import BANK
    tpl = BANK.get("chest_box.png")
    if tpl is None:
        return []

    hits = []
    for s in (0.80, 0.90, 1.0, 1.10):
        cw = int(tpl.shape[1] * scale * s)
        ch = int(tpl.shape[0] * scale * s)
        if roi.shape[0] < ch or roi.shape[1] < cw:
            continue
        ts = cv2.resize(tpl, (cw, ch))
        res = cv2.matchTemplate(roi, ts, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= min_score)
        for y, x in zip(*locs):
            score = float(res[y, x])
            cx = x1 + x + cw // 2
            cy = y1 + y + ch // 2
            hits.append((cx / w, cy / h, score))

    # 3. NMS 去重 (距离阈值 0.08)
    unique = []
    for rx, ry, sc in sorted(hits, key=lambda item: item[2], reverse=True):
        if not any(abs(rx - ux) < 0.08 and abs(ry - uy) < 0.08 for ux, uy, _ in unique):
            unique.append((rx, ry, sc))

    unique.sort(key=lambda item: item[0])
    return unique


# -------------------------------------------------------------
# 灵光一闪「神闪」（金色包边卡牌）检测
# -------------------------------------------------------------
def detect_god_epiphany_card(img_bgr: Optional[np.ndarray]) -> Tuple[Optional[int], Dict[str, Any]]:
    """
    识别 3 选 1 灵光一闪界面中是否存在「神闪」（金色包边专属强化卡牌）。

    主人 2026/10/11 核心规则与真机特征：
    - "又遇到了新的样式，灵感一闪，但是有神闪，看到左边卡的1下面的那个图标没有，而且卡片带有金色的包边就是神闪，
       但是那个神闪的图标不一样都是这个样式，但是你可以通过包边金色来区分"
    - 普通灵光一闪卡牌四周为青色/青蓝包边（HSV H: 75~105），内嵌青色星芒；
    - 「神闪」卡牌四周为耀眼的金色/橙金色包边（HSV H: 12~34, S: 100~255, V: 120~255），
      卡牌编号下方带有金色专属徽章。
    - 返回: `(best_slot, details)`，若存在神闪则 best_slot 为 1、2 或 3（1-based）；否则为 None。
    """
    if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
        return None, {}
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    card_cols = [
        (1, 0.135, 0.355),
        (2, 0.390, 0.610),
        (3, 0.645, 0.865),
    ]

    scale = (576.0 / float(h)) ** 2
    gold_scores: Dict[int, float] = {}
    cyan_scores: Dict[int, float] = {}

    for idx, rx1, rx2 in card_cols:
        x1, x2 = int(rx1 * w), int(rx2 * w)
        y1, y2 = int(0.20 * h), int(0.77 * h)

        # 左右竖向包边带
        left_strip = hsv[y1:y2, max(0, x1 - 3):x1 + 12]
        right_strip = hsv[y1:y2, x2 - 12:min(w, x2 + 3)]
        top_strip = hsv[int(0.19 * h):int(0.22 * h), x1 + 10:x2 - 10]

        g_left = np.count_nonzero(cv2.inRange(left_strip, np.array([12, 100, 120]), np.array([34, 255, 255])))
        c_left = np.count_nonzero(cv2.inRange(left_strip, np.array([75, 100, 120]), np.array([105, 255, 255])))

        g_right = np.count_nonzero(cv2.inRange(right_strip, np.array([12, 100, 120]), np.array([34, 255, 255])))
        g_top = np.count_nonzero(cv2.inRange(top_strip, np.array([12, 100, 120]), np.array([34, 255, 255])))

        # 编号下方专属图标区域
        icon_box = hsv[int(0.32 * h):int(0.44 * h), int((rx1 + 0.015) * w):int((rx1 + 0.055) * w)]
        g_icon = np.count_nonzero(cv2.inRange(icon_box, np.array([12, 100, 120]), np.array([34, 255, 255])))

        total_gold = float((g_left + g_right + g_top + g_icon) * scale)
        gold_scores[idx] = round(total_gold, 1)
        cyan_scores[idx] = round(float(c_left * scale), 1)

    best_slot = max(gold_scores, key=gold_scores.get)
    best_gold = gold_scores[best_slot]
    other_golds = [gold_scores[i] for i in (1, 2, 3) if i != best_slot]
    other_avg = sum(other_golds) / len(other_golds) if other_golds else 0.0

    # 神闪判定门槛：总归一化金色包边像素 >= 1200，且显著高于其余两张卡的平均金色（裕量 >= 800）
    if best_gold >= 1200 and best_gold > other_avg + 800:
        return best_slot, {
            "slot": best_slot,
            "gold": best_gold,
            "other_avg": round(other_avg, 1),
            "scores": gold_scores,
            "cyan_scores": cyan_scores,
        }
    return None, {"scores": gold_scores, "cyan_scores": cyan_scores}



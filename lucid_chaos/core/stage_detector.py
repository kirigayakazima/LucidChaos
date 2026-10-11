import os
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import cv2

class StageMapDetector:
    @classmethod
    def _get_template(cls, name: str) -> Optional[np.ndarray]:
        """统一走全项目共用的模板库。"""
        from .template_bank import BANK
        return BANK.get(name)

    @classmethod
    def _match_multiscale(cls, img: np.ndarray, tpl: np.ndarray, threshold: float = 0.70) -> Optional[Tuple[int, int, Tuple[float, float], float]]:
        if tpl is None or img is None:
            return None
        h, w = img.shape[:2]
        base_scale = h / 576.0
        all_scales = [0.85, 0.95, 1.0, 1.05, 1.15, 1.25]

        # 性能：下面原本是 6 档 **全图** matchTemplate，实测这个函数要 3337ms。
        # 优化：先用 1/4 缩略图把 6 档各粗筛一次（每档约 20ms），然后
        #   ① 全都够不到 threshold-0.12 -> 直接返回 None（没有目标，最常见）
        #   ② 只对**粗筛分接近最高分**（差距 <=0.10）的那几档做全分辨率扫描
        # 有目标、且某档明显最优时，结果与原来一致（回归夹具的点击坐标逐像素相同）。
        keep_scales = all_scales
        try:
            small = cv2.resize(img, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
            coarse = {}
            for s in all_scales:
                tw0 = int(tpl.shape[1] * base_scale * s) // 4
                th0 = int(tpl.shape[0] * base_scale * s) // 4
                if tw0 <= 0 or th0 <= 0 or small.shape[0] < th0 or small.shape[1] < tw0:
                    continue
                ts0 = cv2.resize(tpl, (tw0, th0))
                res0 = cv2.matchTemplate(small, ts0, cv2.TM_CCOEFF_NORMED)
                _, mv0, _, _ = cv2.minMaxLoc(res0)
                coarse[s] = float(mv0)
            if coarse:
                best_c = max(coarse.values())
                if best_c < threshold - 0.12:
                    return None
                keep_scales = [s for s in all_scales if coarse.get(s, 0.0) >= best_c - 0.10]
        except Exception:  # noqa: BLE001
            keep_scales = all_scales

        best_score = 0.0
        best_loc = None
        best_dim = (0, 0)
        for s in keep_scales:
            tw = int(tpl.shape[1] * base_scale * s)
            th = int(tpl.shape[0] * base_scale * s)
            if tw <= 0 or th <= 0 or tw >= w or th >= h:
                continue
            ts = cv2.resize(tpl, (tw, th))
            res = cv2.matchTemplate(img, ts, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            if mv > best_score:
                best_score = mv
                best_loc = ml
                best_dim = (tw, th)
        if best_score >= threshold and best_loc is not None:
            cx = best_loc[0] + best_dim[0] // 2
            cy = best_loc[1] + best_dim[1] // 2
            return (cx, cy, (cx / max(1, w), cy / max(1, h)), best_score)
        return None

    @classmethod
    def find_active_stage_button(cls, img_bgr: np.ndarray, ocr_targets: Optional[List[Any]] = None) -> Optional[Tuple[int, int, Tuple[float, float]]]:
        """
        Three-tier active stage locator on chapter map screen with exclusion masking:
        1. Masking: Detects all checkmark (✓) badges and masks out their entire node capsule,
           ensuring artwork (e.g. flames in cleared stages) is never misdetected.
        2. Tier 1: Golden glowing play button (▶) or orange 'N' badge in unmasked area.
        3. Tier 2: '進行觀測' text / magnifying glass (🔍) search icon in unmasked area.
        4. Tier 3: Uncompleted node between last cleared and first locked stage.
        Returns:
            (cx, cy, (rx, ry)) in client and relative coordinates, or None.
        """
        if img_bgr is None or img_bgr.size == 0:
            return None

        h, w = img_bgr.shape[:2]
        scale = h / 576.0

        # -------------------------------------------------------------
        # 0. MASKING: Find all cleared checkmarks and mask out their capsules
        # -------------------------------------------------------------
        mask_cleared = np.zeros((h, w), dtype=np.uint8)
        tpl_check = cls._get_template("check_btn.png")
        if tpl_check is not None:
            ts_check = cv2.resize(tpl_check, (0, 0), fx=scale, fy=scale)
            res_check = cv2.matchTemplate(img_bgr, ts_check, cv2.TM_CCOEFF_NORMED)
            pts = np.where(res_check >= 0.65)
            seen_checks = []
            for pt in zip(*pts[::-1]):
                if not any(abs(pt[0] - x) < int(30 * scale) and abs(pt[1] - y) < int(30 * scale) for x, y in seen_checks):
                    seen_checks.append(pt)
                    # Mask out entire node capsule: left ~150px, right ~45px, top/down ~35px
                    x1 = max(0, int(pt[0] - 150 * scale))
                    x2 = min(w, int(pt[0] + 45 * scale))
                    y1 = max(0, int(pt[1] - 35 * scale))
                    y2 = min(h, int(pt[1] + 45 * scale))
                    mask_cleared[y1:y2, x1:x2] = 255

        # Also mask top and bottom navigation bars
        mask_cleared[:int(0.18 * h), :] = 255
        mask_cleared[int(0.88 * h):, :] = 255

        unmasked_img = img_bgr.copy()
        unmasked_img[mask_cleared == 255] = 0

        # -------------------------------------------------------------
        # Tier 1: Primary - Golden Play Button (▶) or N-badge (in unmasked area)
        # -------------------------------------------------------------
        tpl_play = cls._get_template("play_btn.png")
        m_play = cls._match_multiscale(unmasked_img, tpl_play, threshold=0.70)
        if m_play and 0.20 < m_play[2][1] < 0.85:
            if mask_cleared[m_play[1], m_play[0]] == 0:
                return (m_play[0], m_play[1], m_play[2])

        # Orange N-badge fallback in unmasked area
        hsv = cv2.cvtColor(unmasked_img, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([8, 180, 200])
        upper_orange = np.array([22, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        contours_n, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_dim = int(8 * scale)
        max_dim = int(22 * scale)
        badge_candidates = []
        for c in contours_n:
            x, y, bw, bh = cv2.boundingRect(c)
            if min_dim <= bw <= max_dim and min_dim <= bh <= max_dim:
                ratio = bw / max(1, bh)
                cx = x + bw // 2
                cy = y + bh // 2
                rx = cx / max(1, w)
                ry = cy / max(1, h)
                if 0.20 < ry < 0.85 and 0.7 < ratio < 1.4:
                    if mask_cleared[cy, cx] == 0:
                        badge_candidates.append((cx, cy, rx, ry, bw, bh))

        if badge_candidates:
            badge_candidates.sort(key=lambda item: abs(item[3] - 0.50))
            best = badge_candidates[0]
            btn_cx = int(best[0] - 15 * scale)
            btn_cy = int(best[1] + 20 * scale)
            return (btn_cx, btn_cy, (btn_cx / max(1, w), btn_cy / max(1, h)))

        # -------------------------------------------------------------
        # Tier 2: Observation / Search Stage (🔍 or '進行觀測')
        # -------------------------------------------------------------
        # 2a. OCR detection of '進行觀測' / '觀測'
        if ocr_targets:
            for t in ocr_targets:
                text = getattr(t, "text", "")
                cx = getattr(t, "center", (0, 0))[0]
                cy = getattr(t, "center", (0, 0))[1]
                cy_rel = getattr(t, "center_rel", (0, 0))[1]
                if any(k in text for k in ["觀測", "观测", "進行觀測", "进行观测"]) and 0.20 < cy_rel < 0.85:
                    if mask_cleared[cy, cx] == 0:
                        return (t.center[0], t.center[1], t.center_rel)

        # 2b. Template matching for search / magnifying glass icon
        tpl_search = cls._get_template("search_icon.png")
        m_search = cls._match_multiscale(unmasked_img, tpl_search, threshold=0.75)
        if m_search and 0.20 < m_search[2][1] < 0.85:
            if mask_cleared[m_search[1], m_search[0]] == 0:
                return (m_search[0], m_search[1], m_search[2])

        # -------------------------------------------------------------
        # Tier 3: Topological Interpolation (Strictly between Cleared and Locked)
        # -------------------------------------------------------------
        tpl_lock = cls._get_template("lock_icon.png")
        m_lock = cls._match_multiscale(img_bgr, tpl_lock, threshold=0.75)
        # Must have an actual locked stage on screen to establish the right boundary!
        if m_lock and 0.20 < m_lock[2][1] < 0.85:
            min_lock_x = m_lock[0]
            max_check_x = (max(p[0] for p in seen_checks) if seen_checks else 0) + int(45 * scale)

            if ocr_targets:
                for t in ocr_targets:
                    text = getattr(t, "text", "")
                    cx = getattr(t, "center", (0, 0))[0]
                    cy = getattr(t, "center", (0, 0))[1]
                    cy_rel = getattr(t, "center_rel", (0, 0))[1]
                    if text.startswith("#") and 0.20 < cy_rel < 0.85:
                        if max_check_x + int(20 * scale) < cx < min_lock_x - int(20 * scale):
                            if mask_cleared[cy, cx] == 0:
                                return (t.center[0], t.center[1], t.center_rel)

        return None

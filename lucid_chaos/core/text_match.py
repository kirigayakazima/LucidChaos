"""
文字匹配工具（L1「文字锁定」的底座）。

为什么单独抽出来：
  1. 原来只有 `VisionService._text_similarity` 一份，藏在核心视觉服务里；
     "场景判定/文字锁定"也要用它，但从 `router` 反过来 import 视觉服务（带 RapidOCR）太重。
  2. 真机教训：OCR 会把文字**认掉一两个字** —— 实测 `配置隊伍` 被读成 **`配置伍`**（掉了"隊"）。
     所以"文字限定死"必须是**模糊匹配 + 组合条件**，不能是精确子串。

相似度规则（与改造前 `VisionService._text_similarity` 完全一致，只是挪了位置）：
  - 繁→简映射后，一边包含另一边 -> 1.0
  - 否则 = 交集字数 / 关键字长度
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

# 繁->简 映射：只覆盖游戏里高频出现的字（与改造前保持一致，未做扩充）
TRAD_SIMP = {'結': '结', '戰': '战', '鬥': '斗', '獲': '获', '點': '点', '門': '门',
             '鎖': '锁', '進': '进', '隊': '队', '伍': '伍'}


def normalize(s: str) -> str:
    return "".join(TRAD_SIMP.get(c, c) for c in (s or ""))


def text_similarity(s1: str, s2: str) -> float:
    """中文模糊相似度（0.0~1.0）。与改造前 `VisionService._text_similarity` 逐位一致。"""
    if not s1 or not s2:
        return 0.0
    a, b = normalize(s1), normalize(s2)
    if a in b or b in a:
        return 1.0
    common = set(a) & set(b)
    return len(common) / max(1, len(a))


def fuzzy_hit(tokens: Iterable[str], keyword: str, min_sim: float = 0.7) -> Optional[str]:
    """
    在 OCR 文本块里找 `keyword`：先精确包含，再模糊匹配。
    命中返回实际那一块的文本（便于写进日志/证据），否则 None。
    """
    key = normalize(keyword)
    best, best_sim = None, 0.0
    for t in tokens:
        if not t:
            continue
        if key and key in normalize(t):
            return t
        sim = text_similarity(keyword, t)
        if sim >= min_sim and sim > best_sim:
            best, best_sim = t, sim
    return best


def lock_hit(tokens: Sequence[str], groups: Sequence[Sequence[str]],
             min_sim: float = 0.7) -> Optional[list]:
    """
    "组合条件"文字锁定：`groups` 里**每一组**都要命中（组内任一词命中即可）。
    返回命中明细 `[(组, 实际文本), ...]`，全部命中才返回；否则 None。
    """
    hits = []
    for group in groups:
        got = None
        for kw in group:
            got = fuzzy_hit(tokens, kw, min_sim=min_sim)
            if got:
                break
        if not got:
            return None
        hits.append((group[0], got))
    return hits

"""
「文字 + 位置校验」单测（`VisionService.pick_text / pick_texts`）：

    python -m lucid_chaos.tests.test_text_match

背景（真机踩坑，值得单独测）：
  关卡详情抽屉里有一句正文 `和阿黛海特一起進入夢中的世界` 含「進入」。
  旧写法 `locate_text('進入')` 会命中**这句正文**（rx 0.742, ry 0.335），
  而真正的【進入】按钮在右下 (rx 0.80, ry 0.92)。
  位置校验 ry>0.75 于是失败 -> 跳过"点進入"分支 -> 反复点节点 -> 抽屉反复打开 -> **永远进不去**。

修复：① 只在"按钮该在的区域"里模糊匹配 ② OCR 没读到就用形状兜底（右下宽按钮）。
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from lucid_chaos.core.vision_service import VisionService, VisionTarget
from lucid_chaos.core.text_match import text_similarity, fuzzy_hit

FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def T(text, rx, ry, conf=0.9):
    return VisionTarget(text=text, confidence=conf, bbox=(0, 0, 10, 10),
                        center=(int(rx * 1000), int(ry * 1000)), center_rel=(rx, ry), source="ocr")


vis = VisionService()

print("=== 1) 抽屉场景：正文与按钮都含「進入」 ===")
targets = [
    T("#06／進入萝境", 0.62, 0.26),            # 抽屉标题（含進入）
    T("和阿黛海特一起進入夢中的世界", 0.742, 0.335),   # 正文（含進入）<- 真机上旧写法就命中这句
    T("進入", 0.802, 0.921),                    # 真正的按钮
]


def old_pick(ts, kw):
    """复现旧 `locate_text` 的挑选规则：confidence*sim 最大者胜（没有区域、没有长度偏好）。"""
    best, bk = None, 0.0
    for t in ts:
        sim = vis._text_similarity(kw, t.text)
        if sim < 0.7 or t.confidence < 0.4:
            continue
        k = t.confidence * sim
        if k > bk:
            best, bk = t, k
    return best


naive = old_pick(targets, "進入")                                   # 旧行为
new_noregion = vis.pick_text(targets, "進入", min_score=0.4)        # 新规则（无区域）
guarded = vis.pick_text(targets, "進入", min_score=0.4, region=(0.55, 0.80, 1.0, 1.0))
print(f"   旧规则            -> {naive.text!r} ({naive.center_rel})")
print(f"   新规则(不加区域)   -> {new_noregion.text!r} ({new_noregion.center_rel})")
print(f"   新规则(加区域)     -> {guarded.text!r} ({guarded.center_rel})")
check("旧规则命中正文/标题（这就是真机上'永远进不去'的原因）", naive.text != "進入", f"{naive.text!r}")
check("新规则即使不加区域也能挑中按钮（多了『按钮名要短』偏好）",
      new_noregion is not None and new_noregion.text == "進入", f"{new_noregion and new_noregion.text!r}")
check("新规则加区域同样挑中按钮", guarded is not None and guarded.text == "進入", f"{guarded and guarded.text!r}")
check("按钮位置满足 ry>0.80", guarded.center_rel[1] > 0.80, f"{guarded.center_rel}")

print("\n=== 2) 结算：多个同义关键字按顺序挑，且限定右下 ===")
settle = [
    T("獎勵", 0.64, 0.72),
    T("確認領取所有獎勵", 0.60, 0.80),      # 正文（含確認）但偏左
    T("確認", 0.877, 0.917),                # 按钮
]
got = vis.pick_texts(settle, ("確認", "确定", "完成"), min_score=0.4, region=(0.55, 0.60, 1.0, 1.0))
print(f"   -> {got.text!r} {got.center_rel}")
check("挑到右下那个【確認】按钮", got is not None and got.text == "確認", f"{got and got.text!r}")

print("\n=== 3) 模糊匹配：OCR 掉字也要能命中 ===")
check("配置隊伍 vs 配置伍 相似度 0.75", abs(text_similarity("配置隊伍", "配置伍") - 0.75) < 1e-6)
check("模糊命中 配置隊伍 -> 配置伍", fuzzy_hit(["配置伍", "進入"], "配置隊伍") == "配置伍")
check("繁体/简体互通（進入/进入）", text_similarity("進入", "进入") == 1.0)

print("\n=== 4) 区域过滤不误伤：区域外有匹配也要拒绝 ===")
only_outside = [T("進入", 0.74, 0.33)]
check("区域外的『進入』被拒绝", vis.pick_text(only_outside, "進入", region=(0.55, 0.72, 1.0, 1.0)) is None)
check("不加区域时能拿到（说明拒绝确实来自区域）",
      vis.pick_text(only_outside, "進入") is not None)

print("\n=== 5) 不重新 OCR：传空列表就直接没有结果 ===")
check("空 targets -> None", vis.pick_text([], "進入") is None)
check("pick_texts 空 targets -> None", vis.pick_texts([], ("確認", "進入")) is None)

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)

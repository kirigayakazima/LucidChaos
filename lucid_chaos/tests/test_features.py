"""
形状特征原语测试（合成图 + 真实夹具基线）：

    python -m lucid_chaos.tests.test_features

两部分：
  A) **合成图**：自己画条/按钮/圆，验证原语真的能认出来（离线可验证）
  B) **真实夹具基线**：跑一遍 8 张夹具，量出"哪些特征本来就一直在（属于 HUD）"、
     "哪些是某界面特有"——这是给各场景定判据前的标定数据。
     已知结论：左上角绿色 HP 条**所有界面都有**（属于常驻 HUD），不能拿它当战斗判据。
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import cv2
import numpy as np

from lucid_chaos.core import features as F

FIXTURES = _HERE.parent / "fixtures"
W, H = 2560, 1440
FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


print("=== A) 合成图：绿色 HP 条 ===")
syn = np.zeros((H, W, 3), dtype=np.uint8)
green_bgr = (80, 200, 80)          # 绿色（BGR）
bar_boxes = [(60, 60, 380, 14), (300, 92, 300, 12), (300, 120, 300, 12)]
for (x, y, w, h) in bar_boxes:
    cv2.rectangle(syn, (x, y), (x + w, y + h), green_bgr, -1)
bars = F.find_horizontal_bars(syn, "green", region=(0.0, 0.0, 0.45, 0.15))
print("   检出条数:", len(bars), [(b['cx'], b['cy'], b['w'], b['h']) for b in bars])
check("三条绿色长条都检出", len(bars) == 3, f"{len(bars)}")
check("长条按从上到下排序", all(bars[i]['cy'] <= bars[i + 1]['cy'] for i in range(len(bars) - 1)))
check("没有绿色时不误报", F.find_horizontal_bars(syn, "cyan", region=(0.0, 0.0, 0.45, 0.15)) == [])

print("\n=== B) 合成图：底部中央橙色按钮 ===")
syn2 = np.zeros((H, W, 3), dtype=np.uint8)
cv2.rectangle(syn2, (950, 1300), (1600, 1400), (0, 140, 240), -1)     # 橙色
btn = F.find_button(syn2, region=(0.30, 0.88, 0.70, 0.98))
print("   检出:", btn)
check("检出橙色按钮且质心落在按钮内",
      btn is not None and btn['color'] == 'orange' and 950 < btn['cx'] < 1600 and 1300 < btn['cy'] < 1400,
      f"{btn}")
check("跨度足够大（>=10%w）", btn is not None and btn['span'] >= 0.10 * int(0.40 * W), f"{btn and btn['span']}")
check("右下角区域的按钮不会被底部中央规则误抓",
      F.find_button(syn2, region=(0.30, 0.88, 0.70, 0.98), min_pixels=8000) is not None)

print("\n=== C) 合成图：圆形按钮（Hough） ===")
syn3 = np.zeros((H, W, 3), dtype=np.uint8)
cv2.circle(syn3, (2300, 1230), 60, (200, 200, 200), 4)
circles = F.find_circles(syn3, region=(0.80, 0.75, 1.0, 0.95))
print("   检出圆:", circles)
check("检出 1 个圆且位置接近", any(abs(cx - 2300) < 30 and abs(cy - 1230) < 30 for cx, cy, _ in circles), f"{circles}")

print("\n=== D) 真实夹具基线（哪些特征常驻、哪些界面特有）===")
hdr = f"{'夹具':<32}{'左上绿条':>8}{'底中橙钮':>9}{'右下圆':>7}{'青色占比':>9}"
print(hdr)
rows = {}
for p in sorted(FIXTURES.glob("*.png")):
    img = imread(p)
    hb = len(F.find_horizontal_bars(img, "green", region=(0.0, 0.0, 0.45, 0.15)))
    # ⚠️ 只查橙色：生产里的"弹窗关闭"通用规则**只用橙色**（底部中央的關閉/確認）。
    # 紫色不算——战斗手牌里有大量紫色卡面，算紫色会误判（本测试曾经因此报过一次假失败）。
    ob = F.find_button(img, region=(0.30, 0.88, 0.70, 0.98), colors=("orange",))
    ob_span = ob['span'] if ob else 0
    ob = ob if (ob and ob_span >= int(0.10 * img.shape[1])) else None
    cc = len(F.find_circles(img, region=(0.80, 0.75, 1.0, 0.95)))
    cy = F.hsv_ratio(img, "cyan", region=(0.01, 0.62, 0.42, 0.98))
    rows[p.name] = dict(hp_bars=hb, orange_btn=ob is not None, circles=cc, cyan=round(cy, 4))
    print(f"{p.name:<32}{hb:>8}{str(ob is not None):>9}{cc:>7}{cy*100:>8.2f}%")

check("左上绿色 HP 条**多数**界面能看到，但压暗界面会看不到（所以它既不是常驻、也不是战斗特有 -> 不能当战斗判据）",
      sum(1 for r in rows.values() if r['hp_bars'] >= 1) >= 6,
      f"{ {k: v['hp_bars'] for k, v in rows.items()} }")
check("底部中央橙钮只在 card_detail_popup 上出现（弹窗关闭规则有效）",
      [k for k, v in rows.items() if v['orange_btn']] == ["card_detail_popup.png"],
      f"{[k for k, v in rows.items() if v['orange_btn']]}")
check("战斗手牌不会误触发'底部中央橙钮'规则（战斗图只有 9 个橙色像素）",
      rows.get("battle_hand_row.png", {}).get('orange_btn') is False)
# 注意（2026/10/6 真机新夹具带来的风险）：
# dialogie_lei（角色对白）的青色占比实测 2.96%，与选路判定用的 3% 阈值只差 0.04%！
# 也就是说"青色面板占比"这个判据对"对白界面"的余量很薄。
# 当前之所以没事：对白界面靠「剧情对白演出 75 分」胜出（那帧肉鸽分 = 0）。
# 记在案：下一步要给选路找更强的 true-negative 判据（例如青色面板的形状/跨度，而不是纯占比）。
check("选路图青色占比 >= 3%",
      all(rows[f"rogue_map_{s}.png"]['cyan'] >= 0.03 for s in ("bright", "dim"))
      and rows["rogue_map_real.png"]['cyan'] >= 0.03,
      f"{ {k: round(v['cyan']*100, 2) for k, v in rows.items() if k.startswith('rogue')} }")
check("（已知薄余量）对白图青色占比 < 3%，仅差 0.04%",
      rows.get("dialogue_lei.png", {}).get('cyan', 1) < 0.03,
      f"dialogue_lei={round(rows.get('dialogue_lei.png', {}).get('cyan', 0)*100, 2)}%")

print("\n=== E) 布局原语：按列聚类 ===")
from collections import namedtuple  # noqa: E402
T = namedtuple("T", "text center_rel")
items = [T("卡1", (0.285, 0.31)), T("卡2", (0.489, 0.31)), T("卡3", (0.728, 0.31)), T("旁白", (0.49, 0.21))]
cols = F.find_columns(items, band=(0.28, 0.65), gap_ratio=0.06)
check("三张卡聚成 3 列", len(cols) == 3, f"{[ [i.text for i in c] for c in cols ]}")
check("竖带外的文本被排除", all(all(i.text != "旁白" for i in c) for c in cols))
check("列按 x 从左到右", [c[0].text for c in cols] == ["卡1", "卡2", "卡3"])

print("\n=== F) 卡牌 N 选 1 布局特征（形状优先，不看关键字）===")
three = [T("摆3张随機中立稀有～傅説卡", (0.287, 0.311)), T("原子分解", (0.489, 0.311)),
         T("满溢的回聲", (0.728, 0.312))]
two = [T("原子分解", (0.20, 0.31)), T("无法获得游侠", (0.50, 0.46))]
uneven = [T("a", (0.10, 0.31)), T("b", (0.30, 0.31)), T("c", (0.75, 0.31))]
lay3 = F.card_layout(three)
lay2 = F.card_layout(two)
lay_un = F.card_layout(uneven)
print("   三列等距 :", lay3)
print("   两列     :", lay2)
print("   三列不等距:", lay_un)
check("三列等距 -> ok", lay3['ok'] is True, f"{lay3}")
check("两列 -> 不成立", lay2['ok'] is False, f"{lay2}")
check("三列但间距不均 -> 不成立（不是所有三列都算卡牌界面）", lay_un['ok'] is False, f"{lay_un}")

print("\n=== G) 真实夹具：布局特征只在 3 选 1 那张成立 ===")
from lucid_chaos.core.vision_service import VisionService  # noqa: E402
_vis = VisionService()
for p in sorted(FIXTURES.glob("*.png")):
    img = imread(p)
    ts = _vis.locate_all_texts(img, min_score=0.5)
    lay = F.card_layout(ts)
    btn = F.find_button(img, region=(0.78, 0.86, 0.99, 0.98))
    print(f"   {p.name:<32} 列数={lay['columns']} ok={lay['ok']} 间距={lay['spacing']} 右下按钮={bool(btn)}")
    if p.name == "card_reward_3choose.png":
        check("3 选 1 那张：布局成立", lay['ok'] is True, f"{lay}")
        check("3 选 1 那张：确认按钮是**灰白色**（不在 orange/purple 里 -> 布局特征不能靠按钮颜色）",
              btn is None, f"{btn and btn['color']}")

print("\n=== H) 亮背景特征：只属于「外侧选关」大地图 ===")
print(hdr)
bright_rows = {}
for p in sorted(FIXTURES.glob("*.png")):
    img = imread(p)
    br = F.bright_ratio(img, region=(0.05, 0.20, 0.95, 0.90))
    bright_rows[p.name] = round(br, 4)
    print(f"{p.name:<32}{'':>8}{'':>9}{'':>7}{br*100:>8.2f}%")
check("选关图亮背景占比 >= 0.60（含细则抽屉那张）",
      bright_rows.get("stage_map_chapter1.png", 0) >= 0.60
      and bright_rows.get("stage_map_drawer.png", 0) >= 0.60,
      f"{bright_rows.get('stage_map_chapter1.png')} / {bright_rows.get('stage_map_drawer.png')}")
# 注意（真机发现）：**配队界面也是亮背景**（同为浅色 UI），所以"亮背景"不能单独区分它和选关大地图。
# 配队界面靠 L1 文字锁定（`配置隊伍`+`進入`）定案，不参与打分池，因此不会被亮背景判据误判。
check("只有 stage_map_* 与配队界面是亮背景（其它全部 <0.60）",
      all(v < 0.60 for k, v in bright_rows.items()
          if not (k.startswith("stage_map") or k == "team_config_enter.png")),
      f"{ {k: v for k, v in bright_rows.items() if v >= 0.60} }")
check("（已知共用）配队界面也是亮背景 -> 证明它必须用文字锁定，不能靠亮背景",
      bright_rows.get("team_config_enter.png", 0) >= 0.60,
      f"team_config={bright_rows.get('team_config_enter.png')}")

print("\n=== I) 过渡/全黑帧判定（真机踩到：黑屏时拿上一帧场景乱点）===")
blank_img = np.zeros((H, W, 3), dtype=np.uint8)
check("纯黑 -> blank", F.is_blank_frame(blank_img) is True,
      f"stats={F.blank_stats(blank_img)}")
mean_rows = {}
for p in sorted(FIXTURES.glob("*.png")):
    img = imread(p)
    ts = _vis.locate_all_texts(img, min_score=0.5)
    mean_rows[p.name] = (round(F.blank_stats(img)[0], 1),
                         F.is_blank_frame(img, targets=ts),
                         F.count_game_texts(ts))
for k, v in mean_rows.items():
    print(f"   {k:<32} 亮度均值={v[0]:>6}  游戏文本={v[2]:>3}  blank={v[1]}")
check("真机过渡帧被认出（transition_black，0 块游戏文本）",
      mean_rows.get("transition_black.png", (0, False, 0))[1] is True)
# ⚠️ 关键回归：黑底白字剧情（均值同样极黑）**不能**被判成过渡帧，否则脚本什么都不做 -> 看门狗硬停
check("黑底剧情帧不被误判为过渡帧（有 2 块游戏文字）",
      mean_rows.get("dialogue_black_bg.png", (0, True, 0))[1] is False,
      f"{mean_rows.get('dialogue_black_bg.png')}")
check("其它界面都不被误判为过渡帧",
      all(not v[1] for k, v in mean_rows.items() if k != "transition_black.png"),
      f"{[k for k, v in mean_rows.items() if v[1]]}")
check("黑底剧情帧的游戏文字数为正（判据来源）",
      mean_rows.get("dialogue_black_bg.png", (0, False, 0))[2] >= 1)

print("\n=== J) 橙色 XP 条：结算界面特有 ===")
xp_rows = {}
for p in sorted(FIXTURES.glob("*.png")):
    img = imread(p)
    xp_rows[p.name] = len(F.find_horizontal_bars(img, "orange", region=(0.55, 0.08, 1.0, 0.30)))
    print(f"   {p.name:<32} 右上橙条={xp_rows[p.name]}")
check("结算图有橙条", xp_rows.get("settlement_stage_reward.png", 0) >= 1)
check("选关抽屉没有橙条（所以两者可区分）", xp_rows.get("stage_map_drawer.png", 0) == 0,
      f"{xp_rows.get('stage_map_drawer.png')}")

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)

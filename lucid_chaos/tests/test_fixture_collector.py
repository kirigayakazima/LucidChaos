"""
自动取证单测（只写临时目录，不碰真实 fixture）：

    python -m lucid_chaos.tests.test_fixture_collector

验证：
  1. 同一幕连续停留 >= 阈值 -> 存一张，且**同一幕只存一次**
  2. 画面变了 -> 重新计；再次卡住 -> 再存一张
  3. 本场首次出现的 (screen, route) 组合 -> 存一张（界面清单）
  4. 上限裁剪：文件数不超过 max_files
  5. 关掉开关后什么都不存
  6. 每张图都有同名 .txt 说明
"""
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

import cv2
import numpy as np

from lucid_chaos.core.fixture_collector import FixtureCollector

FIXTURES = _HERE.parent / "fixtures"
FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)


A = imread(FIXTURES / "rogue_map_bright.png")
B = imread(FIXTURES / "card_reward_3choose.png")
C = imread(FIXTURES / "event_box_survival_signal.png")

tmp = Path(tempfile.mkdtemp(prefix="dsh_fixcol_"))

print("=== 1/2) 卡住取证：同幕只存一次，换画面后能再存 ===")
col = FixtureCollector(tmp / "c1", max_files=16, stuck_threshold=3)
# 先用掉「界面清单」那一次（首次出现该组合必然存一张）
p0 = col.tick(A, screen="EVENT", route="局内互动事件")
check("首次出现的组合按清单存 1 张", p0 is not None, f"{p0 and p0.name}")
saved = [col.tick(A, screen="EVENT", route="局内互动事件", decision="route:局内互动事件") for _ in range(5)]
print("   随后同画面 5 帧 ->", [None if p is None else p.name for p in saved])
check("阈值(3)之前不存", saved[0] is None and saved[1] is None, f"{[None if p is None else p.name for p in saved[:2]]}")
check("达到阈值存 1 张", saved[2] is not None and "stuck3" in saved[2].name, f"{saved[2] and saved[2].name}")
check("同一幕后续不再重复存", all(p is None for p in saved[3:]))
p2 = col.tick(B, screen="EPIPHANY_SELECTION", route="灵光一闪选牌")
check("换画面 -> 先按'新界面'存一张", p2 is not None, f"{p2 and p2.name}")
again = [col.tick(B, screen="EPIPHANY_SELECTION", route="灵光一闪选牌") for _ in range(3)]
check("新画面同幕停留到阈值 -> 再存一张", any(p is not None for p in again),
      f"{[None if p is None else p.name for p in again]}")

print("\n=== 3) 界面清单取证：本场首次出现的组合 ===")
tmp3 = tmp / "c3"
col3 = FixtureCollector(tmp3, max_files=16, stuck_threshold=99)   # 阈值拉高，只看清单逻辑
paths = [col3.tick(A, screen="EVENT", route="局内互动事件"),
         col3.tick(A, screen="EVENT", route="局内互动事件"),
         col3.tick(B, screen="EPIPHANY_SELECTION", route="灵光一闪选牌")]
check("首次出现存 1 张", paths[0] is not None and paths[2] is not None,
      f"{[None if p is None else p.name for p in paths]}")
check("重复组合不重复存", paths[1] is None)
check("每张图都有同名 .txt", all(p.with_suffix(".txt").exists() for p in paths if p is not None))
txt = (paths[0].with_suffix(".txt")).read_text(encoding="utf-8")
check(".txt 里有 screen/route/decision", "screen=EVENT" in txt and "route=局内互动事件" in txt
      and "decision=" in txt, txt.splitlines()[0])

print("\n=== 4) 上限裁剪 ===")
tmp4 = tmp / "c4"
col4 = FixtureCollector(tmp4, max_files=3, stuck_threshold=99)
for i, img in enumerate((A, B, C, A, B, C, A)):
    # 每次换一个新组合，强制每轮都存
    col4.tick(img, screen=f"S{i}", route=f"R{i}")
files = sorted(tmp4.glob("*.png"))
check("png 数量不超过 max_files=3", len(files) <= 3, f"实际 {len(files)}")
check("留下的是最新几张", all("S6" in f.name or "S5" in f.name or "S4" in f.name for f in files),
      f"{[f.name for f in files]}")

print("\n=== 5) 关闭开关 ===")
tmp5 = tmp / "c5"
col5 = FixtureCollector(tmp5, enabled=False, stuck_threshold=2)
r = [col5.tick(A, screen="EVENT", route="X") for _ in range(5)]
check("关掉后不存任何文件", all(p is None for p in r) and not list(tmp5.glob("*.png")) if tmp5.exists() else all(p is None for p in r))

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)

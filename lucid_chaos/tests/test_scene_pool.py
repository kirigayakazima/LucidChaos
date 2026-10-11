"""
打分池决策逻辑单测（纯逻辑，不需要图像）：

    python -m lucid_chaos.tests.test_scene_pool

重点验证防抖（debounce）的**正确语义**——这是真机上踩到的坑：
屏幕上已经是角色对白（剧情对白演出 75 分），只因第二候选 65 分、分差 10 < 15，
旧逻辑就把上一帧的 [外层选关] 强行保留 2 帧 → 脚本去拖动地图 → 看门狗 20 秒超时 ESC。

修正后的规则：
  - 分数够（>= min_threshold）但分差小  -> **采用最强候选**（不能抱住过期场景）
  - 分数不够（< min_threshold）        -> 才允许防抖保留上一稳定场景（最多 2 帧）
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from lucid_chaos.router.scene_pool import ScenePoolClassifier
from lucid_chaos.router.scene_types import SceneType

FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def scored(*pairs):
    """pairs: (SceneType, score) -> (scored_list, scores_map)"""
    lst = [(s, float(v), {}) for s, v in pairs]
    mp = {s.value: float(v) for s, v in pairs}
    # 补足候选数（_decide 只用前两名）
    return lst, mp


print("=== 1) 分数够 + 分差够 -> 直接采用（条件 A）===")
clf = ScenePoolClassifier(min_threshold=50.0, min_margin=15.0)
sl, mp = scored((SceneType.STAGE_MAP, 100), (SceneType.EPIPHANY_CARDS, 30))
res = clf._decide(sl, mp)
check("外层选关 100/30 -> 外层选关", res.scene == SceneType.STAGE_MAP and res.margin == 70, f"{res.scene.value}")

print("\n=== 2) 分数够 + 分差小 -> 必须采用最强候选（真机坑的回归）===")
clf = ScenePoolClassifier(min_threshold=50.0, min_margin=15.0)
clf._last_stable_scene = SceneType.STAGE_MAP          # 上一帧稳定场景 = 外层选关
clf._jitter_count = 0
sl, mp = scored((SceneType.CUTSCENE_DIALOG, 75), (SceneType.ROGUELIKE_MAP, 65))
res = clf._decide(sl, mp)
check("对白 75 / 肉鸽 65（分差 10）-> 应当采信【剧情对白演出】，不能沿用外层选关",
      res.scene == SceneType.CUTSCENE_DIALOG, f"{res.scene.value} conf={res.confidence}")
check("标记了 low_margin 证据", res.details.get("low_margin") == 10.0, f"{res.details}")
check("稳定场景已更新为对白", clf._last_stable_scene == SceneType.CUTSCENE_DIALOG)

print("\n=== 3) 分数不够 -> 仍允许防抖保留上一稳定场景（最多 2 帧）===")
clf = ScenePoolClassifier(min_threshold=50.0, min_margin=15.0)
clf._last_stable_scene = SceneType.STAGE_MAP
clf._jitter_count = 0
sl, mp = scored((SceneType.SETTLEMENT_LOOT, 35), (SceneType.EVENT_CHOICE, 30))
r1 = clf._decide(sl, mp)
r2 = clf._decide(sl, mp)
r3 = clf._decide(sl, mp)
check("第 1 帧防抖 -> 沿用外层选关", r1.scene == SceneType.STAGE_MAP and r1.details.get("debounced"))
check("第 2 帧防抖 -> 仍沿用", r2.scene == SceneType.STAGE_MAP and r2.details.get("debounced"))
check("第 3 帧缓冲用尽 -> 落到 UNKNOWN 兜底", r3.scene == SceneType.UNKNOWN,
      f"{r3.scene.value} {r3.details}")

print("\n=== 4) 高分直通（条件 B，>=80 忽略分差）===")
clf = ScenePoolClassifier(min_threshold=50.0, min_margin=15.0)
sl, mp = scored((SceneType.ROGUELIKE_MAP, 90), (SceneType.STAGE_MAP, 85))
res = clf._decide(sl, mp)
check("肉鸽 90 / 选关 85 -> 肉鸽", res.scene == SceneType.ROGUELIKE_MAP, f"{res.scene.value}")

print("\n=== 5) 全都很低 -> UNKNOWN 兜底 ===")
clf = ScenePoolClassifier(min_threshold=50.0, min_margin=15.0)
sl, mp = scored((SceneType.BATTLE, 20), (SceneType.STAGE_MAP, 10))
res = clf._decide(sl, mp)
check("20/10 -> UNKNOWN", res.scene == SceneType.UNKNOWN, f"{res.scene.value} {res.details}")

print("\n=== 6) 真机日志里的两条序列回放 ===")
# 15:06:36 -> 0/0（防抖第 1 帧）；15:06:51 -> 对白 75 / 肉鸽 65
clf = ScenePoolClassifier(min_threshold=50.0, min_margin=15.0)
clf._last_stable_scene = SceneType.STAGE_MAP
sl0, mp0 = scored((SceneType.STAGE_MAP, 0), (SceneType.EVENT_CHOICE, 0))
a = clf._decide(sl0, mp0)
print(f"   第1帧 全0 -> {a.scene.value} ({a.details})")
check("全 0 时防抖保留外层选关（这一步原来没错）", a.scene == SceneType.STAGE_MAP)
sl1, mp1 = scored((SceneType.CUTSCENE_DIALOG, 75), (SceneType.ROGUELIKE_MAP, 65))
b = clf._decide(sl1, mp1)
print(f"   第2帧 对白75/肉鸽65 -> {b.scene.value}")
check("下一步必须切到对白（修好后不会再拖地图）", b.scene == SceneType.CUTSCENE_DIALOG)

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)

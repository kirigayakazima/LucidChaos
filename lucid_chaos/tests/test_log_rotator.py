"""
日志滚动单测（不碰真实日志，只操作临时副本）：

    python -m lucid_chaos.tests.test_log_rotator

验证主人提的两条要求：
  1. 只保留最近 3 次运行（第 4 次重启，第 1 次的内容被删掉）
  2. 每次运行最多 200 行
"""
import logging
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
try:
    sys.stdout.reconfigure(errors="replace")
except Exception:
    pass

from lucid_chaos.core.log_rotator import (RunCappedFileHandler, load_runs, prepare_log_file,
                                      split_runs, trim_runs, write_runs)

FAILED = []


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def make_run(n_lines: int, tag: str) -> list:
    return ["[00:00:00] [INFO] [ChaosBot] ==========================================",
            f"[00:00:00] [INFO] [ChaosBot]    《卡厄思梦境》PC端自动化挂机脚本启动 {tag}",
            "[00:00:00] [INFO] [ChaosBot] =========================================="] + \
           [f"[00:00:0{i%60}] [INFO] [Orchestrator] line-{tag}-{i}" for i in range(n_lines)]


tmp = Path(tempfile.mkdtemp(prefix="dsh_logtest_"))
log = tmp / "chaos_bot.log"

print("=== 1) 切分运行段 ===")
runs = split_runs(make_run(10, "A") + make_run(5, "B"))
check("切成 2 段", len(runs) == 2, f"实际 {len(runs)}")
check("每段都以启动横幅开头", "挂机脚本启动" in runs[0][1] and "挂机脚本启动" in runs[1][1])

print("\n=== 2) 只保留最近 3 次运行 ===")
all_lines = []
for i in range(1, 6):                     # 5 次运行
    all_lines += make_run(30, f"R{i}")
write_runs(log, [all_lines])
kept = prepare_log_file(log, max_runs=3, max_lines_per_run=200)
runs = load_runs(log)
check("启动时裁到 2 段（留一格给新运行）", len(runs) == 2 and kept == 2, f"实际 {len(runs)}")
check("留下的是最新的 R4 / R5",
      ("R4" in "\n".join(runs[0])) and ("R5" in "\n".join(runs[1])))
check("R1/R2/R3 已被删掉", not any(f"R{i}" in "\n".join(runs[0] + runs[1]) for i in (1, 2, 3)))

print("\n=== 3) 每次运行最多 200 行 ===")
log2 = tmp / "cap.log"
h = RunCappedFileHandler(str(log2), max_runs=3, max_lines_per_run=200)
h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"))
lg = logging.getLogger("captest")
lg.handlers = [h]
lg.setLevel(logging.INFO)
lg.propagate = False
lg.info("==========================================")
lg.info("   《卡厄思梦境》PC端自动化挂机脚本启动")
lg.info("==========================================")
for i in range(500):                       # 远超 200 行
    lg.info(f"noise-{i}")
runs = load_runs(log2)
cur = runs[-1]
check("当前运行段 <= 200 行", len(cur) <= 200, f"实际 {len(cur)}")
check("尾部内容还在（最后一行是 noise-499）", cur[-1].endswith("noise-499"))
check("开头是截断提示（头部保留了启动横幅，提示在前几行内）",
      any("已截断" in x for x in cur[:6]), " / ".join(x[:40] for x in cur[:3]))
check("文件总行数 <= 3*200", sum(len(r) for r in runs) <= 600, f"实际 {sum(len(r) for r in runs)}")

print("\n=== 4) 第 4 次重启后，第一次的内容消失 ===")
h2 = RunCappedFileHandler(str(log2), max_runs=3, max_lines_per_run=200)   # 模拟第 4 次启动
h2.setFormatter(h.formatter)
lg.handlers = [h2]
lg.info("==========================================")
lg.info("   《卡厄思梦境》PC端自动化挂机脚本启动")
lg.info("==========================================")
lg.info("new-run-marker-abc")
runs = load_runs(log2)
check("运行段数 <= 3", len(runs) <= 3, f"实际 {len(runs)}")
check("旧运行（noise-*）已被挤掉到只剩 2 段以内",
      sum(1 for r in runs if any("noise-499" in x for x in r)) <= 1)
check("新运行内容在文件里", any("new-run-marker-abc" in x for r in runs for x in r))

print("\n=== 5) 对真实日志副本试跑（不动原文件）===")
real = _HERE.parent.parent / "chaos_bot.log"
if real.exists():
    import shutil
    copy = tmp / "real_copy.log"
    shutil.copy2(real, copy)
    before = len(copy.read_text(encoding="utf-8", errors="replace").splitlines())
    n_kept = prepare_log_file(copy, max_runs=3, max_lines_per_run=200)
    after_runs = load_runs(copy)
    after = sum(len(r) for r in after_runs)
    print(f"   副本 {before} 行 -> 保留 {n_kept} 段 / {after} 行")
    check("副本被裁到 <= 2*200 行", after <= 400, f"实际 {after}")
    check("原文件未被改动", len(real.read_text(encoding='utf-8', errors='replace').splitlines()) == before)
else:
    print("   （没有真实日志，跳过）")

print("\n=== 6) 往返稳定性：裁剪后再次切分，段数不能变少 ===")
log3 = tmp / "roundtrip.log"
runs_in = [make_run(400, f"S{i}") for i in (1, 2, 3)]
write_runs(log3, runs_in)
prepare_log_file(log3, max_runs=3, max_lines_per_run=200)
r1 = load_runs(log3)
check("首次裁剪后段数 == 2", len(r1) == 2, f"实际 {len(r1)}（截断丢了启动横幅就会并成 1 段）")
check("每段 <= 200 行", all(len(r) <= 200 for r in r1), f"各段 {[len(r) for r in r1]}")
prepare_log_file(log3, max_runs=3, max_lines_per_run=200)
r2 = load_runs(log3)
check("二次裁剪后段数仍 == 2（往返稳定）", len(r2) == 2, f"实际 {len(r2)}")
check("每段仍以启动横幅开头", all(any("挂机脚本启动" in x for x in r[:4]) for r in r2))

# handler 侧同样要保住运行头
log4 = tmp / "handler_head.log"
h3 = RunCappedFileHandler(str(log4), max_runs=3, max_lines_per_run=200)
h3.setFormatter(h.formatter)
lg.handlers = [h3]
lg.info("==========================================")
lg.info("   《卡厄思梦境》PC端自动化挂机脚本启动")
lg.info("==========================================")
for i in range(400):
    lg.info(f"h-noise-{i}")
r4 = load_runs(log4)
check("handler 截断后仍是 1 段且含启动横幅", len(r4) == 1 and any("挂机脚本启动" in x for x in r4[0][:4]),
      f"段数 {len(r4)}")
check("handler 段 <= 200 行", len(r4[-1]) <= 200, f"实际 {len(r4[-1])}")
check("handler 尾部内容在（h-noise-399）", r4[-1][-1].endswith("h-noise-399"))

print(f"\n失败 {len(FAILED)} 项" + (f": {FAILED}" if FAILED else ""))
sys.exit(1 if FAILED else 0)

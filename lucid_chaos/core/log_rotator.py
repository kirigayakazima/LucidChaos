"""
日志滚动：**最多保留最近 N 次运行，每次运行最多 M 行**。

主人要求：`chaos_bot.log` 只留最新的 3 次运行，第 4 次重启就把第 1 次删掉；
每次运行最多 200 行（"不需要那么多，不然很麻烦"）。

设计要点：
  - 一次运行的边界 = 启动横幅（`挂机脚本启动` 那一行，向前吞掉紧邻的 `===` 行）
  - 每次运行内部保留**最后** M 行（尾部最重要：卡死现场就在尾部），并在开头插一行截断提示
  - 写日志时用**内存缓冲 + 重写**：文件总长被硬性限制在 N*M 行以内（3*200=600 行 ≈ 60KB）
  - 阈值都可在 config.json 的 `logging` 段里改
"""

from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence

# 每次启动都会打的一行（见 main.py），用它当「本次运行」的分界
RUN_MARKER = "挂机脚本启动"
TRUNC_NOTE = "... [本段日志已截断，仅保留最后 {n} 行] ..."
_BANNER_ONLY = re.compile(r'={5,}\s*$')   # 注意：真实日志行带 "[时间] [INFO] [ChaosBot] " 前缀，不能只匹配纯 = 行


def split_runs(lines: Sequence[str]) -> List[List[str]]:
    """把日志正文按「运行」切开。"""
    runs: List[List[str]] = []
    cur: List[str] = []
    for idx, ln in enumerate(lines):
        if RUN_MARKER in ln:
            # 向前吞掉紧邻的纯 === 横幅行
            back = len(cur)
            while back > 0 and _BANNER_ONLY.search(cur[back - 1]):
                back -= 1
            tail = cur[back:]
            if back > 0:
                runs.append(cur[:back])
            cur = list(tail)
        cur.append(ln)
    if cur:
        runs.append(cur)
    return runs


def trim_run(run: List[str], max_lines: int) -> List[str]:
    """
    单次运行只保留最后 max_lines 行。

    注意：**必须保住这次运行的"启动横幅"**（否则下次切分时这段会被并进上一段——
    这是实际踩到的坑：真实日志裁完只剩 1 段）。所以截断后保留
    `运行头（含 挂机脚本启动 那一行） + 截断提示 + 尾部内容`，总数不超过 max_lines。
    """
    if max_lines <= 0 or len(run) <= max_lines:
        return list(run)

    # 运行头：到 marker 那一行为止（再带上紧随其后的横幅行）
    head: List[str] = []
    mi = next((i for i, ln in enumerate(run[:6]) if RUN_MARKER in ln), None)
    if mi is not None:
        head = list(run[:mi + 1])
        j = mi + 1
        while j < len(run) and _BANNER_ONLY.search(run[j]):
            head.append(run[j])
            j += 1
    if len(head) + 1 >= max_lines:            # 头太长就不留头（极端情况）
        head = []

    keep_tail = max(0, max_lines - len(head) - 1)
    tail = list(run[-keep_tail:]) if keep_tail else []
    return head + [TRUNC_NOTE.format(n=max_lines)] + tail


def trim_runs(runs: List[List[str]], max_runs: int, max_lines_per_run: int) -> List[List[str]]:
    """保留最后 max_runs 次运行，每次最多 max_lines_per_run 行。"""
    if max_runs > 0:
        runs = runs[-max_runs:]
    return [trim_run(r, max_lines_per_run) for r in runs]


def load_runs(path: Path) -> List[List[str]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return split_runs(text.splitlines())


def write_runs(path: Path, runs: List[List[str]]) -> None:
    # 容错：如果直接传了「行列表」（List[str]），当成单段运行，避免按字符写出去
    if runs and isinstance(runs[0], str):
        runs = [list(runs)]
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(ln for run in runs for ln in run)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def prepare_log_file(path: Path, max_runs: int = 3, max_lines_per_run: int = 200) -> int:
    """
    启动时调用：把已有日志裁到「最多 max_runs-1 次运行」（留一格给即将开始的新运行）。
    返回裁剪后保留的运行段数。
    """
    runs = load_runs(path)
    kept = trim_runs(runs, max(max_runs - 1, 0), max_lines_per_run)
    write_runs(path, kept)
    return len(kept)


class RunCappedFileHandler(logging.Handler):
    """
    只保留最近 max_runs 次运行、每次最多 max_lines_per_run 行的文件日志。

    实现：内存里维护「已保留的旧运行段」+「当前运行段的 deque(maxlen=M)」，
    每条日志后重写文件（总长 ≤ N*M 行，开销可忽略）。
    """

    def __init__(self, filename, max_runs: int = 3, max_lines_per_run: int = 200,
                 encoding: str = "utf-8"):
        super().__init__()
        self.path = Path(filename)
        self.max_runs = max(1, int(max_runs))
        self.max_lines = max(1, int(max_lines_per_run))
        self.encoding = encoding
        self._old_runs: List[List[str]] = load_runs(self.path)
        # 启动时裁掉过老的段；当前这一段由本 handler 自己维护
        self._old_runs = trim_runs(self._old_runs, max(self.max_runs - 1, 0), self.max_lines)
        # 留一行给「已截断」提示，保证一段总共不超过 max_lines
        self._cap = max(1, self.max_lines - 1)
        self._cur: Deque[str] = deque(maxlen=self._cap)
        self._cur_head: List[str] = []      # 本段运行的「启动横幅」（截断后要保住，避免下次切分合并）
        self._truncated = False
        write_runs(self.path, self._old_runs)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            if RUN_MARKER in msg:
                self._cur_head = [msg]      # 新一次运行开始
            elif self._cur_head and _BANNER_ONLY.search(msg) and len(self._cur_head) < 3:
                self._cur_head.append(msg)
            if len(self._cur) == self._cap:
                self._truncated = True
            self._cur.append(msg)
            cur_list = list(self._cur)
            if self._truncated:
                # 保住运行头（含 挂机脚本启动 行），否则下次切分会与上一段合并；
                # 总行数严格不超过 max_lines：head + 提示行 + 正文
                keep = max(0, self.max_lines - len(self._cur_head) - 1)
                cur_list = self._cur_head + [TRUNC_NOTE.format(n=self.max_lines)] + cur_list[-keep:] if keep else \
                    self._cur_head + [TRUNC_NOTE.format(n=self.max_lines)]
            write_runs(self.path, self._old_runs + [cur_list])
        except Exception:  # noqa: BLE001
            self.handleError(record)

    # 便于测试/诊断
    def stats(self) -> dict:
        return {"old_runs": len(self._old_runs), "cur_lines": len(self._cur),
                "max_runs": self.max_runs, "max_lines_per_run": self.max_lines}

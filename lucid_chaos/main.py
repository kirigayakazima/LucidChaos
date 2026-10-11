"""
Main entry point for Chaos Bot PC automation.
"""

import sys
import os
import json
import time
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ctypes
from pathlib import Path
from lucid_chaos.core.log_rotator import RunCappedFileHandler
from lucid_chaos.fsm.orchestrator import ChaosOrchestrator

def disable_quick_edit():
    """Disable Windows CMD QuickEdit mode to prevent mouse clicks from freezing the process."""
    try:
        kernel32 = ctypes.windll.kernel32
        hStdin = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(hStdin, ctypes.byref(mode)):
            # Remove ENABLE_QUICK_EDIT_MODE (0x0040) and keep ENABLE_EXTENDED_FLAGS (0x0080)
            new_mode = (mode.value & ~0x0040) | 0x0080
            kernel32.SetConsoleMode(hStdin, new_mode)
    except Exception:
        pass

def setup_logging(cfg: dict = None):
    # Force line buffering for immediate output
    # errors="replace": 控制台若是 GBK，日志里的 🔍/▷ 等字符只会变成 ?，
    # 不会触发 logging 的 UnicodeEncodeError（文件日志仍按 utf-8 完整记录）
    try:
        sys.stdout.reconfigure(line_buffering=True, errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(errors="replace")
    except Exception:
        pass

    cfg = cfg or {}
    log_cfg = cfg.get("logging", {}) if isinstance(cfg, dict) else {}
    max_runs = int(log_cfg.get("max_runs", 3))
    max_lines = int(log_cfg.get("max_lines_per_run", 200))

    log_path = Path(__file__).resolve().parent / "chaos_bot.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            # 只保留最近 max_runs 次运行、每次最多 max_lines 行
            RunCappedFileHandler(str(log_path), max_runs=max_runs, max_lines_per_run=max_lines),
        ]
    )

def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def main():
    disable_quick_edit()
    config = load_config()          # 先读配置：日志滚动上限也在里面
    setup_logging(config)
    logger = logging.getLogger("ChaosBot")
    logger.info("==========================================")
    logger.info("   《卡厄思梦境》PC端自动化挂机脚本启动")
    logger.info("   (基于 Roadmap v0.1 严格分层规范)")
    logger.info("==========================================")

    orchestrator = ChaosOrchestrator(config)

    try:
        while orchestrator.ctx.running:
            orchestrator.step()
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("用户手动中止挂机脚本，正在退出...")
    except Exception as e:
        logger.exception(f"运行时异常: {e}")
    finally:
        orchestrator.stop()
        logger.info("脚本已安全停止。")

if __name__ == "__main__":
    main()

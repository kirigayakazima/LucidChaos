"""
局内“识别到搜索气泡但点了没反应”取证诊断工具 (Click Effect Diagnosis).

Usage:
    # 1) 只取证（不点击）：抓一张画面 + 跑气泡检测 + 输出标注图
    python -m lucid_chaos.tools.diag_click

    # 2) 后台点击实验：用和挂机脚本完全相同的 PostMessage 点击，逐个对比画面变化
    python -m lucid_chaos.tools.diag_click --click-test

    # 3) 真实鼠标点击实验：把系统光标移到目标上真点一下（会动你的鼠标！）
    python -m lucid_chaos.tools.diag_click --real-click

Outputs (工作区根目录 diag_out/):
    00_frame_a.png        抓到的原始画面
    01_bubbles.png        气泡检测标注图（所有命中框 + 分数 + 相对坐标）
    02_click_*.png        每次实验点击后的画面
    03_real_click_*.png   真实鼠标实验后的画面
    report.txt            文本报告
"""

import sys
import time
import argparse
import ctypes
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 中文控制台(GBK)下遇到无法编码的字符时用 '?' 代替，避免整轮实验中断
try:
    sys.stdout.reconfigure(errors="replace")
    sys.stderr.reconfigure(errors="replace")
except Exception:  # noqa: BLE001
    pass

import cv2
import numpy as np

_orig_imwrite = cv2.imwrite


def _imwrite_unicode(path, img, *args, **kwargs):
    """cv2.imwrite 在 Windows 中文路径下会静默失败 -> 失败时改用 imencode + tofile。"""
    p = str(path)
    try:
        if _orig_imwrite(p, img, *args, **kwargs):
            return True
    except Exception:  # noqa: BLE001
        pass
    suffix = Path(p).suffix or ".png"
    ok, buf = cv2.imencode(suffix, img)
    if ok:
        buf.tofile(p)
        return True
    return False


cv2.imwrite = _imwrite_unicode

from lucid_chaos.core.window import GameWindow
from lucid_chaos.core.capture import ScreenCapture
from lucid_chaos.core.input import InputController
from lucid_chaos.logic.node_engine import NodeDecisionEngine

# 注意: 输出放在工作区根目录下的 diag_out（lucid_chaos 子树在受限会话里不可写）
OUT_DIR = Path(_project_root) / "diag_out"

user32 = ctypes.windll.user32


def frame_diff(a: np.ndarray, b: np.ndarray):
    """Return (changed_ratio, mean_abs_diff) on grayscale frames."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.int16)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.int16)
    if ga.shape != gb.shape:
        return -1.0, -1.0
    d = np.abs(ga - gb)
    return float((d > 18).mean()), float(d.mean())


def region_diff(a: np.ndarray, b: np.ndarray, cx: int, cy: int, half: int = 180):
    """Changed ratio inside a square region around (cx, cy)."""
    h, w = a.shape[:2]
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(w, cx + half), min(h, cy + half)
    return frame_diff(a[y1:y2, x1:x2], b[y1:y2, x1:x2])


def annotate(img: np.ndarray, bubbles, tpl_shape) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]
    for i, b in enumerate(bubbles):
        x, y, bw, bh = b['bbox']
        color = (0, 255, 255) if i == 0 else (0, 255, 0)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 3)
        cv2.drawMarker(vis, (b['cx'], b['cy']), color, cv2.MARKER_CROSS, 24, 2)
        label = f"#{i} {b['score']:.2f} ({b['rx']:.3f},{b['ry']:.3f})"
        ty = max(24, y - 8)
        cv2.putText(vis, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(vis, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.drawMarker(vis, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 40, 2)
    info = f"frame {w}x{h}  tpl {tpl_shape[1]}x{tpl_shape[0]}  bubbles={len(bubbles)}"
    cv2.putText(vis, info, (16, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def move_cursor(win: GameWindow, cx: int, cy: int):
    sx, sy, _, _ = win.get_client_rect()
    user32.SetCursorPos(int(sx + cx), int(sy + cy))
    time.sleep(0.15)


def real_click(win: GameWindow, cx: int, cy: int):
    """Move the physical cursor and click (foreground input)."""
    move_cursor(win, cx, cy)
    time.sleep(0.1)
    user32.mouse_event(0x0002, 0, 0, 0, 0)   # MOUSEEVENTF_LEFTDOWN
    time.sleep(0.08)
    user32.mouse_event(0x0004, 0, 0, 0, 0)   # MOUSEEVENTF_LEFTUP


def park_cursor():
    """Park the physical cursor far away from the game's interactive area."""
    user32.SetCursorPos(40, 40)
    time.sleep(0.1)


def run_probe(win: GameWindow, cap: ScreenCapture, inp: InputController,
              cx: int, cy: int, cw: int, ch: int, log, flush, base_changed: float) -> None:
    """
    对照实验（命中第一个有效方式就停）:
      S1 后台PostMessage，光标停远处、不抢前台   -> 复现挂机脚本的当前行为
      S2 光标移到气泡上 + 后台PostMessage        -> 游戏是否按真实光标位置做命中
      S3 光标移到气泡上 + 真实鼠标点击(窗口未聚焦) -> 是否只有真实输入才被接受
      S4 把游戏切到前台 + 后台PostMessage(光标停远处) -> 是否只要求窗口在前台
    """
    import win32gui

    def effective(prev, cur, px, py):
        changed, _ = frame_diff(prev, cur)
        reg, _ = region_diff(prev, cur, px, py, half=220)
        ok = changed > max(0.005, base_changed * 3) or reg > max(0.01, base_changed * 4)
        return ok, changed, reg

    def set_foreground(hwnd, note=""):
        try:
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.4)
            fg = win32gui.GetForegroundWindow()
            log(f"    [FOCUS] 前台窗口 -> {'游戏' if fg == win.hwnd else hex(fg)} {note}")
        except Exception as e:  # noqa: BLE001
            log(f"    [WARN] 切换前台失败: {e}")

    def defocus_to_console():
        """让游戏失去前台，恢复到“脚本后台运行”的状态。"""
        try:
            cons = user32.GetConsoleWindow()
            if cons:
                win32gui.SetForegroundWindow(cons)
                time.sleep(0.4)
        except Exception:  # noqa: BLE001
            pass

    steps = [
        ("S1 后台PostMessage (光标停远处/不抢前台) = 脚本当前行为", "post"),
        ("S2 光标移到气泡 + 后台PostMessage", "cursor_post"),
        ("S3 光标移到气泡 + 真实鼠标点击 (窗口保持未聚焦)", "real"),
        ("S4 游戏切到前台 + 后台PostMessage (光标停远处)", "focus_post"),
    ]

    prev = cap.capture()
    for i, (label, kind) in enumerate(steps, start=1):
        log(f"\n[PROBE {i}] {label}  目标=({cx},{cy})")
        if kind in ("post", "cursor_post"):
            defocus_to_console()
        try:
            if kind == "post":
                park_cursor()
                inp.click_client(cx, cy, delay_after=1.5, jitter=4)
            elif kind == "cursor_post":
                park_cursor()
                move_cursor(win, cx, cy)
                inp.click_client(cx, cy, delay_after=1.5, jitter=4)
            elif kind == "real":
                defocus_to_console()
                park_cursor()
                real_click(win, cx, cy)
                time.sleep(1.5)
            elif kind == "focus_post":
                park_cursor()
                set_foreground(win.hwnd, "(光标仍停在远处)")
                inp.click_client(cx, cy, delay_after=1.5, jitter=4)
        except Exception as e:  # noqa: BLE001
            log(f"    [ERROR] {e}")
        time.sleep(0.6)
        cur = cap.capture()
        if cur is None:
            log("    抓屏失败")
            continue
        ok, changed, reg = effective(prev, cur, cx, cy)
        cv2.imwrite(str(OUT_DIR / f"10_probe_{i}.png"), cur)
        log(f"    全屏变化={changed * 100:.2f}%  目标区域={reg * 100:.2f}%  "
            f"=> {'【生效】点击生效' if ok else '【无效】无效果'}")
        prev = cur
        if ok:
            log(f"    ===> 结论: [{label}] 能让游戏响应，脚本需要按这种方式发点击。")
            break
    else:
        log("    ===> 四种方式都无效：怀疑画面本身卡死/有不可见弹窗，或点击没进入该窗口。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--click-test", action="store_true", help="PostMessage 后台点击实验（与挂机脚本同一路径）")
    ap.add_argument("--real-click", action="store_true", help="真实鼠标点击实验（会移动系统光标！）")
    ap.add_argument("--probe", action="store_true",
                    help="对照实验: 后台点击/光标位置/前台焦点/真实输入 四选一，找到能让游戏响应的方式")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []

    def log(msg: str = ""):
        print(msg)
        report.append(msg)

    def flush():
        (OUT_DIR / "report.txt").write_text("\n".join(report), encoding="utf-8")

    win = GameWindow()
    if not win.find_window():
        log("[FAIL] 找不到游戏窗口《卡厄思梦境》，请先启动游戏。")
        flush()
        return 1

    cap = ScreenCapture(win)
    inp = InputController(win)
    engine = object.__new__(NodeDecisionEngine)

    info = win.get_info()
    log(f"[WINDOW] {info}")

    img = cap.capture()
    if img is None or img.size == 0:
        log("[FAIL] 抓屏失败。")
        flush()
        return 1

    # DPI sanity: capture must be 1:1 with the client rect
    sx, sy, cw, ch = win.get_client_rect()
    log(f"[DPI] client_rect={cw}x{ch}  capture={img.shape[1]}x{img.shape[0]}  "
        f"{'OK (1:1)' if (cw == img.shape[1] and ch == img.shape[0]) else 'MISMATCH! (点击坐标会偏移)'}")
    log(f"[DPI] screen_origin=({sx},{sy})  process_dpi_aware={user32.IsProcessDPIAware()}")

    cv2.imwrite(str(OUT_DIR / "00_frame_a.png"), img)

    # ---- bubble detection ----
    tpl = NodeDecisionEngine._get_search_bubble_template()
    if tpl is None:
        log("[FAIL] 读取模板 templates/explore_search_bubble.png 失败。")
    bubbles = engine.detect_search_bubbles(img)
    log(f"[BUBBLE] detect_search_bubbles -> {len(bubbles)} 个 (阈值 0.70, 5 档缩放)")
    for i, b in enumerate(bubbles):
        log(f"    #{i} score={b['score']:.3f} rel=({b['rx']:.3f},{b['ry']:.3f}) px=({b['cx']},{b['cy']}) bbox={b['bbox']}")

    if tpl is not None:
        ih, iw = img.shape[:2]
        scale = ih / 576.0
        th, tw = tpl.shape[:2]
        for s in [0.85, 0.95, 1.0, 1.05, 1.15]:
            cw_, ch_ = int(tw * scale * s), int(th * scale * s)
            if cw_ <= 0 or ch_ <= 0 or cw_ >= iw or ch_ >= ih:
                continue
            ts = cv2.resize(tpl, (cw_, ch_))
            res = cv2.matchTemplate(img, ts, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            log(f"[SWEEP] scale={s:.2f} best={mv:.3f} at ({ml[0] + cw_ // 2},{ml[1] + ch_ // 2})")

    if tpl is not None:
        cv2.imwrite(str(OUT_DIR / "01_bubbles.png"), annotate(img, bubbles, tpl.shape))
    log(f"[SAVE] {OUT_DIR / '00_frame_a.png'} / 01_bubbles.png")

    if not (args.click_test or args.real_click or args.probe):
        log("[INFO] 仅取证模式（未发送任何点击）。--probe 对照实验 / --click-test 后台点击 / --real-click 真实鼠标。")
        flush()
        return 0

    # ---- window sanity: how many top-level windows share the title? ----
    try:
        import win32gui
        same = []

        def _cb(h, _):
            if win32gui.IsWindowVisible(h) and win.GetWindowText if False else False:
                return True
            try:
                t = win32gui.GetWindowText(h)
            except Exception:  # noqa: BLE001
                t = ""
            if t and (win.title_keyword in t):
                same.append((hex(h), t))
            return True

        win32gui.EnumWindows(_cb, None)
        log(f"[WINDOW] 标题包含“{win.title_keyword}”的可见顶层窗口共 {len(same)} 个: {same}")
        fg = win32gui.GetForegroundWindow()
        log(f"[WINDOW] 当前前台窗口 = {hex(fg)} '{win32gui.GetWindowText(fg)}'")
    except Exception as e:  # noqa: BLE001
        log(f"[WINDOW] 枚举窗口失败: {e}")

    # ---- baseline: scene animation drift without any click ----
    time.sleep(1.0)
    base_b = cap.capture()
    base_changed, base_mean = frame_diff(img, base_b)
    log(f"\n[BASELINE] 不点击时 1 秒内的自然画面变化: {base_changed * 100:.2f}% (mean={base_mean:.2f}) "
        f"-> 低于此值视为“点击无效”")

    # ---- discriminating probe ----
    if args.probe:
        if not bubbles:
            log("[FAIL] 未检测到气泡，没有点击目标。")
            flush()
            return 1
        b0 = bubbles[0]
        run_probe(win, cap, inp, b0['cx'], b0['cy'], cw, ch, log, flush, base_changed)
        flush()
        log(f"\n[SAVE] {OUT_DIR / 'report.txt'}")
        return 0

    # ---- experiment sequence ----
    seq = []
    if bubbles:
        for i, b in enumerate(bubbles):
            seq.append((f"bubble#{i} @({b['rx']:.3f},{b['ry']:.3f})", b['cx'], b['cy']))
    seq.append(("story advance (0.92,0.85)", int(0.92 * cw), int(0.85 * ch)))
    seq.append(("centre click (0.50,0.50)", cw // 2, ch // 2))

    prev = cap.capture()
    prefix = "03_real" if args.real_click else "02_click"
    for idx, (label, cx, cy) in enumerate(seq, start=1):
        log(f"\n[CLICK {idx}] {label} -> {'真实鼠标' if args.real_click else 'PostMessage 后台'}")
        if args.real_click:
            log("    （请勿触碰鼠标，3 秒后点击）")
            time.sleep(3.0)
            real_click(win, cx, cy)
            time.sleep(1.5)
        else:
            inp.click_client(cx, cy, delay_after=1.5, jitter=4, label=label)
            time.sleep(0.6)
        cur = cap.capture()
        if cur is None:
            log("    抓屏失败")
            continue
        changed, mean_d = frame_diff(prev, cur)
        reg, _ = region_diff(prev, cur, cx, cy, half=220)
        cv2.imwrite(str(OUT_DIR / f"{prefix}_{idx}.png"), cur)
        effective = changed > max(0.005, base_changed * 3) or reg > max(0.01, base_changed * 4)
        verdict = "画面明显变化 -> 点击生效" if effective else "画面基本没变 -> 点击无效"
        log(f"    全屏变化={changed * 100:.2f}%  目标区域变化={reg * 100:.2f}%  => {verdict}")
        prev = cur
        if effective and idx <= len(bubbles):
            log("    （气泡点击已生效，后续实验仍会继续，便于对比）")

    flush()
    log(f"\n[SAVE] {OUT_DIR / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

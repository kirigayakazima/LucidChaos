"""
Non-intrusive input simulator for Windows background window operations.
Sends Win32 PostMessage directly to the target window handle (HWND) without
moving the user's physical mouse cursor or stealing focus.
"""

import time
import win32gui
import win32con
import win32api
import random
from typing import Tuple, Optional, Any
from .window import GameWindow


class InputController:
    def __init__(self, window: GameWindow, overlay: Optional[Any] = None):
        self.window = window
        self.overlay = overlay
        # 最近一次动作的可读描述（形如 "click_client(1980,617) 【搜索】气泡#0 x3"）。
        # 供"全局卡死守卫"判断"同一个动作是不是在反复重复"（强循环信号）。
        self.last_action: str = ""

    @staticmethod
    def _make_lparam(cx: int, cy: int) -> int:
        return (cy << 16) | (cx & 0xFFFF)

    def click_client(self, cx: int, cy: int, delay_after: float = 0.2, jitter: int = 4, label: Optional[str] = None,
                     press_ms: int = 80, move_gap_ms: int = 20,
                     clicks: int = 1, click_gap_ms: int = 90):
        """
        Click at client pixel coordinates with natural random jitter tolerance.

        press_ms     : 每次 WM_LBUTTONDOWN -> WM_LBUTTONUP 的按住时长。默认 80ms（≈人点 50~120ms）。
        move_gap_ms  : WM_MOUSEMOVE -> 第一次按下 的间隔，留给游戏渲染一帧 hover 状态。
        clicks       : 在同一个点上「快速连点」几次（默认 1 次）。用于「第一下被吃掉、第二下才生效」
                       或「第一下选中、第二下确认」这类交互。
        click_gap_ms : 连点之间的间隔（默认 90ms，接近人手双击的节奏）。

        以上参数的默认值 == 历史行为，所以其它调用点行为完全不变；
        只有「局内搜索图标」那条路径会显式传参（见 NodeDecisionEngine._BUBBLE_CLICK_*）。

        注意：本方法只发窗口消息，**绝不移动真实光标、不抢前台**。
        """
        if not self.window.is_valid:
            if not self.window.find_window():
                raise RuntimeError("Game window not found for click.")

        if jitter > 0:
            cx += random.randint(-jitter, jitter)
            cy += random.randint(-jitter, jitter)

        # Show on overlay if label provided
        if getattr(self, "overlay", None) and getattr(self.overlay, "enabled", True) and label:
            try:
                self.overlay.show_point(cx, cy, label=label, duration=0.8)
            except Exception:
                pass

        hwnd = self.window.hwnd
        lparam = self._make_lparam(cx, cy)
        self.last_action = f"click_client({cx},{cy}) {label or ''}".strip()

        try:
            # 「人手式靠近」：当调用方显式要求较长的 hover 前置时间（见 _BUBBLE_CLICK_*）时，
            # 先发一个**偏离目标几十像素的中间点**，再发目标点，最后才按下 —— 完全模拟人手
            # "移过去 → 停一下 → 点"的过程。
            # 动机（真机观察）：场景里的问号 🔍 是**世界空间**交互，主人发现"把真实鼠标移过去才生效"，
            # 说明它可能依赖 hover 状态/射线拾取；多给两个 move 事件 + 更多帧时间，是最不侵入的尝试
            # （依然**不移动真实光标**）。
            if move_gap_ms >= 200:
                try:
                    _, _, cw, ch = self.window.get_client_rect()
                except Exception:  # noqa: BLE001
                    cw, ch = 0, 0
                off_x = random.choice((-1, 1)) * random.randint(28, 56)
                off_y = random.choice((-1, 1)) * random.randint(18, 40)
                ax, ay = cx + off_x, cy + off_y
                if cw and ch:
                    ax = max(0, min(ax, cw - 1))
                    ay = max(0, min(ay, ch - 1))
                win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, self._make_lparam(ax, ay))
                time.sleep(move_gap_ms / 3000.0)
                win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
                time.sleep(move_gap_ms / 1000.0 * (2.0 / 3.0))
            else:
                # Send mouse move, then N quick down/up pairs
                win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, lparam)
                time.sleep(move_gap_ms / 1000.0)
            for i in range(max(1, int(clicks))):
                if i:  # 连点之间留一点间隔，但不要长到被当成两次独立点击
                    time.sleep(max(0.0, click_gap_ms) / 1000.0)
                win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, lparam)
                time.sleep(press_ms / 1000.0)  # Hold click momentarily for game engine recognition
                win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, lparam)
            time.sleep(delay_after)
        except Exception as e:
            # Handle Windows UIPI Access is denied (WinError 5)
            import logging
            logging.getLogger("Input").error(f"后台点击被拦截: {e}。请确保终端是以【管理员身份运行】！")

    def click_rel(self, rx: float, ry: float, delay_after: float = 0.2, jitter_rel: float = 0.006, jitter: int = 0, label: Optional[str] = None):
        """Click at relative coordinates (0.0 ~ 1.0) with relative bounding jitter."""
        self.last_action = f"click_rel({rx:.3f},{ry:.3f}) {label or ''}".strip()
        if jitter_rel > 0:
            rx += random.uniform(-jitter_rel, jitter_rel)
            ry += random.uniform(-jitter_rel, jitter_rel)
        rx = max(0.0, min(1.0, rx))
        ry = max(0.0, min(1.0, ry))
        cx, cy = self.window.rel_to_client(rx, ry)
        self.click_client(cx, cy, delay_after=delay_after, jitter=jitter, label=label or "点击")

    def click_area_rel(self, rx1: float, ry1: float, rx2: float, ry2: float, delay_after: float = 0.2, margin_ratio: float = 0.15, label: Optional[str] = None):
        """
        Click randomly within a relative rectangular area [rx1, ry1] ~ [rx2, ry2].
        Uses margin_ratio to stay inside the button's safe clickable bounds.
        """
        self.last_action = (f"click_area_rel({(rx1+rx2)/2:.3f},{(ry1+ry2)/2:.3f}) {label or ''}").strip()
        w = max(0.001, rx2 - rx1)
        h = max(0.001, ry2 - ry1)
        mx = w * margin_ratio
        my = h * margin_ratio
        target_rx = random.uniform(rx1 + mx, rx2 - mx)
        target_ry = random.uniform(ry1 + my, ry2 - my)

        # Show bounding box on overlay HUD
        x1, y1 = self.window.rel_to_client(rx1, ry1)
        x2, y2 = self.window.rel_to_client(rx2, ry2)
        bw = max(20, x2 - x1)
        bh = max(20, y2 - y1)
        tx, ty = self.window.rel_to_client(target_rx, target_ry)

        if getattr(self, "overlay", None) and getattr(self.overlay, "enabled", True):
            try:
                self.overlay.show_boxes([{
                    'bbox': (x1, y1, bw, bh),
                    'label': label or '交互区域',
                    'click': (tx, ty),
                    'color': '#00FF66'
                }], duration=0.8)
            except Exception:
                pass

        self.click_client(tx, ty, delay_after=delay_after, jitter=0)

    def click_box(self, bbox: Tuple[int, int, int, int], delay_after: float = 0.2, margin_ratio: float = 0.15, label: Optional[str] = None):
        """Click randomly within a bounding box (x, y, w, h)."""
        x, y, w, h = bbox
        mx = max(1, int(w * margin_ratio))
        my = max(1, int(h * margin_ratio))
        tx = random.randint(x + mx, max(x + mx, x + w - mx))
        ty = random.randint(y + my, max(y + my, y + h - my))

        # Show bounding box on on-screen overlay HUD
        if getattr(self, "overlay", None):
            try:
                self.overlay.show_boxes([{
                    'bbox': bbox,
                    'label': label or '点击目标',
                    'click': (tx, ty),
                    'color': '#00FF66'
                }], duration=0.8)
            except Exception:
                pass

        self.click_client(tx, ty, delay_after=delay_after, jitter=0)

    def click_target(self, target: Any, delay_after: float = 0.2, margin_ratio: float = 0.15,
                     label: Optional[str] = None):
        """
        Click safely within the bounding box of a detected VisionTarget or UIElement.
        Falls back to center point with jitter if bbox is not present.
        """
        name = label or getattr(target, "name", None) or getattr(target, "text", None) or "目标"
        if hasattr(target, "bbox") and target.bbox:
            self.click_box(target.bbox, delay_after=delay_after, margin_ratio=margin_ratio, label=str(name))
        elif hasattr(target, "center"):
            cx, cy = target.center[0], target.center[1]
            if getattr(self, "overlay", None):
                try:
                    self.overlay.show_boxes([{
                        'bbox': (cx - 16, cy - 16, 32, 32),
                        'label': str(name),
                        'click': (cx, cy),
                        'color': '#00FF66'
                    }], duration=0.8)
                except Exception:
                    pass
            self.click_client(cx, cy, delay_after=delay_after, jitter=6, label=str(name))
        else:
            raise ValueError(f"Target {target} does not have bbox or center.")

    def drag_client(self, start_cx: int, start_cy: int, end_cx: int, end_cy: int, steps: int = 15,
                    duration: float = 0.3, label: Optional[str] = None):
        """Perform a background swipe/drag operation."""
        if label and getattr(self, "overlay", None):
            try:
                self.overlay.show_boxes([{
                    'bbox': (start_cx - 12, start_cy - 12, 24, 24),
                    'label': label,
                    'click': (start_cx, start_cy),
                    'color': '#FF66FF'
                }], duration=0.8)
            except Exception:
                pass
        if not self.window.is_valid:
            if not self.window.find_window():
                raise RuntimeError("Game window not found for drag.")

        hwnd = self.window.hwnd
        start_lparam = self._make_lparam(start_cx, start_cy)

        try:
            win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, 0, start_lparam)
            time.sleep(0.02)
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONDOWN, win32con.MK_LBUTTON, start_lparam)
            time.sleep(0.05)

            dt = duration / max(1, steps)
            for i in range(1, steps + 1):
                curr_x = int(start_cx + (end_cx - start_cx) * (i / steps))
                curr_y = int(start_cy + (end_cy - start_cy) * (i / steps))
                win32gui.PostMessage(hwnd, win32con.WM_MOUSEMOVE, win32con.MK_LBUTTON, self._make_lparam(curr_x, curr_y))
                time.sleep(dt)

            end_lparam = self._make_lparam(end_cx, end_cy)
            time.sleep(0.05)
            win32gui.PostMessage(hwnd, win32con.WM_LBUTTONUP, 0, end_lparam)
            time.sleep(0.1)
        except Exception as e:
            import logging
            logging.getLogger("Input").error(f"后台拖拽被拦截: {e}。请确保控制台是以【管理员身份运行】！")

    def drag_rel(self, s_rx: float, s_ry: float, e_rx: float, e_ry: float, steps: int = 15,
                 duration: float = 0.3, label: Optional[str] = None):
        """Drag between relative coordinates."""
        self.last_action = f"drag_rel({s_rx:.3f},{s_ry:.3f}->{e_rx:.3f},{e_ry:.3f}) {label or ''}".strip()
        scx, scy = self.window.rel_to_client(s_rx, s_ry)
        ecx, ecy = self.window.rel_to_client(e_rx, e_ry)
        self.drag_client(scx, scy, ecx, ecy, steps=steps, duration=duration, label=label)

    def tap_action(self, action: str, delay_after: float = 0.3, label: Optional[str] = None) -> bool:
        """
        按"语义动作"发键（见 `core/keys.ACTION_KEYS`）：advance_dialogue / skip_cutscene /
        toggle_speed / confirm / cancel / end_turn / auto_battle / quick_menu / hide_ui ...

        ⚠️ 切换型动作用户需谨慎：toggle_speed(CTRL)、auto_battle(R) 都是"开关"，
        调用方必须先确认当前状态（例如靠金色高亮判断），否则会把已经开着的关掉。
        返回 True 表示成功发出。
        """
        from .keys import action_key, ACTION_KEYS
        vk = action_key(action)
        if vk is None:
            return False
        self.send_key(vk, delay_after=delay_after,
                      label=label or f"【按键】{action}({ACTION_KEYS.get(action, '?')})")
        return True

    # 数字键"选择"类动作需要按几下。
    # ⚠️ 主人 2026/10/6 实测：**要按两下**才生效（第一下选中、第二下确认）：
    #    "选择事件选项，比如事件1，按两下1就是选择了" / "打牌，按两下1就是把牌1打出去了"
    # 单下只会高亮选中，不会真的执行 —— 这是之前"明明按了却没反应"的原因之一。
    DIGIT_PRESS_TIMES = 2
    DIGIT_PRESS_GAP_MS = 130

    def tap_card(self, index_1based: int, delay_after: float = 0.6, label: Optional[str] = None,
                 times: Optional[int] = None) -> bool:
        """
        按编号键（1..9 / 10 -> 0），**默认按两下**（选中 + 确认）。

        同一个键在不同界面含义不同，而游戏把含义直接标在界面上：
          - 战斗手牌：使用N號卡牌（1~10，第10张是 0 键）
          - 灵光一闪选牌：選擇N號
          - 局内事件选项框：选择第N个选项
          - 肉鸽选路：选择第N条路线（▶ 旁边标着 1/2/3）
        """
        from .keys import card_key, KEYS
        vk = card_key(index_1based)
        if vk is None:
            return False
        name = "0" if index_1based == 10 else str(index_1based)
        n = self.DIGIT_PRESS_TIMES if times is None else max(1, int(times))
        base = label or f"【按键】编号 {index_1based} (键 {name})"
        for i in range(n):
            self.send_key(vk, delay_after=(0.12 if i < n - 1 else delay_after),
                          hold_ms=45, label=f"{base} 第{i + 1}/{n}下")
            if i < n - 1:
                time.sleep(self.DIGIT_PRESS_GAP_MS / 1000.0)
        return True

    def tap_hero_skill(self, slot_1based: int, delay_after: float = 0.5) -> bool:
        """触发第 N 号角色的自我意識技能（F1/F2/F3）——战斗界面左侧就标着 F1/F2/F3。"""
        from .keys import resolve
        vk = resolve(f"F{slot_1based}") if 1 <= slot_1based <= 3 else None
        if vk is None:
            return False
        self.send_key(vk, delay_after=delay_after, label=f"【按键】{slot_1based}号角色技能 (F{slot_1based})")
        return True

    def send_key(self, vk_code: int, delay_after: float = 0.1, hold_ms: int = 50, label: Optional[str] = None):
        """
        发送一次按键（**投递到游戏窗口的消息队列，不抢前台、不动真实键盘/光标**）。

        lParam 按规范填：重复次数=1 | 扫描码<<16 |（抬起时再置 30/31 位）。
        不少程序（含部分 GLFW/Unity 版本）会校验扫描码，lParam=0 的写法可能被忽略。
        """
        self.last_action = f"send_key({vk_code}) {label or ''}".strip()
        if not self.window.is_valid:
            return
        hwnd = self.window.hwnd
        try:
            scan = win32api.MapVirtualKey(vk_code, 0)
        except Exception:  # noqa: BLE001
            scan = 0
        down = 1 | (scan << 16)
        up = 1 | (scan << 16) | (1 << 30) | (1 << 31)
        try:
            win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, vk_code, down)
            time.sleep(max(0.01, hold_ms / 1000.0))
            win32gui.PostMessage(hwnd, win32con.WM_KEYUP, vk_code, up)
            time.sleep(delay_after)
        except Exception as e:
            import logging
            logging.getLogger("Input").error(
                f"后台按键发送被系统拦截 ({e})。"
                f"这是因为游戏以管理员权限运行，请确保控制台同样以【管理员身份运行】！"
            )

"""
Real-time Transparent Screen Overlay for Game Window (HUD / 锚框遮罩).
Floats directly over the game window on the user's desktop:
- Transparent & Click-through (WS_EX_LAYERED | WS_EX_TRANSPARENT)
- Zero disk I/O: pure in-memory GDI canvas rendering (< 1ms latency)
- Dynamic geometry tracking (automatically pins to the game window client rect)
- Displays bounding boxes (锚框), confidence scores, and click target crosshairs.
"""

import threading
import queue
import time
import logging
import tkinter as tk
import ctypes
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger("ScreenOverlay")

user32 = ctypes.windll.user32
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_TOPMOST = 0x00000008
WS_EX_NOACTIVATE = 0x08000000


class ScreenOverlay:
    enabled: bool = True  # Global Master Switch (全局开关)

    def __init__(self, window_getter=None, enabled: bool = True):
        """
        window_getter: Callable returning (x, y, width, height) of the target game client area.
        enabled: Master toggle for whether bounding boxes are displayed.
        """
        self.window_getter = window_getter
        self.enabled = enabled
        self.cmd_queue = queue.Queue()
        self.ready_event = threading.Event()
        self.running = True
        self._last_rect = None

        self._thread = threading.Thread(target=self._run_loop, name="OverlayThread", daemon=True)
        self._thread.start()
        self.ready_event.wait(timeout=2.0)
        logger.info(f"实时屏幕锚框遮罩 (Screen Overlay) 初始化完成 [状态: {'开启' if self.enabled else '关闭'}]")

    def _run_loop(self):
        try:
            self.root = tk.Tk()
            self.root.overrideredirect(True)
            self.root.attributes("-topmost", True)

            # Chromakey transparency
            self.trans_color = "#010203"
            self.root.attributes("-transparentcolor", self.trans_color)
            self.root.config(bg=self.trans_color)

            self.canvas = tk.Canvas(self.root, bg=self.trans_color, highlightthickness=0)
            self.canvas.pack(fill="both", expand=True)

            self.root.update()
            hwnd = user32.GetParent(self.root.winfo_id()) or self.root.winfo_id()

            # Set Win32 styles: Layered, Transparent (click-through), Toolwindow, No-activate
            style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            user32.SetWindowLongW(
                hwnd,
                GWL_EXSTYLE,
                style | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE
            )

            self.clear_timer = None
            self.current_badge = None
            self.ready_event.set()

            # Start event pump
            self._pump_events()
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Overlay loop error: {e}")
            self.ready_event.set()

    def _pump_events(self):
        if not self.running:
            return

        try:
            while not self.cmd_queue.empty():
                cmd, args = self.cmd_queue.get_nowait()
                if cmd == "draw":
                    self._do_draw(*args)
                elif cmd == "set_badge":
                    self._do_set_badge(*args)
                elif cmd == "clear":
                    self._do_clear()
                elif cmd == "sync":
                    self._do_sync(*args)
                elif cmd == "close":
                    self.root.destroy()
                    return
        except Exception:
            pass

        if self.running:
            self.root.after(30, self._pump_events)

    def _do_sync(self, x: int, y: int, w: int, h: int):
        new_rect = (x, y, w, h)
        if new_rect != self._last_rect:
            self.root.geometry(f"{w}x{h}+{x}+{y}")
            self._last_rect = new_rect

    def _draw_badge(self, text: str):
        # 彻底禁用在游戏画面上绘制文本徽章，避免 mss 截屏抓到后引起 PaddleOCR 自激反馈死循环
        pass

    def _do_set_badge(self, text: str):
        pass

    def _do_clear(self):
        self.canvas.delete("all")

    def _do_draw(self, boxes: List[Dict[str, Any]], duration: float):
        self.canvas.delete("all")

        for b in boxes:
            # Parse bbox coordinates
            if "bbox" in b and b["bbox"]:
                bx, by, bw, bh = b["bbox"]
                x1, y1 = bx, by
                x2, y2 = bx + bw, by + bh
            else:
                x1, y1 = b.get("x1", 0), b.get("y1", 0)
                x2, y2 = b.get("x2", 0), b.get("y2", 0)

            color = b.get("color", "#00FF00")
            width = b.get("width", 2)

            # 1. Draw Bounding Box (锚框矩形)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)

            # 2. Draw Corner Accents (四角高亮瞄准标)
            corner_len = min(12, max(5, (x2 - x1) // 4), max(5, (y2 - y1) // 4))
            # Top-left
            self.canvas.create_line(x1, y1, x1 + corner_len, y1, fill=color, width=width + 1)
            self.canvas.create_line(x1, y1, x1, y1 + corner_len, fill=color, width=width + 1)
            # Top-right
            self.canvas.create_line(x2, y1, x2 - corner_len, y1, fill=color, width=width + 1)
            self.canvas.create_line(x2, y1, x2, y1 + corner_len, fill=color, width=width + 1)
            # Bottom-left
            self.canvas.create_line(x1, y2, x1 + corner_len, y2, fill=color, width=width + 1)
            self.canvas.create_line(x1, y2, x1, y2 - corner_len, fill=color, width=width + 1)
            # Bottom-right
            self.canvas.create_line(x2, y2, x2 - corner_len, y2, fill=color, width=width + 1)
            self.canvas.create_line(x2, y2, x2, y2 - corner_len, fill=color, width=width + 1)

            # 3. Text labels disabled: do not draw text on canvas to prevent OCR interference
            pass

            # 4. Click crosshair target (准星)
            click = b.get("click")
            if click:
                cx, cy = int(click[0]), int(click[1])
                r = 6
                self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#FF2255", width=2)
                self.canvas.create_line(cx - 12, cy, cx + 12, cy, fill="#FF2255", width=2)
                self.canvas.create_line(cx, cy - 12, cx, cy + 12, fill="#FF2255", width=2)

        # Set auto-fade timer
        if duration and duration > 0:
            if self.clear_timer:
                try:
                    self.root.after_cancel(self.clear_timer)
                except Exception:
                    pass
            self.clear_timer = self.root.after(int(duration * 1000), self._do_clear)

    def _sync_window_position(self):
        """Query target game window client rect and sync geometry."""
        if self.window_getter:
            try:
                rect = self.window_getter()
                if rect:
                    x, y, w, h = rect
                    if w > 10 and h > 10:
                        self.cmd_queue.put(("sync", (x, y, w, h)))
            except Exception:
                pass

    def show_badge(self, text: str):
        """
        Show top HUD status badge, e.g. '【路由: 肉鸽选路 (92分)】'
        """
        if not self.running or not self.enabled:
            return
        self._sync_window_position()
        self.cmd_queue.put(("set_badge", (text,)))

    def clear_badge(self):
        """Clear top HUD status badge."""
        if not self.running:
            return
        self.cmd_queue.put(("set_badge", ("",)))

    def show_boxes(self, boxes: List[Dict[str, Any]], duration: float = 1.2):
        """
        Show list of bounding boxes directly on screen.
        """
        if not self.running or not self.enabled:
            return
        self._sync_window_position()
        self.cmd_queue.put(("draw", (boxes, duration)))

    def show_target(self, 
                    bbox: Tuple[int, int, int, int], 
                    label: Optional[str] = None, 
                    click: Optional[Tuple[int, int]] = None,
                    color: str = "#00FF00", 
                    duration: float = 1.2):
        """
        Convenience method to show a single target bounding box on screen.
        """
        if not self.enabled:
            return
        item = {
            "bbox": bbox,
            "label": label,
            "click": click,
            "color": color
        }
        self.show_boxes([item], duration=duration)

    def show_point(self, 
                   cx: int, 
                   cy: int, 
                   label: str = "点击", 
                   color: str = "#00FF66", 
                   radius: int = 22, 
                   duration: float = 0.8):
        """
        Quickly flash a target box & crosshair around a single click point.
        """
        if not self.enabled:
            return
        bbox = (cx - radius, cy - radius, radius * 2, radius * 2)
        self.show_boxes([{
            "bbox": bbox,
            "label": label,
            "click": (cx, cy),
            "color": color
        }], duration=duration)

    def toggle(self, state: Optional[bool] = None):
        """
        Toggle overlay master switch.
        """
        if state is None:
            self.enabled = not self.enabled
        else:
            self.enabled = bool(state)
        if not self.enabled:
            self.clear()
        logger.info(f"屏幕锚框遮罩开关已切换: {'【开启】' if self.enabled else '【关闭】'}")

    def clear(self):
        """Clear all overlay graphics immediately."""
        if self.running:
            self.cmd_queue.put(("clear", ()))

    def close(self):
        """Shutdown overlay window."""
        self.running = False
        try:
            self.cmd_queue.put(("close", ()))
        except Exception:
            pass

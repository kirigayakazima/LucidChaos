"""
Window manager for finding, tracking and coordinating the game window.
Handles multi-monitor coordinates, DPI scaling, and relative (0.0~1.0) resolution mapping.
"""

import ctypes
from ctypes import wintypes
import win32gui
import win32process
import win32con
from typing import Optional, Tuple, Dict, Any

user32 = ctypes.windll.user32

class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]

class POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


class GameWindow:
    def __init__(self, title_keyword: str = "卡厄思梦境"):
        self.title_keyword = title_keyword
        self.hwnd: Optional[int] = None
        self.pid: Optional[int] = None
        self._ensure_default_desktop()

    def _ensure_default_desktop(self):
        """Ensure thread is attached to interactive Default desktop."""
        try:
            hdesk = user32.OpenDesktopW("Default", 0, False, 0x01FF)
            if hdesk:
                user32.SetThreadDesktop(hdesk)
                user32.CloseDesktop(hdesk)
        except Exception:
            pass

    def find_window(self) -> bool:
        """Find the game window by title keyword, strictly avoiding self/GUI process."""
        self._ensure_default_desktop()
        import os
        current_pid = os.getpid()

        exact_hwnds = []
        fuzzy_hwnds = []

        def enum_cb(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd).strip()
            if not title:
                return True

            # Exclude our own GUI / control console window and developer tools
            try:
                _, w_pid = win32process.GetWindowThreadProcessId(hwnd)
            except Exception:
                return True
            if w_pid == current_pid:
                return True
            if any(k in title for k in ["控制台", "ChaosBot", "自动化", "run_gui", "cmd.exe", "PowerShell", "Visual Studio", "Cursor"]):
                return True

            if title == self.title_keyword:
                exact_hwnds.append((hwnd, w_pid))
            elif self.title_keyword in title:
                fuzzy_hwnds.append((hwnd, w_pid))
            return True

        win32gui.EnumWindows(enum_cb, None)

        candidates = exact_hwnds or fuzzy_hwnds
        if not candidates:
            self.hwnd = None
            self.pid = None
            return False

        # Pick the exact matching game window
        self.hwnd, self.pid = candidates[0]
        return True

    @property
    def is_valid(self) -> bool:
        if not self.hwnd:
            return False
        return bool(win32gui.IsWindow(self.hwnd) and win32gui.IsWindowVisible(self.hwnd))

    def bring_to_front(self) -> bool:
        """Bring target game window to foreground."""
        if not self.is_valid:
            if not self.find_window():
                return False
        try:
            if win32gui.IsIconic(self.hwnd):
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.hwnd)
            return True
        except Exception:
            return False

    def get_client_rect(self) -> Tuple[int, int, int, int]:
        """Returns (x, y, width, height) of the client area in screen coordinates."""
        if not self.is_valid:
            if not self.find_window():
                raise RuntimeError(f"Window '{self.title_keyword}' not found.")

        rect = RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(rect))
        w = rect.right - rect.left
        h = rect.bottom - rect.top

        pt = POINT(0, 0)
        user32.ClientToScreen(self.hwnd, ctypes.byref(pt))
        return pt.x, pt.y, w, h

    def rel_to_client(self, rx: float, ry: float) -> Tuple[int, int]:
        """Convert relative coordinates (0.0 ~ 1.0) to client area pixel (x, y)."""
        _, _, w, h = self.get_client_rect()
        cx = int(round(rx * w))
        cy = int(round(ry * h))
        return cx, cy

    def rel_to_screen(self, rx: float, ry: float) -> Tuple[int, int]:
        """Convert relative coordinates (0.0 ~ 1.0) to absolute screen pixel (x, y)."""
        sx, sy, w, h = self.get_client_rect()
        cx = int(round(rx * w))
        cy = int(round(ry * h))
        return sx + cx, sy + cy

    def get_info(self) -> Dict[str, Any]:
        """Get diagnostic information about the window."""
        if not self.is_valid:
            self.find_window()
        if not self.is_valid:
            return {"status": "not_found"}
        sx, sy, w, h = self.get_client_rect()
        title = win32gui.GetWindowText(self.hwnd)
        return {
            "hwnd": hex(self.hwnd),
            "pid": self.pid,
            "title": title,
            "screen_x": sx,
            "screen_y": sy,
            "width": w,
            "height": h,
            "aspect_ratio": f"{w / max(1, h):.3f}"
        }

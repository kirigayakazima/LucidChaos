"""
High-Precision 2K Template Cropping & Calibration Utility (高精度 2K 模板辅助截取工具).

Usage:
1. Capture from current running game window:
   python -m lucid_chaos.tools.crop_template_helper --capture --save-as cyan_arrow_2k.png
2. Load an existing screenshot:
   python -m lucid_chaos.tools.crop_template_helper --file path/to/screenshot.png --save-as rogue_sword_banner_2k.png
3. Automatically isolates white icons / creates RGBA transparency if --transparent flag is set.
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np

def crop_interactive(img_bgr: np.ndarray, save_name: str, transparent: bool = False, category: str = ""):
    h, w = img_bgr.shape[:2]
    print(f"\n[INFO] 当前载入图像尺寸: {w}x{h}")
    print("[提示] 请在弹出窗口中使用鼠标拖拽矩形框选中模板目标，完成后按【SPACE】或【ENTER】确认，按【C】取消。\n")

    # Interactive ROI selection
    roi = cv2.selectROI("Template Cropper - Press SPACE/ENTER to Confirm", img_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    x, y, rw, rh = roi
    if rw <= 0 or rh <= 0:
        print("[WARN] 未选定任何区域，已取消截取。")
        return

    cropped = img_bgr[y:y+rh, x:x+rw].copy()

    # Apply transparency if requested
    if transparent:
        # Convert to BGRA
        b, g, r = cv2.split(cropped)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # Threshold: bright white foreground retains full alpha
        _, alpha = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        # Optional morphology smoothing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
        cropped_rgba = cv2.merge([b, g, r, alpha])
        out_img = cropped_rgba
        print("[INFO] 已自动提取前景白色高光，生成透明通道 (RGBA)。")
    else:
        out_img = cropped

    # Determine save path
    templates_dir = Path(__file__).resolve().parent.parent / "templates"
    if category:
        save_dir = templates_dir / category
    else:
        save_dir = templates_dir

    save_dir.mkdir(parents=True, exist_ok=True)
    target_path = save_dir / save_name

    # Save lossless PNG
    is_success, buf = cv2.imencode(".png", out_img)
    if is_success:
        with open(str(target_path), "wb") as f:
            f.write(buf)
        print("=" * 60)
        print(f"[SUCCESS] 高精度 2K 模板成功保存至:")
        print(f"  -> {target_path}")
        print(f"  -> 尺寸: {rw}x{rh} (像素)")
        print("=" * 60)
    else:
        print(f"[ERROR] 模板保存失败: {target_path}")

def main():
    parser = argparse.ArgumentParser(description="2K High-Precision Template Cropping Tool")
    parser.add_argument("--capture", action="store_true", help="Capture screenshot directly from live game window")
    parser.add_argument("--file", type=str, default="", help="Path to an existing screenshot")
    parser.add_argument("--save-as", type=str, required=True, help="Target template filename (e.g. cyan_arrow.png)")
    parser.add_argument("--category", type=str, default="", help="Sub-folder inside templates (e.g. rogue)")
    parser.add_argument("--transparent", action="store_true", help="Extract foreground transparency (RGBA)")

    args = parser.parse_args()

    img = None
    if args.capture:
        # Add project root to sys.path
        root_dir = str(Path(__file__).resolve().parent.parent.parent)
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)

        from lucid_chaos.core.window import GameWindow
        from lucid_chaos.core.capture import ScreenCapture

        gw = GameWindow()
        if not gw.find_window():
            print("[ERROR] 未找到运行中的游戏窗口《卡厄思梦境》！请先启动游戏或使用 --file 参数指定截图。")
            return
        sc = ScreenCapture(gw)
        img = sc.capture()
        if img is None:
            print("[ERROR] 截取游戏窗口失败！")
            return
    elif args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"[ERROR] 指定截图文件不存在: {p}")
            return
        with open(str(p), "rb") as f:
            img = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        print("[ERROR] 请指定 --capture 或 --file 参数！")
        return

    crop_interactive(img, args.save_as, transparent=args.transparent, category=args.category)

if __name__ == "__main__":
    main()

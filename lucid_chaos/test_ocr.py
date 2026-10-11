import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lucid_chaos.core.vision import VisionEngine

def test_ocr():
    img_path = r"C:\Users\xuaner\.gemini\antigravity\brain\26aeca17-ff86-4336-9a26-16438e2eb52e\.user_uploaded\media_1791179035297.jpg"
    img = cv2.imread(img_path)
    if img is None:
        print("Test image not found.")
        return

    print("Running RapidOCR on your game screenshot...")
    v = VisionEngine()
    items = v.ocr_image(img)
    print(f"\nRecognized {len(items)} text elements:")
    for it in items[:15]:
        cx, cy = it["center_client"]
        rx, ry = it["center_rel"]
        print(f" - [{it['text']}] | conf: {it['score']:.2f} | client: ({cx}, {cy}) | rel: ({rx:.3f}, {ry:.3f})")

if __name__ == "__main__":
    test_ocr()

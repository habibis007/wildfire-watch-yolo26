"""Draw merged labels on random images from each source, and print per-split/class counts.

    python scripts/spot_check.py [n_per_source] [out.png]
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wildfire import Detection, draw_detections  # noqa: E402

MERGED = ROOT / "data" / "merged"


def main(n: int = 8, out: str = "data/spot_check.png") -> None:
    random.seed(0)
    rows: dict[str, list[np.ndarray]] = {}
    imgs_per_split, boxes = Counter(), Counter()
    for split_dir in sorted(p for p in MERGED.iterdir() if p.is_dir()):
        coco = json.loads((split_dir / "_annotations.coco.json").read_text())
        names = {c["id"]: c["name"] for c in coco["categories"]}
        anns: dict[int, list] = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
            boxes[(split_dir.name, names[a["category_id"]])] += 1
        imgs_per_split[split_dir.name] = len(coco["images"])
        for img in coco["images"]:
            src = img["file_name"].split("__")[1]
            rows.setdefault(src, []).append((split_dir / img["file_name"], [
                Detection(names[a["category_id"]], 1.0, a["bbox"][0], a["bbox"][1],
                          a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]) for a in anns[img["id"]]]))
    print("images per split:", dict(imgs_per_split), "total", sum(imgs_per_split.values()))
    for split in imgs_per_split:
        print(f"  {split:5s} boxes: fire {boxes[(split, 'fire')]:5d}  smoke {boxes[(split, 'smoke')]:5d}")
    for src, items in rows.items():
        print(f"  source {src}: {len(items)} images")

    tiles = []
    for src, items in rows.items():
        row = []
        for path, dets in random.sample(items, min(n, len(items))):
            im = draw_detections(cv2.imread(str(path)), dets, thickness=3)
            row.append(cv2.resize(im, (240, 180)))
        row += [np.zeros((180, 240, 3), np.uint8)] * (n - len(row))
        strip = np.hstack(row)
        cv2.putText(strip, src[:40], (5, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        tiles.append(strip)
    cv2.imwrite(str(ROOT / out), np.vstack(tiles))
    print("wrote", out)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8, *(sys.argv[2:3] or []))

import json
import sys
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_dataset as pd  # noqa: E402


def test_build_remaps_and_drops_classes(tmp_path, monkeypatch):
    monkeypatch.setattr(pd, "DATA", tmp_path)
    monkeypatch.setattr(pd, "MERGED", tmp_path / "merged")
    monkeypatch.setattr(pd, "rf", lambda: None)  # raw data already "downloaded" below
    src = pd.Source("ws", "proj", 1, max_images=10, class_map={"flame": "fire", "smoke": "smoke"})
    monkeypatch.setattr(pd, "SOURCES", [src])

    split = tmp_path / "raw" / "ws__proj" / "train"
    split.mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(split / f"img{i}.jpg"), np.zeros((100, 200, 3), np.uint8))
    (split / "_annotations.coco.json").write_text(json.dumps({
        "categories": [{"id": 0, "name": "Flame"}, {"id": 1, "name": "smoke"}, {"id": 2, "name": "person"}],
        "images": [{"id": i, "file_name": f"img{i}.jpg", "width": 200, "height": 100} for i in range(3)],
        "annotations": [
            {"id": 0, "image_id": 0, "category_id": 0, "bbox": [10, 10, 20, 20]},
            {"id": 1, "image_id": 0, "category_id": 2, "bbox": [50, 50, 5, 5]},   # person: dropped
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, 100, 50]},
            {"id": 3, "image_id": 2, "category_id": 2, "bbox": [1, 1, 2, 2]},     # image 2 has no kept box
        ],
    }))

    pd.build()

    coco = json.loads((tmp_path / "merged" / "train" / "_annotations.coco.json").read_text())
    assert [c["name"] for c in coco["categories"]] == ["fire", "smoke"]
    assert len(coco["images"]) == 2
    assert sorted(a["category_id"] for a in coco["annotations"]) == [0, 1]
    for img in coco["images"]:
        assert (tmp_path / "merged" / "train" / img["file_name"]).exists()

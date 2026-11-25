"""Build the merged two-class (fire, smoke) dataset and upload it to Roboflow.

    python scripts/prepare_dataset.py inspect          # list classes/counts of each source (no writes)
    python scripts/prepare_dataset.py build            # download, remap, subsample -> data/merged (local)
    python scripts/prepare_dataset.py upload           # upload data/merged to a new project in your workspace

The API key is read from the ROBOFLOW_API_KEY environment variable or a local
.env file (git-ignored). It is never printed.

Why build locally instead of forking in the web UI? It keeps the class
merging reproducible: the mapping below is the single source of truth, and
anything not mapped (e.g. "person", "car") is dropped from the annotations.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MERGED = DATA / "merged"
TARGET_PROJECT = "wildfire-watch-yolo26"
SEED = 26
CATEGORIES = [{"id": 0, "name": "fire", "supercategory": "none"}, {"id": 1, "name": "smoke", "supercategory": "none"}]
CAT_ID = {"fire": 0, "smoke": 1}


@dataclass
class Source:
    workspace: str
    project: str
    version: int
    max_images: int
    # source class name (case-insensitive) -> "fire" | "smoke"; anything else is dropped
    class_map: dict[str, str] = field(default_factory=dict)
    note: str = ""
    # skip images whose ORIGINAL filename starts with any of these (e.g. D-Fire's
    # "WEB"/"AoF" images that also appear in another source)
    exclude_prefixes: tuple[str, ...] = ()
    # Roboflow exports name augmented copies "<orig>_jpg.rf.<hash>.jpg"; keep one per original
    dedupe_augmented: bool = False


FIRE_SMOKE = {"fire": "fire", "smoke": "smoke", "wildfire": "fire"}

# Chosen after the licence/quality review in docs/DATASETS.md. Target ~3,000 images.
SOURCES: list[Source] = [
    Source("master-candidate", "forest-fire-and-smoke-rf4pd", 23, max_images=1500, class_map=FIRE_SMOKE,
           note="aerial/oblique drone forest fires, CC BY 4.0",
           exclude_prefixes=("WEB", "AoF"), dedupe_augmented=True),
    Source("fire-detection-nlrcv", "9.5ai-d-fire", 1, max_images=1200, class_map=FIRE_SMOKE,
           note="D-Fire (ground-level base), CC BY 4.0 on Universe / CC0 upstream"),
    Source("tim-4ijf0", "drone-fire-detection-byija", 1, max_images=152, class_map=FIRE_SMOKE,
           note="top-down drone, fire only, CC BY 4.0"),
]


def original_stem(file_name: str) -> str:
    """'img_001_jpg.rf.ab12.jpg' -> 'img_001' ; 'WEB123.jpg' -> 'WEB123'."""
    name = Path(file_name).name
    if ".rf." in name:
        name = name.split(".rf.")[0]
        for ext in ("_jpg", "_jpeg", "_png"):
            if name.lower().endswith(ext):
                return name[: -len(ext)]
        return name
    return Path(name).stem


def api_key() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY in your environment or in .env (see README).")
    return key


def rf():
    from roboflow import Roboflow

    return Roboflow(api_key=api_key())


def inspect_sources() -> None:
    r = rf()
    for s in SOURCES:
        p = r.workspace(s.workspace).project(s.project)
        v = p.version(s.version)
        print(f"\n{s.workspace}/{s.project} v{s.version}  ({s.note})")
        print("  classes:", getattr(p, "classes", "?"))
        print("  images in version:", getattr(v, "splits", "?"))
        unmapped = [c for c in (getattr(p, "classes", {}) or {}) if c.lower() not in s.class_map]
        if unmapped:
            print("  will DROP:", unmapped)


def _load_coco_split(split_dir: Path) -> dict | None:
    ann = split_dir / "_annotations.coco.json"
    return json.loads(ann.read_text()) if ann.exists() else None


def build() -> None:
    random.seed(SEED)
    r = rf()
    if MERGED.exists():
        shutil.rmtree(MERGED)
    totals: dict[str, dict[str, int]] = {}
    merged: dict[str, dict] = {}
    for s in SOURCES:
        tag = f"{s.workspace}__{s.project}".replace("/", "_")
        dl_dir = DATA / "raw" / tag
        if not dl_dir.exists():
            r.workspace(s.workspace).project(s.project).version(s.version).download("coco", location=str(dl_dir))
        # gather (split, image_record, annotations) for images that keep >= 1 box after remapping
        pool = []
        seen: set[str] = set()
        for split in ("train", "valid", "test"):
            coco = _load_coco_split(dl_dir / split)
            if coco is None:
                continue
            cats = {c["id"]: s.class_map.get(c["name"].lower()) for c in coco["categories"]}
            by_img: dict[int, list] = {}
            for a in coco["annotations"]:
                new = cats.get(a["category_id"])
                if new:
                    by_img.setdefault(a["image_id"], []).append({**a, "_cls": new})
            for img in coco["images"]:
                if img["id"] not in by_img:
                    continue
                stem = original_stem(img["file_name"])
                if s.exclude_prefixes and stem.startswith(s.exclude_prefixes):
                    continue
                if s.dedupe_augmented:
                    if stem in seen:
                        continue
                    seen.add(stem)
                pool.append((split, dl_dir / split / img["file_name"], img, by_img[img["id"]]))
        random.shuffle(pool)
        chosen = pool[: s.max_images]
        counts = {"images": len(chosen), "fire": 0, "smoke": 0}
        for split, path, img, anns in chosen:
            coco = merged.setdefault(split, {"images": [], "annotations": [], "categories": CATEGORIES})
            out = MERGED / split
            out.mkdir(parents=True, exist_ok=True)
            fname = f"{tag}__{Path(img['file_name']).name}"
            shutil.copy(path, out / fname)
            img_id = len(coco["images"])
            coco["images"].append({"id": img_id, "file_name": fname, "width": img["width"], "height": img["height"]})
            for a in anns:
                counts[a["_cls"]] += 1
                coco["annotations"].append({
                    "id": len(coco["annotations"]), "image_id": img_id, "category_id": CAT_ID[a["_cls"]],
                    "bbox": a["bbox"], "area": a.get("area", a["bbox"][2] * a["bbox"][3]), "iscrowd": 0,
                })
        totals[tag] = counts
        print(f"{tag}: kept {len(chosen)} of {len(pool)} usable images  {counts}")
    # Roboflow's own COCO layout: <split>/_annotations.coco.json next to the images.
    for split, coco in merged.items():
        (MERGED / split / "_annotations.coco.json").write_text(json.dumps(coco))
    (MERGED / "manifest.json").write_text(json.dumps({"seed": SEED, "sources": [s.__dict__ for s in SOURCES], "totals": totals}, indent=2))
    print("\nTotal images:", sum(t["images"] for t in totals.values()))


def upload() -> None:
    if not MERGED.exists():
        sys.exit("Run `build` first.")
    ws = rf().workspace()
    print(f"Uploading {MERGED} to {ws.url}/{TARGET_PROJECT} ...")
    ws.upload_dataset(str(MERGED), TARGET_PROJECT, project_license="CC BY 4.0", project_type="object-detection")
    print("Done. Next: python scripts/train_on_roboflow.py version")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    {"inspect": inspect_sources, "build": build, "upload": upload}[cmd]()

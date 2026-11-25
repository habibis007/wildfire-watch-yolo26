"""Generate the notebooks from this file so they stay readable in code review.

    python scripts/build_notebooks.py

Edit the cells here, not in the .ipynb, then re-run.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = "habibis007/wildfire-watch-yolo26"
ROOT = Path(__file__).resolve().parents[1]


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.strip("\n").splitlines(keepends=True)}


def notebook(cells: list[dict], gpu: bool = True) -> dict:
    meta = {
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    }
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    return {"nbformat": 4, "nbformat_minor": 0, "metadata": meta, "cells": cells}


BADGE = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/notebooks/{{nb}})"

# ---------------------------------------------------------------------------
# Demo notebook
# ---------------------------------------------------------------------------
demo = [
    md(f"""
# Wildfire and smoke detection from drone video (YOLO26)

{BADGE.format(nb="wildfire_demo.ipynb")}

This notebook takes a drone video, detects **fire** and **smoke** frame by frame
with a YOLO26 model fine-tuned on Roboflow, draws boxes, raises an alert only when
detections **persist**, and writes an annotated output video.

| Step | What happens |
|---|---|
| 1. Setup | Install packages, read the Roboflow API key from Colab Secrets |
| 2. Frame extraction | Read the video with `cv2.VideoCapture`, sample every Nth frame |
| 3. Preprocessing | Resize, and optionally apply CLAHE contrast enhancement |
| 4. Detection | Load the trained model by its Roboflow ID and run it locally |
| 5. Visualization | Colour-coded boxes: fire vs smoke |
| 6. Alert logic | Alert only after N consecutive sampled frames, with timestamps |
| 7. Output | Write the annotated video and close the writer properly |

**Before you run:** *Runtime → Change runtime type → T4 GPU*, and add your Roboflow
API key as a Colab Secret named `ROBOFLOW_API_KEY` (key icon in the left sidebar).
"""),
    md("## 1. Setup"),
    code(f"""
import os, subprocess, sys

IN_COLAB = "google.colab" in sys.modules
HAS_GPU = subprocess.run("nvidia-smi", shell=True, capture_output=True).returncode == 0

# inference-gpu runs the model with CUDA; plain inference falls back to CPU.
!pip install -q {{"inference-gpu" if HAS_GPU else "inference"}} supervision opencv-python

if IN_COLAB and not os.path.exists("/content/wildfire-watch-yolo26"):
    !git clone -q --depth 1 https://github.com/{REPO}.git /content/wildfire-watch-yolo26
REPO_DIR = "/content/wildfire-watch-yolo26" if IN_COLAB else os.path.abspath("..")
sys.path.insert(0, os.path.join(REPO_DIR, "src"))
print("GPU:", HAS_GPU, "| repo:", REPO_DIR)
"""),
    code("""
# The API key is read from Colab Secrets (or an environment variable locally),
# so it never appears in the notebook.
try:
    from google.colab import userdata
    ROBOFLOW_API_KEY = userdata.get("ROBOFLOW_API_KEY")
except Exception:
    ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
print("API key loaded:", bool(ROBOFLOW_API_KEY))
"""),
    code("""
# ---- Settings you might change on stage ----
MODEL_ID        = "wildfire-watch-yolo26/1"   # <workspace project>/<version> on Roboflow
VIDEO           = "samples/clip_01.mp4"       # relative to the repo
EVERY_N         = 5      # run the model on every 5th frame
MIN_CONSECUTIVE = 5      # alert after 5 sampled frames in a row
CONFIDENCE      = 0.40   # ignore weaker detections
USE_CLAHE       = False  # contrast enhancement before detection
INFER_SIZE      = 640    # the model was trained at 640

import time, json
import cv2, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from wildfire import (AnnotatedVideoWriter, Detection, PersistenceAlert, apply_clahe,
                      draw_alert_banner, draw_detections, from_inference_result,
                      resize_keep_aspect, sample_frames, to_h264, video_info)

VIDEO_PATH = Path(REPO_DIR) / VIDEO
OUT_DIR = Path(REPO_DIR) / "outputs"; OUT_DIR.mkdir(exist_ok=True)
def show(img_bgr, title="", ax=None):
    ax = ax or plt.gca(); ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); ax.set_title(title); ax.axis("off")
"""),
    md("## 2. Frame extraction"),
    code("""
info = video_info(VIDEO_PATH)
print({k: round(v, 2) if isinstance(v, float) else v for k, v in info.items()})

# A strip of 6 evenly spaced sampled frames
sampled = list(sample_frames(VIDEO_PATH, every_n=EVERY_N))
picks = np.linspace(0, len(sampled) - 1, 6).astype(int)
fig, axes = plt.subplots(1, 6, figsize=(20, 3))
for ax, i in zip(axes, picks):
    show(sampled[i].image, f"t={sampled[i].timestamp:.1f}s", ax)
plt.show()
print(f"{len(sampled)} of {info['frame_count']} frames will go to the model (every {EVERY_N}th).")
"""),
    md("""
## 3. Preprocessing

The model resizes to 640 internally, but resizing first means less data to move to the GPU.
**CLAHE** (Contrast Limited Adaptive Histogram Equalisation) boosts local contrast, which can
make thin, hazy smoke easier to see. Compare:
"""),
    code("""
frame = sampled[picks[2]].image
small, scale = resize_keep_aspect(frame, INFER_SIZE)
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
show(small, f"resized {small.shape[1]}x{small.shape[0]}", axes[0])
show(apply_clahe(small), "resized + CLAHE", axes[1])
plt.show()
"""),
    md("""
## 4. Detection

`get_model` downloads the trained weights from Roboflow once, then runs them **locally** on
this machine (free; no per-call API charges). If the model can't be loaded (no key or no
network), the notebook falls back to detections cached from an earlier run, so the
demo still works.
"""),
    code("""
model = None
try:
    from inference import get_model
    model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
    print("Loaded", MODEL_ID)
except Exception as e:
    print("Could not load model, using cached detections instead:", repr(e)[:200])

CACHE = VIDEO_PATH.with_suffix(".detections.json")
cache = {int(k): [Detection.from_dict(d) for d in v] for k, v in json.loads(CACHE.read_text()).items()} if CACHE.exists() else {}

def detect(index, image):
    \"\"\"Detections in ORIGINAL frame coordinates.\"\"\"
    if model is None:
        return cache.get(index, [])
    small, scale = resize_keep_aspect(image, INFER_SIZE)
    if USE_CLAHE:
        small = apply_clahe(small)
    result = model.infer(small, confidence=CONFIDENCE)[0]
    return [d.scaled(1 / scale, 1 / scale) for d in from_inference_result(result, CONFIDENCE)]

sf = sampled[picks[2]]
t0 = time.perf_counter(); dets = detect(sf.index, sf.image); dt = time.perf_counter() - t0
print(f"{len(dets)} detections in {dt*1000:.0f} ms:", [(d.class_name, round(d.confidence, 2)) for d in dets])
"""),
    md("## 5. Visualization\n\nFire is drawn in **red-orange**, smoke in **slate blue-grey**."),
    code("""
plt.figure(figsize=(12, 7)); show(draw_detections(sf.image, dets), f"t={sf.timestamp:.1f}s"); plt.show()
"""),
    md(f"""
## 6 & 7. Alert logic and annotated output

One detection is not enough to raise an alarm: sun glare or a red roof can fool the model
for a frame. We alert only when the same class appears in **`MIN_CONSECUTIVE`** sampled
frames in a row, and print when it was first seen.

We read **every** frame so the output keeps the original speed, run the model on every
`EVERY_N`-th frame, and reuse the latest boxes in between. The writer is a context manager,
so the file is always closed and complete, even if you interrupt the cell.
"""),
    code("""
alert = PersistenceAlert(min_consecutive=MIN_CONSECUTIVE, min_confidence=CONFIDENCE)
raw_path = OUT_DIR / (VIDEO_PATH.stem + "_annotated_raw.mp4")
new_cache, last = {}, []
t0 = time.perf_counter()

with AnnotatedVideoWriter(raw_path, fps=info["fps"]) as writer:
    for f in sample_frames(VIDEO_PATH, every_n=EVERY_N, include_unsampled=True):
        if f.is_sampled:
            last = detect(f.index, f.image)
            new_cache[f.index] = [d.to_dict() for d in last]
            for ev in alert.update(f.timestamp, last):
                print(f"🔥 ALERT at {ev.raised_time:6.2f}s: {ev.class_name.upper()} "
                      f"(seen since {ev.start_time:.2f}s, conf {ev.peak_confidence:.2f})")
        out = draw_detections(f.image, last)
        out = draw_alert_banner(out, alert.status_text(), active=bool(alert.active_classes))
        writer.write(out)

elapsed = time.perf_counter() - t0
print(f"\\nWrote {writer.frames_written} frames in {elapsed:.1f}s "
      f"({writer.frames_written / elapsed:.0f} fps)")
print("\\nAlert summary:\\n" + alert.summary())

if model is not None:  # refresh the offline fallback for next time
    CACHE.write_text(json.dumps(new_cache))
"""),
    code("""
# OpenCV writes mp4v, which browsers can't play; convert to H.264 for viewing.
from IPython.display import HTML
from base64 import b64encode
final = to_h264(raw_path, OUT_DIR / (VIDEO_PATH.stem + "_annotated.mp4"))
print("Saved", final)
HTML(f'<video width=800 controls autoplay muted loop src="data:video/mp4;base64,{b64encode(final.read_bytes()).decode()}">')
"""),
    code("""
# Optional: download the annotated video
if IN_COLAB:
    from google.colab import files
    files.download(str(final))
"""),
    md("""
## Things to try

- Set `USE_CLAHE = True` and re-run steps 4 onwards. Does it find hazy smoke earlier?
- Lower `MIN_CONSECUTIVE` to 1: how many false alarms appear?
- Change `EVERY_N`: faster processing vs slower reaction.
"""),
]

# ---------------------------------------------------------------------------
# Fallback training notebook (only if Roboflow Train credits run short)
# ---------------------------------------------------------------------------
train = [
    md(f"""
# Fallback: train YOLO26 in Colab with Ultralytics

{BADGE.format(nb="train_fallback_ultralytics.ipynb")}

Use this only if Roboflow Train credits run short. It downloads the dataset version from
Roboflow, fine-tunes YOLO26 on a free T4, and uploads the weights back to the same Roboflow
version, so the demo notebook can still load the model by its ID.

Runtime → T4 GPU, and add `ROBOFLOW_API_KEY` as a Colab Secret.
"""),
    code("""
!pip install -q ultralytics roboflow
import os
try:
    from google.colab import userdata
    API_KEY = userdata.get("ROBOFLOW_API_KEY")
except Exception:
    API_KEY = os.environ["ROBOFLOW_API_KEY"]
"""),
    code("""
WORKSPACE = "your-workspace"          # from the Roboflow URL
PROJECT   = "wildfire-watch-yolo26"
VERSION   = 1
MODEL     = "yolo26n.pt"              # or yolo26s.pt
EPOCHS    = 60

from roboflow import Roboflow
version = Roboflow(api_key=API_KEY).workspace(WORKSPACE).project(PROJECT).version(VERSION)
# YOLO26 reads the same images/labels txt layout as the "yolov8" export.
ds = version.download("yolov8", location="/content/dataset")
"""),
    code("""
from ultralytics import YOLO
model = YOLO(MODEL)  # COCO-pretrained starting point: this is the transfer-learning step
results = model.train(data=f"{ds.location}/data.yaml", epochs=EPOCHS, imgsz=640, batch=32,
                      patience=15, project="/content/runs", name="wildfire", exist_ok=True)
"""),
    code("""
from IPython.display import Image, display
run = "/content/runs/wildfire"
for f in ["results.png", "confusion_matrix.png", "PR_curve.png"]:
    if os.path.exists(f"{run}/{f}"):
        display(Image(f"{run}/{f}", width=900))
metrics = YOLO(f"{run}/weights/best.pt").val(data=f"{ds.location}/data.yaml")
print(f"precision {metrics.box.mp:.3f}  recall {metrics.box.mr:.3f}  "
      f"mAP50 {metrics.box.map50:.3f}  mAP50-95 {metrics.box.map:.3f}")
"""),
    code("""
# Upload the weights to the same Roboflow version, so get_model("<project>/<version>") works.
version.deploy(model_type="yolo26", model_path=run, filename="weights/best.pt")
"""),
]

if __name__ == "__main__":
    for name, cells in [("wildfire_demo.ipynb", demo), ("train_fallback_ultralytics.ipynb", train)]:
        path = ROOT / "notebooks" / name
        path.write_text(json.dumps(notebook(cells), indent=1) + "\n")
        print("wrote", path.relative_to(ROOT))

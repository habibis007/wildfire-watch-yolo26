# How to run and use Wildfire Watch

This guide shows how to run the fire and smoke detector, use it on your own video, adjust its
behaviour, and retrain it. For background on how it works, see the [README](README.md).

**You will need**

- A Google account, for the Colab route. Or, to run locally, Python 3.10 or later.
- A free [Roboflow](https://roboflow.com) account and its API key: in Roboflow, open
  *Settings → API Keys* and copy the private key. Without a key the demo still plays,
  using detections saved from an earlier run.

---

## Option A: run it in Google Colab (easiest)

1. Open the notebook:
   <https://colab.research.google.com/github/habibis007/wildfire-watch-yolo26/blob/main/notebooks/wildfire_demo.ipynb>
2. Choose *Runtime → Change runtime type → T4 GPU*, then *Save*.
   A GPU is optional (a CPU also works), but the package install is faster.
3. Add your key as a secret. Click the key icon in the left sidebar, then *Add new secret*.
   Set the name to `ROBOFLOW_API_KEY`, paste your key as the value, and switch on
   *Notebook access*. Never paste the key into a code cell.
4. Choose *Runtime → Run all*.
5. Read the output as it runs:
   - **Step 2** shows a strip of sampled frames.
   - **Step 4** prints `Loaded wildfire-watch-yolo26-cicrc/1`. If it says *using cached
     detections* instead, the key or network is missing, but the demo still works.
   - **Step 6** prints a line such as `🔥 ALERT at 11.xx s: SMOKE`, followed by an alert summary.
   - **Step 7** plays the annotated video in the notebook. The last cell downloads it.

The first run takes a few minutes, mostly installing packages. After that, the detection loop
over a 30-second clip takes seconds.

## Option B: run it on your own computer

```bash
git clone https://github.com/habibis007/wildfire-watch-yolo26.git
cd wildfire-watch-yolo26
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyter
```

Give the notebook your key through an environment variable, in the same terminal:

```bash
export ROBOFLOW_API_KEY=your_key_here     # Windows PowerShell: $env:ROBOFLOW_API_KEY="your_key_here"
```

Then start Jupyter from the `notebooks/` folder and run every cell:

```bash
cd notebooks
jupyter notebook wildfire_demo.ipynb
```

The annotated videos are saved to `outputs/`, a folder git ignores. You get
`*_annotated_raw.mp4`, and a browser-friendly H.264 copy, `*_annotated.mp4`.

> **Note:** the H.264 copy needs `ffmpeg`. Without it, only the `_raw` file is written; it
> plays in VLC. Install ffmpeg with `brew install ffmpeg` on macOS or
> `sudo apt install ffmpeg` on Linux. Colab already has it.

## Use it on your own video

1. Put your video in the `samples/` folder, or anywhere inside the repo.
2. In the **Settings** cell (step 1), set `VIDEO` to its path, relative to the repo root.
   For example:
   ```python
   VIDEO = "samples/my_drone_clip.mp4"
   ```
3. Run the notebook from that cell onwards.

In Colab, first upload the file to `/content/wildfire-watch-yolo26/samples/` using the file
browser on the left.

Tips:
- The model was trained mostly on **aerial and surveillance views**. It works best on drone
  or tower-camera footage, and is weak on hazy ground-level shots.
- Videos without a saved `.detections.json` file need a working key: the offline fallback only
  covers the three sample clips.
- After a successful run, the notebook saves a new `.detections.json` next to your video,
  so the next run works offline.

## Settings you can change

These are all in the **Settings** cell. Re-run the notebook from step 4 after changing one.

| Setting | Default | What it does |
|---|---|---|
| `VIDEO` | `samples/clip_01_drone_smoke_onset.mp4` | The input video |
| `EVERY_N` | `5` | Run the model on every Nth frame. Higher is faster, but alerts react more slowly |
| `MIN_CONSECUTIVE` | `5` | Sampled frames in a row needed before an alert. `1` gives many false alarms |
| `CONFIDENCE` | `0.40` | Ignore detections below this score. Lower finds more, with more mistakes |
| `USE_CLAHE` | `False` | Contrast enhancement before detection. Helps people see haze, not the model |
| `INFER_SIZE` | `640` | Frame size sent to the model. Keep 640, the size it was trained at |
| `MODEL_ID` | `wildfire-watch-yolo26-cicrc/1` | Use another Roboflow model by its `project/version` ID |

Three experiments worth trying in a class:
- Set `MIN_CONSECUTIVE = 1` and count the false alarms.
- Set `VIDEO = "samples/clip_03_ground_haze.mp4"`, then turn `USE_CLAHE` on and off and
  compare when the alert fires.
- Set `EVERY_N = 15` and see how much later the alert comes.

## Use the detector in your own Python code

The code in `src/wildfire/` is a small library. The notebook is one way to use it; here is a
minimal script that does the same thing:

```python
import os, sys
sys.path.insert(0, "src")
from inference import get_model
from wildfire import (AnnotatedVideoWriter, PersistenceAlert, draw_alert_banner, draw_detections,
                      from_inference_result, resize_keep_aspect, sample_frames, video_info)

model = get_model("wildfire-watch-yolo26-cicrc/1", api_key=os.environ["ROBOFLOW_API_KEY"])
video = "samples/clip_01_drone_smoke_onset.mp4"
alert, last = PersistenceAlert(min_consecutive=5, min_confidence=0.4), []

with AnnotatedVideoWriter("outputs/my_run.mp4", fps=video_info(video)["fps"]) as writer:
    for f in sample_frames(video, every_n=5, include_unsampled=True):
        if f.is_sampled:
            small, scale = resize_keep_aspect(f.image, 640)
            result = model.infer(small, confidence=0.4)[0]
            last = [d.scaled(1 / scale, 1 / scale) for d in from_inference_result(result, 0.4)]
            for ev in alert.update(f.timestamp, last):
                print(f"ALERT {ev.class_name} at {ev.raised_time:.1f}s")
        writer.write(draw_alert_banner(draw_detections(f.image, last), alert.status_text(),
                                       active=bool(alert.active_classes)))
print(alert.summary())
```

Run it from the repo root with your virtual environment active.

`get_model` downloads the weights once, then runs them on your own machine. There is no
per-image charge.

## Retrain or improve the model

Retrain if the detector misses your kind of footage, for example ground-level views. These
steps use Roboflow credits; `train` asks before it spends any.

```bash
echo "ROBOFLOW_API_KEY=your_key_here" > .env   # read by the scripts; .env is git-ignored

python scripts/prepare_dataset.py inspect      # list the source datasets and their classes
python scripts/prepare_dataset.py build        # download, remap to fire/smoke -> data/merged
python scripts/spot_check.py                   # draw labels on random images -> data/spot_check.png
python scripts/prepare_dataset.py upload       # upload to a project in your workspace
python scripts/train_on_roboflow.py version    # resize to 640 and add augmentation
python scripts/train_on_roboflow.py train 1 yolo26n   # confirm, then train
python scripts/train_on_roboflow.py status 1   # check progress and metrics
```

To add your own footage:
1. Extract frames from it.
2. Label fire and smoke in Roboflow Annotate.
3. Add them to the project before running `version`.

When training finishes, put the new `project/version` ID in `MODEL_ID`.

**Out of credits?** Open `notebooks/train_fallback_ultralytics.ipynb` in Colab. It trains on
Colab's free GPU and uploads the weights back to the same Roboflow version.

## Check that everything works

```bash
pytest
```

The tests cover the alert logic, frame sampling, and the video writer. They need no key and
no network.

## Troubleshooting

| Problem | Fix |
|---|---|
| `API key loaded: False` | Colab: check the secret's name is exactly `ROBOFLOW_API_KEY` and *Notebook access* is on. Locally: `export` the key in the same terminal before starting Jupyter |
| `Could not load model, using cached detections` | Missing key or no network. The demo still runs on the sample clips |
| The video does not play in the notebook | `ffmpeg` is missing; open the `_annotated_raw.mp4` file in VLC |
| No alert on your video | Lower `CONFIDENCE` to 0.3, or `MIN_CONSECUTIVE` to 3. Check the footage resembles aerial views |
| Too many alerts | Raise `MIN_CONSECUTIVE` or `CONFIDENCE` |
| `ModuleNotFoundError: wildfire` | Run from the repo root (scripts) or from `notebooks/` (the notebook) |

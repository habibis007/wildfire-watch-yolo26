"""Generate a dataset version and train YOLO26 on Roboflow's hosted GPUs.

    python scripts/train_on_roboflow.py version          # resize 640 + light augmentation
    python scripts/train_on_roboflow.py train [N] [yolo26n|yolo26s] [epochs]
    python scripts/train_on_roboflow.py status [N]

Training uses Roboflow credits. `train` asks for confirmation before starting.
"""

from __future__ import annotations

import sys

from prepare_dataset import TARGET_PROJECT, rf

# Light augmentation: flips, brightness and blur (blur mimics haze / motion).
# "versions": 2 -> each training image appears twice (original + 1 augmented copy),
# which keeps the dataset (and training cost) modest.
VERSION_SETTINGS = {
    "preprocessing": {
        "auto-orient": True,
        "resize": {"width": 640, "height": 640, "format": "Stretch to"},
    },
    "augmentation": {
        "image": {"versions": 2},
        "flip": {"horizontal": True, "vertical": False},
        "brightness": {"brighten": True, "darken": True, "percent": 20},
        "blur": {"pixels": 1.5},
    },
}


def project():
    return rf().workspace().project(TARGET_PROJECT)


def make_version() -> None:
    n = project().generate_version(settings=VERSION_SETTINGS)
    print(f"Generating version {n} of {TARGET_PROJECT}. Model ID will be '{TARGET_PROJECT}/{n}'.")


def train(version: int, model_type: str = "yolo26n", epochs: int | None = None) -> None:
    ans = input(f"Start Roboflow training of {TARGET_PROJECT}/{version} as {model_type}? This spends credits. [y/N] ")
    if ans.strip().lower() != "y":
        print("Cancelled.")
        return
    project().version(version).train(model_type=model_type, epochs=epochs)
    print(f"Training finished. Load it with get_model('{TARGET_PROJECT}/{version}').")


def status(version: int) -> None:
    v = project().version(version)
    print("model:", getattr(v, "model", None))


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:] or ["version"]
    if cmd == "version":
        make_version()
    elif cmd == "train":
        train(int(rest[0]), rest[1] if len(rest) > 1 else "yolo26n", int(rest[2]) if len(rest) > 2 else None)
    elif cmd == "status":
        status(int(rest[0]))
    else:
        sys.exit(__doc__)

import cv2
import numpy as np

from wildfire import (
    AnnotatedVideoWriter,
    Detection,
    apply_clahe,
    draw_alert_banner,
    draw_detections,
    from_inference_result,
    resize_keep_aspect,
    sample_frames,
    video_info,
)


def make_video(path, n=30, size=(320, 240), fps=10):
    with AnnotatedVideoWriter(path, fps=fps) as w:
        for i in range(n):
            frame = np.full((size[1], size[0], 3), i * 8 % 255, np.uint8)
            w.write(frame)
    return path


def test_writer_produces_complete_readable_file(tmp_path):
    p = make_video(tmp_path / "v.mp4", n=30)
    info = video_info(p)
    assert info["frame_count"] == 30
    assert (info["width"], info["height"]) == (320, 240)


def test_writer_releases_on_exception(tmp_path):
    p = tmp_path / "crash.mp4"
    try:
        with AnnotatedVideoWriter(p, fps=10) as w:
            for i in range(12):
                w.write(np.zeros((120, 160, 3), np.uint8))
            raise KeyboardInterrupt  # simulate stopping the cell mid-run
    except KeyboardInterrupt:
        pass
    assert video_info(p)["frame_count"] == 12


def test_sample_frames_every_n(tmp_path):
    p = make_video(tmp_path / "v.mp4", n=30, fps=10)
    sampled = list(sample_frames(p, every_n=5))
    assert [f.index for f in sampled] == [0, 5, 10, 15, 20, 25]
    assert sampled[1].timestamp == 0.5
    full = list(sample_frames(p, every_n=5, include_unsampled=True))
    assert len(full) == 30 and sum(f.is_sampled for f in full) == 6


def test_sample_frames_max_seconds(tmp_path):
    p = make_video(tmp_path / "v.mp4", n=30, fps=10)
    assert len(list(sample_frames(p, every_n=1, max_seconds=1.0))) == 11


def test_resize_and_clahe_shapes():
    img = np.random.randint(0, 255, (1080, 1920, 3), np.uint8)
    small, scale = resize_keep_aspect(img, 640)
    assert small.shape == (360, 640, 3) and abs(scale - 1 / 3) < 1e-9
    assert apply_clahe(small).shape == small.shape
    same, s = resize_keep_aspect(small, 640)
    assert s == 1.0 and same is small


def test_draw_does_not_mutate_input():
    img = np.zeros((240, 320, 3), np.uint8)
    out = draw_detections(img, [Detection("fire", 0.9, 10, 10, 100, 100), Detection("smoke", 0.5, 300, 200, 400, 300)])
    out = draw_alert_banner(out, "ALERT: FIRE", active=True)
    assert img.sum() == 0 and out.sum() > 0


def test_from_inference_result_dict():
    result = {"predictions": [
        {"x": 50, "y": 40, "width": 20, "height": 10, "class": "fire", "confidence": 0.9},
        {"x": 5, "y": 5, "width": 2, "height": 2, "class": "smoke", "confidence": 0.1},
    ]}
    dets = from_inference_result(result, min_confidence=0.25)
    assert dets == [Detection("fire", 0.9, 40, 35, 60, 45)]
    assert dets[0].scaled(2, 2).x2 == 120

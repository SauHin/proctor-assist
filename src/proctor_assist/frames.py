"""Extract frames from the webcam videos (D18). Run: python -m proctor_assist.frames

Writes ROOT/data/processed/frames/subjectN/<k>.jpg. Frame k is at k / 5 seconds of the video.
Writes the frame counts and sizes to results/frames/. The subject splits must exist first.
"""
import json

import cv2
import pandas as pd

from proctor_assist.env import REPO, ROOT, env_info
from proctor_assist.labels import OEP, SUBJECTS
from proctor_assist.splits import load_splits

VIDEO_FPS = 25
STEP = 5  # keep every 5th frame (D18)
FPS = VIDEO_FPS // STEP  # frame rate of the stored frames
SIZE = (224, 224)  # the whole image, no crop (D18)
QUALITY = 90  # JPEG quality
OUT = ROOT / "data" / "processed" / "frames"
RESULTS = REPO / "results" / "frames"


def preprocess(frame):
    """Resize one video frame (BGR) to the stored size. The app must use this same function."""
    return cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA)


def extract(video, out_dir):
    """Save every STEP-th frame of one video as a JPEG file. Return the counts and the size."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FPS)) != VIDEO_FPS:
        raise SystemExit(f"{video}: cannot open the file, or it does not have {VIDEO_FPS} fps")
    header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_dir.mkdir(parents=True, exist_ok=True)
    decoded = saved = size = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if decoded % STEP == 0:
            path = out_dir / f"{saved:06d}.jpg"
            if not cv2.imwrite(str(path), preprocess(frame), [cv2.IMWRITE_JPEG_QUALITY, QUALITY]):
                raise SystemExit(f"Cannot write {path}")
            size += path.stat().st_size
            saved += 1
        decoded += 1
    cap.release()
    return {"frames_in_header": header, "frames_decoded": decoded, "frames_saved": saved, "bytes": size}


def main():
    # Data leakage rule: each subject has its split before a frame is extracted.
    in_splits = {s for part in load_splits()["protocol_a"].values() for s in part}
    if in_splits != set(SUBJECTS):
        raise SystemExit("The subject splits do not cover all subjects. Run proctor_assist.splits first.")
    rows = []
    for s in SUBJECTS:
        out_dir = OUT / f"subject{s}"
        done = out_dir / "_done.json"  # written last, so an interrupted subject is extracted again
        if not done.exists():
            done.write_text(json.dumps(extract(OEP / f"subject{s}" / "webcam.avi", out_dir)))
        rows.append({"subject": s} | json.loads(done.read_text()))
        print(f"subject{s}: {rows[-1]['frames_saved']} frames", flush=True)
    table = pd.DataFrame(rows)
    table["seconds"] = table.frames_saved // FPS  # whole seconds that have all their frames
    info = env_info()
    summary = {
        "fps": FPS,
        "size": SIZE,
        "jpeg_quality": QUALITY,
        "frames_saved": int(table.frames_saved.sum()),
        "seconds": int(table.seconds.sum()),
        "gigabytes": round(table.bytes.sum() / 2**30, 3),
        "subjects_where_header_and_decoded_count_differ": int((table.frames_in_header != table.frames_decoded).sum()),
        "run": {k: info[k] for k in ("time_utc", "git_sha", "git_dirty", "python")} | {"opencv": cv2.__version__},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "summary.csv", index=False, lineterminator="\n")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    print(f"Frames are in {OUT}. Wrote the summary to {RESULTS}.")


if __name__ == "__main__":
    main()

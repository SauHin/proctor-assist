"""Face, hand, and body landmarks for E2 (D32). Run: python -m proctor_assist.landmarks

Reads every 5th frame of each webcam video, so row k belongs to stored frame k (D18). Stores all
raw points in data/processed/landmarks/subjectN.npz (never committed: landmarks are personal
data). Writes the detection rates per subject to results/landmarks/summary.csv.
`frame_features` turns the raw points into the model input. The app must use the same function.
"""
import json
import math
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from proctor_assist import frames
from proctor_assist.download import sha256
from proctor_assist.env import FAST_DATA, REPO, ROOT, env_info
from proctor_assist.labels import OEP, SUBJECTS
from proctor_assist.splits import load_splits

MODELS = ROOT / "data" / "models" / "mediapipe"
MODEL_SHA256 = {  # D25 and D32
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "hand_landmarker.task": "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    "pose_landmarker_full.task": "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
}
OUT = FAST_DATA / "processed" / "landmarks"
RESULTS = REPO / "results" / "landmarks"
SIZE = np.array([640, 480])  # all 24 webcam videos have this size (results/eda/summary.md)
# Face mesh points: nose tip, eye corners and lids, inner lips, and mouth corners.
FACE_POINTS = [1, 33, 133, 159, 145, 362, 263, 386, 374, 13, 14, 61, 291]
# Body points: nose, ears, shoulders, elbows, and wrists.
POSE_POINTS = [0, 7, 8, 11, 12, 13, 14, 15, 16]
N_FEATURES = 3 + 2 * len(FACE_POINTS) + 2 * len(POSE_POINTS) + 2 * 2 * 21 + 4


def _tools():
    from mediapipe.tasks.python import BaseOptions, vision  # only the laptop has MediaPipe (D25)

    for name, digest in MODEL_SHA256.items():
        if sha256(MODELS / name) != digest:
            raise SystemExit(f"{MODELS / name} is not the model file of D25 or D32.")
    base = lambda name: BaseOptions(model_asset_path=str(MODELS / name))
    return (
        vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base("face_landmarker.task"), output_facial_transformation_matrixes=True)
        ),
        vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(base_options=base("hand_landmarker.task"), num_hands=2)),
        vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=base("pose_landmarker_full.task"))),
    )


def extract(subject, n_frames):
    """Raw landmarks of the first n_frames stored frames of one video. A part that is not found is NaN."""
    import mediapipe as mp

    face_tool, hand_tool, pose_tool = _tools()
    raw = {
        "face": np.full((n_frames, 478, 3), np.nan, np.float16),
        "face_matrix": np.full((n_frames, 4, 4), np.nan, np.float32),
        "pose": np.full((n_frames, 33, 4), np.nan, np.float16),  # x, y, z, visibility
        "hands": np.full((n_frames, 2, 21, 3), np.nan, np.float16),  # sorted from left to right in the image
    }
    cap = cv2.VideoCapture(str(OEP / f"subject{subject}" / "webcam.avi"))
    i = k = 0
    while k < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % frames.STEP == 0:
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame[:, :, ::-1]))
            face, hands, pose = face_tool.detect(image), hand_tool.detect(image), pose_tool.detect(image)
            if face.face_landmarks:
                raw["face"][k] = [(p.x, p.y, p.z) for p in face.face_landmarks[0]]
                raw["face_matrix"][k] = np.asarray(face.facial_transformation_matrixes[0])
            if pose.pose_landmarks:
                raw["pose"][k] = [(p.x, p.y, p.z, p.visibility) for p in pose.pose_landmarks[0]]
            for slot, hand in enumerate(sorted(hands.hand_landmarks, key=lambda h: h[0].x)[:2]):
                raw["hands"][k, slot] = [(p.x, p.y, p.z) for p in hand]
            k += 1
        i += 1
    cap.release()
    if k != n_frames:
        raise SystemExit(f"subject{subject}: the video has {k} of the {n_frames} stored frames.")
    return raw


def frame_features(raw):
    """Model input per frame (D32): N_FEATURES values, as float32.

    Head yaw, pitch, and roll (degrees / 90). Selected face and body points, and all hand points, in
    pixels, minus the middle of the shoulders, divided by the shoulder width. Then one flag each for
    face, body, and two hands. A part that is not found, or no body to normalize with, gives zeros.
    """
    r = raw["face_matrix"][:, :3, :3].astype(np.float64)
    yaw = np.degrees(np.arctan2(r[:, 0, 2], r[:, 2, 2]))
    pitch = np.degrees(np.arcsin(np.clip(-r[:, 1, 2], -1, 1)))
    roll = np.degrees(np.arctan2(r[:, 1, 0], r[:, 1, 1]))
    head = np.stack([yaw, pitch, roll], 1) / 90

    px = lambda points: points[..., :2].astype(np.float64) * SIZE
    shoulders = px(raw["pose"][:, [11, 12]])
    origin = shoulders.mean(1)
    width = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    width = np.where(width > 1, width, np.nan)  # no body found, or a width of less than one pixel
    norm = lambda points: ((px(points) - origin[:, None]) / width[:, None, None]).reshape(len(width), -1)
    parts = [
        head,
        norm(raw["face"][:, FACE_POINTS]),
        norm(raw["pose"][:, POSE_POINTS]),
        norm(raw["hands"].reshape(len(width), 42, 3)),
    ]
    flags = np.stack([found(raw["face"]), found(raw["pose"]), found(raw["hands"][:, 0]), found(raw["hands"][:, 1])], 1)
    return np.nan_to_num(np.concatenate(parts + [flags], 1)).astype(np.float32)


def found(part):
    """True for each frame where this part has at least one point."""
    return ~np.isnan(part.reshape(len(part), -1).astype(np.float64)).all(1)


def _one(subject, n_frames):
    path = OUT / f"subject{subject}.npz"
    if not path.exists():
        raw = extract(subject, n_frames)
        np.savez(OUT / f"subject{subject}.partial.npz", **raw)  # renamed last, so a stopped run starts again
        (OUT / f"subject{subject}.partial.npz").replace(path)
    raw = np.load(path)
    return {
        "subject": subject,
        "frames": len(raw["face"]),
        "face_found": found(raw["face"]).mean(),
        "body_found": found(raw["pose"]).mean(),
        "hand_1_found": found(raw["hands"][:, 0]).mean(),
        "hand_2_found": found(raw["hands"][:, 1]).mean(),
    }


def main(subjects=SUBJECTS):
    # Data leakage rule: each subject has its split before a landmark is extracted.
    in_splits = {s for part in load_splits()["protocol_a"].values() for s in part}
    if in_splits != set(SUBJECTS):
        raise SystemExit("The subject splits do not cover all subjects. Run proctor_assist.splits first.")
    n_frames = pd.read_csv(frames.RESULTS / "summary.csv").set_index("subject").frames_saved
    OUT.mkdir(parents=True, exist_ok=True)
    # One process per video. 4 processes leave cores free on the 16-core laptop.
    with ProcessPoolExecutor(4) as pool:
        rows = list(pool.map(_one, subjects, [int(n_frames[s]) for s in subjects]))
    for row in rows:
        print(f"subject{row['subject']}: face {row['face_found']:.3f}, body {row['body_found']:.3f}", flush=True)
    if list(subjects) != list(SUBJECTS):
        return
    info = env_info()
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS / "summary.csv", index=False, float_format="%.4f", lineterminator="\n")
    run = {k: info[k] for k in ("time_utc", "git_sha", "git_dirty", "python")} | {"n_features": N_FEATURES}
    (RESULTS / "run.json").write_text(json.dumps(run, indent=2) + "\n", newline="\n")
    print(f"Landmarks are in {OUT}. Wrote the detection rates to {RESULTS}.")


if __name__ == "__main__":
    import sys

    main([int(s) for s in sys.argv[1:]] or SUBJECTS)

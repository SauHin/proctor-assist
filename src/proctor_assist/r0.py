"""R0: a head-turn rule on the MediaPipe face pose, with no training (D25).

Run: python -m proctor_assist.r0

Step 1 reads the head pose of one frame per second from each webcam video. It stores the pose in
data/processed/pose/ (never committed). Step 2 computes the suspicion score per second and
writes results/r0/: the scores, the ROC-AUC for each split part, and the tables per subject and
per type code.
"""
import json
import math

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from proctor_assist import frames
from proctor_assist.download import sha256
from proctor_assist.env import REPO, ROOT, env_info
from proctor_assist.evaluate import per_code, per_subject
from proctor_assist.labels import OEP, SUBJECTS, clean_labels, load_labels
from proctor_assist.splits import load_splits
from proctor_assist.windows import window_table

MODEL = ROOT / "data" / "models" / "mediapipe" / "face_landmarker.task"
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
POSE = ROOT / "data" / "processed" / "pose"
RESULTS = REPO / "results" / "r0"
FULL_TURN = 30.0  # degrees from the normal position that give the score 1 (D25)
FRAME_IN_SECOND = frames.STEP * (frames.FPS - 1)  # video frame of the last stored frame of a second


def head_pose(matrix):
    """Yaw and pitch in degrees from the face transformation matrix of MediaPipe.

    Yaw is the turn to the left or right. Pitch is the turn up or down.
    """
    r = np.asarray(matrix)
    return math.degrees(math.atan2(r[0, 2], r[2, 2])), math.degrees(math.asin(max(-1.0, min(1.0, -r[1, 2]))))


def landmarker():
    """The MediaPipe face landmarker of D25."""
    from mediapipe.tasks.python import BaseOptions, vision

    if sha256(MODEL) != MODEL_SHA256:
        raise SystemExit(f"{MODEL} is not the model file of D25.")
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL)), output_facial_transformation_matrixes=True
    )
    return vision.FaceLandmarker.create_from_options(options)


def pose_row(tool, frame, t):
    """The head pose of one video frame (BGR) of second t. The app uses this function too."""
    import mediapipe as mp

    found = tool.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame[:, :, ::-1])))
    face = bool(found.face_landmarks)
    yaw, pitch = head_pose(found.facial_transformation_matrixes[0]) if face else (None, None)
    return {"t": t, "face": face, "yaw": yaw, "pitch": pitch}


def read_poses(subject, n_seconds):
    """Head pose of one frame per second of one video: columns t, face, yaw, pitch."""
    cap = cv2.VideoCapture(str(OEP / f"subject{subject}" / "webcam.avi"))
    rows, i = [], 0
    with landmarker() as tool:  # only the laptop has MediaPipe (optional extra "landmarks")
        while len(rows) < n_seconds:
            ok, frame = cap.read()
            if not ok:
                break
            if i % frames.VIDEO_FPS == FRAME_IN_SECOND:
                rows.append(pose_row(tool, frame, i // frames.VIDEO_FPS))
            i += 1
    cap.release()
    return pd.DataFrame(rows)


def suspicion(pose):
    """Suspicion score per second from the pose table of one subject (D25).

    The normal position is the median of the seconds with a face up to the current second. A
    second with no face gets the score of the last second with a face, or 0 at the start.
    """
    yaw = pose.yaw.where(pose.face).astype(float)
    pitch = pose.pitch.where(pose.face).astype(float)
    distance = np.hypot(yaw - yaw.expanding().median(), pitch - pitch.expanding().median())
    return (distance / FULL_TURN).clip(upper=1).ffill().fillna(0).to_numpy()


def main():
    n_seconds = pd.read_csv(frames.RESULTS / "summary.csv").set_index("subject").seconds
    labels = clean_labels(load_labels())
    POSE.mkdir(parents=True, exist_ok=True)
    tables, faces = [], {}
    for s in SUBJECTS:
        path = POSE / f"subject{s}.csv"
        if not path.exists():
            read_poses(s, n_seconds[s]).to_csv(path, index=False)
            print(f"subject{s}: pose read", flush=True)
        pose = pd.read_csv(path)
        faces[s] = float(pose.face.mean())
        # The same seconds as for all models: a full window before them, and a target (D18).
        seconds = window_table([s], labels, n_seconds)
        tables.append(seconds.assign(p0=1 - suspicion(pose)[seconds.t]))
    scores = pd.concat(tables, ignore_index=True)

    splits = load_splits()
    named = [("A", splits["protocol_a"]), *((f"B fold {k}", f) for k, f in enumerate(splits["protocol_b"], 1))]
    rows = []
    for name, split in named:
        for part in ("val", "test"):
            g = scores[scores.subject.isin(split[part])]
            rows.append({"split": name, "part": part, "seconds": len(g), "roc_auc_cheat": roc_auc_score(g.target > 0, 1 - g.p0)})
    info = env_info()
    summary = {
        "model": "r0",
        "full_turn_degrees": FULL_TURN,
        "seconds": len(scores),
        "roc_auc_cheat_all_subjects": float(roc_auc_score(scores.target > 0, 1 - scores.p0)),
        "share_of_seconds_with_a_face": {"min": min(faces.values()), "mean": float(np.mean(list(faces.values())))},
        "run": {k: info[k] for k in ("time_utc", "git_sha", "git_dirty", "python")},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS / "scores.csv", index=False, float_format="%.5f", lineterminator="\n")
    pd.DataFrame(rows).to_csv(RESULTS / "roc_auc_by_split.csv", index=False, float_format="%.4f", lineterminator="\n")
    table = per_subject(scores.assign(part="all"))
    table["share_of_seconds_with_a_face"] = table.subject.map(faces)
    table.to_csv(RESULTS / "per_subject.csv", index=False, float_format="%.4f", lineterminator="\n")
    per_code(scores.assign(part="all"), labels).to_csv(RESULTS / "per_code.csv", index=False, float_format="%.4f", lineterminator="\n")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    print(f"Wrote the R0 scores and tables to {RESULTS}")


if __name__ == "__main__":
    main()

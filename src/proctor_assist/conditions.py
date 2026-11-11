"""Tests under changed conditions (D37, D40), on the test subjects of the 4 folds of protocol B.

Run on the laptop (MediaPipe, D25):
  python -m proctor_assist.conditions extract
  python -m proctor_assist.conditions score --runs "G:/My Drive/proctor-assist/runs"

An image condition changes each video frame (BGR, 640 x 480) before every other step, as a worse
camera would. `extract` stores, per condition and subject, the E2 input of each stored frame, the
face matrix for R0, and the scoring frame of each second for P1 and B1 (data.expand), in
data/processed/conditions/ (never committed). A lower frame rate shows each stored frame for
longer, so it uses the data without a change and needs no extraction.

`score` scores the test subjects of each fold with P1, B1, E2, and R0. The threshold of each run
comes from its validation subjects without a change (D37), with the code of proctor_assist.events.
The condition "none" scores the data without a change with the same code, as the reference.
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image

from proctor_assist import events, frames, landmarks, r0
from proctor_assist.data import frame_table, transform
from proctor_assist.env import FAST_DATA, REPO, ROOT, env_info
from proctor_assist.features import SequenceDataset
from proctor_assist.labels import SUBJECTS
from proctor_assist.models import build
from proctor_assist.splits import load_splits
from proctor_assist.train import predict

CONDITIONS = {  # two levels each, moderate and strong (D37). The values: D40.
    "low_light": (0.5, 0.25),  # factor on the brightness of each pixel
    "blur": (8, 16),  # sigma of a Gaussian blur, in pixels of the 640 x 480 frame
    "jpeg": (15, 7),  # JPEG quality
    "fps": (2.5, 1),  # frames per second of the camera
}
LEVELS = ("moderate", "strong")
NONE = ("none", "fps", frames.FPS)  # the stored data without a change
MODELS = ("r0", "b1", "p1", "e2")
OUT = FAST_DATA / "processed" / "conditions"
RESULTS = REPO / "results" / "conditions"


def named():
    """[(name, kind, value)] of the 8 conditions, for example ("blur_strong", "blur", 6)."""
    return [(f"{kind}_{level}", kind, v) for kind, values in CONDITIONS.items() for level, v in zip(LEVELS, values)]


def change(frame, kind, value):
    """One video frame (BGR, uint8) under an image condition."""
    if kind == "low_light":
        return cv2.convertScaleAbs(frame, alpha=value)
    if kind == "blur":
        return cv2.GaussianBlur(frame, (0, 0), value)
    if kind == "jpeg":
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, value])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    raise ValueError(f"{kind} is not an image condition")


def held(k, fps):
    """The stored frame that a camera with `fps` frames per second shows at stored frame k."""
    step = frames.FPS / fps
    return (np.floor(np.asarray(k) / step) * step).astype(int)


def _extract(name, kind, value, subject, n_frames):
    out = OUT / name / f"subject{subject}"
    if (out / "landmarks.npz").exists():
        return name, subject
    out.mkdir(parents=True, exist_ok=True)

    def keep(k, frame):  # the scoring frame of each second for P1 and B1 (data.expand)
        if k % frames.FPS == frames.FPS - 1:
            path = out / f"{k:06d}.jpg"
            if not cv2.imwrite(str(path), frames.preprocess(frame), [cv2.IMWRITE_JPEG_QUALITY, frames.QUALITY]):
                raise SystemExit(f"Cannot write {path}")

    raw = landmarks.extract(subject, n_frames, lambda f: change(f, kind, value), keep)
    np.savez(out / "landmarks.partial.npz", features=landmarks.frame_features(raw), face_matrix=raw["face_matrix"])
    (out / "landmarks.partial.npz").replace(out / "landmarks.npz")  # written last, so a stopped run starts again
    return name, subject


def extract(processes):
    n_frames = pd.read_csv(frames.RESULTS / "summary.csv").set_index("subject").frames_saved
    jobs = [(name, kind, v, s, int(n_frames[s])) for name, kind, v in named() if kind != "fps" for s in SUBJECTS]
    with ProcessPoolExecutor(processes) as pool:
        for name, s in pool.map(_extract, *zip(*jobs)):
            print(f"{name} subject{s}: done", flush=True)


class _Frames(Dataset):
    """The scoring frames of the rows of a frame table, from one folder per subject."""

    def __init__(self, table, root):
        self.rows = [(root / f"subject{s}" / f"{k:06d}.jpg", y) for s, k, y in zip(table.subject, table.frame, table.target)]
        self.transform = transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, y = self.rows[i]
        return self.transform(decode_image(str(path))), y


def inputs(name, kind, value, subjects):
    """{subject: (E2 input, face matrix)} of all stored frames of each subject under a condition."""
    if kind != "fps":
        return {s: tuple(np.load(OUT / name / f"subject{s}" / "landmarks.npz")[k] for k in ("features", "face_matrix")) for s in subjects}
    out = {}
    for s in subjects:
        raw = np.load(landmarks.OUT / f"subject{s}.npz")
        idx = held(np.arange(len(raw["face"])), value)
        out[s] = (landmarks.frame_features(raw)[idx], raw["face_matrix"][idx])
    return out


def r0_p0(table, matrices):
    """R0 scores (as P(no cheat)) of the rows of a table, from the face matrices of each subject."""
    out = []
    for s, g in table.groupby("subject", sort=False):
        m = matrices[s]
        rows = []
        for t in range(len(m) // frames.FPS):
            k = frames.FPS * t + frames.FPS - 1  # the frame that R0 reads (r0.FRAME_IN_SECOND)
            face = not np.isnan(m[k]).all()
            yaw, pitch = r0.head_pose(m[k]) if face else (None, None)
            rows.append({"t": t, "face": face, "yaw": yaw, "pitch": pitch})
        out.append(1 - r0.suspicion(pd.DataFrame(rows))[g.t.to_numpy()])
    return np.concatenate(out)


def model_p0(model, dataset, workers):
    _, probs, _ = predict(model, DataLoader(dataset, 256, num_workers=workers), "cpu")
    return probs[:, 0]


def checkpoint(name, places):
    for place in places:
        if (place / name / "best.pt").exists():
            return place / name / "best.pt"
    raise SystemExit(f"No best.pt for {name} in {places}")


def threshold(val):
    val = val.assign(suspicion=1 - val.p0)
    return events.choose(events.curve(events.prepare(val), events.thresholds(val)))


def score(runs_dir):
    info = env_info()  # before the outputs are written
    places = [Path(runs_dir), ROOT / "runs"] if runs_dir else [ROOT / "runs"]
    r0_all = pd.read_csv(REPO / "results" / "r0" / "scores.csv")
    (RESULTS / "scores").mkdir(parents=True, exist_ok=True)
    rows, checks = [], {}
    for fold, split in enumerate(load_splits()["protocol_b"], 1):
        test = frame_table(split["test"], train=False)  # subject, t, target, frame
        runs = {m: next(f for f in sorted(REPO.glob(f"results/{m}_b_f{fold}_*")) if (f / "scores_val.csv").exists()) for m in MODELS[1:]}
        saved = {m: pd.read_csv(runs[m] / "scores_test.csv").p0.to_numpy() for m in runs}
        saved["r0"] = r0_all.set_index(["subject", "t"]).loc[list(zip(test.subject, test.t))].p0.to_numpy()
        th = {m: threshold(pd.read_csv(runs[m] / "scores_val.csv")) for m in runs}
        th["r0"] = threshold(r0_all[r0_all.subject.isin(split["val"])])
        models = {}
        for m, run in runs.items():
            model = build(yaml.safe_load((run / "config.yaml").read_text()), pretrained=False)
            model.load_state_dict(torch.load(checkpoint(run.name, places), map_location="cpu"))
            models[m] = model.eval()
        for name, kind, value in [NONE] + named():
            path = RESULTS / "scores" / f"scores_f{fold}_{name}.csv"
            if path.exists():
                p0 = pd.read_csv(path)
            else:
                data = inputs(name, kind, value, split["test"])
                if kind == "fps":
                    root, table = frames.OUT, test.assign(frame=held(test.frame, value))
                else:
                    root, table = OUT / name, test
                p0 = test[["subject", "t", "target"]].copy()
                p0["r0"] = r0_p0(test, {s: d[1] for s, d in data.items()})
                p0["e2"] = model_p0(models["e2"], SequenceDataset(test, {s: torch.from_numpy(d[0]) for s, d in data.items()}), 0)
                for m in ("b1", "p1"):
                    p0[m] = model_p0(models[m], _Frames(table, root), 4)  # 4 processes decode the JPEG files
                p0.to_csv(path, index=False, float_format="%.5f", lineterminator="\n")
            if name == "none":  # the CPU scores without a change against the saved scores of the runs
                checks[f"fold {fold}"] = {m: float(np.abs(p0[m].to_numpy() - saved[m]).max()) for m in MODELS}
            for m in MODELS:
                scores = test[["subject", "t", "target"]].assign(suspicion=1 - p0[m].to_numpy())
                n, _ = events.count(events.prepare(scores), th[m])
                rows.append(
                    {"model": m, "fold": fold, "condition": name, "kind": kind, "value": value, "threshold": th[m],
                     "roc_auc_cheat": roc_auc_score(scores.target > 0, scores.suspicion)} | n | events.rates(n)
                )
            print(f"fold {fold} {name}: done", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "summary.csv", index=False, float_format="%.4f", lineterminator="\n")
    keys = ["roc_auc_cheat", "detected_share", "false_alarms_per_hour"]
    g = table.groupby(["model", "condition"], sort=False)[keys]
    pd.concat({"mean": g.mean(), "std": g.std()}, axis=1).to_csv(RESULTS / "protocol_b.csv", float_format="%.4f", lineterminator="\n")
    run = {k: info[k] for k in ("time_utc", "git_sha", "git_dirty", "python")} | {
        "conditions": CONDITIONS, "torch_threads": torch.get_num_threads(),
        "none_max_abs_diff_of_p0_vs_saved_scores": checks,
    }
    (RESULTS / "run.json").write_text(json.dumps(run, indent=2) + "\n", newline="\n")
    print(f"Wrote summary.csv and protocol_b.csv to {RESULTS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["extract", "score"])
    parser.add_argument("--processes", type=int, default=6)
    parser.add_argument("--runs", help="folder with the run folders of P1 and B1, for example the Drive folder")
    args = parser.parse_args()
    if args.step == "extract":
        extract(args.processes)
    else:
        score(args.runs)


if __name__ == "__main__":
    main()

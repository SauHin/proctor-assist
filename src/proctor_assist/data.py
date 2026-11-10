"""Frame tables and the dataset for the per-frame models (D18, D21).

Run: python -m proctor_assist.data prints one hash of the decoded pixels of some frames. The
hash must be the same on the laptop and in Colab (D20).
"""
import hashlib

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

from proctor_assist import frames
from proctor_assist.frames import FPS
from proctor_assist.labels import SUBJECTS, clean_labels, load_labels
from proctor_assist.windows import WINDOW_S, window_table

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet statistics of the pretrained weights


def expand(windows, train):
    """Turn rows of (subject, t, target) into rows with a `frame` index.

    For training: all FPS frames of the second (D21). For a score: the last frame of the second.
    """
    if not train:
        return windows.assign(frame=windows.t * FPS + FPS - 1).reset_index(drop=True)
    rows = windows.loc[windows.index.repeat(FPS)]
    return rows.assign(frame=rows.t * FPS + rows.groupby(level=0).cumcount()).reset_index(drop=True)


def frame_table(subjects, train):
    """Frames and targets of `subjects` for a per-frame model.

    Training uses each second that has a target. A score uses the seconds that have a full window,
    so that all models are scored on the same seconds.
    """
    labels = clean_labels(load_labels())
    n_seconds = pd.read_csv(frames.RESULTS / "summary.csv").set_index("subject").seconds
    return expand(window_table(subjects, labels, n_seconds, length_s=1 if train else WINDOW_S), train)


def transform(augment=None):
    """Tensor transform. `augment` holds degrees, translate, scale, and hflip (D21), or is None."""
    steps = []
    if augment:
        d, t, s = augment["degrees"], augment["translate"], augment["scale"]
        steps.append(v2.RandomAffine(degrees=d, translate=(t, t), scale=(1 - s, 1 + s)))
        if augment["hflip"]:
            steps.append(v2.RandomHorizontalFlip())
    return v2.Compose(steps + [v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


def read_frame(subject, frame):
    """One stored frame as a uint8 tensor: RGB, 3 x 224 x 224."""
    return decode_image(str(frames.OUT / f"subject{subject}" / f"{frame:06d}.jpg"))


class FrameDataset(Dataset):
    def __init__(self, table, transform):
        self.rows = list(zip(table.subject.tolist(), table.frame.tolist(), table.target.tolist()))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        subject, frame, target = self.rows[i]
        return self.transform(read_frame(subject, frame)), target


if __name__ == "__main__":
    h = hashlib.sha256()
    for s in SUBJECTS:
        for k in (0, 500, 1000):
            h.update(read_frame(s, k).numpy().tobytes())
    print("decoded pixels of 72 frames, sha256:", h.hexdigest())

"""Frame features from a trained P1 backbone, and the window dataset of E1 (D29).

The features of a frame are the output of the last hidden dense layer of P1. They are computed
one time per P1 run and stored in data/processed/features/<P1 run>/subjectN.npy (never committed).
"""
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from proctor_assist import frames, landmarks
from proctor_assist.data import FrameDataset, transform
from proctor_assist.env import FAST_DATA, ROOT
from proctor_assist.labels import SUBJECTS
from proctor_assist.models import build
from proctor_assist.windows import frame_ids


def p1_run(cfg):
    """The run folder of the model named in cfg["features"], for the same protocol and fold."""
    fold = f"f{cfg['fold']}_" if cfg["protocol"] == "b" else ""
    pattern = f"{cfg['features']}_{cfg['protocol']}_{fold}2*"  # a run folder name goes on with its date
    found = sorted(p for p in (ROOT / "runs").glob(pattern) if (p / "best.pt").exists())
    if len(found) != 1:
        raise SystemExit(f"Expected one finished run for {pattern} in {ROOT / 'runs'}, found {len(found)}.")
    return found[0]


@torch.no_grad()
def load(cfg, device):
    """Return {subject: features of all its stored frames}. Compute and store the missing files."""
    if cfg["features"] == "landmarks":  # E2: the landmark files come from proctor_assist.landmarks (D32)
        return {s: torch.from_numpy(landmarks.frame_features(np.load(landmarks.OUT / f"subject{s}.npz"))) for s in SUBJECTS}
    run = p1_run(cfg)
    out = FAST_DATA / "processed" / "features" / run.name
    out.mkdir(parents=True, exist_ok=True)
    model = None
    for s in SUBJECTS:
        path = out / f"subject{s}.npy"
        if path.exists():
            continue
        if model is None:
            model = build(yaml.safe_load((run / "config.yaml").read_text()), pretrained=False).to(device).eval()
            model.load_state_dict(torch.load(run / "best.pt", map_location=device))
        n = sum(1 for _ in (frames.OUT / f"subject{s}").glob("*.jpg"))
        table = pd.DataFrame({"subject": s, "frame": range(n), "target": 0})
        batches = DataLoader(FrameDataset(table, transform()), 256, num_workers=cfg.get("workers", 8))
        feats = [model.head[:-1](model.features(x.to(device))).cpu() for x, _ in batches]
        np.save(path, torch.cat(feats).numpy())
    return {s: torch.from_numpy(np.load(out / f"subject{s}.npy")) for s in SUBJECTS}


class SequenceDataset(Dataset):
    """One item per window: the features of its frames (frames x feature size), and its target."""

    def __init__(self, table, feats):
        self.rows = list(zip(table.subject.tolist(), table.t.tolist(), table.target.tolist()))
        self.feats = feats

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        subject, t, target = self.rows[i]
        ids = frame_ids(t)
        return self.feats[subject][ids.start : ids.stop], target

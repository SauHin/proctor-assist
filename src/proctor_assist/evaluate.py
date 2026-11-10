"""Look closer at the scores of one run: per subject, and per OEP type code.

Run: python -m proctor_assist.evaluate results/<run> [--checkpoint <best.pt>]
     python -m proctor_assist.evaluate --summary p1_b      (all runs with this prefix, after the line above)

Writes per_subject.csv and per_code.csv into the folder, from its scores_*.csv files. With
--checkpoint, the script first scores the training subjects with the same steps as validation
(scores_train.csv and metrics_train.json). This shows the gap between the subjects that the
model saw in training and the subjects that it did not see.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from proctor_assist.data import FrameDataset, frame_table, transform
from proctor_assist.env import REPO
from proctor_assist.labels import REAL, clean_labels, load_labels
from proctor_assist.metrics import report
from proctor_assist.models import MODELS
from proctor_assist.train import predict, split_of


def per_subject(scores):
    """One row per part and subject: how well the suspicion score separates cheating seconds."""
    rows = []
    for (part, subject), g in scores.groupby(["part", "subject"], sort=False):
        cheat, suspicion = g.target > 0, 1 - g.p0
        rows.append(
            {
                "part": part,
                "subject": subject,
                "group": "real" if subject in REAL else "actor",
                "seconds": len(g),
                "cheat_seconds": int(cheat.sum()),
                "roc_auc_cheat": roc_auc_score(cheat, suspicion) if 0 < cheat.sum() < len(g) else None,
                "mean_suspicion_cheat": suspicion[cheat].mean(),
                "mean_suspicion_no_cheat": suspicion[~cheat].mean(),
            }
        )
    return pd.DataFrame(rows)


def per_code(scores, labels):
    """One row per part and OEP type code: the suspicion on the seconds that the code covers.

    A second with two codes counts for both. Code 0 stands for the seconds with target "no cheat".
    """
    spans = [(r.subject, t, r.code) for r in labels.itertuples() for t in range(r.start_s, r.end_s)]
    coded = pd.DataFrame(spans, columns=["subject", "t", "code"]).drop_duplicates()
    joined = scores.assign(suspicion=1 - scores.p0).merge(coded, on=["subject", "t"], how="left")
    joined["code"] = joined.code.where(joined.target > 0, 0).astype(int)
    table = joined.groupby(["part", "code"], sort=False).agg(
        seconds=("suspicion", "size"),
        subjects=("subject", "nunique"),
        mean_suspicion=("suspicion", "mean"),
        share_suspicion_over_half=("suspicion", lambda s: (s > 0.5).mean()),
    )
    return table.reset_index().sort_values(["part", "code"])


def score_train(folder, checkpoint):
    cfg = yaml.safe_load((folder / "config.yaml").read_text())
    table = frame_table(split_of(cfg)["train"], train=False)
    model = MODELS[cfg["model"]](dense_layers=cfg["dense_layers"], units=cfg["units"], pretrained=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    _, probs, targets = predict(model.to(device), DataLoader(FrameDataset(table, transform()), 128), device)
    scores = table[["subject", "t", "target"]].reset_index(drop=True)
    scores[[f"p{c}" for c in range(probs.shape[1])]] = probs
    scores.to_csv(folder / "scores_train.csv", index=False, float_format="%.5f")
    (folder / "metrics_train.json").write_text(json.dumps(report(targets, probs), indent=2) + "\n")


def summary(prefix, results=REPO / "results"):
    """Write results/<prefix>_summary.csv: one row per run folder, then the mean and the standard deviation.

    Also write results/<prefix>_test_per_subject.csv: the test rows per subject of all runs, with the
    R0 value for the same seconds next to them.
    """
    rows, subjects = [], []
    for folder in sorted(f for f in results.glob(prefix + "*") if (f / "metrics.json").exists()):
        m = json.loads((folder / "metrics.json").read_text())
        row = {"run": folder.name, "epochs_run": m["epochs_run"], "best_epoch": m["best_epoch"]}
        row["train_minutes"] = m["train_seconds"] / 60
        for part in ("val", "test"):
            for key in ("macro_f1", "baseline_macro_f1", "accuracy", "baseline_accuracy", "roc_auc_cheat"):
                row[f"{part}_{key}"] = m[part][key]
        rows.append(row)
        per = pd.read_csv(folder / "per_subject.csv")
        subjects.append(per[per.part == "test"].assign(run=folder.name))
    table = pd.DataFrame(rows)
    numbers = table.drop(columns="run")
    stats = [numbers.mean().to_frame().T.assign(run="mean"), numbers.std().to_frame().T.assign(run="std")]
    pd.concat([table, *stats], ignore_index=True).to_csv(
        results / f"{prefix}_summary.csv", index=False, float_format="%.4f", lineterminator="\n"
    )
    r0 = pd.read_csv(results / "r0" / "per_subject.csv").set_index("subject").roc_auc_cheat
    per = pd.concat(subjects, ignore_index=True).drop(columns="part").sort_values("subject")
    per.assign(r0_roc_auc_cheat=per.subject.map(r0)).to_csv(
        results / f"{prefix}_test_per_subject.csv", index=False, float_format="%.4f", lineterminator="\n"
    )
    print(f"Wrote {prefix}_summary.csv and {prefix}_test_per_subject.csv for {len(rows)} runs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?")
    parser.add_argument("--checkpoint")
    parser.add_argument("--summary", help="prefix of the run folders in results/, for example p1_b")
    args = parser.parse_args()
    if args.summary:
        return summary(args.summary)
    folder = Path(args.folder)
    if args.checkpoint:
        score_train(folder, args.checkpoint)
    parts = [p for p in ("train", "val", "test") if (folder / f"scores_{p}.csv").exists()]
    scores = pd.concat([pd.read_csv(folder / f"scores_{p}.csv").assign(part=p) for p in parts], ignore_index=True)
    per_subject(scores).to_csv(folder / "per_subject.csv", index=False, float_format="%.4f", lineterminator="\n")
    per_code(scores, clean_labels(load_labels())).to_csv(
        folder / "per_code.csv", index=False, float_format="%.4f", lineterminator="\n"
    )
    print(f"Wrote per_subject.csv and per_code.csv for {parts} to {folder}")


if __name__ == "__main__":
    main()

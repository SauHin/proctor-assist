"""Train one model from a YAML config and write a run folder (D3, D21).

Run: python -m proctor_assist.train configs/p1_a.yaml
     python -m proctor_assist.train configs/p1_b.yaml --fold 2                (protocol B, fold 2)
     python -m proctor_assist.train configs/p1_a.yaml --resume <run folder>   (after a disconnect)
     python -m proctor_assist.train configs/p1_a.yaml --smoke                 (short check, CPU is enough)

The run folder gets config.yaml, env.json, history.csv with one row per epoch, last.pt, best.pt,
metrics.json, and the class probabilities per second of the validation and test subjects.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from proctor_assist.data import FrameDataset, frame_table, transform
from proctor_assist.env import new_run_dir
from proctor_assist.metrics import report
from proctor_assist.models import build
from proctor_assist.splits import load_splits


def split_of(cfg):
    splits = load_splits()
    return splits["protocol_a"] if cfg["protocol"] == "a" else splits["protocol_b"][cfg["fold"] - 1]


@torch.no_grad()
def predict(model, loader, device):
    """Return the mean loss, the class probabilities, and the targets for all rows of the loader."""
    model.eval()
    loss, probs, targets = 0.0, [], []
    for x, y in loader:
        logits = model(x.to(device))
        loss += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
        probs.append(logits.softmax(1).cpu())
        targets.append(y)
    targets = torch.cat(targets).numpy()
    return loss / len(targets), torch.cat(probs).numpy(), targets


def run(cfg, run_dir, smoke=False):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = split_of(cfg)
    tables = {part: frame_table(split[part], train=part == "train") for part in ("train", "val", "test")}
    if smoke:  # a few rows and 2 epochs: this only shows that the pipeline works
        tables = {p: t.sample(n=min(len(t), 320 if p == "train" else 96), random_state=0) for p, t in tables.items()}
        cfg = cfg | {"epochs": 2, "pretrained": False}
    workers = 0 if smoke else cfg.get("workers", min(8, os.cpu_count()))

    counts = tables["train"].target.value_counts()
    sampler = None
    if cfg["balanced_sampling"]:  # each class has the same chance in a batch (D21)
        weights = (1 / counts[tables["train"].target]).tolist()
        generator = torch.Generator().manual_seed(cfg["seed"])
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), generator=generator)
    train_data = FrameDataset(tables["train"], transform(cfg["augment"]))
    loaders = {p: DataLoader(FrameDataset(tables[p], transform()), 128, num_workers=workers) for p in ("val", "test")}
    loaders["train"] = DataLoader(
        train_data, cfg["batch_size"], sampler=sampler, shuffle=sampler is None, num_workers=workers, drop_last=True
    )

    model = build(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    state = {"epoch": 0, "best_val_loss": float("inf"), "best_epoch": 0, "train_seconds": 0.0, "history": []}
    last = run_dir / "last.pt"
    if last.exists():  # continue a run after a disconnect
        saved = torch.load(last, map_location=device)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]

    # Early stopping: stop after `patience` epochs without a lower validation loss.
    while state["epoch"] < cfg["epochs"] and state["epoch"] - state["best_epoch"] < cfg["patience"]:
        start = time.time()
        model.train()
        loss_sum, preds, targets = 0.0, [], []
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
        preds, targets = torch.cat(preds), torch.cat(targets)
        val_loss, val_probs, val_targets = predict(model, loaders["val"], device)
        state["epoch"] += 1
        state["train_seconds"] += time.time() - start
        state["history"].append(
            {
                "epoch": state["epoch"],
                "train_loss": loss_sum / len(targets),
                "train_macro_f1": f1_score(targets, preds, average="macro"),
                "val_loss": val_loss,
                "val_macro_f1": f1_score(val_targets, val_probs.argmax(1), average="macro"),
                "seconds": time.time() - start,
            }
        )
        if val_loss < state["best_val_loss"]:
            state["best_val_loss"], state["best_epoch"] = val_loss, state["epoch"]
            torch.save(model.state_dict(), run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "state": state}, last)
        pd.DataFrame(state["history"]).to_csv(run_dir / "history.csv", index=False)
        print(json.dumps(state["history"][-1]), flush=True)

    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device))
    metrics = {
        "model": cfg["model"],
        "device": device,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs_run": state["epoch"],
        "best_epoch": state["best_epoch"],
        "train_seconds": state["train_seconds"],
        "train_frames_per_class": {int(c): int(n) for c, n in counts.sort_index().items()},
    }
    for part in ("val", "test"):
        _, probs, targets = predict(model, loaders[part], device)
        metrics[part] = report(targets, probs, majority_class=int(counts.idxmax()))
        scores = tables[part][["subject", "t", "target"]].reset_index(drop=True)
        scores[[f"p{c}" for c in range(probs.shape[1])]] = probs
        scores.to_csv(run_dir / f"scores_{part}.csv", index=False, float_format="%.5f")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--resume", help="run folder to continue")
    parser.add_argument("--smoke", action="store_true", help="short check with a few rows")
    parser.add_argument("--fold", type=int, help="fold of protocol B, from 1 to 4")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.fold:
        cfg = cfg | {"fold": args.fold, "name": f"{cfg['name']}_f{args.fold}"}
    if args.resume:
        run_dir = Path(args.resume)
    else:
        run_dir = new_run_dir(cfg["name"] + ("-smoke" if args.smoke else ""))
        # The values that the run uses, with the fold. The file in configs/ has the comments.
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics = run(cfg, run_dir, args.smoke)
    print(f"run folder: {run_dir}")
    print(json.dumps({part: {k: metrics[part][k] for k in ("macro_f1", "baseline_macro_f1", "roc_auc_cheat")} for part in ("val", "test")}))


if __name__ == "__main__":
    main()

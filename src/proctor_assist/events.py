"""Event-level evaluation (D37): detected cheating events against false alarms per hour.

Run: python -m proctor_assist.events      (after proctor_assist.evaluate, on the local score files)

An event is a run of consecutive scored seconds with a cheat target. A missing second (masked,
D14) ends a run. An alarm second is a scored second with a suspicion at or above the threshold.
An event is detected if one of its seconds, widened by TOLERANCE seconds at each end, is an alarm
second. Each run of alarm seconds outside all widened events is one false alarm (D38), so one
long alarm over the whole exam gives many false alarms. False alarms per hour use all scored
seconds, so the value is the load per exam taker that a proctor sees.

The threshold of each run comes from its validation subjects only (CLAUDE.md, PLAN §6): the one
with the most detected events at no more than BUDGET false alarms per hour.
"""
import json

import numpy as np
import pandas as pd

from proctor_assist.env import REPO, env_info
from proctor_assist.labels import REAL
from proctor_assist.splits import load_splits

TOLERANCE = 2  # seconds, D37
BUDGET = 2.0  # false alarms per hour per exam taker, D37
GRID = np.linspace(0, 1, 1001)  # quantiles of the validation scores that serve as thresholds
RESULTS = REPO / "results"
OUT = RESULTS / "events"
MODELS = ("r0", "b1", "p1", "p1s", "e1", "e2")


def runs(t, flag):
    """(starts, ends) of the runs of consecutive seconds where flag is true. The ends are inclusive."""
    t = np.asarray(t)[np.asarray(flag, bool)]
    if len(t) == 0:
        return t, t
    breaks = np.diff(t) != 1
    return t[np.r_[True, breaks]], t[np.r_[breaks, True]]


def prepare(scores, tolerance=TOLERANCE):
    """Per subject: seconds, suspicion, event starts, the row indices of each widened event, and
    the mask of the seconds near an event. These do not depend on the threshold."""
    out = []
    for _, g in scores.groupby("subject", sort=False):
        t = g.t.to_numpy()
        starts, ends = runs(t, g.target > 0)
        inside = [np.flatnonzero((t >= a - tolerance) & (t <= b + tolerance)) for a, b in zip(starts, ends)]
        near = np.zeros(len(t), bool)
        for idx in inside:
            near[idx] = True
        out.append({"t": t, "suspicion": g.suspicion.to_numpy(), "starts": starts, "inside": inside, "near": near})
    return out


def count(subjects, threshold):
    """Events, detected events, false alarms, scored seconds, and delays, over the prepared subjects.

    The delay is the first alarm second inside the widened event minus the event start. It can be
    negative, down to -TOLERANCE.
    """
    n = {"events": 0, "detected": 0, "false_alarms": 0, "seconds": 0}
    delays = []
    for s in subjects:
        alarm = s["suspicion"] >= threshold
        for start, idx in zip(s["starts"], s["inside"]):
            hits = idx[alarm[idx]]
            if len(hits):
                n["detected"] += 1
                delays.append(int(s["t"][hits[0]] - start))
        n["events"] += len(s["starts"])
        n["false_alarms"] += len(runs(s["t"], alarm & ~s["near"])[0])
        n["seconds"] += len(s["t"])
    return n, delays


def rates(n):
    return {
        "detected_share": n["detected"] / n["events"] if n["events"] else np.nan,
        "false_alarms_per_hour": n["false_alarms"] / (n["seconds"] / 3600),
    }


def curve(subjects, thresholds):
    return pd.DataFrame([{"threshold": th, **rates(count(subjects, th)[0])} for th in thresholds])


def choose(val_curve, budget=BUDGET):
    """The threshold with the most detected events at no more than budget false alarms per hour.

    Ties go to the higher threshold. The last threshold (infinity) gives no alarm, so a choice exists.
    """
    ok = val_curve[val_curve.false_alarms_per_hour <= budget]
    return ok.sort_values(["detected_share", "threshold"]).threshold.iloc[-1]


def thresholds(val):
    return np.r_[np.unique(np.quantile(val.suspicion, GRID)), np.inf]


def evaluate(val, test):
    """Threshold from val only, then the counts on test. Returns the summary row and the curves."""
    grid = thresholds(val)
    val_subjects, test_subjects = prepare(val), prepare(test)
    val_curve = curve(val_subjects, grid)
    th = choose(val_curve)
    row = {"threshold": th}
    row |= {f"val_{k}": v for k, v in rates(count(val_subjects, th)[0]).items()}
    n, delays = count(test_subjects, th)
    row |= {f"test_{k}": v for k, v in n.items()} | {f"test_{k}": v for k, v in rates(n).items()}
    row["test_median_delay_s"] = np.median(delays) if delays else np.nan
    for group, part in test.groupby(test.subject.isin(REAL).map({True: "real", False: "actor"})):
        row |= {f"test_{group}_{k}": v for k, v in rates(count(prepare(part), th)[0]).items()}
    curves = pd.concat([val_curve.assign(part="val"), curve(test_subjects, grid).assign(part="test")])
    return row, curves


def run_folders():
    """(model, protocol, fold, name, val scores, test scores) for every trained run and for R0."""
    splits = load_splits()
    r0 = pd.read_csv(RESULTS / "r0" / "scores.csv")
    parts = [("a", 0, splits["protocol_a"])] + [("b", i + 1, s) for i, s in enumerate(splits["protocol_b"])]
    for protocol, fold, split in parts:
        yield "r0", protocol, fold, "r0", *(r0[r0.subject.isin(split[p])] for p in ("val", "test"))
    for model in MODELS[1:]:
        for folder in sorted(RESULTS.glob(f"{model}_[ab]_2*")) + sorted(RESULTS.glob(f"{model}_b_f*")):
            protocol = folder.name.split("_")[1]
            fold = int(folder.name.split("_")[2][1]) if protocol == "b" else 0
            yield model, protocol, fold, folder.name, *(pd.read_csv(folder / f"scores_{p}.csv") for p in ("val", "test"))


def main():
    info = env_info()  # before the outputs are written, so they do not make the tree dirty
    rows, curves = [], []
    for model, protocol, fold, name, val, test in run_folders():
        val, test = (s.assign(suspicion=1 - s.p0) for s in (val, test))
        row, c = evaluate(val, test)
        rows.append({"model": model, "protocol": protocol, "fold": fold, "run": name} | row)
        curves.append(c.assign(model=model, protocol=protocol, fold=fold))
        print(f"{name} {protocol}{fold}: test detected {row['test_detected_share']:.3f} at "
              f"{row['test_false_alarms_per_hour']:.2f} false alarms per hour", flush=True)
    table = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "summary.csv", index=False, float_format="%.4f", lineterminator="\n")
    keys, values = ["model", "protocol", "fold", "part"], ["detected_share", "false_alarms_per_hour"]
    c = pd.concat(curves, ignore_index=True)
    c = c[c.groupby(keys)[values].diff().fillna(1).abs().sum(axis=1) != 0]  # drop points that repeat the previous point
    c[keys + ["threshold"] + values].to_csv(
        OUT / "curves.csv", index=False, float_format="%.4f", lineterminator="\n"
    )
    b = table[table.protocol == "b"].drop(columns=["protocol", "fold", "run"]).groupby("model", sort=False)
    pd.concat({"mean": b.mean(), "std": b.std()}, axis=1).to_csv(
        OUT / "protocol_b.csv", float_format="%.4f", lineterminator="\n"
    )
    run = {k: info[k] for k in ("time_utc", "git_sha", "git_dirty", "python")} | {"tolerance_s": TOLERANCE, "budget_per_hour": BUDGET}
    (OUT / "run.json").write_text(json.dumps(run, indent=2) + "\n", newline="\n")
    print(f"Wrote summary.csv, curves.csv, and protocol_b.csv to {OUT}")


if __name__ == "__main__":
    main()

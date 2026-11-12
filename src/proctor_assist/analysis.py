"""Error analysis of milestone 7 (D42), on the test subjects of the 4 folds of protocol B.

Run: python -m proctor_assist.analysis      (after proctor_assist.events and proctor_assist.conditions)

Each run keeps the threshold that proctor_assist.events chose on its validation subjects
(results/events/summary.csv). Every subject is a test subject in exactly one fold, so the tables
pool the 24 subjects. Writes to results/analysis/:
- events.csv: one row per model and test event, with its class, OEP codes, length, and result.
- by_class.csv, by_code.csv, by_length.csv, by_group.csv: the share of events detected per group.
- per_subject.csv: events, detected events, and false alarms per model and subject.
- false_alarms.csv: one row per model and false alarm, with its time, and the second and value of
  its peak suspicion. The frame review of D43 takes the 20 rows of P1 and E1 with the highest peak.
- confusion.csv: the confusion matrix per second (largest probability) of the trained models.
- faces.csv: the share of stored frames with a face, per image condition (D40) and subject.
"""
import json

import numpy as np
import pandas as pd

from proctor_assist import conditions, events, landmarks
from proctor_assist.env import REPO, env_info
from proctor_assist.labels import CLASSES, PRIORITY, REAL, SUBJECTS, clean_labels, load_labels
from proctor_assist.splits import load_splits

OUT = REPO / "results" / "analysis"
LENGTHS = [0, 2, 5, 10, 30, np.inf]  # bins of event length in seconds: 1-2, 3-5, 6-10, 11-30, over 30


def test_scores():
    """(model, fold, threshold, test scores with suspicion) for every protocol B run."""
    summary = pd.read_csv(events.OUT / "summary.csv")
    r0 = pd.read_csv(REPO / "results" / "r0" / "scores.csv")
    splits = load_splits()["protocol_b"]
    for row in summary[summary.protocol == "b"].itertuples():
        if row.model == "r0":
            scores = r0[r0.subject.isin(splits[row.fold - 1]["test"])]
        else:
            scores = pd.read_csv(REPO / "results" / row.run / "scores_test.csv")
        yield row.model, row.fold, row.threshold, row, scores.assign(suspicion=1 - scores.p0)


def event_class(targets):
    """The class with the most seconds in an event. A tie goes to the first class in PRIORITY."""
    counts = np.bincount(targets, minlength=len(CLASSES))
    return max(PRIORITY, key=lambda c: (counts[c], -PRIORITY.index(c)))


def analyse(scores, threshold, codes):
    """Rows per test event, per subject, and per false alarm, for one run at its threshold."""
    ev, subj, fa = [], [], []
    for s, g in scores.groupby("subject", sort=False):
        t, target, sus = g.t.to_numpy(), g.target.to_numpy(), g.suspicion.to_numpy()
        alarm = sus >= threshold
        near = np.zeros(len(t), bool)
        detected = 0
        for a, b in zip(*events.runs(t, target > 0)):
            inside = (t >= a - events.TOLERANCE) & (t <= b + events.TOLERANCE)
            near |= inside
            hits = t[inside & alarm]
            own = (t >= a) & (t <= b)
            detected += len(hits) > 0
            ev.append({
                "subject": s, "start_s": a, "end_s": b, "length_s": b - a + 1,
                "class": CLASSES[event_class(target[own])],
                "codes": " ".join(str(c) for c in sorted({c for x in t[own] for c in codes.get((s, x), ())})),
                "detected": len(hits) > 0, "delay_s": hits[0] - a if len(hits) else np.nan,
            })
        starts, ends = events.runs(t, alarm & ~near)
        for a, b in zip(starts, ends):
            span = (t >= a) & (t <= b)
            fa.append({"subject": s, "start_s": a, "end_s": b, "length_s": b - a + 1,
                       "peak_t": t[span][sus[span].argmax()], "peak_suspicion": sus[span].max()})
        subj.append({"subject": s, "events": len(events.runs(t, target > 0)[0]), "detected": detected,
                     "false_alarms": len(starts), "seconds": len(t)})
    fa = pd.DataFrame(fa, columns=["subject", "start_s", "end_s", "length_s", "peak_t", "peak_suspicion"])  # R0 has none
    return pd.DataFrame(ev), pd.DataFrame(subj), fa.astype({c: int for c in fa.columns[:5]} | {"peak_suspicion": float})


def share(table, by):
    g = table.groupby(["model", by], sort=False, observed=True).detected
    return g.agg(events="size", detected="sum", detected_share="mean").reset_index()


def faces():
    """Share of stored frames with a face (the face matrix of MediaPipe exists), per condition and subject."""
    rows = []
    for s in SUBJECTS:
        rows.append({"condition": "none", "subject": s, "face_share": float(
            (~np.isnan(np.load(landmarks.OUT / f"subject{s}.npz")["face_matrix"]).all(axis=(1, 2))).mean())})
    for name, kind, _ in conditions.named():
        if kind == "fps":
            continue  # a lower frame rate repeats frames, so the share does not change
        for s in SUBJECTS:
            m = np.load(conditions.OUT / name / f"subject{s}" / "landmarks.npz")["face_matrix"]
            rows.append({"condition": name, "subject": s, "face_share": float((~np.isnan(m).all(axis=(1, 2))).mean())})
    return pd.DataFrame(rows).assign(group=lambda d: np.where(d.subject.isin(REAL), "real", "actor"))


def main():
    info = env_info()  # before the outputs are written
    labels = clean_labels(load_labels())
    codes = {}
    for r in labels.itertuples():
        for x in range(r.start_s, r.end_s):
            codes.setdefault((r.subject, x), set()).add(r.code)
    ev, subj, fa, conf = [], [], [], []
    for model, fold, th, row, scores in test_scores():
        e, s, f = analyse(scores, th, codes)
        if (e.detected.sum(), s.false_alarms.sum()) != (row.test_detected, row.test_false_alarms):
            raise SystemExit(f"{model} fold {fold}: the counts differ from results/events/summary.csv")
        tag = {"model": model, "fold": fold}
        ev.append(e.assign(**tag))
        subj.append(s.assign(**tag))
        fa.append(f.assign(**tag))
        if "p3" in scores:
            pred = scores[["p0", "p1", "p2", "p3"]].to_numpy().argmax(axis=1)
            m = pd.crosstab(scores.target, pred).reindex(index=range(4), columns=range(4), fill_value=0)
            conf += [{"model": model, "fold": fold, "true": CLASSES[i], "predicted": CLASSES[j], "seconds": int(m.loc[i, j])}
                     for i in range(4) for j in range(4)]
    OUT.mkdir(exist_ok=True)
    ev = pd.concat(ev, ignore_index=True)
    ev["group"] = np.where(ev.subject.isin(REAL), "real", "actor")
    ev["length_bin"] = pd.cut(ev.length_s, LENGTHS, labels=["1-2", "3-5", "6-10", "11-30", "over 30"])
    subj = pd.concat(subj, ignore_index=True)
    subj["group"] = np.where(subj.subject.isin(REAL), "real", "actor")
    subj["false_alarms_per_hour"] = subj.false_alarms / (subj.seconds / 3600)
    codes_long = ev.assign(code=ev.codes.str.split()).explode("code")
    group = share(ev, "group").merge(
        subj.groupby(["model", "group"]).agg(false_alarms=("false_alarms", "sum"), seconds=("seconds", "sum")).reset_index())
    group["false_alarms_per_hour"] = group.false_alarms / (group.seconds / 3600)
    tables = {
        "events": ev, "per_subject": subj, "false_alarms": pd.concat(fa, ignore_index=True),
        "by_class": share(ev, "class"), "by_code": share(codes_long, "code"),
        "by_length": share(ev, "length_bin"), "by_group": group,
        "confusion": pd.DataFrame(conf).groupby(["model", "true", "predicted"], sort=False).seconds.sum().reset_index(),
        "faces": faces(),
    }
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, float_format="%.4f", lineterminator="\n")
    run = {k: info[k] for k in ("time_utc", "git_sha", "git_dirty", "python")} | {"length_bins_s": "1-2, 3-5, 6-10, 11-30, over 30"}
    (OUT / "run.json").write_text(json.dumps(run, indent=2) + "\n", newline="\n")
    print(f"Wrote {', '.join(tables)} to {OUT}")


if __name__ == "__main__":
    main()

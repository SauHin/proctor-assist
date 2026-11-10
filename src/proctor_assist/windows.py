"""Windows over the stored frames (D18): one window per second, with the class of its last second.

A window is formed when the data is read, so its length can change without a new extraction.
Run: python -m proctor_assist.windows writes the number of windows per class for each split
to results/windows/summary.csv.
"""
import pandas as pd

from proctor_assist.env import REPO
from proctor_assist.frames import FPS
from proctor_assist.frames import RESULTS as FRAME_RESULTS
from proctor_assist.labels import CLASSES, MASK, clean_labels, load_labels, second_labels
from proctor_assist.splits import load_splits

WINDOW_S = 3  # window length of the temporal models, in seconds (D18)


def frame_ids(t, length_s=WINDOW_S):
    """Indices of the stored frames in the window that ends with second t."""
    return range(FPS * (t - length_s + 1), FPS * (t + 1))


def window_table(subjects, labels, n_seconds, length_s=WINDOW_S):
    """One row per window: subject, t (its last second), and target (the D14 class of second t).

    Only `subjects` give windows, so training windows come from training subjects only. A second
    without a full window before it gives no window. A second with a masked target gives none.
    """
    rows = []
    for s in subjects:
        y = second_labels(labels[labels.subject == s], n_seconds[s])
        rows += [(s, t, int(y[t])) for t in range(length_s - 1, len(y)) if y[t] != MASK]
    return pd.DataFrame(rows, columns=["subject", "t", "target"])


def main():
    labels = clean_labels(load_labels())
    n_seconds = pd.read_csv(FRAME_RESULTS / "summary.csv").set_index("subject").seconds
    splits = load_splits()
    named = [("A", splits["protocol_a"]), *((f"B fold {k}", f) for k, f in enumerate(splits["protocol_b"], 1))]
    rows = []
    for name, split in named:
        for part in ("train", "val", "test"):
            counts = window_table(split[part], labels, n_seconds).target.value_counts()
            row = {"split": name, "part": part, "windows": int(counts.sum())}
            rows.append(row | {f"class_{c}": int(counts.get(c, 0)) for c in CLASSES})
    out = REPO / "results" / "windows" / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

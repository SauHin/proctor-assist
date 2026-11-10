"""EDA of the OEP labels and webcam videos (milestone 2). Run: python -m proctor_assist.eda

Tables, summary.md, and figures go to results/eda/ (committed). Contact sheets of sample frames
and the list of overlapping rows go to ROOT/data/eda/ (never committed: the frames show faces).

The length of an interval is end - start. It is not known if the annotators meant the end
second as part of the interval, so summary.json has the total for both readings.
"""
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from proctor_assist.env import REPO, ROOT, env_info
from proctor_assist.labels import OEP, SUBJECTS, load_labels

OUT = REPO / "results" / "eda"
PRIVATE = ROOT / "data" / "eda"
# Reference values from Atoum et al. 2017 (section IV and Fig. 9), for paper types 1 to 5.
# These are the paper's numbers, not our results.
PAPER = {
    "instances": 569,
    "cheat_seconds": 7235,
    "share_instances_pct": [34, 50, 10, 3, 4],
    "share_seconds_pct": [16, 50, 21, 1, 12],
}
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order, one color per type code
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
KEY = ["subject", "start_s", "end_s", "code"]


def video_table():
    """Read the properties of each webcam video from its header, and try to read the last frame."""
    rows = []
    for s in SUBJECTS:
        cap = cv2.VideoCapture(str(OEP / f"subject{s}" / "webcam.avi"))
        if not cap.isOpened():
            raise SystemExit(f"Cannot open the webcam video of subject{s}")
        fps, frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames - 1)
        rows.append(
            {
                "subject": s,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": fps,
                "frames": frames,
                "codec": int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little").decode("latin-1"),
                "last_frame_readable": cap.read()[0],
                "duration_s": frames / fps,
            }
        )
        cap.release()
    return pd.DataFrame(rows).set_index("subject")


def per_code(valid):
    """Count and length of the intervals per type code: all subjects, actors, and real exam takers."""
    out = []
    for group, part in (("all", valid), ("actor", valid[~valid.real]), ("real", valid[valid.real])):
        g = part.groupby("code").dur_s.agg(
            instances="size", seconds="sum", min_s="min", median_s="median", max_s="max"
        )
        g.insert(1, "subjects", part.groupby("code").subject.nunique())
        g["share_instances_pct"] = 100 * g.instances / g.instances.sum()
        g["share_seconds_pct"] = 100 * g.seconds / g.seconds.sum()
        out.append(g.reset_index().assign(group=group))
    return pd.concat(out, ignore_index=True)


def overlap_pairs(valid):
    """List each pair of intervals of one subject that share time. Touching intervals do not count."""
    rows = []
    for s, part in valid.drop_duplicates(KEY).groupby("subject"):
        recs = part.sort_values(["start_s", "end_s"]).to_dict("records")
        for i, a in enumerate(recs):
            for b in recs[i + 1 :]:  # b starts at or after the start of a
                shared = min(a["end_s"], b["end_s"]) - b["start_s"]
                if shared > 0:
                    lo, hi = sorted((a, b), key=lambda r: r["code"])
                    rows.append(
                        {
                            "subject": s,
                            "code_a": lo["code"],
                            "code_b": hi["code"],
                            "shared_s": shared,
                            "line_a": lo["line"],
                            "raw_a": lo["raw"],
                            "line_b": hi["line"],
                            "raw_b": hi["raw"],
                        }
                    )
    return pd.DataFrame(rows, columns=["subject", "code_a", "code_b", "shared_s", "line_a", "raw_a", "line_b", "raw_b"])


def cheat_seconds(part):
    """Number of seconds of one subject that at least one interval covers."""
    return len(set().union(*(range(a, b) for a, b in zip(part.start_s, part.end_s))))


def per_subject(labels, valid, videos):
    t = videos.copy()
    t.insert(0, "group", ["real" if r else "actor" for r in labels.groupby("subject").real.first()])
    t["rows"] = labels.groupby("subject").size()
    t["invalid_rows"] = (labels.problem != "").groupby(labels.subject).sum()
    counts = valid.groupby(["subject", "code"]).size().unstack(fill_value=0).add_prefix("code_")
    t = t.join(counts)
    t["cheat_s"] = valid.groupby("subject")[["start_s", "end_s"]].apply(cheat_seconds)
    t["cheat_share_pct"] = 100 * t.cheat_s / t.duration_s
    t["last_end_s"] = valid.groupby("subject").end_s.max()
    t["last_end_in_video"] = t.last_end_s <= t.duration_s
    # Test of the other reading: if a time were plain seconds, 0935 would mean 935 s, not 575 s.
    t["last_end_in_video_if_plain_seconds"] = (t.last_end_s // 60 * 100 + t.last_end_s % 60) <= t.duration_s
    return t


def md(df):
    """Format a table as Markdown, so that no number in a document is typed by hand."""
    text = df.round(1).astype(str)
    lines = ["| " + " | ".join(text.columns) + " |", "|" + " --- |" * len(text.columns)]
    lines += ["| " + " | ".join(row) + " |" for row in text.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def write_summary_md(summary, t):
    codes, subj = t["per_code"], t["per_subject"].set_index("subject")
    paper = pd.DataFrame(
        {
            "paper type": range(1, 6),
            "share_instances_pct": PAPER["share_instances_pct"],
            "share_seconds_pct": PAPER["share_seconds_pct"],
        }
    )
    facts = pd.DataFrame(
        [(k, round(v, 1) if isinstance(v, float) else v) for k, v in summary.items() if not isinstance(v, dict)],
        columns=["item", "value"],
    )
    videos = subj.groupby(["width", "height", "fps", "codec", "last_frame_readable"]).size().rename("subjects")
    code_cols = [c for c in subj.columns if c.startswith("code_")]
    shown = subj.assign(duration_min=subj.duration_s / 60).reset_index()[
        ["subject", "group", "duration_min", "rows", "invalid_rows", *code_cols, "cheat_s", "cheat_share_pct", "last_end_s", "last_end_in_video", "last_end_in_video_if_plain_seconds"]
    ]
    parts = [
        "# OEP EDA summary\n",
        f"Generated by `python -m proctor_assist.eda` at git `{summary['run']['git_sha']}` "
        f"(uncommitted changes: {summary['run']['git_dirty']}). Do not edit by hand.\n",
        "The length of an interval is end - start, in seconds. `code` is the number in `gt.txt`. "
        "Its meaning is in `docs/decisions.md` (D12).\n",
        "## Rows in gt.txt and totals\n",
        md(facts),
        "## Per type code, all subjects\n",
        md(codes[codes.group == "all"].drop(columns="group")),
        "## Reference: paper Fig. 9, by paper type (the paper's numbers, not our results)\n",
        f"The paper reports {PAPER['instances']} instances and {PAPER['cheat_seconds']} seconds of cheating.\n",
        md(paper),
        "## Per type code, actors and real exam takers\n",
        md(codes[codes.group != "all"][["group", *codes.columns.drop("group")]]),
        "## Intervals that share time, by code pair (exact duplicate rows removed first)\n",
        md(t["overlaps_by_code"]),
        "## Rows that break the format\n",
        md(t["invalid_rows"]),
        "## Exact duplicate rows\n",
        md(t["duplicate_rows"]),
        "## Longest intervals\n",
        md(t["longest_intervals"]),
        "## Rows that share time with two or more rows of the same code\n",
        md(t["rows_in_two_or_more_same_code_overlaps"]),
        "## Videos\n",
        md(videos.reset_index()),
        "## Per subject\n",
        md(shown),
    ]
    (OUT / "summary.md").write_text("\n".join(parts), encoding="utf-8", newline="\n")


def figures(valid, codes, subj, colors):
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
        }
    )
    names = [f"code {c}" for c in colors]
    n_real = int((subj.group == "real").sum())
    groups = (("actor", PALETTE[0], f"actors ({len(subj) - n_real} subjects)"), ("real", PALETTE[1], f"real exam takers ({n_real} subjects)"))

    # 1. Intervals and seconds per type code, actors next to real exam takers.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(axes, ("instances", "seconds"), ("Intervals per type code", "Labeled seconds per type code")):
        for k, (group, color, label) in enumerate(groups):
            vals = codes[codes.group == group].set_index("code")[col].reindex(list(colors), fill_value=0)
            bars = ax.bar([i + (k - 0.5) * 0.4 for i in range(len(vals))], vals, width=0.38, color=color, label=label)
            ax.bar_label(bars, fmt="%d", fontsize=8, color=MUTED, padding=2)
        ax.set_xticks(range(len(names)), names)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.1))
    fig.savefig(OUT / "code_counts.png")
    plt.close(fig)

    # 2. Timeline: one row per subject, one lane per type code inside the row.
    order = subj.sort_values("group", kind="stable").index
    pos = {s: y for y, s in enumerate(order)}
    lane_h = 0.8 / len(colors)
    fig, ax = plt.subplots(figsize=(11, 13))
    ax.barh(list(pos.values()), subj.duration_s[order] / 60, height=0.86, color="#efeee9", label="video length")
    for lane, (c, color) in enumerate(colors.items()):
        part = valid[valid.code == c]
        ax.barh(
            part.subject.map(pos) - 0.4 + (lane + 0.5) * lane_h,
            part.dur_s / 60,
            left=part.start_s / 60,
            height=lane_h,
            color=color,
            label=f"code {c}",
        )
    ax.set_yticks(list(pos.values()), [f"subject{s}" + (" (real)" if subj.group[s] == "real" else "") for s in order])
    ax.invert_yaxis()
    ax.margins(y=0.01)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("minutes from the start of the video")
    ax.set_title("Labeled intervals per subject (lanes inside a row: one per type code)", loc="left")
    ax.legend(ncols=len(colors) + 1, loc="lower left", bbox_to_anchor=(0, 1.03))
    fig.savefig(OUT / "timeline.png")
    plt.close(fig)

    # 3. Interval length per type code.
    fig, ax = plt.subplots(figsize=(7, 4))
    boxes = ax.boxplot(
        [valid.dur_s[valid.code == c] for c in colors],
        positions=range(len(colors)),
        patch_artist=True,
        medianprops={"color": INK},
        flierprops={"markersize": 3, "markeredgecolor": MUTED},
    )
    for patch, color in zip(boxes["boxes"], colors.values()):
        patch.set(facecolor=color, edgecolor=color)
    ax.set_xticks(range(len(names)), names)
    ax.set_yscale("log")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("interval length in seconds (log scale)")
    ax.set_title("Length of the labeled intervals per type code", loc="left")
    fig.savefig(OUT / "interval_lengths.png")
    plt.close(fig)


def contact_sheets(valid, n=8, seed=0):
    """Save one image per type code with the middle frame of n intervals, from as many subjects as possible."""
    out = PRIVATE / "frames"
    out.mkdir(parents=True, exist_ok=True)
    for c, part in valid.groupby("code"):
        part = part.sample(frac=1, random_state=seed)
        pick = part.assign(k=part.groupby("subject").cumcount()).sort_values("k", kind="stable").head(n)
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        for ax in axes.flat:
            ax.axis("off")
        for ax, r in zip(axes.flat, pick.itertuples()):
            t = (r.start_s + r.end_s) / 2
            cap = cv2.VideoCapture(str(OEP / f"subject{r.subject}" / "webcam.avi"))
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            cap.release()
            if ok:
                ax.imshow(frame[:, :, ::-1])  # BGR to RGB
            ax.set_title(f"subject{r.subject}, line {r.line}, frame at {int(t) // 60:02d}:{int(t) % 60:02d}", fontsize=9)
        fig.suptitle(f"Type code {c}: middle frame of {len(pick)} intervals")
        fig.savefig(out / f"code{c}.jpg", dpi=100)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    labels = load_labels()
    valid = labels[labels.problem == ""].astype({"start_s": int, "end_s": int, "code": int})
    valid["dur_s"] = valid.end_s - valid.start_s
    colors = dict(zip(sorted(map(int, valid.code.unique())), PALETTE, strict=True))

    codes = per_code(valid)
    subj = per_subject(labels, valid, video_table())
    pairs = overlap_pairs(valid)
    pairs_by_code = (
        pairs.groupby(["code_a", "code_b"])
        .agg(pairs=("shared_s", "size"), subjects=("subject", "nunique"), shared_s=("shared_s", "sum"))
        .reset_index()
    )
    # A row that shares time with two or more rows of the same code can be a typing error.
    same = pairs[pairs.code_a == pairs.code_b]
    names = ["subject", "line", "raw"]
    in_same = pd.concat([same[["subject", f"line_{x}", f"raw_{x}"]].set_axis(names, axis=1) for x in "ab"])
    in_same = in_same.assign(raw=in_same.raw.map(repr)).value_counts().rename("same_code_pairs").reset_index()
    shown = labels.assign(raw=labels.raw.map(repr), dur_s=valid.dur_s)  # repr shows the tabs of a row
    tables = {
        "per_code": codes,
        "per_subject": subj.reset_index(),
        "overlaps_by_code": pairs_by_code,
        "invalid_rows": shown[shown.problem != ""][["subject", "line", "raw", "problem"]],
        "duplicate_rows": shown.loc[valid.index[valid.duplicated(KEY, keep=False)], ["subject", "line", "raw"]],
        "longest_intervals": shown.nlargest(8, "dur_s")[["subject", "line", "raw", "dur_s"]],
        "rows_in_two_or_more_same_code_overlaps": in_same[in_same.same_code_pairs >= 2],
    }
    info = env_info()
    summary = {
        "rows": len(labels),
        "valid_rows": len(valid),
        "invalid_rows": len(tables["invalid_rows"]),
        "codes": sorted(colors),
        "exact_duplicate_rows": int(valid.duplicated(KEY).sum()),
        "rows_that_start_before_the_row_above": int((valid.groupby("subject").start_s.diff() < 0).sum()),
        "interval_seconds_end_minus_start": int(valid.dur_s.sum()),
        "interval_seconds_end_second_included": int(valid.dur_s.sum() + len(valid)),
        "cheat_seconds_without_double_count": int(subj.cheat_s.sum()),
        "video_seconds": float(subj.duration_s.sum()),
        "cheat_share_pct": float(100 * subj.cheat_s.sum() / subj.duration_s.sum()),
        "mean_video_minutes": float(subj.duration_s.mean() / 60),
        "overlapping_pairs": len(pairs),
        "overlapping_pairs_of_different_codes": int((pairs.code_a != pairs.code_b).sum()),
        "subjects_with_last_end_in_video": int(subj.last_end_in_video.sum()),
        "subjects_with_last_end_in_video_if_plain_seconds": int(subj.last_end_in_video_if_plain_seconds.sum()),
        "paper": PAPER,
        "run": {k: info[k] for k in ("time_utc", "git_sha", "git_dirty", "python")}
        | {"opencv": cv2.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
    }

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", newline="\n")
    for name, table in tables.items():
        table.to_csv(OUT / f"{name}.csv", index=False, float_format="%.3f", lineterminator="\n")
    write_summary_md(summary, tables)
    pairs.to_csv(PRIVATE / "overlap_pairs.csv", index=False, lineterminator="\n")
    figures(valid, codes, subj, colors)
    contact_sheets(valid)
    print(f"Wrote tables and figures to {OUT}, and private files to {PRIVATE}.")


if __name__ == "__main__":
    main()

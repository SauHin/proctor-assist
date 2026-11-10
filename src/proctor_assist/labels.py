"""Read the OEP ground-truth files (gt.txt): one cheating interval per line.

Line format: start<TAB>end<TAB>type code. A time is MMSS from the start of the video, with a
resolution of one second. docs/decisions.md (D12) explains the type codes.
"""
import re

import pandas as pd

from proctor_assist.env import ROOT

OEP = ROOT / "data" / "raw" / "oep"
SUBJECTS = range(1, 25)
REAL = frozenset({10, 11, 12, 13, 14, 15, 16, 18, 19})  # real exam takers (READ_ME.txt). The others are actors.
_MMSS = re.compile(r"[0-9]{2}[0-5][0-9]")


def parse_gt(text):
    """Parse the text of one gt.txt into rows: line, raw, start_s, end_s, code, problem.

    Spaces and tabs around a field are ignored. A line that breaks the format stays in the output
    with the reason in `problem` and without values. The parser never repairs a label, because
    labels are an owner decision (CLAUDE.md).
    """
    rows = []
    for n, raw in enumerate(text.splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        row = {"line": n, "raw": raw, "start_s": None, "end_s": None, "code": None, "problem": ""}
        if len(parts) != 3:
            row["problem"] = f"{len(parts)} fields, expected 3"
        elif not (_MMSS.fullmatch(parts[0]) and _MMSS.fullmatch(parts[1])):
            row["problem"] = "time is not MMSS"
        elif not parts[2].isdigit():
            row["problem"] = "type code is not a number"
        else:
            start, end = (int(p[:2]) * 60 + int(p[2:]) for p in parts[:2])
            if end < start:
                row["problem"] = "end before start"
            else:
                row.update(start_s=start, end_s=end, code=int(parts[2]))
        rows.append(row)
    return rows


def load_labels(root=OEP):
    """Return the gt.txt rows of all 24 subjects as one table, with `subject` and `real` columns."""
    tables = [
        pd.DataFrame(parse_gt((root / f"subject{s}" / "gt.txt").read_text(encoding="latin-1"))).assign(
            subject=s, real=s in REAL
        )
        for s in SUBJECTS
    ]
    return pd.concat(tables, ignore_index=True).astype({c: "Int64" for c in ("start_s", "end_s", "code")})

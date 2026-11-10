"""Read the OEP ground-truth files (gt.txt): one cheating interval per line.

Line format: start<TAB>end<TAB>type code. A time is MMSS from the start of the video, with a
resolution of one second. docs/decisions.md (D12) explains the type codes.
"""
import re

import numpy as np
import pandas as pd

from proctor_assist.env import ROOT

OEP = ROOT / "data" / "raw" / "oep"
SUBJECTS = range(1, 25)
REAL = frozenset({10, 11, 12, 13, 14, 15, 16, 18, 19})  # real exam takers (READ_ME.txt). The others are actors.
_MMSS = re.compile(r"[0-9]{2}[0-5][0-9]")

# Class mapping and label cleaning. The owner decided every value below (docs/decisions.md, D14).
# Do not change a value here without a new owner decision.
CLASSES = {0: "no cheat", 1: "looks at an object", 2: "talks to a person", 3: "phone call"}
CODE_TO_CLASS = {1: 1, 6: 1, 2: 2, 5: 3}
MASKED_CODES = frozenset({3})  # Internet use: not visible in the webcam
PRIORITY = (3, 1, 2)  # where classes share a second, the first class in this list wins
MASK = -1  # a masked second is left out of training and evaluation
# (subject, line): (row in gt.txt, row that the code reads instead). The raw files are never edited (D11).
FIXES = {(7, 14): ("0900\t00903\t6", "0900\t0903\t6"), (1, 6): ("0230\t0336\t2", "0230\t0236\t2")}
_KEY = ["subject", "start_s", "end_s", "code"]


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


def clean_labels(labels):
    """Apply the label decisions of the owner (D14) to the table from load_labels.

    The result has the corrected rows, one copy of each duplicate row, and no row with a problem.
    """
    out = labels.copy()
    for (subject, line), (old, new) in FIXES.items():
        hit = (out.subject == subject) & (out.line == line)
        if out.raw[hit].tolist() != [old]:
            raise ValueError(f"subject{subject} line {line} is not {old!r}. FIXES does not match the data.")
        (row,) = parse_gt(new)
        for col in ("start_s", "end_s", "code", "problem"):
            out.loc[hit, col] = row[col]
    if (out.problem != "").any():
        raise ValueError("A label row has a problem, and no owner decision covers it.")
    out = out.drop_duplicates(_KEY).reset_index(drop=True)
    return out.astype({"start_s": int, "end_s": int, "code": int})


def second_labels(rows, n_seconds):
    """Give each second of one video its class (D14): 0, a cheating class, or MASK.

    `rows` are the cleaned rows of one subject. An interval is [start, end).
    """
    spans = list(zip(rows.start_s, rows.end_s, rows.code))
    unknown = {c for _, _, c in spans} - CODE_TO_CLASS.keys() - MASKED_CODES
    if unknown:
        raise ValueError(f"Type codes without an owner decision: {sorted(unknown)}")
    out = np.zeros(n_seconds, dtype=np.int8)
    for start, end, code in spans:
        if code in MASKED_CODES:
            out[start:end] = MASK
    # A later write replaces an earlier one. A visible class replaces the mask, and the first class
    # in PRIORITY is written last.
    for cls in reversed(PRIORITY):
        for start, end, code in spans:
            if CODE_TO_CLASS.get(code) == cls:
                out[start:end] = cls
    return out

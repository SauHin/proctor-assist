import pandas as pd
import pytest

from proctor_assist import labels


def test_parse_gt_reads_mmss_and_ignores_whitespace():
    # CRLF, a trailing tab, a space after a field, and no newline at the end, as in the real files.
    rows = labels.parse_gt("0135\t0204\t1\r\n0030\t0038 \t2\r\n0105\t0115\t3\t\r\n\r\n1024\t1054\t3")
    assert [(r["line"], r["start_s"], r["end_s"], r["code"], r["problem"]) for r in rows] == [
        (1, 95, 124, 1, ""),
        (2, 30, 38, 2, ""),
        (3, 65, 75, 3, ""),
        (5, 624, 654, 3, ""),
    ]


@pytest.mark.parametrize(
    "line, problem",
    [
        ("0900\t00903\t6", "time is not MMSS"),  # five digits
        ("0160\t0200\t1", "time is not MMSS"),  # 60 seconds
        ("0200\t0130\t1", "end before start"),
        ("0100\t0130", "2 fields, expected 3"),
        ("0100\t0130\tx", "type code is not a number"),
    ],
)
def test_parse_gt_flags_a_bad_line_and_does_not_repair_it(line, problem):
    (row,) = labels.parse_gt(line)
    assert row["problem"] == problem
    assert row["raw"] == line
    assert row["start_s"] is None and row["end_s"] is None and row["code"] is None


def _rows(*spans):
    return pd.DataFrame(spans, columns=["start_s", "end_s", "code"])


def test_second_labels_follow_the_owner_decisions_d14():
    M = labels.MASK
    # Code 6 joins class 1, code 5 is class 3, and the end second is not part of the interval.
    assert labels.second_labels(_rows((1, 3, 6), (4, 5, 5)), 6).tolist() == [0, 1, 1, 0, 3, 0]
    # Class 1 wins over class 2, and class 3 wins over class 1.
    assert labels.second_labels(_rows((0, 4, 2), (1, 2, 1), (2, 3, 5), (2, 3, 6)), 4).tolist() == [2, 1, 3, 2]
    # Code 3 alone is masked. A visible code wins over the mask. A row past the end is cut.
    assert labels.second_labels(_rows((0, 3, 3), (2, 9, 2)), 4).tolist() == [M, M, 2, 2]
    with pytest.raises(ValueError, match="4"):
        labels.second_labels(_rows((0, 1, 4)), 2)


def test_clean_labels_applies_the_fixes_of_d14():
    if not labels.OEP.exists():
        pytest.skip("OEP data is not downloaded")
    raw = labels.load_labels()
    df = labels.clean_labels(raw)
    row = lambda s, line: df[(df.subject == s) & (df.line == line)][["start_s", "end_s", "code"]].values.tolist()
    assert row(7, 14) == [[540, 543, 6]]  # 0900 0903 6
    assert row(1, 6) == [[150, 156, 2]]  # 0230 0236 2
    assert row(8, 3) == [[72, 619, 3]]  # 0112 1019 3, kept as it is
    assert (df.problem == "").all() and not df.duplicated(["subject", "start_s", "end_s", "code"]).any()
    assert len(df) == len(raw) - 3  # three duplicate rows of subject7


def test_clean_labels_fails_if_a_fix_does_not_match_the_data():
    table = pd.DataFrame([{"subject": 7, "line": 14, "raw": "0900\t0903\t6", "start_s": 540, "end_s": 543, "code": 6, "problem": ""}])
    with pytest.raises(ValueError, match="FIXES"):
        labels.clean_labels(table)


def test_load_labels_covers_all_subjects():
    if not labels.OEP.exists():
        pytest.skip("OEP data is not downloaded")
    df = labels.load_labels()
    assert set(df.subject) == set(labels.SUBJECTS)
    assert df[df.real].subject.nunique() == 9

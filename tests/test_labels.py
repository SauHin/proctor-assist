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


def test_load_labels_covers_all_subjects():
    if not labels.OEP.exists():
        pytest.skip("OEP data is not downloaded")
    df = labels.load_labels()
    assert set(df.subject) == set(labels.SUBJECTS)
    assert df[df.real].subject.nunique() == 9

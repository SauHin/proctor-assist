import pandas as pd

from proctor_assist import eda


def test_overlap_pairs_and_cheat_seconds_on_toy_rows():
    rows = pd.DataFrame(
        [(1, 1, 10, 20, 1), (1, 2, 20, 30, 1), (1, 3, 25, 40, 2), (1, 4, 25, 40, 2), (2, 1, 0, 5, 3)],
        columns=["subject", "line", "start_s", "end_s", "code"],
    ).assign(raw="")
    pairs = eda.overlap_pairs(rows)
    # Lines 1 and 2 only touch. Line 4 repeats line 3 and is dropped. Lines 2 and 3 share 25 to 30.
    assert pairs[["subject", "code_a", "code_b", "shared_s", "line_a", "line_b"]].values.tolist() == [[1, 1, 2, 5, 2, 3]]
    assert eda.cheat_seconds(rows[rows.subject == 1]) == 30  # 10 to 40, each second counted once

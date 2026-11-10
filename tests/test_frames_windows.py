import cv2
import numpy as np
import pandas as pd

from proctor_assist import frames, labels, splits, windows


def test_extract_keeps_every_fifth_frame_at_the_stored_size(tmp_path):
    # A made-up 25 fps video of 12 frames. Each frame has one gray level, so the kept frames are known.
    video = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (640, 480))
    for i in range(12):
        writer.write(np.full((480, 640, 3), i * 20, np.uint8))
    writer.release()

    got = frames.extract(video, tmp_path / "out")
    assert (got["frames_decoded"], got["frames_saved"]) == (12, 3)  # frames 0, 5, and 10
    saved = [cv2.imread(str(tmp_path / "out" / f"{k:06d}.jpg")) for k in range(3)]
    assert all(img.shape == (224, 224, 3) for img in saved)
    assert [round(img.mean() / 20) for img in saved] == [0, 5, 10]


def test_frame_ids_of_a_window():
    assert windows.frame_ids(2) == range(0, 15)  # seconds 0, 1, 2 at 5 fps
    assert windows.frame_ids(5, length_s=1) == range(25, 30)


def test_window_table_uses_the_last_second_and_skips_masked_targets():
    rows = pd.DataFrame(
        [(1, 3, 5, 2), (1, 5, 7, 3), (2, 0, 9, 1)], columns=["subject", "start_s", "end_s", "code"]
    )
    table = windows.window_table([1], rows, {1: 8, 2: 9})
    # Subject 1: seconds 0 to 7. Seconds 3 and 4 are class 2, and seconds 5 and 6 are masked (code 3).
    assert table.subject.unique().tolist() == [1]  # subject 2 was not asked for
    assert table.t.tolist() == [2, 3, 4, 7]  # no full window before second 2, no masked target
    assert table.target.tolist() == [0, 2, 2, 0]


def test_windows_of_a_split_part_come_from_its_subjects_only():
    empty = pd.DataFrame(columns=["subject", "start_s", "end_s", "code"])
    n_seconds = {s: 5 for s in labels.SUBJECTS}
    s = splits.load_splits()
    for split in [s["protocol_a"], *s["protocol_b"]]:
        seen = {part: set(windows.window_table(split[part], empty, n_seconds).subject) for part in split}
        assert all(seen[part] == set(split[part]) for part in split)
        assert not (seen["train"] & seen["val"]) and not (seen["train"] & seen["test"]) and not (seen["val"] & seen["test"])

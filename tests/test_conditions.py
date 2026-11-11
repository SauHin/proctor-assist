import numpy as np
import pandas as pd

from proctor_assist import conditions

# The frames in this file are made up for the tests. They are not data.


def test_there_are_4_kinds_at_2_levels():
    names = [n for n, _, _ in conditions.named()]
    assert len(names) == len(set(names)) == 8
    assert {"low_light_moderate", "blur_strong", "jpeg_moderate", "fps_strong"} <= set(names)


def test_held_frames_follow_the_camera_frame_rate():
    k = np.arange(10)
    assert conditions.held(k, 2.5).tolist() == [0, 0, 2, 2, 4, 4, 6, 6, 8, 8]
    assert conditions.held(k, 1).tolist() == [0] * 5 + [5] * 5
    assert conditions.held(k, 5).tolist() == k.tolist()  # the condition "none"


def test_image_conditions_keep_the_size_and_change_the_frame():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    dark = conditions.change(np.full((4, 4, 3), 200, np.uint8), "low_light", 0.25)
    assert (dark == 50).all()
    for kind, value in [("blur", 3), ("jpeg", 7)]:
        out = conditions.change(frame, kind, value)
        assert out.shape == frame.shape and out.dtype == np.uint8 and not (out == frame).all()
    assert conditions.change(frame, "blur", 3).std() < frame.std() / 2


def test_r0_scores_use_the_last_frame_of_each_second():
    turned = np.eye(4)
    c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
    turned[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]  # a head turn of 40 degrees
    m = np.stack([np.eye(4)] * 25)  # 5 seconds. The normal position is the median of the seconds so far.
    m[19] = turned  # second 3 ends with frame 19
    m[20] = turned  # frame 20 is not the last frame of a second
    p0 = conditions.r0_p0(pd.DataFrame({"subject": 1, "t": [2, 3, 4]}), {1: m})
    assert p0.tolist() == [1.0, 0.0, 1.0]

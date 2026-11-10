import math

import numpy as np
import pandas as pd
import pytest

from proctor_assist import r0


def _rotation(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    turn = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    nod = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return turn @ nod


def test_head_pose_reads_yaw_and_pitch_from_a_rotation():
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation(20, -10)
    assert r0.head_pose(matrix) == pytest.approx((20, -10))


def test_suspicion_uses_the_own_median_and_keeps_the_last_score_without_a_face():
    pose = pd.DataFrame(
        {
            "t": range(7),
            "face": [False, True, True, True, False, True, True],
            "yaw": [None, 10, 10, 25, None, 10, 70],
            "pitch": [None, 5, 5, 5, None, 5, 5],
        }
    )
    score = r0.suspicion(pose)
    assert score[0] == 0  # no face yet
    assert score[1] == 0 and score[2] == 0  # at the normal position
    assert score[3] == pytest.approx(15 / 30)  # the median of 10, 10, 25 is 10, so the turn is 15 degrees
    assert score[4] == score[3]  # no face: the last score stays
    assert score[5] == 0 and score[6] == 1  # back to normal, then a turn of 60 degrees is cut at 1
